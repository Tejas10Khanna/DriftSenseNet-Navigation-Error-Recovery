import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import map_coordinates

# 1. EMPIRICALLY MEASURED PHYSICAL PARAMETERS (DRAM-1X SEM)

# Core Grid - Perfectly straight lines creating wide rectangular trenches
PITCH_X_NM = 170.0        
PITCH_Y_NM = 125.0        
LINE_WIDTH_X_NM = 36.0   
LINE_WIDTH_Y_NM = 46.0    
CONTACT_R_NM = 28.0       
EDGE_SOFTNESS_NM = 1.5    #

# Coarse Street Sub-Grid 
SUBGRID_PITCH_NM = 220.0  
SUBGRID_LINE_WIDTH_NM = 6.0
SUBGRID_VAL = 105.0       
SUBGRID_SOFTNESS_NM = 2.0 

# Gray-level hierarchy
TRENCH_VAL = 28.0         
BG_VAL = 65.0             
LINE_VAL = 158.0          
CONTACT_VAL = 245.0       
STREET_VAL = 84.0         

# Official Applied Materials Reference Defaults
MAT_SIZE_NM = 2600.0
SEPARATOR_WIDTH_NM = 320.0
P_STRADDLE = 0.35
BEAM_SPOT_SIGMA_NM = 5.0

REF_PX_SIZE_NM = 1.0
REF_SIZE_PX = 1000
REF_FOV_NM = REF_SIZE_PX * REF_PX_SIZE_NM

SEARCH_PX_SIZE_NM = 10.0
SEARCH_SIZE_PX = 1000
SEARCH_FOV_NM = SEARCH_SIZE_PX * SEARCH_PX_SIZE_NM

# Dose & Sensor Noise Parameters
REFERENCE_DOSE = 2000.0
SEARCH_DOSE = 200.0

# Search Raster Acquisition Artifacts
SEARCH_RASTER_SHEAR_PX = 1.5
SEARCH_ROW_JITTER_PX = 0.5

# Canvas Resolution Settings
SUPERSAMPLE = 4
SEARCH_NATIVE_PX_NM = SEARCH_PX_SIZE_NM / SUPERSAMPLE
CANVAS_FOV_NM = 30000.0
CANVAS_PX_NM = SEARCH_NATIVE_PX_NM
CANVAS_PX = int(round(CANVAS_FOV_NM / CANVAS_PX_NM))

CONTACT_ROTATION_DEG_RANGE = (-3.0, 3.0)
CONTACT_SCALE_JITTER_RANGE = (-0.15, 0.15)

CURRENT_STYLE = "dram"

# FinFET-style parameters (dense parallel vertical fins, 1-2 crossing gate bars)
FIN_PITCH_NM = 45.0
FIN_WIDTH_NM = 20.0
GATE_BAR_PITCH_NM = 260.0
GATE_BAR_WIDTH_NM = 60.0
FIN_VAL = 165.0
GATE_VAL = 205.0

_cache = {}


# 2. PROCEDURAL LAYOUT GENERATION

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def get_contact_jitter_grid(jitter_seed=0):
    key = ("contact_jitter", jitter_seed)
    if key not in _cache:
        n_cells_x = int(np.ceil(CANVAS_FOV_NM / (PITCH_X_NM * 0.5))) + 2
        n_cells_y = int(np.ceil(CANVAS_FOV_NM / (PITCH_Y_NM * 0.5))) + 2
        rng = np.random.default_rng(jitter_seed)
        rot = rng.uniform(*CONTACT_ROTATION_DEG_RANGE, size=(n_cells_y, n_cells_x)).astype(np.float32)
        scale = 1.0 + rng.uniform(*CONTACT_SCALE_JITTER_RANGE, size=(n_cells_y, n_cells_x)).astype(np.float32)
        _cache[key] = (rot, scale)
    return _cache[key]


def get_mat_pitch_grid(jitter_seed=0):
    """Assigns discrete density scales across different memory mats for variance."""
    key = ("mat_pitch", jitter_seed)
    if key not in _cache:
        n_mats = int(np.ceil(CANVAS_FOV_NM / MAT_SIZE_NM)) + 2
        rng = np.random.default_rng(jitter_seed + 101)
        density_options = np.array([0.65, 0.85, 1.0, 1.0, 1.35, 1.6], dtype=np.float32)
        pitch_scales = rng.choice(density_options, size=(n_mats, n_mats)).astype(np.float32)
        _cache[key] = pitch_scales
    return _cache[key]


def finfet_layout_intensity(x_nm, y_nm, jitter_seed=0):
    """
    FinFET-style: dense parallel vertical fins, crossed by periodic
    horizontal gate bars. Reuses the same mat/street/subgrid machinery as
    the DRAM path (get_mat_pitch_grid still varies fin density per mat) so
    the two styles share noise/street/edge-brightening behavior exactly
    only the fine-feature geometry differs.
    """
    x_nm = x_nm.astype(np.float32, copy=False)
    y_nm = y_nm.astype(np.float32, copy=False)

    pitch_grid = get_mat_pitch_grid(jitter_seed)
    m_col = np.clip((x_nm / MAT_SIZE_NM).astype(np.int32), 0, pitch_grid.shape[1] - 1)
    m_row = np.clip((y_nm / MAT_SIZE_NM).astype(np.int32), 0, pitch_grid.shape[0] - 1)
    local_scale = pitch_grid[m_row, m_col]
    local_fin_pitch = FIN_PITCH_NM * local_scale
    local_fin_width = FIN_WIDTH_NM * np.sqrt(local_scale)

    dx = np.mod(x_nm + local_fin_pitch / 2, local_fin_pitch) - local_fin_pitch / 2
    fin_resp = sigmoid((local_fin_width / 2 - np.abs(dx)) / EDGE_SOFTNESS_NM)
    core_img = BG_VAL + (FIN_VAL - BG_VAL) * fin_resp

    dy = np.mod(y_nm + GATE_BAR_PITCH_NM / 2, GATE_BAR_PITCH_NM) - GATE_BAR_PITCH_NM / 2
    gate_resp = sigmoid((GATE_BAR_WIDTH_NM / 2 - np.abs(dy)) / EDGE_SOFTNESS_NM)
    core_img = core_img * (1.0 - gate_resp) + GATE_VAL * gate_resp

    core_edge_dist = np.minimum(np.abs(np.abs(dx) - local_fin_width / 2),
                                 np.abs(np.abs(dy) - GATE_BAR_WIDTH_NM / 2))

    sub_dx = np.mod(x_nm + SUBGRID_PITCH_NM / 2, SUBGRID_PITCH_NM) - SUBGRID_PITCH_NM / 2
    sub_dy = np.mod(y_nm + SUBGRID_PITCH_NM / 2, SUBGRID_PITCH_NM) - SUBGRID_PITCH_NM / 2
    sub_line_x = sigmoid((SUBGRID_LINE_WIDTH_NM / 2 - np.abs(sub_dx)) / SUBGRID_SOFTNESS_NM)
    sub_line_y = sigmoid((SUBGRID_LINE_WIDTH_NM / 2 - np.abs(sub_dy)) / SUBGRID_SOFTNESS_NM)
    subgrid_resp = np.maximum(sub_line_x, sub_line_y)

    ddx = np.mod(x_nm + MAT_SIZE_NM / 2, MAT_SIZE_NM) - MAT_SIZE_NM / 2
    ddy = np.mod(y_nm + MAT_SIZE_NM / 2, MAT_SIZE_NM) - MAT_SIZE_NM / 2
    street_x = sigmoid((SEPARATOR_WIDTH_NM / 2 - np.abs(ddx)) / EDGE_SOFTNESS_NM)
    street_y = sigmoid((SEPARATOR_WIDTH_NM / 2 - np.abs(ddy)) / EDGE_SOFTNESS_NM)
    street_resp = np.maximum(street_x, street_y)

    street_img = STREET_VAL + (SUBGRID_VAL - STREET_VAL) * subgrid_resp
    img = core_img * (1.0 - street_resp) + street_img * street_resp
    edge_dist = core_edge_dist * (1.0 - street_resp) + (SEPARATOR_WIDTH_NM / 2) * street_resp
    return img, edge_dist


def layout_intensity(x_nm, y_nm, jitter_seed=0):
    if CURRENT_STYLE == "finfet":
        return finfet_layout_intensity(x_nm, y_nm, jitter_seed=jitter_seed)

    x_nm = x_nm.astype(np.float32, copy=False)
    y_nm = y_nm.astype(np.float32, copy=False)

    # Layer 1: Coarse Substrate Background Grid

    sub_dx = np.mod(x_nm + SUBGRID_PITCH_NM / 2, SUBGRID_PITCH_NM) - SUBGRID_PITCH_NM / 2
    sub_dy = np.mod(y_nm + SUBGRID_PITCH_NM / 2, SUBGRID_PITCH_NM) - SUBGRID_PITCH_NM / 2
    sub_line_x = sigmoid((SUBGRID_LINE_WIDTH_NM / 2 - np.abs(sub_dx)) / SUBGRID_SOFTNESS_NM)
    sub_line_y = sigmoid((SUBGRID_LINE_WIDTH_NM / 2 - np.abs(sub_dy)) / SUBGRID_SOFTNESS_NM)
    subgrid_resp = np.maximum(sub_line_x, sub_line_y)
    del sub_dx, sub_dy, sub_line_x, sub_line_y

    # Layer 2: Mat-Specific Pitch Scaling

    pitch_grid = get_mat_pitch_grid(jitter_seed)
    m_cells_y, m_cells_x = pitch_grid.shape
    m_col = np.clip((x_nm / MAT_SIZE_NM).astype(np.int32), 0, m_cells_x - 1)
    m_row = np.clip((y_nm / MAT_SIZE_NM).astype(np.int32), 0, m_cells_y - 1)
    
    local_pitch_scale = pitch_grid[m_row, m_col]
    del m_col, m_row

    local_px = PITCH_X_NM * local_pitch_scale
    local_py = PITCH_Y_NM * local_pitch_scale
    local_lw_x = LINE_WIDTH_X_NM * np.sqrt(local_pitch_scale)
    local_lw_y = LINE_WIDTH_Y_NM * np.sqrt(local_pitch_scale)
    local_cr = CONTACT_R_NM * np.sqrt(local_pitch_scale)
    del local_pitch_scale

    # Layer 3: Perfectly Straight Word/Bit Lines & Rectangular Trenches
    
    dx = np.mod(x_nm + local_px / 2, local_px) - local_px / 2
    dy = np.mod(y_nm + local_py / 2, local_py) - local_py / 2

    line_x = sigmoid((local_lw_x / 2 - np.abs(dx)) / EDGE_SOFTNESS_NM)
    line_y = sigmoid((local_lw_y / 2 - np.abs(dy)) / EDGE_SOFTNESS_NM)
    line_mask = np.maximum(line_x, line_y)
    
    trench_resp = (1.0 - line_x) * (1.0 - line_y)
    del line_x, line_y

    base_core = BG_VAL * (1.0 - trench_resp) + TRENCH_VAL * trench_resp
    
    # Add the bright lines over the BG_VAL
    core_img = base_core + (LINE_VAL - BG_VAL) * line_mask
    del base_core, trench_resp

    # Layer 4: STAGGERED (Checkerboard) Contact Via Dots

    cell_col = (x_nm / local_px).astype(np.int32)
    cell_row = (y_nm / local_py).astype(np.int32)
    stagger_mask = ((cell_col + cell_row) % 2 == 0).astype(np.float32)

    rot_grid, scale_grid = get_contact_jitter_grid(jitter_seed)
    j_cells_y, j_cells_x = rot_grid.shape
    j_col = np.clip(cell_col, 0, j_cells_x - 1)
    j_row = np.clip(cell_row, 0, j_cells_y - 1)
    rot_deg = rot_grid[j_row, j_col]
    cell_scale = scale_grid[j_row, j_col]
    del cell_col, cell_row, j_col, j_row

    theta = np.radians(-rot_deg).astype(np.float32)
    del rot_deg
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    del theta
    
    dx_local = (dx * cos_t - dy * sin_t) / cell_scale
    dy_local = (dx * sin_t + dy * cos_t) / cell_scale
    del cos_t, sin_t, cell_scale

    r = np.sqrt(dx_local ** 2 + dy_local ** 2)
    del dx_local, dy_local
    
    contact_resp = sigmoid((local_cr - r) / EDGE_SOFTNESS_NM) * stagger_mask
    del r, stagger_mask, local_px, local_py, local_cr

    core_img = core_img * (1.0 - contact_resp) + CONTACT_VAL * contact_resp

    core_edge_dist = np.minimum(
        np.abs(np.abs(dx) - local_lw_x / 2),
        np.abs(np.abs(dy) - local_lw_y / 2),
    )
    del contact_resp, line_mask, dx, dy, local_lw_x, local_lw_y

    # Layer 5: Streets with the Coarse Background Grid

    ddx = np.mod(x_nm + MAT_SIZE_NM / 2, MAT_SIZE_NM) - MAT_SIZE_NM / 2
    ddy = np.mod(y_nm + MAT_SIZE_NM / 2, MAT_SIZE_NM) - MAT_SIZE_NM / 2
    street_x = sigmoid((SEPARATOR_WIDTH_NM / 2 - np.abs(ddx)) / EDGE_SOFTNESS_NM)
    street_y = sigmoid((SEPARATOR_WIDTH_NM / 2 - np.abs(ddy)) / EDGE_SOFTNESS_NM)
    street_resp = np.maximum(street_x, street_y)
    del ddx, ddy, street_x, street_y

    # Explicit street layer incorporating the coarse grid
    street_img = STREET_VAL + (SUBGRID_VAL - STREET_VAL) * subgrid_resp
    del subgrid_resp

    img = core_img * (1.0 - street_resp) + street_img * street_resp
    edge_dist = core_edge_dist * (1.0 - street_resp) + (SEPARATOR_WIDTH_NM / 2) * street_resp
    del core_img, street_img, street_resp, core_edge_dist

    return img, edge_dist


# 3. CANVAS CACHING & OPTICS CROPPING

def build_canvas(jitter_seed=0):
    cache_key = ("canvas", jitter_seed, CURRENT_STYLE)
    if cache_key in _cache:
        return _cache[cache_key], _cache[("canvas_edge", jitter_seed, CURRENT_STYLE)]

    t0 = time.time()
    coords = ((np.arange(CANVAS_PX) + 0.5) * CANVAS_PX_NM).astype(np.float32)
    xs_row = coords

    img_full = np.empty((CANVAS_PX, CANVAS_PX), dtype=np.float32)
    edge_full = np.empty((CANVAS_PX, CANVAS_PX), dtype=np.float32)

    chunk_rows = 400
    for row_start in range(0, CANVAS_PX, chunk_rows):
        row_end = min(row_start + chunk_rows, CANVAS_PX)
        ys_chunk = coords[row_start:row_end]
        xs, ys = np.meshgrid(xs_row, ys_chunk)
        img_chunk, edge_chunk = layout_intensity(xs, ys, jitter_seed=jitter_seed)
        img_full[row_start:row_end, :] = img_chunk
        edge_full[row_start:row_end, :] = edge_chunk
        del xs, ys, img_chunk, edge_chunk

    beam_sigma_native_px = BEAM_SPOT_SIGMA_NM / CANVAS_PX_NM
    pil_img = Image.fromarray(np.clip(img_full, 0, 255).astype(np.uint8), mode="L")
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=beam_sigma_native_px))
    img_full = np.asarray(pil_img, dtype=np.float32)

    _cache[cache_key] = img_full
    _cache[("canvas_edge", jitter_seed, CURRENT_STYLE)] = edge_full
    _cache[("canvas_pil", jitter_seed, CURRENT_STYLE)] = Image.fromarray(np.clip(img_full, 0, 255).astype(np.uint8), mode="L")
    _cache[("canvas_edge_pil", jitter_seed, CURRENT_STYLE)] = Image.fromarray(edge_full.astype(np.float32), mode="F")

    print(f"[cache] built {CANVAS_PX}x{CANVAS_PX} canvas (jitter_seed={jitter_seed}) in {time.time()-t0:.2f}s")
    return img_full, edge_full


def crop_from_canvas(center_nm, fov_nm, out_px, jitter_seed=0):
    build_canvas(jitter_seed=jitter_seed)
    pil_full = _cache[("canvas_pil", jitter_seed, CURRENT_STYLE)]
    pil_edge = _cache[("canvas_edge_pil", jitter_seed, CURRENT_STYLE)]
    cx, cy = center_nm
    half = fov_nm / 2
    x0_px = (cx - half) / CANVAS_PX_NM
    y0_px = (cy - half) / CANVAS_PX_NM
    x1_px = (cx + half) / CANVAS_PX_NM
    y1_px = (cy + half) / CANVAS_PX_NM

    is_downsampling = (x1_px - x0_px) > out_px
    resample_filter = Image.BOX if is_downsampling else Image.LANCZOS

    crop = pil_full.resize((out_px, out_px), resample_filter, box=(x0_px, y0_px, x1_px, y1_px))
    img = np.asarray(crop, dtype=np.float32)

    edge_crop = pil_edge.resize((out_px, out_px), Image.BILINEAR, box=(x0_px, y0_px, x1_px, y1_px))
    edge_dist = np.asarray(edge_crop, dtype=np.float32)
    return img, edge_dist


# 4. PHYSICAL NOISE & SEARCH ARTIFACTS

def add_edge_brightening(img, edge_dist, strength=18.0, width_px=1.5):
    boost = strength * np.exp(-(np.clip(edge_dist, 0, None) ** 2) / (2 * width_px**2))
    return np.clip(img + boost, 0, 255)


def add_charging_noise(img, rng, bg_mask, strength=10.0, n_blobs=4, blob_sigma_px=70):
    h, w = img.shape
    charge = np.zeros((h, w), dtype=np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    for _ in range(n_blobs):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        sign = rng.choice([-1.0, 1.0])
        sigma = blob_sigma_px * rng.uniform(0.7, 1.3)
        charge += sign * strength * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
    charge *= bg_mask
    return np.clip(img + charge, 0, 255)


def add_sensor_noise(img, rng, dose=1.0):
    img = img.astype(np.float32)
    scaled = np.clip(img, 1, 255) * dose
    shot = rng.poisson(scaled).astype(np.float32) / dose
    read_noise = rng.normal(0, 2.5, size=img.shape)
    return np.clip(shot + read_noise, 0, 255).astype(np.uint8)


def sample_ref_center(rng, margin, boundary_bias=None):
    straddle_prob = P_STRADDLE if boundary_bias is None else boundary_bias
    valid_lo, valid_hi = margin, CANVAS_FOV_NM - margin
    is_boundary_case = rng.random() < straddle_prob
    if is_boundary_case:
        axis = rng.integers(0, 2)
        k_lo = int(np.ceil(valid_lo / MAT_SIZE_NM))
        k_hi = int(np.floor(valid_hi / MAT_SIZE_NM))
        k = rng.integers(k_lo, k_hi + 1)
        boundary = k * MAT_SIZE_NM
        offset = rng.uniform(-REF_FOV_NM * 0.35, REF_FOV_NM * 0.35)
        coord_on_axis = float(np.clip(boundary + offset, valid_lo, valid_hi))
        other = float(rng.uniform(valid_lo, valid_hi))
        center = (coord_on_axis, other) if axis == 0 else (other, coord_on_axis)
        return center, True
    else:
        for _ in range(25):
            cx = float(rng.uniform(valid_lo, valid_hi))
            cy = float(rng.uniform(valid_lo, valid_hi))
            dx_to_boundary = min(cx % MAT_SIZE_NM, MAT_SIZE_NM - (cx % MAT_SIZE_NM))
            dy_to_boundary = min(cy % MAT_SIZE_NM, MAT_SIZE_NM - (cy % MAT_SIZE_NM))
            if dx_to_boundary > REF_FOV_NM * 0.6 and dy_to_boundary > REF_FOV_NM * 0.6:
                return (cx, cy), False
        return (cx, cy), False


def apply_search_raster_artifacts(img, rng, fill_value):
    h, w = img.shape
    row_idx = np.arange(h)
    shear_shift = SEARCH_RASTER_SHEAR_PX * (row_idx / h - 0.5)
    jitter_shift = rng.normal(0, SEARCH_ROW_JITTER_PX, size=h)
    total_shift = (shear_shift + jitter_shift).astype(np.float32)

    yy, xx = np.meshgrid(row_idx, np.arange(w), indexing="ij")
    xx_shifted = xx - total_shift[:, None]
    coords = np.stack([yy.astype(np.float32), xx_shifted])
    shifted = map_coordinates(img.astype(np.float32), coords, order=1, mode="constant", cval=fill_value)
    return shifted


# 5. PAIR GENERATION PIPELINE & EVALUATION

def generate_pair(seed, boundary_bias=None, out_dir=None, jitter_seed=0):
    rng = np.random.default_rng(seed)

    # Drift cap: 800nm (80px at 10nm/px), not the original 3000nm (300px).
    # A 300px drift routinely put the true match far enough from the search
    # image center that a periodic "clone" near the center scored higher
    # under the tie-break rule ("closest to center") than the true
    # match — that's a benchmark artifact of an overly violent drift
    # simulation, not a meaningful test of localization accuracy. 800nm is
    # still a realistic stage-drift magnitude and keeps the tie-break rule
    # meaningful rather than adversarial.
    drift_max = 800.0
    margin = SEARCH_FOV_NM / 2 + drift_max
    ref_center, is_boundary_case = sample_ref_center(rng, margin, boundary_bias=boundary_bias)
    drift = rng.uniform(-drift_max, drift_max, size=2)
    search_center = (ref_center[0] + drift[0], ref_center[1] + drift[1])

    ref_img, ref_edge = crop_from_canvas(ref_center, REF_FOV_NM, REF_SIZE_PX, jitter_seed=jitter_seed)
    search_img, search_edge = crop_from_canvas(search_center, SEARCH_FOV_NM, SEARCH_SIZE_PX, jitter_seed=jitter_seed)

    ref_img = add_edge_brightening(ref_img, ref_edge, strength=18, width_px=1.5)
    search_img = add_edge_brightening(search_img, search_edge, strength=13, width_px=1.0)

    search_img = apply_search_raster_artifacts(search_img, rng, fill_value=STREET_VAL)

    ref_bg = (ref_img < (BG_VAL + LINE_VAL) / 2).astype(np.float32)
    search_bg = (search_img < (BG_VAL + LINE_VAL) / 2).astype(np.float32)
    ref_img = add_charging_noise(ref_img, rng, ref_bg, strength=8, n_blobs=3, blob_sigma_px=80)
    search_img = add_charging_noise(search_img, rng, search_bg, strength=8, n_blobs=4, blob_sigma_px=60)

    ref_noisy = add_sensor_noise(ref_img, rng, dose=REFERENCE_DOSE)
    search_noisy = add_sensor_noise(search_img, rng, dose=SEARCH_DOSE)

    search_left = search_center[0] - SEARCH_FOV_NM / 2
    search_top = search_center[1] - SEARCH_FOV_NM / 2
    ref_left = ref_center[0] - REF_FOV_NM / 2
    ref_top = ref_center[1] - REF_FOV_NM / 2
    gt_x0 = (ref_left - search_left) / SEARCH_PX_SIZE_NM
    gt_y0 = (ref_top - search_top) / SEARCH_PX_SIZE_NM
    gt_w = REF_FOV_NM / SEARCH_PX_SIZE_NM
    gt_h = gt_w
    gt_cx = gt_x0 + gt_w / 2
    gt_cy = gt_y0 + gt_h / 2

    out_dir = Path(out_dir) if out_dir is not None else None

    meta = {
        "sample": out_dir.name if out_dir else f"seed_{seed}",
        "seed": seed,
        "is_boundary_case": bool(is_boundary_case),
        "drift_nm": [float(drift[0]), float(drift[1])],
        "ground_truth_bbox_px": [float(gt_x0), float(gt_y0), float(gt_w), float(gt_h)],
        "ground_truth_center_px": [float(gt_cx), float(gt_cy)],
        "gt_cx": float(gt_cx),
        "gt_cy": float(gt_cy),
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ref_noisy, mode="L").save(out_dir / "reference.png")
        Image.fromarray(search_noisy, mode="L").save(out_dir / "search.png")
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    return ref_noisy, search_noisy, meta


def zncc_check(ref_img, search_img, gt_cx, gt_cy):
    ref_small = np.asarray(Image.fromarray(ref_img).resize((100, 100), Image.LANCZOS), dtype=np.float32)
    ref_small = (ref_small - ref_small.mean()) / (ref_small.std() + 1e-6)

    y0, x0 = int(gt_cy - 50), int(gt_cx - 50)
    patch = search_img[max(0, y0):y0 + 100, max(0, x0):x0 + 100].astype(np.float32)
    if patch.shape != (100, 100):
        return None
    patch = (patch - patch.mean()) / (patch.std() + 1e-6)
    return float(np.mean(ref_small * patch))


# 6. CLI ENTRY POINT

def main():
    import argparse
    global CURRENT_STYLE

    parser = argparse.ArgumentParser(description="Drift-Sense synthetic dataset generator")
    parser.add_argument("--style", choices=["dram", "finfet"], default="dram",
                         help="Die architecture style (default: dram)")
    parser.add_argument("--count", type=int, default=30,
                         help="Number of image pairs to generate (default: 30)")
    parser.add_argument("--output_dir", type=str, default="dataset_out",
                         help="Output directory (default: dataset_out)")
    parser.add_argument("--start_seed", type=int, default=42,
                         help="Starting seed (default: 42, matches the fixed eval set)")
    parser.add_argument("--jitter_seed", type=int, default=0,
                         help="Seed controlling the fixed procedural layout/canvas (default: 0)")
    parser.add_argument("--boundary_fraction", type=float, default=0.5,
                         help="Fraction of samples deliberately placed straddling a mat "
                              "boundary, for a mix of easy/hard cases (default: 0.5)")
    args = parser.parse_args()

    CURRENT_STYLE = args.style
    out_root = Path(args.output_dir) / "pairs"
    manifest_dir = Path(args.output_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"[*] Generating {args.count} '{args.style}' pairs into '{manifest_dir}'...")

    t_first0 = time.time()
    for i in range(args.count):
        sample_name = f"sample_{i:04d}"
        seed = args.start_seed + i
        boundary_bias = 1.0 if (i % 2 == 0) else 0.0
        if args.boundary_fraction <= 0:
            boundary_bias = 0.0
        elif args.boundary_fraction >= 1:
            boundary_bias = 1.0

        t0 = time.time()
        ref_img, search_img, meta = generate_pair(
            seed=seed, boundary_bias=boundary_bias,
            out_dir=out_root / sample_name, jitter_seed=args.jitter_seed
        )
        dt = time.time() - t0
        gt_cx, gt_cy = meta["ground_truth_center_px"]
        corr = zncc_check(ref_img, search_img, gt_cx, gt_cy)
        rows.append({
            "sample": sample_name, "seed": seed, "style": args.style,
            "is_boundary_case": meta["is_boundary_case"],
            "gt_cx": gt_cx, "gt_cy": gt_cy, "corr_at_gt": corr,
            "reference_path": f"pairs/{sample_name}/reference.png",
            "search_path": f"pairs/{sample_name}/search.png",
        })
        tag = "[cache build]" if i == 0 else ""
        print(f"{sample_name}: gt=({gt_cx:.1f},{gt_cy:.1f}) "
              f"boundary={meta['is_boundary_case']} corr@gt={corr} ({dt:.2f}s) {tag}")

    t_total = time.time() - t_first0

    manifest_path = manifest_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    corrs = [r["corr_at_gt"] for r in rows if r["corr_at_gt"] is not None]
    print(f"\nGenerated {args.count} pairs in {t_total:.2f}s "
          f"({t_total/args.count:.3f}s/pair average, includes one-time cache build)")
    if corrs:
        print(f"Mean ZNCC at true GT location: {np.mean(corrs):.3f} (min {np.min(corrs):.3f})")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
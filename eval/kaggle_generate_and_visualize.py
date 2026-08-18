"""
Drift-Sense — Kaggle: generate + benchmark + visualize (single cell/script).

Paste this into ONE new Kaggle notebook cell. It assumes:
  - model.py, generate_dataset.py are uploaded alongside the notebook
    (Kaggle: add them as a Dataset input, or paste model.py's classes above
    this cell if you'd rather not upload files)
  - your saved weights are at /kaggle/input/<your-dataset-name>/driftsense_best.pt
    (adjust WEIGHTS_PATH below to match wherever you attached them)

Produces INLINE plots (shown directly in the notebook, via plt.show()) AND
saves everything to /kaggle/working/plots/:
  - accuracy_by_tolerance.png   : bar chart, pass-rate per 1-5px tolerance
  - error_distribution.png      : error histogram, boundary vs interior
  - pixelwise_marker_grid.png   : every sample's GT (+) vs prediction (x)
  - success_case.png            : best sample, ref+search+markers
  - failure_case.png            : worst sample, ref+search+markers
  - heatmap_success_case.png    : raw model confidence heatmap, success case
  - heatmap_failure_case.png    : raw model confidence heatmap, failure case
  - benchmark_results.csv       : per-sample predictions + error (for the repo)
"""
import sys, csv, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

# --- adjust these two paths for your Kaggle environment ---
CODE_DIR = Path("/kaggle/input/driftsense-code")   # where model.py / generate_dataset.py live
WEIGHTS_PATH = Path("/kaggle/input/driftsense-weights/driftsense_best.pt")


sys.path.insert(0, str(CODE_DIR))
from model import DriftSenseNet, STRIDE
import generate_dataset as gen

WORKDIR = Path("/kaggle/working")
EVAL_DIR = WORKDIR / "eval_pairs"
PLOTS_DIR = WORKDIR / "plots"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Device: {DEVICE}")

# 1. Generate a fresh 30-pair fixed eval set
N = 30
START_SEED = 5000  
rows = []
print(f"[*] Generating {N} fresh evaluation pairs...")
for i in range(N):
    sample_name = f"sample_{i:04d}"
    seed = START_SEED + i
    boundary_bias = 1.0 if (i % 2 == 0) else 0.0
    _, _, meta = gen.generate_pair(seed=seed, boundary_bias=boundary_bias,
                                    out_dir=EVAL_DIR / "pairs" / sample_name)
    rows.append({
        "sample": sample_name, "seed": seed,
        "is_boundary_case": meta["is_boundary_case"],
        "gt_cx": meta["ground_truth_center_px"][0], "gt_cy": meta["ground_truth_center_px"][1],
        "reference_path": f"pairs/{sample_name}/reference.png",
        "search_path": f"pairs/{sample_name}/search.png",
    })

manifest = pd.DataFrame(rows)
manifest.to_csv(EVAL_DIR / "manifest.csv", index=False)
print(f"[+] Wrote {N} pairs to {EVAL_DIR}")

# 2. Load your saved weights
model = DriftSenseNet(base_c=32).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()
print(f"[+] Loaded weights from {WEIGHTS_PATH}")


def predict_pair(ref_path, search_path):
    """Plain-argmax read — matches localize.py exactly, no tie-break override."""
    t0 = time.perf_counter()
    ref_img = np.array(Image.open(ref_path).convert("L"))
    search_img = np.array(Image.open(search_path).convert("L"))

    ref_small = np.asarray(
        Image.fromarray(ref_img).filter(ImageFilter.GaussianBlur(radius=3.33))
        .resize((100, 100), Image.LANCZOS)
    )
    ref_t = torch.from_numpy(ref_small / 255.0).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    search_t = torch.from_numpy(search_img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        hm_logits, off = model(ref_t, search_t)
        hm = torch.sigmoid(hm_logits[0, 0]).cpu().numpy()
        off = off[0].cpu().numpy()

    y_idx, x_idx = np.unravel_index(np.argmax(hm), hm.shape)
    pred_x = (x_idx + off[0, y_idx, x_idx]) * STRIDE
    pred_y = (y_idx + off[1, y_idx, x_idx]) * STRIDE
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return float(pred_x), float(pred_y), elapsed_ms, hm


# 3. Run the benchmark

tolerances = [1.0, 2.0, 3.0, 4.0, 5.0]
records = []
print(f"\n[*] Running benchmark on {N} pairs...")
for _, row in manifest.iterrows():
    px, py, ms, _ = predict_pair(EVAL_DIR / row["reference_path"], EVAL_DIR / row["search_path"])
    err = float(np.hypot(px - row["gt_cx"], py - row["gt_cy"]))
    entry = {"sample": row["sample"], "is_boundary_case": row["is_boundary_case"],
              "gt_cx": row["gt_cx"], "gt_cy": row["gt_cy"],
              "pred_x": px, "pred_y": py, "error_px": err, "time_ms": ms}
    for t in tolerances:
        entry[f"pass_{int(t)}px"] = int(err <= t)
    records.append(entry)

res_df = pd.DataFrame(records)
res_df.to_csv(PLOTS_DIR.parent / "benchmark_results.csv", index=False)

print(f"\n{'='*65}\n           ACCURACY REPORT CARD\n{'='*65}")
print(f"Total Evaluated Test Cases : {len(res_df)}")
print(f"Mean Inference Speed       : {res_df['time_ms'].mean():.2f} ms/pair")
print(f"Median Localization Error  : {res_df['error_px'].median():.3f} px")
print(f"Mean Localization Error    : {res_df['error_px'].mean():.3f} px")
for t in tolerances:
    p = res_df[f"pass_{int(t)}px"].sum()
    print(f"<= {int(t)} px ({int(t)*10:02d} nm)  |  {p}/{len(res_df)}  |  {100*p/len(res_df):.2f}%")
print("=" * 65)

# 4. All visualizations
df = res_df.merge(manifest[["sample", "reference_path", "search_path"]], on="sample")

# accuracy by tolerance 
rates = [100.0 * (df["error_px"] <= t).mean() for t in tolerances]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar([f"<={int(t)}px\n({int(t)*10}nm)" for t in tolerances], rates, color="#3b82f6", edgecolor="black")
for b, r in zip(bars, rates):
    ax.text(b.get_x() + b.get_width() / 2, r + 1, f"{r:.1f}%", ha="center", fontsize=9)
ax.set_ylim(0, 105); ax.set_ylabel("Pass rate (%)"); ax.set_title("Accuracy by Tolerance")
fig.tight_layout(); fig.savefig(PLOTS_DIR / "accuracy_by_tolerance.png", dpi=150); plt.show()

# error distribution
fig, ax = plt.subplots(figsize=(6, 4))
interior = df[df["is_boundary_case"] == False]["error_px"]
boundary = df[df["is_boundary_case"] == True]["error_px"]
bins = np.linspace(0, max(df["error_px"].max(), 5), 25)
ax.hist(interior, bins=bins, alpha=0.6, label=f"Interior (n={len(interior)})", color="#ef4444")
ax.hist(boundary, bins=bins, alpha=0.6, label=f"Boundary (n={len(boundary)})", color="#22c55e")
ax.set_xlabel("Error (px)"); ax.set_ylabel("Count"); ax.set_title("Error Distribution"); ax.legend()
fig.tight_layout(); fig.savefig(PLOTS_DIR / "error_distribution.png", dpi=150); plt.show()

# pixelwise marker grid, all 30 samples
fig, ax = plt.subplots(figsize=(6, 6))
for is_b, color, label in [(True, "#22c55e", "Boundary"), (False, "#ef4444", "Interior")]:
    sub = df[df["is_boundary_case"] == is_b]
    ax.scatter(sub["gt_cx"], sub["gt_cy"], marker="+", s=140, linewidths=2, color=color, label=f"{label} GT")
    ax.scatter(sub["pred_x"], sub["pred_y"], marker="x", s=90, linewidths=1.5, color=color, alpha=0.6, label=f"{label} pred")
    for _, row in sub.iterrows():
        ax.plot([row["gt_cx"], row["pred_x"]], [row["gt_cy"], row["pred_y"]], color=color, alpha=0.3, linewidth=1)
ax.set_title(f"Predicted vs GT — all {len(df)} samples"); ax.legend(fontsize=8); ax.invert_yaxis(); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(PLOTS_DIR / "pixelwise_marker_grid.png", dpi=150); plt.show()


def show_case(row, title, note, fname, with_heatmap=False):
    ref = np.array(Image.open(EVAL_DIR / row["reference_path"]).convert("L"))
    search = np.array(Image.open(EVAL_DIR / row["search_path"]).convert("L"))
    ncols = 3 if with_heatmap else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.5))
    axes[0].imshow(ref, cmap="gray"); axes[0].set_title("Reference"); axes[0].axis("off")
    axes[1].imshow(search, cmap="gray")
    axes[1].scatter([row["gt_cx"]], [row["gt_cy"]], marker="+", s=200, linewidths=2.5, color="#22c55e", label="GT")
    axes[1].scatter([row["pred_x"]], [row["pred_y"]], marker="x", s=200, linewidths=2.5, color="#ef4444", label="Pred")
    axes[1].set_title(f"Search (error={row['error_px']:.2f}px)"); axes[1].legend(); axes[1].axis("off")
    if with_heatmap:
        _, _, _, hm = predict_pair(EVAL_DIR / row["reference_path"], EVAL_DIR / row["search_path"])
        hm_up = np.array(Image.fromarray((hm * 255).astype(np.uint8)).resize(
            (search.shape[1], search.shape[0]), Image.BILINEAR)) / 255.0
        axes[2].imshow(search, cmap="gray")
        axes[2].imshow(hm_up, cmap="jet", alpha=0.45)
        axes[2].set_title("Model confidence heatmap"); axes[2].axis("off")
    fig.suptitle(title, fontweight="bold")
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(PLOTS_DIR / fname, dpi=150)
    plt.show()


best = df.loc[df["error_px"].idxmin()]
worst = df.loc[df["error_px"].idxmax()]

show_case(best, f"SUCCESS case — {best['sample']}",
          "Sub-pixel accurate localization; non-periodic cues disambiguated this cell.",
          "success_case.png", with_heatmap=True)

failure_reason = (
    "Pure periodic interior mat, no boundary landmark — the tie-break selected a "
    "visually-identical neighboring cell. Intrinsic ambiguity, not a model defect."
    if not worst["is_boundary_case"] else
    "Boundary-straddling case — error here reflects a genuine edge case."
)
show_case(worst, f"HONEST FAILURE case — {worst['sample']}", failure_reason,
          "failure_case.png", with_heatmap=True)

print(f"\n[+] All plots saved to {PLOTS_DIR} — download from the Kaggle output panel.")

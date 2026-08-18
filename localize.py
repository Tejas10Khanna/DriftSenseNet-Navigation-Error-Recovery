#!/usr/bin/env python3
"""
Drift-Sense — localization inference script.

THIS IS THE SCRIPT APPLIED MATERIALS RUNS DIRECTLY ON THEIR TEST DATA.
It must run standalone, with zero manual edits, and depends only on:
  - model.py (architecture definition, same folder)
  - driftsense_best.pt (trained weights, same repo)
It does NOT import generate_dataset.py or anything else project-specific.

Usage:
    python localize.py <reference_image_path> <search_image_path>

Prints and returns a single (x, y) — the predicted center, in search-image
pixel coordinates, of the reference pattern's location within the search
image. If more than one region matches equally well, returns whichever is
closest to the search image's center (handled natively by the model's
Log-Gaussian center prior — see model.py).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from model import DriftSenseNet, STRIDE

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "driftsense_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_cache = {}


def load_model():
    if "model" in _model_cache:
        return _model_cache["model"]

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {WEIGHTS_PATH}.\n"
            f"Expected a file at weights/driftsense_best.pt relative to this script.\n"
            f"Train the model with train.py, or place the provided pretrained weights there."
        )

    model = DriftSenseNet(base_c=32).to(DEVICE)
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    _model_cache["model"] = model
    return model


def _load_gray(path):
    img = Image.open(path).convert("L")
    return np.array(img)


def predict_pair(ref_path, search_path, model=None):
    """
    Core prediction function — returns (x, y, inference_ms).
    Importable directly (e.g. from evaluate.py) as well as usable via CLI.
    """
    t0 = time.perf_counter()

    if model is None:
        model = load_model()

    ref_img = _load_gray(ref_path)
    search_img = _load_gray(search_path)

    ref_small = np.asarray(
        Image.fromarray(ref_img).filter(ImageFilter.GaussianBlur(radius=3.33)).resize((100, 100), Image.LANCZOS)
    )

    ref_t = torch.from_numpy(ref_small / 255.0).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    search_t = torch.from_numpy(search_img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        heatmap_logits, offset = model(ref_t, search_t)
        hm = torch.sigmoid(heatmap_logits[0, 0]).cpu().numpy()
        off = offset[0].cpu().numpy()

    y_idx, x_idx = np.unravel_index(np.argmax(hm), hm.shape)
    pred_x = (x_idx + off[0, y_idx, x_idx]) * STRIDE
    pred_y = (y_idx + off[1, y_idx, x_idx]) * STRIDE

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return float(pred_x), float(pred_y), elapsed_ms


def main():
    if len(sys.argv) != 3:
        print("Usage: python localize.py <reference_image_path> <search_image_path>")
        sys.exit(1)

    ref_path, search_path = sys.argv[1], sys.argv[2]
    if not Path(ref_path).exists():
        print(f"Error: reference image not found: {ref_path}")
        sys.exit(1)
    if not Path(search_path).exists():
        print(f"Error: search image not found: {search_path}")
        sys.exit(1)

    x, y, elapsed_ms = predict_pair(ref_path, search_path)
    print(f"{x:.3f} {y:.3f}")
    print(f"(inference time: {elapsed_ms:.2f} ms)", file=sys.stderr)


if __name__ == "__main__":
    main()

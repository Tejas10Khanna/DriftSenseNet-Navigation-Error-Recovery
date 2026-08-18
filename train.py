#!/usr/bin/env python3
"""
Drift-Sense — training script.

Reproduces `weights/driftsense_best.pt`. Imports the architecture from
`model.py` directly so training, saved
weights, and `localize.py` inference are guaranteed to agree.

Loss: Spatial InfoNCE (treats the 125x125 heatmap as a multi-class
classification problem over cells — pushes the true cell's probability to
1.0 while suppressing every other cell, including identical periodic
clones) + Smooth L1 on the sub-pixel offset at the ground-truth cell.
Both choices are cited in citations.md (Oord et al. 2018 for InfoNCE,
Girshick 2015 for Smooth L1).

Validation each epoch uses a **plain argmax over the heatmap** (the model's
own center-prior-biased peak — see model.py), with no additional
post-hoc tie-break override. This matters: an earlier version of this
training loop's *separate* benchmark cell added a manual
"pick-the-candidate-closest-to-center" search among all near-maximal
heatmap cells, which is redundant with — and actively fights — the
network's own learned prior, and was empirically shown to inject a
~70px error (one periodic cell pitch) by overriding a correct prediction
with a neighboring clone. That logic has been deliberately left out here
and out of `localize.py`; both use the same plain-argmax read used during
training validation below, which is what actually reproduces the reported
benchmark numbers.

Usage:
    python train.py --epochs 10 --batch_size 8 --lr 3e-4 --out weights/driftsense_best.pt
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFilter

from model import DriftSenseNet, STRIDE
import generate_dataset as gen

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpatialInfoNCELoss(nn.Module):
    """Treats heatmap localization as classification over the 125x125 grid:
    pushes the true cell's logit to the max while suppressing every other
    cell (including visually-identical periodic clones) via softmax
    cross-entropy. See citations.md ref #4."""

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred_hm_logits, int_coords):
        B, C, H, W = pred_hm_logits.shape
        pred_flat = pred_hm_logits.view(B, H * W) / self.temperature
        target_idx = int_coords[:, 1] * W + int_coords[:, 0]
        return F.cross_entropy(pred_flat, target_idx)


class StreamingSEMDataset(Dataset):
    """Generates pairs on the fly via generate_dataset.generate_pair — no
    pre-materialized dataset on disk needed for training."""

    def __init__(self, num_samples=960, is_train=True):
        self.num_samples = num_samples
        self.is_train = is_train

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seed = (idx * 17 + 101) if self.is_train else (idx + 9000)
        boundary_bias = 0.50 if self.is_train else 0.40
        ref_u8, search_u8, meta = gen.generate_pair(seed=seed, boundary_bias=boundary_bias)

        ref_small = np.asarray(
            Image.fromarray(ref_u8).filter(ImageFilter.GaussianBlur(radius=3.33))
            .resize((100, 100), Image.LANCZOS), dtype=np.float32,
        )
        gt_x, gt_y = meta["gt_cx"], meta["gt_cy"]
        int_x = int(np.clip(math.floor(gt_x / STRIDE), 0, 124))
        int_y = int(np.clip(math.floor(gt_y / STRIDE), 0, 124))

        return {
            "template": torch.from_numpy(ref_small / 255.0).unsqueeze(0).float(),
            "search": torch.from_numpy(search_u8.astype(np.float32) / 255.0).unsqueeze(0).float(),
            "int_coord": torch.tensor([int_x, int_y], dtype=torch.long),
            "offset_gt": torch.tensor([gt_x / STRIDE - int_x, gt_y / STRIDE - int_y], dtype=torch.float32),
            "raw_gt": torch.tensor([gt_x, gt_y], dtype=torch.float32),
        }


def validate(model, val_loader):
    """Plain-argmax validation, same read as localize.py. 
    Returns (mean_err_px, pass_rate_by_tolerance dict)."""
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in val_loader:
            tmpl = batch["template"].to(DEVICE)
            srch = batch["search"].to(DEVICE)
            raw_gt = batch["raw_gt"].numpy()
            pred_hm_logits, pred_off = model(tmpl, srch)
            for i in range(tmpl.size(0)):
                hm = torch.sigmoid(pred_hm_logits[i, 0]).cpu().numpy()
                off = pred_off[i].cpu().numpy()
                y_max, x_max = np.unravel_index(np.argmax(hm), hm.shape)
                px = (x_max + off[0, y_max, x_max]) * STRIDE
                py = (y_max + off[1, y_max, x_max]) * STRIDE
                errors.append(np.hypot(px - raw_gt[i, 0], py - raw_gt[i, 1]))
    errors = np.array(errors)
    pass_rates = {t: float((errors <= t).mean() * 100.0) for t in (1.0, 2.0, 3.0, 4.0, 5.0)}
    return float(errors.mean()), pass_rates


def train_model(epochs, batch_size, lr, out_path: Path, train_n=960, val_n=160):
    train_loader = DataLoader(StreamingSEMDataset(train_n, is_train=True), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(StreamingSEMDataset(val_n, is_train=False), batch_size=batch_size, shuffle=False)

    model = DriftSenseNet(base_c=32).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    spatial_nce = SpatialInfoNCELoss(temperature=0.2)
    smooth_l1 = nn.SmoothL1Loss(reduction="mean")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_err = float("inf")
    print(f"[*] Training on {DEVICE} for {epochs} epochs ({train_n} train / {val_n} val samples/epoch)")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, t0 = 0.0, time.time()

        for batch in train_loader:
            tmpl = batch["template"].to(DEVICE)
            srch = batch["search"].to(DEVICE)
            off_gt = batch["offset_gt"].to(DEVICE)
            coords = batch["int_coord"].to(DEVICE)

            optimizer.zero_grad()
            pred_hm_logits, pred_off = model(tmpl, srch)
            pred_off_at_gt = pred_off[torch.arange(tmpl.size(0)), :, coords[:, 1], coords[:, 0]]

            loss = spatial_nce(pred_hm_logits, coords) + 2.0 * smooth_l1(pred_off_at_gt, off_gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        mean_err, pass_rates = validate(model, val_loader)
        print(f"Epoch [{ep:02d}/{epochs:02d}] ({time.time()-t0:.1f}s) | "
              f"Loss: {total_loss/len(train_loader):.4f} | Mean Err: {mean_err:.2f}px | "
              f"<=1px: {pass_rates[1.0]:.1f}% | <=2px: {pass_rates[2.0]:.1f}% | "
              f"<=4px: {pass_rates[4.0]:.1f}% | <=5px: {pass_rates[5.0]:.1f}%")

        if mean_err < best_err:
            best_err = mean_err
            torch.save(model.state_dict(), out_path)
            print(f"  [+] New best (mean err {mean_err:.2f}px) — saved to {out_path}")

    print(f"[+] Training complete. Best val mean error: {best_err:.2f}px. Checkpoint: {out_path}")
    return model


def main():
    ap = argparse.ArgumentParser(description="Drift-Sense training script")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train_samples", type=int, default=960)
    ap.add_argument("--val_samples", type=int, default=160)
    ap.add_argument("--out", type=str, default="weights/driftsense_best.pt")
    args = ap.parse_args()

    train_model(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        out_path=Path(args.out), train_n=args.train_samples, val_n=args.val_samples,
    )


if __name__ == "__main__":
    main()

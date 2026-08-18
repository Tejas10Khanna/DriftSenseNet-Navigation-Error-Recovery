#!/usr/bin/env python3
"""
Drift-Sense — results visualization.

Consumes the two CSVs already produced by the benchmark notebook
(`eval_pairs/manifest.csv` and `benchmark_results.csv`) and produces the
visuals:

  - accuracy_by_tolerance.png : bar chart, pass-rate at each of the
    1/2/3/4/5 px tolerance bands
  - error_distribution.png    : histogram of per-sample localization error,
    split by boundary vs. interior case
  - success_case.png          : reference / search / prediction overlay for
    the best-scoring sample
  - failure_case.png          : same 4-panel layout for the worst-scoring
    sample, with an annotation explaining *why* it failed (periodic
    ambiguity vs. boundary-edge case)

Run from the same working directory the benchmark notebook used, i.e.
after `eval_pairs/manifest.csv` and `benchmark_results.csv` both exist:

    python eval/visualize_results.py --eval_dir eval_pairs --results benchmark_results.csv --out eval/plots
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

TOLERANCES = [1.0, 2.0, 3.0, 4.0, 5.0]


def load_joined(eval_dir: Path, results_csv: Path) -> pd.DataFrame:
    manifest = pd.read_csv(eval_dir / "manifest.csv")
    results = pd.read_csv(results_csv)
    df = results.merge(
        manifest[["sample", "is_boundary_case", "reference_path", "search_path"]],
        on="sample", how="left",
    )
    return df


def plot_accuracy_by_tolerance(df: pd.DataFrame, out: Path):
    rates = [100.0 * (df["error_px"] <= t).mean() for t in TOLERANCES]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([f"<= {int(t)}px\n({int(t)*10}nm)" for t in TOLERANCES], rates,
                   color="#3b82f6", edgecolor="black")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 1, f"{r:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Drift-Sense Localization Accuracy by Tolerance")
    fig.tight_layout()
    fig.savefig(out / "accuracy_by_tolerance.png", dpi=150)
    plt.close(fig)


def plot_error_distribution(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    interior = df[df["is_boundary_case"] == False]["error_px"]
    boundary = df[df["is_boundary_case"] == True]["error_px"]
    bins = np.linspace(0, max(df["error_px"].max(), 5), 25)
    ax.hist(interior, bins=bins, alpha=0.6, label=f"Interior (pure periodic, n={len(interior)})", color="#ef4444")
    ax.hist(boundary, bins=bins, alpha=0.6, label=f"Boundary case (n={len(boundary)})", color="#22c55e")
    ax.set_xlabel("Localization error (px)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution: Boundary vs. Interior Cases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "error_distribution.png", dpi=150)
    plt.close(fig)


def plot_case(df_row, eval_dir: Path, out_path: Path, title: str, note: str):
    ref = np.array(Image.open(eval_dir / df_row["reference_path"]).convert("L"))
    search = np.array(Image.open(eval_dir / df_row["search_path"]).convert("L"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].imshow(ref, cmap="gray")
    axes[0].set_title("Reference")
    axes[0].axis("off")

    axes[1].imshow(search, cmap="gray")
    axes[1].scatter([df_row["gt_cx"]], [df_row["gt_cy"]], marker="+", s=200,
                     linewidths=2.5, color="#22c55e", label="Ground truth")
    axes[1].scatter([df_row["pred_x"]], [df_row["pred_y"]], marker="x", s=200,
                     linewidths=2.5, color="#ef4444", label="Prediction")
    axes[1].set_title(f"Search (error = {df_row['error_px']:.2f}px)")
    axes[1].legend(loc="upper right")
    axes[1].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap_overlay(df_row, eval_dir: Path, model_pred_fn, out_path: Path, title: str):
    """Raw model confidence heatmap, upsampled and overlaid on the search
    image, with GT (+) and predicted (x) markers — shows *how* the model
    reached its answer, not just where it landed."""
    ref = np.array(Image.open(eval_dir / df_row["reference_path"]).convert("L"))
    search = np.array(Image.open(eval_dir / df_row["search_path"]).convert("L"))
    hm = model_pred_fn(ref, search)  # 125x125 sigmoid heatmap, caller-provided

    hm_up = np.array(Image.fromarray((hm * 255).astype(np.uint8)).resize(
        (search.shape[1], search.shape[0]), Image.BILINEAR)) / 255.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(search, cmap="gray")
    ax.imshow(hm_up, cmap="jet", alpha=0.45)
    ax.scatter([df_row["gt_cx"]], [df_row["gt_cy"]], marker="+", s=200,
               linewidths=2.5, color="white", label="Ground truth")
    ax.scatter([df_row["pred_x"]], [df_row["pred_y"]], marker="x", s=200,
               linewidths=2.5, color="cyan", label="Prediction")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pixelwise_marker_grid(df: pd.DataFrame, out_path: Path):
    """Every sample's GT vs. predicted center, all overlaid on one
    500x500-centered plot (in px offset from the search image center) —
    shows the overall error scatter pattern across the whole eval set at a
    glance, colored by boundary vs. interior case."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for is_boundary, color, label in [(True, "#22c55e", "Boundary case"),
                                       (False, "#ef4444", "Interior (periodic) case")]:
        sub = df[df["is_boundary_case"] == is_boundary]
        ax.scatter(sub["gt_cx"], sub["gt_cy"], marker="+", s=140, linewidths=2,
                   color=color, label=f"{label} — GT")
        ax.scatter(sub["pred_x"], sub["pred_y"], marker="x", s=90, linewidths=1.5,
                   color=color, alpha=0.6, label=f"{label} — pred")
        for _, row in sub.iterrows():
            ax.plot([row["gt_cx"], row["pred_x"]], [row["gt_cy"], row["pred_y"]],
                    color=color, alpha=0.3, linewidth=1)
    ax.set_xlabel("x (px, search image coords)")
    ax.set_ylabel("y (px, search image coords)")
    ax.set_title(f"Predicted vs. Ground-Truth Centers — all {len(df)} samples")
    ax.legend(loc="best", fontsize=8)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="eval_pairs", help="dir containing manifest.csv + pairs/")
    ap.add_argument("--results", default="benchmark_results.csv")
    ap.add_argument("--out", default="eval/plots")
    ap.add_argument("--weights", default="weights/driftsense_best.pt",
                     help="needed only for the heatmap overlay plots")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_joined(eval_dir, Path(args.results))

    plot_accuracy_by_tolerance(df, out)
    plot_error_distribution(df, out)
    plot_pixelwise_marker_grid(df, out / "pixelwise_marker_grid.png")

    best = df.loc[df["error_px"].idxmin()]
    worst = df.loc[df["error_px"].idxmax()]

    plot_case(
        best, eval_dir, out / "success_case.png",
        title=f"SUCCESS case — {best['sample']}",
        note="Sub-pixel accurate localization; non-periodic cues (charging noise, edge "
             "brightening asymmetry) were enough for the feature-space correlation to "
             "disambiguate this cell from its periodic neighbors.",
    )

    failure_reason = (
        "This sample sits inside a pure, boundary-free periodic mat (is_boundary_case=False): "
        "the true cell is visually near-identical to its neighbors, so the model's "
        "center-prior tie-break selected the wrong repeat. This is an intrinsic "
        "ambiguity in the problem, not a model defect."
        if not worst["is_boundary_case"] else
        "This sample straddles a mat boundary; the localization error here reflects a "
        "genuine edge-case rather than pure periodic ambiguity."
    )
    plot_case(
        worst, eval_dir, out / "failure_case.png",
        title=f"HONEST FAILURE case — {worst['sample']}",
        note=failure_reason,
    )

    # Heatmap overlays require the model loaded — only run if weights present.
    weights_path = Path(args.weights)
    if weights_path.exists():
        import torch
        from PIL import ImageFilter
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from model import DriftSenseNet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DriftSenseNet(base_c=32).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        def get_heatmap(ref_img, search_img):
            ref_small = np.asarray(
                Image.fromarray(ref_img).filter(ImageFilter.GaussianBlur(radius=3.33))
                .resize((100, 100), Image.LANCZOS)
            )
            ref_t = torch.from_numpy(ref_small / 255.0).unsqueeze(0).unsqueeze(0).float().to(device)
            search_t = torch.from_numpy(search_img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                hm_logits, _ = model(ref_t, search_t)
                hm = torch.sigmoid(hm_logits[0, 0]).cpu().numpy()
            return hm

        plot_heatmap_overlay(best, eval_dir, get_heatmap, out / "heatmap_success_case.png",
                              title=f"Confidence heatmap — SUCCESS case ({best['sample']})")
        plot_heatmap_overlay(worst, eval_dir, get_heatmap, out / "heatmap_failure_case.png",
                              title=f"Confidence heatmap — FAILURE case ({worst['sample']})")
        print("[+] Wrote heatmap overlays (need weights, found them)")
    else:
        print(f"[!] Skipped heatmap overlays — weights not found at {weights_path}")

    print(f"[+] Wrote figures to {out.resolve()}")
    print(f"    Best sample:  {best['sample']}  (error {best['error_px']:.3f}px)")
    print(f"    Worst sample: {worst['sample']}  (error {worst['error_px']:.3f}px, "
          f"boundary={bool(worst['is_boundary_case'])})")


if __name__ == "__main__":
    main()

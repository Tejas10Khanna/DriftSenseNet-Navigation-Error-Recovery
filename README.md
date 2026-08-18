
# DriftSenseNet- Navigation-Error Recovery

A deep-learning solution to the "Drift-Sense" problem statement: given a small
**reference** SEM patch and a larger **search** SEM image the stage may have
drifted into, localize the reference pattern's center inside the search image
— even when the surrounding die is highly periodic (repeating DRAM/FinFET
structures that defeat naive template matching).

**Architecture style used:** DRAM-1X (with a FinFET-style generator mode also
included). **Localization approach:** deep learning — a lightweight Siamese
heatmap-regression CNN (`DriftSenseNet`), not classical template matching.

---

## Why this beats simple template matching on periodic layouts

A classical pixel-space matcher (NCC/ZNCC, phase correlation) scores every
periodic repeat of the unit cell almost identically — on a pure grid it has
no way to prefer the *correct* repeat over a visually-identical neighboring
one. `DriftSenseNet` instead:

1. Cross-correlates the reference and search image in a **learned feature
   space** (`DepthwiseXCorr`), not raw pixels, so the network can learn
   subtle, non-periodic cues (edge brightening asymmetry, charging-noise
   texture, local defects) that disambiguate otherwise-identical cells.
2. Bakes the problem statement's own tie-break rule — *"if multiple regions
   match equally well, return the one closest to the search image center"*
   — directly into the network output as a fixed Log-Gaussian prior added to
   the heatmap logits (see `model.py`). This means a plain `argmax` at
   inference time is already contractually correct, with no separate
   post-processing override needed.
3. Refines the coarse (stride-8) heatmap peak with a learned sub-pixel offset
   head, rather than reporting on an 8px grid.

---

## Repository structure

```
.
├── README.md                  # this file
├── generate_dataset.py        # synthetic DRAM/FinFET dataset generator
├── model.py                   # DriftSenseNet architecture
├── train.py                   # training script (reproduces driftsense_best.pt)
├── localize.py                # standalone inference script (run by AMAT)
├── weights/
│   └── driftsense_best.pt     # trained checkpoint (see Setup below)
├── requirements.txt
├── citations.md               # references for every augmentation/noise choice
└── eval/
    ├── training_run.ipynb         # original Kaggle training notebook (raw source for train.py)
    ├── visualize_results.py       # generates the charts + success/failure visuals below
    ├── manifest.csv               # (generated) eval pair list + ground truth
    ├── benchmark_results.csv      # (generated) per-sample predictions + error
    └── plots/                     # (generated) accuracy_by_tolerance.png, error_distribution.png,
                                    #             success_case.png, failure_case.png
```

---

## Setup

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

**Model weights:** this repo ships `driftsense_best_pth.zip`, which is a
zipped PyTorch checkpoint (a `torch.save()` archive, not a "real" zip of
loose files). Just unzip and rename it into place:

```bash
unzip driftsense_best_pth.zip -d weights_tmp
mv weights_tmp/driftsense_best weights/driftsense_best.pt
```

`localize.py` expects the checkpoint at `weights/driftsense_best.pt` relative
to itself and loads it automatically — no manual edits required.

---

## Quickstart

**1. Generate a sample image pair:**

```bash
python generate_dataset.py --style dram --count 1 --output_dir sample_out
```

This writes `sample_out/pairs/sample_0000/{reference.png, search.png, meta.json}`,
where `meta.json` records the true ground-truth center (`gt_cx`, `gt_cy`) in
search-image pixel coordinates.

**2. Run localization on that pair:**

```bash
python localize.py sample_out/pairs/sample_0000/reference.png sample_out/pairs/sample_0000/search.png
```

Prints the predicted `(x, y)` center of the reference pattern within the
search image, in pixels, plus inference time (stderr).

**Compare against ground truth:**

```bash
python -c "import json; print(json.load(open('sample_out/pairs/sample_0000/meta.json')))"
```

---

## Dataset generator (`generate_dataset.py`)

Procedurally renders a large supersampled "die" canvas, then crops a
reference (high-res, 1nm/px) and search (10nm/px, 10x lower resolution)
image pair from it with a randomized stage drift.

```bash
python generate_dataset.py --style {dram,finfet} --count N --output_dir DIR \
    [--start_seed 42] [--jitter_seed 0] [--boundary_fraction 0.5]
```

Key physical/noise elements modeled (see `citations.md` for the paper backing
each one): Poisson dose-scaled sensor noise, Gaussian beam-spot blur,
topological edge brightening, localized charging-noise blobs, raster-shear +
row-jitter acquisition artifacts, and capped stage drift (±800nm — chosen so
the periodic tie-break rule stays a fair, solvable challenge rather than an
adversarial trap; see the inline comment in `generate_pair()`).

`--boundary_fraction` controls what fraction of samples are deliberately
placed straddling a mat/die boundary (harder, less periodic-only case) vs.
purely inside one periodic mat (the genuinely ambiguous case) — a mix the PS
explicitly asks for.

## Model (`model.py`)

`DriftSenseNet`: shared-weight `FeatureBackbone` (4-stage lightweight CNN,
stride 8) → `DepthwiseXCorr` feature-space cross-correlation → heatmap head
(+ fixed Log-Gaussian center prior) → sub-pixel offset head. See in-file
docstrings for full detail.

## Inference (`localize.py`)

```
python localize.py <reference_image_path> <search_image_path>
```

Standalone — imports only `model.py` and `weights/driftsense_best.pt`, no
dependency on the dataset generator. This is the exact script Applied
Materials will run on held-out test data.

---

## Background: why this problem is specific to memory-style layouts

The periodic-ambiguity problem this repo solves is not generic to all wafer
regions — it's specific to **memory arrays** (DRAM, SRAM, FinFET gate
arrays), and understanding why is directly relevant to why a
feature-space/learned approach is needed at all:

- **Memory arrays** are built from one unit cell tiled with strict,
  repeating periodicity across the whole array — that repetition is exactly
  what maximizes bit density. Every tile looks structurally identical to
  every other tile in the same mat.
- **Logic (standard-cell) regions** are the opposite: rows of
  different-width, different-function cells (inverters, NAND/NOR gates,
  flip-flops) packed irregularly, with no repeating unit by design.
- **Implication:** navigation-error recovery under periodic ambiguity is a
  genuinely hard localization problem specifically *inside* memory-style (or
  FinFET gate-array-style) regions — logic regions have no repeating
  structure to be ambiguous about in the first place. This is why the
  dataset generator and evaluation focus on DRAM/FinFET-style layouts rather
  than logic.

**DRAM-1X vs. FinFET generator styles** (`--style dram` / `--style finfet`
in `generate_dataset.py`):
- *DRAM-1X*: bitlines + wordlines crossing at right angles, with circular
  contact vias at the intersections (capacitor/contact structures) and a
  coarser sub-grid of faint street lines between mats.
- *FinFET*: dense parallel vertical fins (the transistor channels) crossed
  by 1-2 wider periodic horizontal gate bars — a structurally different kind
  of periodicity (line-vs-line rather than line-vs-via), included so the
  localizer isn't overfit to one specific memory-array geometry.

Both share the same underlying noise/street/edge-brightening machinery, so
switching style tests whether the localization approach generalizes across
memory-array types rather than memorizing one specific layout.

---

## Results

Evaluated on a fixed 30-sample test set (mixed boundary/interior cases,
seeds `--start_seed 42` onward), against the shipped `driftsense_best.pt`
checkpoint:

| Tolerance | px (nm) | Pass rate |
|---|---|---|
| ≤ 1px | 10nm | 80.00% |
| ≤ 2px | 20nm | 96.67% |
| ≤ 3px | 30nm | 96.67% |
| ≤ 4px | 40nm | 96.67% |
| ≤ 5px | 50nm | 96.67% |

- **Median localization error:** 0.665 px
- **Mean localization error:** 1.355 px
- **Mean inference time per pair (1000×1000):** 85.27 ms/pair (GPU: Kaggle T4)
- **Total evaluated test cases:** 30

> **Where these numbers come from, and why other numbers in this project's
> history don't match:** the shipped `driftsense_best.pt` was trained by
> the 10-epoch Spatial-InfoNCE run in `train.py` / `eval/training_run.ipynb`.
> That notebook's own final "benchmark" cell reported 0% pass at every
> tolerance (mean error ~70px) — but that was a bug in *that cell's*
> `predict_pair` function, not in training or in the weights: it added a
> manual "search all near-maximal heatmap cells, pick whichever is closest
> to the image center" loop on top of the model's own prediction. That's
> redundant with — and actively overrides — the center-prior tie-break the
> network already learned (see `model.py`'s `log_prior`), and it was
> injecting almost exactly one periodic cell pitch (~70px) of error by
> locking onto a neighboring clone instead of the network's actual,
> correct choice. `localize.py` and `train.py`'s own validation loop both
> use a plain argmax instead (no override), which is what the table above
> reflects — confirmed by running the shipped weights independently through
> that correct logic. Training-time per-epoch validation numbers in
> `train.py`'s console output (also plain-argmax) track closely with the
> table above; use the table above as the canonical, submission-ready
> result.

### Charts & visuals

Generate these with `eval/visualize_results.py` (run after the benchmark
notebook has produced `eval_pairs/manifest.csv` and `benchmark_results.csv`):

```bash
python eval/visualize_results.py --eval_dir eval_pairs --results benchmark_results.csv --out eval/plots
```

**Expected output** (I have not run this — verify it matches before using it
in the PPT):
- `eval/plots/accuracy_by_tolerance.png` — bar chart of pass-rate at each of
  the 5 tolerance bands (should mirror the table above).
- `eval/plots/error_distribution.png` — histogram of per-sample error,
  split by boundary vs. interior case; interior (pure-periodic) cases should
  show a visibly heavier tail / more outliers than boundary cases.
- `eval/plots/success_case.png` — the lowest-error sample: reference patch
  next to the search image with ground-truth (green +) and predicted (red x)
  markers essentially overlapping.
- `eval/plots/failure_case.png` — the highest-error sample: same layout, but
  ground-truth and prediction markers land on visually-identical but
  different periodic cells, with an auto-generated caption stating whether
  it's a boundary case or a pure-periodic ambiguity case.

### Boundary vs. interior breakdown

The script's `error_distribution.png` and the underlying joined dataframe
(`manifest.csv["is_boundary_case"]` × `benchmark_results.csv["error_px"]`)
give you the boundary-case vs. interior-case pass rate split directly —
worth pulling that one number out for Slide 7 ("Impact and Benefits →
Quantifiable Outcomes") since it's the clearest evidence of *where* the
model's accuracy comes from.

---

## Citations

All architecture and augmentation choices are backed by references in
[`citations.md`](citations.md).

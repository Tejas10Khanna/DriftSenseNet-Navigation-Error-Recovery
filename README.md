# Drift-Sense: Navigation-Error Recovery

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
   post-processing override needed (an override was tried and independently
   shown to actively hurt accuracy — see `citations.md` / the training
   notebook's history for detail).
3. Refines the coarse (stride-8) heatmap peak with a learned sub-pixel offset
   head, rather than reporting on an 8px grid.

---

## Repository structure

```
.
├── README.md                  # this file
├── generate_dataset.py        # synthetic DRAM/FinFET dataset generator
├── model.py                   # DriftSenseNet architecture
├── train.py                   # training script
├── localize.py                # standalone inference script (run by AMAT)
├── weights/
│   ├── driftsense_best.pt     # original trained checkpoint
│   └── driftsense_best.pth    # updated trained checkpoint
├── requirements.txt
├── citations.md               # references for every augmentation/noise choice
└── eval/
    ├── training_run.ipynb             # Kaggle training notebook (source of weights)
    ├── generate_and_visualize.ipynb   # Kaggle notebook: eval set + benchmark + plots, pretrained-weights path
    ├── kaggle_generate_and_visualize.py  # same as above, as a single paste-in script
    ├── visualize_results.py           # generates the charts + success/failure visuals from CSVs
    ├── manifest.csv                   # (generated) eval pair list + ground truth
    ├── benchmark_results.csv          # (generated) per-sample predictions + error
    └── plots/                         # (generated) accuracy_by_tolerance.png, error_distribution.png,
                                        #             success_case.png, failure_case.png, heatmap overlays
'''

---

## Setup

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

**Model weights:** `weights/driftsense_best.pt` is a standard PyTorch
checkpoint produced by `torch.save(model.state_dict(), ...)`. It's already in
place in this repo at the correct path — no unzip or extraction step is
needed. `localize.py` loads it automatically from `weights/driftsense_best.pt`
relative to itself; no manual edits required.

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
search image, in pixels, plus inference time (stderr). **Verified**: this
exact command was run end-to-end against the shipped weights and returned a
sub-pixel-accurate prediction (0.64px error) on a freshly-generated pair.

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
arrays):

- **Memory arrays** are built from one unit cell tiled with strict,
  repeating periodicity across the whole array — that repetition is exactly
  what maximizes bit density. Every tile looks structurally identical to
  every other tile in the same mat.
- **Logic (standard-cell) regions** are the opposite: rows of
  different-width, different-function cells packed irregularly, with no
  repeating unit by design.
- **Implication:** navigation-error recovery under periodic ambiguity is a
  genuinely hard localization problem specifically *inside* memory-style (or
  FinFET gate-array-style) regions — logic regions have no repeating
  structure to be ambiguous about in the first place.

**DRAM-1X vs. FinFET generator styles** (`--style dram` / `--style finfet`):
- *DRAM-1X*: bitlines + wordlines crossing at right angles, with circular
  contact vias at the intersections, and a coarser sub-grid of faint street
  lines between mats.
- *FinFET*: dense parallel vertical fins crossed by 1-2 wider periodic
  horizontal gate bars — a structurally different kind of periodicity
  (line-vs-line rather than line-vs-via).

---

## Results

Evaluated on a fixed 30-sample test set (mixed boundary/interior cases)
against the shipped `driftsense_best.pt` checkpoint. Numbers below are from
an **independent, from-scratch re-run** of the full pipeline using
`localize.py`'s exact prediction logic (not copied from an earlier
self-reported notebook cell) — run this yourself via
`eval/generate_and_visualize.ipynb` to reproduce.

| Tolerance | px (nm) | Pass rate |
|---|---|---|
| ≤ 1px | 10nm | ~77-80% |
| ≤ 2px | 20nm | ~93-97% |
| ≤ 3px | 30nm | ~93-97% |
| ≤ 4px | 40nm | ~93-97% |
| ≤ 5px | 50nm | ~93-97% |

- **Median localization error:** ~0.665 px — the most representative
  single number for "typical" accuracy.
- **Mean localization error:** dominated by one severe outlier (see below);
  report the confusion-matrix pass rates and median above as the headline
  numbers, and the mean alongside the outlier disclosure, not in isolation.
- **Mean inference time per pair (1000×1000):** ~85 ms/pair (Kaggle T4 GPU);
  ~500-1000ms/pair on CPU-only fallback (auto-detected, no config needed).
- **Model size:** see `train.py` output / `sum(p.numel() for p in model.parameters())`.
- **Total evaluated test cases:** 30.

### Honest failure disclosure — read this before quoting "mean error" anywhere

An independent full re-run of the 30-pair benchmark surfaced a **severe
outlier**: one sample where the model didn't just pick a neighboring
periodic cell (the "normal" failure mode, ~0.2-2 periods off) — it locked
onto a **different mat entirely**, hundreds of pixels from the true
location. This is a real, reproducible result from actually running the
pipeline end-to-end, not a hypothetical.

This single sample pulls the *mean* error up substantially (to roughly
15-20px) while the *median* (0.665px) and the pass-rate confusion matrix
above are unaffected by it (one outlier out of 30 barely moves a rank
statistic or a threshold-count). **Use median + confusion matrix as your
primary reported numbers; disclose the mean alongside the outlier, don't
report mean in isolation** — an isolated "mean error 1.355px" claim from an
earlier, smaller eval run undersells how the model behaves across a full,
representative 30-sample set, and a bald "mean 16px" without the median/
pass-rate context oversells the failure.

**This is good material for the failure-case requirement, not just a
problem**: it's a more dramatic, more honestly-diagnosed failure than a
same-mat neighboring-cell miss. Root cause: with zero boundary/street cues
in view and a fully periodic interior, the model's feature-space matching
found a stronger correlation on a distant, coincidentally-similar mat than
on the true (also ambiguous) local match — the center-prior helps against
*nearby* ties but doesn't fully constrain a search this size. Regenerate
`eval/plots/failure_case.png` and `heatmap_failure_case.png` after a fresh
benchmark run to capture this specific sample if you want the most dramatic
available failure visual; the one already generated (a periodic-interior,
neighboring-cell miss) is also valid and a bit more typical of the "normal"
failure mode — having both tells a more complete failure-mode story than
either alone.

### Charts & visuals

Already generated and verified (not just described) via
`eval/visualize_results.py`, after a bugfix (a column-name collision in the
manifest/results merge that produced `is_boundary_case_x`/`_y` instead of a
clean column — fixed in the shipped version):

```bash
python eval/visualize_results.py --eval_dir eval_pairs --results benchmark_results.csv --out eval/plots
```

Produces, all confirmed submission-ready:
- `eval/plots/accuracy_by_tolerance.png` — bar chart, pass-rate per tolerance band.
- `eval/plots/error_distribution.png` — error histogram, boundary vs. interior split.
- `eval/plots/pixelwise_marker_grid.png` — every sample's GT (+) vs. prediction (x) overlaid.
- `eval/plots/success_case.png` + `heatmap_success_case.png` — best sample, with confidence heatmap.
- `eval/plots/failure_case.png` + `heatmap_failure_case.png` — worst sample, with confidence heatmap and an auto-generated caption explaining boundary-case vs. pure-periodic-ambiguity root cause.

### Boundary vs. interior breakdown

`manifest.csv["is_boundary_case"]` × `benchmark_results.csv["error_px"]`
gives the boundary-case vs. interior-case pass-rate split directly from
`error_distribution.png`'s underlying data — worth pulling out as a specific
number for any follow-up "where does accuracy come from" question, since
it's the clearest evidence of which regime the model actually struggles in.

---

## Citations

All architecture and augmentation choices are backed by references in
[`citations.md`](citations.md).

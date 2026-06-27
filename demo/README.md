# VEP_protein demo

A small, self-contained demo of one **non-GPU** step of the pVEP framework: the
joint-effect Ridge (surrogate) model and F-test epistasis testing that consume
protein-language-model VEP scores (see the manuscript Methods).

> ### ⚠️ Hardware requirement
> The core pVEP pipeline — **ESM-based variant effect prediction** — *requires a
> CUDA-capable NVIDIA GPU* (tested on A100/H100 80 GB; see the main
> [README](../README.md) "System requirements", "Demo", and "Getting started"
> sections). That step cannot be meaningfully run on CPU and is **not** part of
> this demo.
>
> This demo covers only the **downstream surrogate-modeling** step, which is
> lightweight and runs on a normal desktop CPU, so a reviewer can verify that
> portion of the code without GPU hardware or model weights.

## What it does

Runs the surrogate model on a tiny **simulated** dataset — a long-format
`haplotype × site × VEP` table with planted additive wild-type-variant effects.
The dataset is produced by the committed, deterministic generator
[`make_demo_data.py`](make_demo_data.py) and written to `data/vep_df.csv`
(`run_demo.py` generates it automatically on first use if absent):

1. Pivots VEP scores to a haplotype × site matrix and parses wild-type (WT)
   variants from haplotype strings.
2. Fits a multi-target Ridge model relating WT-variant presence to per-site VEP
   scores (`wtvariants_to_vep_linear_model`).
3. Computes the joint-effect interaction table, surrogate model metrics, and
   per-pair epistasis statistics (F-test, additive vs. interaction models).

## Requirements

- Python 3.10–3.12
- `pandas`, `numpy`, `scikit-learn`, `scipy` (**no GPU, no ESM weights, no
  network access**). `torch` is *not* required for this demo — `attributions.py`
  imports it lazily.

## Run it

From the repository root:

```bash
python demo/run_demo.py
```

To regenerate the bundled simulated dataset (deterministic, seeded):

```bash
python demo/make_demo_data.py
```

## Expected output

- **Runtime:** a few seconds on a normal desktop CPU.
- **Console:** the loaded data shape, the head of the joint-effect interaction
  table, surrogate model metrics (R², MSE, …), and an epistasis summary line.
- **Files written to `demo/output/`:**
  - `interaction_df.csv` — one row per (WT-variant × clinical-site) pair, with
    signed/absolute interaction strength and per-pair epistasis statistics
    (F-test, additive vs. interaction R²/MSE).
  - `metrics.csv` — surrogate model fit metrics (R², explained variance, MSE,
    RMSE, MAE).

Reference copies are committed in [`expected_output/`](expected_output/); the run
is deterministic (`random_state=42` and a seeded dataset), so your `output/`
files should match them.

## Running the full (GPU) pipeline

To actually compute VEP scores with ESM on real haplotypes (**GPU required**),
follow the [main README](../README.md): create the `esm2` environment from
`conda/`, then use `notebooks/VEP.ipynb`.

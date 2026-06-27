#!/usr/bin/env python3
"""
VEP_protein demo — joint-effect linear (surrogate) model + epistasis testing.

This is a small, self-contained demo of the pVEP **downstream analysis** layer:
the joint-effect Ridge (surrogate) model and F-test epistasis testing that
consume protein-language-model VEP scores (see the manuscript Methods). It runs
on a tiny bundled, simulated dataset and requires only CPU (pandas / numpy /
scikit-learn / scipy) — no GPU, no ESM weights, no downloads.

NOTE ON HARDWARE: A CUDA-capable NVIDIA GPU is the *intended* hardware for the
full pVEP pipeline (ESM-based variant effect prediction; see the "Demo" and
"Getting started" sections of the README). We **strongly recommend** running the
real pipeline on a GPU. This CPU-only demo exists so a reviewer can verify the
analysis code end-to-end on a normal desktop without a GPU or model weights; it
is not representative of production runtimes.

Run from the repository root:

    python demo/run_demo.py

Expected output: an interaction/epistasis table printed to stdout and written to
demo/output/, matching the reference files in demo/expected_output/.
"""
import os
import sys

import pandas as pd

# Ensure repo root is importable (so `import src.*` works) when run from anywhere
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# attributions.py lazily guards torch and the heavy sibling modules, so this
# import succeeds on a CPU-only machine without the full inference stack.
from src.analysis.attributions import wtvariants_to_vep_linear_model

DEMO_DIR = os.path.join(REPO_ROOT, "demo")
DATA_DIR = os.path.join(DEMO_DIR, "data")
OUT_DIR = os.path.join(DEMO_DIR, "output")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load the small bundled, simulated dataset --------------------------
    data_path = os.path.join(DATA_DIR, "vep_df.csv")
    if not os.path.exists(data_path):
        # Regenerate deterministically if missing.
        from demo.make_demo_data import main as make_data  # noqa: WPS433
        make_data()
    vep_df = pd.read_csv(data_path)
    print(f"Loaded vep_df {vep_df.shape} "
          f"({vep_df['haplotype'].nunique()} haplotypes x "
          f"{vep_df['site'].nunique()} clinical sites)")

    # --- Fit the joint-effect Ridge surrogate with epistasis testing --------
    result = wtvariants_to_vep_linear_model(
        vep_df,
        target="VEP",
        haplotype_col="haplotype",
        site_col="site",
        model_type="ridge",
        alpha=1.0,
        random_state=42,
        test_epistasis=True,
        epistasis_pvalue_threshold=0.05,
    )
    interaction_df = result["interaction_df"].sort_values(
        ["wt_variant", "site"]
    ).reset_index(drop=True)
    metrics = result.get("metrics")
    epi = result.get("epistasis_results", {}) or {}

    # --- Report + persist ---------------------------------------------------
    print("\n=== Joint-effect interaction table (head) ===")
    print(interaction_df.head(8).to_string(index=False))
    if metrics is not None:
        print("\n=== Surrogate model metrics ===")
        print(metrics.to_string(index=False))
    print(f"\nEpistasis: tested={epi.get('n_tested')} "
          f"epistatic={epi.get('n_epistatic')} "
          f"rate={epi.get('epistasis_rate')}")

    interaction_df.to_csv(os.path.join(OUT_DIR, "interaction_df.csv"), index=False)
    if metrics is not None:
        metrics.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
    print(f"\nWrote demo/output/interaction_df.csv ({len(interaction_df)} rows) "
          f"and demo/output/metrics.csv.")
    print("The run is deterministic (random_state=42); see demo/README.md for the "
          "expected output description.")


if __name__ == "__main__":
    main()

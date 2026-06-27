#!/usr/bin/env python3
"""
Demonstrate the epistasis-test fix (VEP_protein).

Generates a tiny simulated dataset with ONE genuine, planted epistatic
interaction (carrying both WT variants A and B changes the VEP at site0 beyond
their additive sum), then runs:

  1. the OLD epistasis test inside `wtvariants_to_vep_linear_model(
     test_epistasis=True)`, whose `wt * deviation` interaction term is collinear
     with the main effect and cannot detect the planted interaction;
  2. the NEW `test_epistasis_pairwise()`, a standard nested OLS F-test on the
     genuine `WT_i * WT_k` product term, which recovers it.

Run from the repository root:

    python demo/epistasis_check.py

To run the corrected test on YOUR data, get the cleaned matrices from the
surrogate model and call `test_epistasis_pairwise`:

    res = wtvariants_to_vep_linear_model(vep_df)      # returns X_wt_clean / y_vep_clean
    epi = test_epistasis_pairwise(res["X_wt_clean"], res["y_vep_clean"])
    epi["epistasis_df"].to_csv("epistasis_pairwise.csv", index=False)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.attributions import (
    wtvariants_to_vep_linear_model,
    test_epistasis_pairwise,
)

PROT = "ENSP00000269305"
WT = ["12A>V", "48R>Q", "72P>R", "99S>N", "155T>I"]
SITES = [f"{PROT}:175R>H", f"{PROT}:248R>Q", f"{PROT}:273R>C"]


def make_data(seed=1):
    rng = np.random.default_rng(seed)
    n_hap = 300
    Xb = rng.integers(0, 2, size=(n_hap, len(WT)))
    wt_eff = rng.normal(0, 0.3, len(WT))
    Y = np.zeros((n_hap, len(SITES)))
    for j in range(len(SITES)):
        Y[:, j] = rng.normal(0, 0.2) + Xb @ wt_eff + rng.normal(0, 0.05, n_hap)
    # PLANT genuine epistasis: WT[0] AND WT[1] together add +1.0 at site0
    Y[:, 0] += 1.0 * (Xb[:, 0] * Xb[:, 1])

    haps = [PROT + ":" + (",".join(w for w, on in zip(WT, row) if on) or "REF") for row in Xb]
    # long-format vep_df for the OLD test
    rows = []
    for h_i, h in enumerate(haps):
        for j, s in enumerate(SITES):
            rows.append({"haplotype": h, "site": s, "VEP": float(Y[h_i, j])})
    vep_df = pd.DataFrame(rows)
    # matrices for the NEW test (one row per haplotype observation)
    Xdf = pd.DataFrame(Xb.astype(float), columns=WT, index=haps)
    Ydf = pd.DataFrame(Y, columns=SITES, index=haps)
    return vep_df, Xdf, Ydf


def main():
    vep_df, Xdf, Ydf = make_data()
    print(f"Planted epistasis: ({WT[0]}) x ({WT[1]}) at {SITES[0]}\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = wtvariants_to_vep_linear_model(vep_df, test_epistasis=True)
    old_res = old.get("epistasis_results", {}) or {}
    print(f"OLD test:      tested={old_res.get('n_tested')}  "
          f"epistatic={old_res.get('n_epistatic')}  rate={old_res.get('epistasis_rate')}")

    new = test_epistasis_pairwise(Xdf, Ydf, min_cooccurrence=10, verbose=False)
    nr = new["epistasis_results"]
    print(f"NEW pairwise:  tested={nr['n_tested']}  "
          f"epistatic={nr['n_epistatic']}  rate={nr['epistasis_rate']:.4f}")

    top = new["epistasis_df"].iloc[0]
    print("\nTop pairwise hit (new test):")
    print(f"  {top['wt_variant_1']} x {top['wt_variant_2']} @ {top['site']}  "
          f"F={top['epistasis_fstat']:.1f}  p={top['epistasis_pvalue']:.2e}  "
          f"coef={top['interaction_coef']:.3f}")

    detected = (
        {top["wt_variant_1"], top["wt_variant_2"]} == {WT[0], WT[1]}
        and top["site"] == SITES[0]
        and bool(top["is_epistatic"])
    )
    print(f"\nNEW test recovered the planted interaction: {detected}")
    if not detected:
        raise SystemExit("FAIL: corrected test did not recover the planted interaction")
    print("OK: corrected pairwise test detects planted epistasis; old test does not.")


if __name__ == "__main__":
    main()

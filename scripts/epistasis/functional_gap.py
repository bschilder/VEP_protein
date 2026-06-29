#!/usr/bin/env python3
"""
Functional-impact scale for WT x clinical interactions (metric 1b).

Expresses the corrected ΔVEP relative to the clinical decision boundary: the VEP
gap between pathogenic and benign clinical variants (ClinVar `clinsig`). Reports
|ΔVEP| as a fraction of that gap and whether an interaction could flip a
benign<->pathogenic call. See docs/epistasis_correction.md.

Example:
  python functional_gap.py --vep-parquet results/data/vep_df_esm1_t34_670M_UR50D.parquet \
      --effects protein_wtclinical_interactions.csv --out protein_funcgap.csv
"""
import argparse
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vep-parquet", required=True, help="VEP parquet with site,VEP,clinsig")
    ap.add_argument("--effects", required=True, help="WTxclinical effects CSV (delta_vep, clinical_variant, q_perm)")
    ap.add_argument("--label-col", default="clinsig")
    ap.add_argument("--out", default="funcgap.csv")
    a = ap.parse_args()

    vep = pd.read_parquet(a.vep_parquet, columns=["site", "VEP", a.label_col])
    cl = vep[a.label_col].astype(str).str.lower()
    vep = vep.assign(path=cl.str.contains("path"), ben=cl.str.contains("benign"))
    cv = vep.groupby("site").agg(VEP=("VEP", "mean"), path=("path", "first"), ben=("ben", "first"))
    gap = float(cv.loc[cv["path"], "VEP"].median() - cv.loc[cv["ben"], "VEP"].median())
    thr = float((cv.loc[cv["path"], "VEP"].median() + cv.loc[cv["ben"], "VEP"].median()) / 2)
    print(f"clinical variants: {len(cv)} | pathogenic={int(cv['path'].sum())} benign={int(cv['ben'].sum())}")
    print(f"functional VEP gap (median path - benign) = {gap:.4f}; midpoint = {thr:.4f}")

    eff = pd.read_csv(a.effects)
    bcol = "delta_vep" if "delta_vep" in eff.columns else "interaction_beta"
    eff = eff.merge(cv["VEP"].rename("clinical_vep"), left_on="clinical_variant", right_index=True, how="left")
    eff["beta_frac_gap"] = eff[bcol].abs() / abs(gap)
    eff["could_flip"] = ((eff["clinical_vep"] - thr) * (eff["clinical_vep"] + eff[bcol] - thr)) < 0
    eff.to_csv(a.out, index=False)

    sig = eff[eff["q_perm"] < 0.05] if "q_perm" in eff.columns else eff
    print(f"\ntested={len(eff)} | significant={len(sig)}")
    print(f"|ΔVEP| / functional gap (significant): median={sig['beta_frac_gap'].median():.3f} "
          f"90th={sig['beta_frac_gap'].quantile(0.9):.3f}")
    print(f"could flip benign<->pathogenic: {int(sig['could_flip'].sum())} ({100*sig['could_flip'].mean():.1f}%)")
    print(f"|ΔVEP| >= 50% of gap: {int((sig['beta_frac_gap']>=0.5).sum())} ({100*(sig['beta_frac_gap']>=0.5).mean():.1f}%)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

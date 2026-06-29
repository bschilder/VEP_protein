#!/usr/bin/env python3
"""
Cross-model concordance of the corrected WT x clinical effect (ΔVEP).

A biological interaction should reproduce across independent VEP models; a
model/statistical artifact should not. For each model's long-format VEP parquet
(vep_df_<model>.parquet, columns [Gene, haplotype, site, VEP]) this computes the
marginal ΔVEP per (gene, wt, site) on testable pairs (>=min_group carriers),
then reports Spearman concordance and top-1% |ΔVEP| reproducibility across
models. See docs/epistasis_correction.md.

Example:
  python cross_model_concordance.py --data-dir results/data \
      --models esm1_t34_670M_UR50D esm2_t33_650M_UR50D esm1v_t33_650M_UR90S_1 esmc_600m
"""
import argparse, os, re, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr


def parse_wt(h):
    s = str(h); rhs = s.split(":", 1)[1] if ":" in s else s
    if rhs.strip().upper() in ("REF", "", "NAN"):
        return []
    return [v.strip() for v in re.split(r"[,|]", rhs) if v.strip()]


def model_betas(parquet, min_group):
    vep = pd.read_parquet(parquet, columns=["Gene", "haplotype", "site", "VEP"])
    out = {}
    for g, sub in vep.groupby("Gene"):
        y = sub.pivot_table(index="haplotype", columns="site", values="VEP", aggfunc="mean")
        hp = list(y.index); vs = {h: set(parse_wt(h)) for h in hp}
        allwt = sorted(set().union(*vs.values())) if vs else []
        if not allwt or y.shape[1] == 0:
            continue
        Xm = np.zeros((len(hp), len(allwt))); idx = {w: j for j, w in enumerate(allwt)}
        for i, h in enumerate(hp):
            for v in vs[h]:
                if v in idx:
                    Xm[i, idx[v]] = 1.0
        Yv = y.values.astype(float); sites = list(y.columns)
        present = (Xm > 0.5); ncar = present.sum(0)
        valid = (ncar >= min_group) & ((len(hp) - ncar) >= min_group)
        for j in np.where(valid)[0]:
            c = present[:, j]
            d = np.nanmean(Yv[c], 0) - np.nanmean(Yv[~c], 0)
            for si, s in enumerate(sites):
                if np.isfinite(d[si]):
                    out[(g, allwt[j], s)] = d[si]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir with vep_df_<model>.parquet")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--min-group", type=int, default=5)
    ap.add_argument("--out", default="cross_model_betas.csv")
    a = ap.parse_args()

    betas = {}
    for m in a.models:
        pq = os.path.join(a.data_dir, f"vep_df_{m}.parquet")
        if not os.path.exists(pq):
            print(f"  skip {m}: {pq} missing"); continue
        b = model_betas(pq, a.min_group)
        print(f"  {m}: {len(b)} testable pairs")
        if b:
            betas[m] = b
    models = list(betas.keys())
    keys = sorted(set().union(*[set(b) for b in betas.values()]))
    df = pd.DataFrame(index=keys)
    for m in models:
        df[m] = pd.Series(betas[m])
    df.to_csv(a.out)

    print(f"\nunion pairs: {len(df)}")
    print("Spearman concordance of ΔVEP (shared pairs):")
    for x, y in itertools.combinations(models, 2):
        s = df[[x, y]].dropna()
        if len(s) > 50:
            print(f"  {x[:20]:20} vs {y[:20]:20}: rho={spearmanr(s[x], s[y]).correlation:.3f} (n={len(s)})")
    print("\nTop-1% |ΔVEP| reproducibility (A's top-1% also in B's):")
    vals = []
    for x, y in itertools.combinations(models, 2):
        s = df[[x, y]].dropna()
        if len(s) < 100:
            continue
        tx = s[x].abs() >= s[x].abs().quantile(0.99)
        ty = s[y].abs() >= s[y].abs().quantile(0.99)
        f = (tx & ty).sum() / max(tx.sum(), 1); vals.append(f)
        print(f"  {x[:20]:20} vs {y[:20]:20}: {100*f:.1f}%")
    if vals:
        print(f"  mean top-1% reproducibility: {100*np.mean(vals):.1f}%")
    sign_ok = (np.sign(df[models]).sum(1).abs() >= max(2, len(models) - 1)) & (df[models].notna().sum(1) >= max(2, len(models) - 1))
    print(f"\nconsistent-sign ΔVEP across >= {max(2, len(models)-1)} models: {int(sign_ok.sum())} pairs")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Corrected WT x clinical-variant epistasis test, full set.

For each (background WT variant i, clinical variant/site j):
    OLS  y_vep[:, j] ~ 1 + WT_i  ,  t-test on the slope beta_ij (= ΔVEP)
with effect sizes, isolation flag, and both Benjamini-Hochberg and a
permutation FDR (per-site y permutation; preserves within-site dependence).
See docs/epistasis_correction.md for rationale.

Two input modes:
  --mode spliceai : per-site windowed surrogate models (pickles with
                    'Xwt_clean' [hap x WT] and 'y_vep_clean' [hap x 1 site]);
                    sample is parsed from the haplotype index (e.g. HG00096_0)
                    for cluster-robust SEs.
  --mode protein  : a long-format VEP parquet with columns
                    [Gene, haplotype, site, VEP]; per gene, y is the pivot and
                    X is parsed from the haplotype strings ('PROT:v1,v2').

Outputs a per-pair CSV with: wt_variant, clinical_variant, n_with, delta_vep,
frac_carriers_isolated, abs_t, pvalue, q_bh, q_perm (+ positions).
"""
import argparse, os, glob, pickle, re
import numpy as np
import pandas as pd
from scipy import stats


def parse_wt(h):
    s = str(h); rhs = s.split(":", 1)[1] if ":" in s else s
    if rhs.strip().upper() in ("REF", "", "NAN"):
        return []
    return [v.strip() for v in re.split(r"[,|]", rhs) if v.strip()]


def firstint(s):
    m = re.search(r"(\d+)", str(s).split(":")[-1])
    return int(m.group(1)) if m else np.nan


def site_t(Xs, ys, valid, n):
    """Vectorised marginal-OLS |t| and beta for all WT columns at one site."""
    present = Xs.sum(0)
    xbar = present / n
    ybar = ys.mean()
    Sxx = present - n * xbar ** 2
    Sxy = (Xs * ys[:, None]).sum(0) - n * xbar * ybar
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = Sxy / Sxx
        ic = ybar - beta * xbar
        rss = ((ys[:, None] - (ic[None, :] + beta[None, :] * Xs)) ** 2).sum(0)
        se = np.sqrt((rss / (n - 2)) / Sxx)
        t = beta / se
    return beta, np.where(valid & np.isfinite(t), t, np.nan)


def process(X, y, B, min_group, rng):
    Xv = X.values.astype(float)
    hap_nvar = Xv.sum(1)
    isolated = (Xv * (hap_nvar == 1)[:, None]).sum(0)
    wt = list(X.columns)
    recs, null_t = [], []
    for site in y.columns:
        ya = y[site].values.astype(float)
        ym = ~np.isnan(ya)
        Xs, ys, nn = Xv[ym], ya[ym], int(ym.sum())
        if nn - 2 <= 0:
            continue
        present = Xs.sum(0); absent = nn - present
        valid = (present >= min_group) & (absent >= min_group) & (present > 0) & (absent > 0)
        if not valid.any():
            continue
        beta, t = site_t(Xs, ys, valid, nn)
        cp = firstint(site)
        for j in np.where(valid & np.isfinite(t))[0]:
            recs.append(dict(wt_variant=wt[j], clinical_variant=site,
                             wt_position=firstint(wt[j]), clinical_position=cp,
                             n_with=int(present[j]), delta_vep=float(beta[j]),
                             frac_carriers_isolated=float(isolated[j] / present[j]),
                             abs_t=float(abs(t[j])),
                             pvalue=float(2 * stats.t.sf(abs(t[j]), nn - 2))))
        for _ in range(B):
            _, tp = site_t(Xs, rng.permutation(ys), valid, nn)
            null_t.append(np.abs(tp[np.isfinite(tp)]))
    return recs, (np.concatenate(null_t) if null_t else np.array([]))


def bh(p):
    p = np.asarray(p, float); m = len(p)
    if m == 0:
        return p
    o = np.argsort(p); r = p[o] * m / (np.arange(m) + 1); q = np.empty(m)
    q[o] = np.clip(np.minimum.accumulate(r[::-1])[::-1], 0, 1)
    return q


def perm_fdr(abs_t, null_t, B):
    if not len(abs_t) or not len(null_t):
        return np.full(len(abs_t), np.nan)
    order = np.argsort(-abs_t); s = abs_t[order]
    ns = np.sort(null_t)
    exp_false = (len(ns) - np.searchsorted(ns, s, "left")) / max(B, 1)
    fdr = np.minimum(exp_false / np.arange(1, len(s) + 1), 1.0)
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    q = np.empty(len(abs_t)); q[order] = fdr
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["spliceai", "protein"], required=True)
    ap.add_argument("--models-dir", help="dir of per-site surrogate pickles (spliceai)")
    ap.add_argument("--vep-parquet", help="long-format VEP parquet (protein)")
    ap.add_argument("--out", default="wtclinical_interactions.csv")
    ap.add_argument("--B", type=int, default=20, help="permutations for permutation FDR")
    ap.add_argument("--min-group", type=int, default=5)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    recs, null = [], []

    if a.mode == "spliceai":
        for p in sorted(glob.glob(os.path.join(a.models_dir, "*.pkl"))):
            d = pickle.load(open(p, "rb"))
            X, y = d.get("Xwt_clean"), d.get("y_vep_clean")
            if X is None or y is None:
                continue
            r, nt = process(X, y, a.B, a.min_group, rng)
            recs += r
            if len(nt):
                null.append(nt)
    else:
        vep = pd.read_parquet(a.vep_parquet, columns=["Gene", "haplotype", "site", "VEP"])
        for g, sub in vep.groupby("Gene"):
            y = sub.pivot_table(index="haplotype", columns="site", values="VEP", aggfunc="mean")
            hp = list(y.index); vs = {h: set(parse_wt(h)) for h in hp}
            allwt = sorted(set().union(*vs.values())) if vs else []
            if not allwt or y.shape[1] == 0:
                continue
            X = pd.DataFrame(0.0, index=hp, columns=allwt)
            for h in hp:
                for v in vs[h]:
                    if v in X.columns:
                        X.at[h, v] = 1.0
            r, nt = process(X, y, a.B, a.min_group, rng)
            for d in r:
                d["gene"] = g
            recs += r
            if len(nt):
                null.append(nt)

    df = pd.DataFrame(recs)
    null_t = np.concatenate(null) if null else np.array([])
    if len(df):
        df["q_bh"] = bh(df["pvalue"].values)
        df["q_perm"] = perm_fdr(df["abs_t"].values, null_t, a.B)
        df["abs_delta_vep"] = df["delta_vep"].abs()
    df.to_csv(a.out, index=False)
    n = len(df)
    print(f"{a.mode}: tested={n} | BH<0.05={int((df.get('q_bh', pd.Series()) < 0.05).sum())} "
          f"| permFDR<0.05={int((df.get('q_perm', pd.Series()) < 0.05).sum())} -> {a.out}")


if __name__ == "__main__":
    main()

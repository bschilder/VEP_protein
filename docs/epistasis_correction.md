# Epistasis test: correction, validation, and biological-meaningfulness analysis

This document records a correction to the WT × clinical-variant **epistasis test**
and the analyses used to establish that the corrected interactions are
biologically meaningful. It applies to both `VEP_DNA` (SpliceAI) and
`VEP_protein` (ESM); the corrected function lives in
`src/analysis/attribution.py::test_wt_clinical_interaction`
(`src/analysis/attributions.py` in `VEP_protein`).

## 1. The bug

The original epistasis test inside
`wtvariants_to_vep_linear_model(test_epistasis=True)` and
`test_epistasis_across_models` built its "interaction" regressor as

```python
deviation      = joint_effect - expected_additive   # a per-pair SCALAR
epistasis_term = WT * deviation                      # = scalar * WT  → collinear with WT
```

`epistasis_term` is a scalar multiple of the main `WT` term, so the two columns
are **perfectly collinear**. Consequences:

- Under **OLS** a collinear column adds exactly zero explanatory power
  (`ΔR² = 0`), so the nested F-test has no power.
- Under **Ridge** (as used) the duplicate column instead *reduces* the L2
  penalty (the "grouping effect"; `min β₁²+β₂² s.t. β₁+cβ₂=s = s²/(1+c²)`,
  i.e. an effective penalty `λ/(1+c²)`), producing a **spurious** `ΔR²` /
  F-stat that scales with the clinical-effect magnitude — not with any genuine
  non-additivity.

Verified directly (BRCA1): for the 40 most-significant old hits, Ridge `ΔR²`
was 0.06–0.25 (p≈1e-16) while **OLS `ΔR²` was exactly 0**. The classical
partial-F reference distribution is only valid for OLS (Janson–Fithian–Hastie
2015; R. Tibshirani 2015; ESL §3.4.1); plugging Ridge fits into it is invalid
(Cule et al. 2011). So the old `epistasis_fstat` / `is_epistatic` numbers are
artifacts and should not be reported.

## 2. The corrected test (`test_wt_clinical_interaction`)

Because each VEP score `y_vep[h, j]` is itself the **effect (delta/LLR)** of
clinical variant *j* in haplotype background *h* (SpliceAI `max_delta_score`;
ESM masked-marginal LLR), the dependence of that effect on a background variant
`WT_i` **is** the `WT_i × clinical_j` interaction (a cross-derivative). The
corrected test, per (WT_i, clinical_j) pair:

```
OLS:  y_vep[:, j] ~ 1 + WT_i      →  t-test on the slope β_ij (= ΔVEP)
```

- **OLS** (not Ridge) so the reference distribution is exact.
- Multiple testing: **Benjamini–Hochberg** and a **permutation FDR**
  (per-site y permutation; preserves within-site dependence). Optional
  **cluster-robust SE** (`cluster_labels`, e.g. by sample → handles the
  2-haplotypes-per-sample ploidy) and Freedman–Lane **permutation null**
  (`n_permutations`).
- Null calibration verified (~5% at p<0.05; the old test failed this).

## 3. Two-track reporting (testability + power)

Testability requires a background variant carried by **≥ `min_group`** and
absent in **≥ `min_group`** haplotypes — most background variants are rarer than
that, so only a fraction of candidate pairs are testable. Report the denominator
explicitly.

| | total candidate | testable (≥5/≥5) | interacting (permutation FDR<0.05) |
|---|---|---|---|
| SpliceAI (chr17) | 104,824 | 30,529 (29%) | 11,987 (39% of testable) |
| Protein (ESM, all genes) | 3,311,739 | 109,808 (3.3%) | 9,867 (9% of testable) |

`min_group` is a power/coverage knob (lower → more rare variants, weaker
per-test power). At `min_group=1`, ~94% of protein tests rest on ≤2 carriers and
the permutation FDR is correctly ≈0 (a single point has no replication) — so
**singletons are reported as effect sizes, not significances** (see §4).

## 4. Biological meaningfulness (independent of p-values)

A p-value only says β≠0; with large n almost everything is "significant". The
corrected function and analyses establish meaningfulness via effect size,
reproducibility, and model-blind corroboration:

- **1a — `beta_rel_clinical`**: ΔVEP relative to the clinical variant's own effect.
- **1b — functional impact** (`scripts/epistasis/functional_gap.py`): of 9,867
  significant protein interactions, median |ΔVEP| is 1% of the ClinVar
  pathogenic–benign gap, but **285 (2.9%) are large enough to flip a
  benign↔pathogenic call**.
- **1c — within-site outlier**: `beta_zscore_within_site` (robust z) and
  `p_within_site` (empirical rank p) use the per-site distribution of all
  background variants' effects as a null — **valid even for singletons** (no
  replication needed).
- **2a — cross-model reproducibility**
  (`scripts/epistasis/cross_model_concordance.py`): the strongest interactions
  reproduce ~**80%** (top-1% |ΔVEP|) across independent ESM models, with 12,218
  consistent-sign across ≥4 models; the weak tail (ρ≈0.3) does not — so report
  the strong, reproducible interactions, not the tail.
- **structural** (prior analysis): top-1% interactions are **~10× enriched** for
  AlphaFold2 3D contacts.

**Recommended framing:** prioritize interactions by effect size, not
significance; the strong interactions are reproducible across models, enriched
for 3D contacts, and a small fraction (~3%) are clinically impactful —
establishing biological meaningfulness independent of p.

## 5. Reproducing

```bash
# corrected WT×clinical test, full set (effect sizes + BH & permutation FDR)
python scripts/epistasis/run_wt_clinical_interaction.py --mode spliceai
python scripts/epistasis/run_wt_clinical_interaction.py --mode protein \
    --vep-parquet <vep_df_<model>.parquet>
# cross-model concordance (protein) and functional-gap impact (protein)
python scripts/epistasis/cross_model_concordance.py --data-dir <results/data>
python scripts/epistasis/functional_gap.py --vep-parquet <vep_df_<model>.parquet> \
    --effects <protein_wtclinical.csv>
```

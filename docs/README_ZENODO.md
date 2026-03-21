# Zenodo data archives for personalizedVEP

This document describes the contents of the Zenodo archives associated with the personalizedVEP manuscript. Two zip files are uploaded to the **personalizedVEP** record:

- **personalizedVEP_data.zip** — Variant effect prediction (VEP) data, haplotype data, surrogate models, and misc tables.
- **personalizedFolding_data.zip** — ColabFold / structure-related outputs (large; separate zip due to size).

Paths below are relative to the root of each zip.

---

## personalizedVEP_data.zip

### `vep/esm/`

Variant effect prediction (VEP) scores from protein language models (ESM). One Parquet file per model and optionally per scoring strategy.

- **Files:** `vep_df_esm*.parquet` (e.g. `vep_df_esm2_t33_650M_UR50D.parquet`, `vep_df_esm1b_t33_650M_UR50S.parquet`).
- **Contents:** Long-format tables with columns for protein, haplotype, sample, site, variant, scoring strategy (e.g. wt-marginals, masked-marginals, pseudo-ppl), and VEP score. Produced by the VEP pipeline; used for manuscript figures and downstream analysis.

### `vep/spliceai`

SpliceAI-based variant annotations merged with ClinVar.

- **File:** `spliceai_clinvar_merged.parquet`
- **Contents:** SpliceAI scores and ClinVar metadata for variants (e.g. 1000 Genomes on GRCh38). Used for splicing-related figures and comparisons with protein VEP.

### `vep/flashzoi`

FlashZoi (UTR) variant effect predictions.

- **File:** `vep_df.parquet`
- **Contents:** Variant effect predictions for ClinVar UTR SNVs from the FlashZoi pipeline. Used for UTR/splicing analyses.

### `haplosaurus/`

Haplotype data from Ensembl Haplosaurus (1000 Genomes or similar cohort).

- **Files:** One gzipped JSON per transcript (e.g. `ENST00000000233.json.gz`).
- **Contents:** Per-transcript haplotype information: protein and/or CDS haplotype sequences, sample IDs, and metadata. Produced by Haplosaurus; used by the VEP pipeline and haplotype-aware analyses. See Ensembl [Haplosaurus](https://useast.ensembl.org/info/docs/tools/vep/haplo/index.html).

### `surrogate/esm/`

Fitted linear (Ridge) models for joint wild-type variant–VEP effects per gene (protein ESM).

- **Files:** `wtvariants_to_vep_*.pkl` (e.g. `wtvariants_to_vep_linear_model_out_BRCA1.pkl`, `wtvariants_to_vep_linear_model_out_TP53.pkl`).
- **Contents:** Serialized model output from the ESM-based surrogate (e.g. `wtvariants_to_vep_linear_model`). Used for interaction / epistasis-style analyses and figures.

### `surrogate/spliceai/`

Fitted Ridge surrogate models for SpliceAI.

- **Files:** `ridge_model_*.pkl` (one per gene or locus).
- **Contents:** Serialized Ridge surrogate models trained on SpliceAI outputs. Used for DNA-level surrogate analyses and comparison with protein ESM surrogates.

### `misc/`

Miscellaneous tables and summaries used across the manuscript and notebooks.

| File | Description |
|------|-------------|
| `freq_df.parquet` | Haplotype or variant frequency table (e.g. from 1000 Genomes or cohort). Used for weighted analyses. |
| `normality_results.parquet` | Results of normality tests on score distributions (e.g. VEP score distributions per site or model). |
| `*_n_variants_df.parquet` | Variant counts or score summaries per variant/site for SpliceAI (`spliceai_n_variants_df.parquet`) and FlashZoi (`flashzoi_n_variants_df.parquet`). Used for splicing-related figures. |
| `vep_ecdf.parquet` | Empirical cumulative distribution function (ECDF) of VEP scores. Used for ECDF/percentile figures. |
| `vep_ecdf_umap.tsv.gz` | UMAP embedding of VEP ECDF or score summaries. Used for embedding/visualization figures. |

---

## personalizedFolding_data.zip

### `colabfold/`

ColabFold (AlphaFold2-style) structure prediction outputs.

- **Contents:** Directory tree produced by ColabFold runs: input MSAs, structure PDB/CIF files, confidence metrics, and logs. Typically one subfolder per protein or run; many files (e.g. >8k) and large total size (~19 GB). Used for structure-based analyses and categorical Jacobian / sensitivity maps in the manuscript.

---

## Regenerating and uploading

- **personalizedVEP_data.zip:** Build and upload via `notebooks/Zenodo.ipynb` using `zenodo.upload_zip_to_draft(..., zip_filename="personalizedVEP_data.zip")` with the `items` dict defined there. VEP parquet files come from the VEP pipeline; haplosaurus and surrogate paths point to external data dirs.
- **personalizedFolding_data.zip:** Same notebook; separate `upload_zip_to_draft(..., zip_filename="personalizedFolding_data.zip")` with `items={"colabfold/": "~/projects/data/colabfold/"}` (or your local path).

See `notebooks/Zenodo.ipynb` and `src/zenodo.py` for upload/download helpers and token setup.

import os
import glob
from re import T
import pandas as pd
import numpy as np
from tqdm import tqdm
import pooch
import seaborn as sns
import matplotlib.pyplot as plt


import src.utils as utils

# Categories:
# - pathogenicity
# - splicing 
# - protein
# - conservation
# - MPRA
# - population genetics
# - regulatory
# - gene constraint
# - clinical
# - other

ANNOT_DICT = {
    # Pathogenicity predictors
    "CADD_PHRED": "pathogenicity",
    "CADD_RAW": "pathogenicity",
    "ClinPred": "pathogenicity", 
    "BayesDel_addAF_score": "pathogenicity",
    "BayesDel_noAF_score": "pathogenicity",
    "DANN_score": "pathogenicity",
    "Eigen-PC-phred_coding": "pathogenicity",
    "Eigen-PC-raw_coding": "pathogenicity",
    "Eigen-phred_coding": "pathogenicity",
    "Eigen-raw_coding": "pathogenicity",
    "MPC_score": "pathogenicity",
    "MetaLR_score": "pathogenicity",
    "MetaRNN_score": "pathogenicity",
    "MetaSVM_score": "pathogenicity",
    "MutationTaster_converted_rankscore": "pathogenicity",
    "Reliability_index": "pathogenicity",

    # Splicing
    "rf_score": "splicing",
    "ada_score": "splicing",
    "MaxEntScan_ref": "splicing",
    "MaxEntScan_alt": "splicing",
    "MaxEntScan_diff": "splicing",
    "SpliceAI_pred_DP_AG": "splicing",
    "SpliceAI_pred_DP_AL": "splicing",
    "SpliceAI_pred_DP_DG": "splicing",
    "SpliceAI_pred_DP_DL": "splicing",
    "SpliceAI_pred_DS_AG": "splicing",
    "SpliceAI_pred_DS_AL": "splicing",
    "SpliceAI_pred_DS_DG": "splicing",
    "SpliceAI_pred_DS_DL": "splicing",

    # Protein 

    "gMVP_score": "protein",
    "fathmm-XF_coding_score": "protein",
    "PROVEAN_converted_rankscore": "protein",
    "PROVEAN_pred": "protein",
    "PROVEAN_score": "protein",
    "MutationAssessor_score": "protein",
    "MVP_score": "protein",
    "LIST-S2_score": "protein",
    "VARITY_ER_LOO_score": "protein",
    "VARITY_ER_score": "protein",
    "VARITY_R_LOO_score": "protein",
    "VARITY_R_score": "protein",
    "DEOGEN2_score": "protein",
    "PrimateAI_pred": "protein",
    "PrimateAI_score": "protein",
    "MutFormer_score": "protein",
    "ESM1b_score": "protein",
    "REVEL": "protein",
    "EVE_SCORE": "protein",
    "am_pathogenicity": "protein",
    "SIFT_score": "protein",
    "PolyPhen_score": "protein",
    "BLOSUM62": "protein",
    "MOTIF_SCORE_CHANGE": "protein",
    "mutfunc_exp": "protein",
    "mutfunc_int": "protein",
    "mutfunc_mod": "protein",
    "mutfunc_motif": "protein",

    # Conservation
    "GERP++_NR": "conservation",
    "GERP++_RS": "conservation",
    "GERP_91_mammals": "conservation",
    "phastCons100way_vertebrate": "conservation",
    "phastCons17way_primate": "conservation",
    "phastCons470way_mammalian": "conservation",
    "phyloP100way_vertebrate": "conservation",
    "phyloP17way_primate": "conservation",
    "phyloP470way_mammalian": "conservation",

    # MPRA/experimental
    "MaveDB_score_mean": "MPRA",
    "MaveDB_score_abs_mean": "MPRA",
    # "MaveDB_score_min": "MPRA",
    # "MaveDB_score_max": "MPRA",

    # Regulatory
    "OpenTargets_l2g": "regulatory",
    "Enformer_SAD": "regulatory",
    "Enformer_SAR": "regulatory",

    # Population genetics/allele frequency
    "AF": "population",
    "AFR_AF": "population",
    "AMR_AF": "population",
    "EAS_AF": "population",
    "EUR_AF": "population",
    "SAS_AF": "population",
    "gnomADe_AF": "population",
    "gnomADe_AFR_AF": "population",
    "gnomADe_AMR_AF": "population",
    "gnomADe_ASJ_AF": "population",
    "gnomADe_EAS_AF": "population",
    "gnomADe_FIN_AF": "population",
    "gnomADe_MID_AF": "population",
    "gnomADe_NFE_AF": "population",
    "gnomADe_REMAINING_AF": "population",
    "gnomADe_SAS_AF": "population",
    "gnomADg_AF": "population",
    "gnomADg_AFR_AF": "population",
    "gnomADg_AMI_AF": "population",
    "gnomADg_AMR_AF": "population",
    "gnomADg_ASJ_AF": "population",
    "gnomADg_EAS_AF": "population",
    "gnomADg_FIN_AF": "population",
    "gnomADg_MID_AF": "population",
    "gnomADg_NFE_AF": "population",
    "gnomADg_REMAINING_AF": "population",
    "gnomADg_SAS_AF": "population",
    "AF_TGP": "population",
    "AllOfUs_gvs_all_af": "population",
    "AllOfUs_gvs_max_af": "population",
    "AllOfUs_gvs_afr_af": "population",
    "AllOfUs_gvs_amr_af": "population",
    "AllOfUs_gvs_eas_af": "population",
    "AllOfUs_gvs_eur_af": "population",
    "AllOfUs_gvs_mid_af": "population",
    "AllOfUs_gvs_oth_af": "population",
    "AllOfUs_gvs_sas_af": "population",
    "1000Gp3_AF": "population",
    "1000Gp3_AFR_AF": "population",
    "1000Gp3_AMR_AF": "population",
    "1000Gp3_EAS_AF": "population",
    "1000Gp3_EUR_AF": "population",
    "1000Gp3_SAS_AF": "population",
    "ALFA_African_AF": "population",
    "ALFA_African_American_AF": "population",
    "ALFA_African_Others_AF": "population",
    "ALFA_Asian_AF": "population",
    "ALFA_East_Asian_AF": "population",
    "ALFA_European_AF": "population",
    "ALFA_Latin_American_1_AF": "population",
    "ALFA_Latin_American_2_AF": "population",
    "ALFA_Other_AF": "population",
    "ALFA_Other_Asian_AF": "population",
    "ALFA_South_Asian_AF": "population",
    "ALFA_Total_AF": "population",
    "TOPMed_frz8_AF": "population",

    # Constraint
    "LOEUF": "constraint",
    "pHaplo": "constraint",
    "pTriplo": "constraint",
    "bStatistic": "constraint",
    "bStatistic_converted_rankscore": "constraint", 

    # Clinical/phenotype
    "Geno2MP_HPO_count": "clinical",

    # Other
    # (add more as needed)
}

ANNOT_COLS = list(ANNOT_DICT.keys())

def filter_annotations(
    df,
    df_filters={"Location": "Location"},
    # cache_search=os.path.join(pooch.os_cache("pooch"), "clinvar_by_chrom", "clinvarVEP*"),
    cache_search=os.path.join(os.path.expanduser("~/projects/data/ensemblVEP"), "clinvarVEP*"),
    cached_file=None,
    force=False
):
    """
    Filter and annotate [Ensembl VEP (Variant Effect Predictor)](https://useast.ensembl.org/Tools/VEP) 
    annotation files for ClinVar variants.

    This function loads, filters, and processes VEP annotation files for ClinVar variants,
    optionally caching the merged and filtered results for future use. It also extracts
    and computes additional annotation columns such as SIFT/PolyPhen scores and MaveDB statistics.

    Args:
        df (pd.DataFrame): DataFrame containing the variants of interest. Used to filter the VEP annotations.
        df_filters (dict, optional): Dictionary mapping column names in `df` to column names in the VEP annotation files
            for filtering. Default is {"Location": "Location"}.
        cached_file (str, optional): Path to the cached, merged, and filtered annotation file. If the file exists and
            `force` is False, it will be loaded instead of recomputing. Default is a file in the pooch cache directory.
        force (bool, optional): If True, forces regeneration of the merged and filtered annotation file even if the
            cached file exists. Default is False.

    Returns:
        pd.DataFrame: The merged, filtered, and annotated DataFrame containing VEP annotations for the variants of interest.

    Notes:
        - The function expects VEP annotation files to be located in the "clinvar_by_chrom" subdirectory of the pooch cache.
        - Additional columns are computed:
            - "mutant": Concatenation of reference amino acid, protein position, and alternate amino acid.
            - "ENST": Transcript ID without version.
            - "SIFT_score" and "PolyPhen_score": Extracted numeric scores from the respective columns.
            - "MaveDB_score_mean", "MaveDB_score_abs_mean", "MaveDB_score_min", "MaveDB_score_max": Statistics computed from comma-separated MaveDB scores.
    """
    if cached_file is not None and os.path.exists(cached_file) and not force:
        print(f"Reading from {cached_file}")
        cv_annot = pd.read_parquet(cached_file)
    else:
        # Get all VEP annotation files in the cache directory
        cv_vep_files = glob.glob(cache_search)
        if len(cv_vep_files) == 0:
            raise FileNotFoundError(f"No VEP annotation files found in {pooch.os_cache('pooch')}/clinvar_by_chrom")
        
        # Check that each of the filters are present in the VEP annotation files
        for col, filter_col in df_filters.items():
            if col not in df.columns:
                raise ValueError(f"Filter column {col} not found in `df`")

        cv_annot = []
        for f in tqdm(cv_vep_files, desc="Processing VEP annotation files"):
            cv_tmp = pd.read_csv(f, sep="\t", 
                                 low_memory=False, 
                                 na_values=['-'],
                                #  dtype={"Amino_acids": str,
                                #         "Protein_position": str}
                                 ) 
            cv_tmp["chrom"] = "chr" + cv_tmp["Location"].str.split(":").str[0]
            cv_tmp["chromStart"] = cv_tmp["Location"].str.split(":").str[1].str.split("-").str[0]  
            
            # Convert all columns to string/object dtype before creating polars DataFrame to avoid ArrowTypeError
            for col in cv_tmp.columns:
                cv_tmp[col] = cv_tmp[col].astype(str)
            
            # Add variant name column
            cv_tmp = utils.add_variant_name(cv_tmp,  
                            chrom_col='chrom',
                            start_col='chromStart',
                            end_col=None,
                            ref_col='REF_ALLELE',
                            alt_col='Allele',
                            alias='site')
            
            # Filter according to df_filters
            for col, filter_col in df_filters.items():
                cv_tmp = cv_tmp.loc[
                    cv_tmp[filter_col].isin(df[col].unique().tolist())
                ]

            # Add to list
            print(os.path.basename(f),cv_tmp.shape)
            cv_annot.append(cv_tmp)

        # Merge and save all filtered annotations
        cv_annot = pd.concat(cv_annot)
        
        # Convert numeric columns that are currently object type to float
        # This is necessary for parquet conversion (e.g., mutfunc_mod, mutfunc_exp, etc.)
        # Try to convert all object columns that contain numeric strings
        print("Converting numeric string columns to float")
        for col in cv_annot.columns:
            if cv_annot[col].dtype == 'object':
                # Check if column appears to contain numeric values
                # Sample a few non-null values to check
                sample_values = cv_annot[col].dropna().head(100)
                if len(sample_values) > 0:
                    # Try to convert a sample to see if it's numeric
                    try:
                        test_convert = pd.to_numeric(sample_values, errors='coerce')
                        # If most values can be converted, convert the whole column
                        if test_convert.notna().sum() / len(sample_values) > 0.5:
                            cv_annot[col] = pd.to_numeric(cv_annot[col], errors='coerce')
                    except Exception:
                        # If conversion fails, keep as object
                        pass
        
        if cached_file is not None:
            print(f"Caching merged and filtered annotations --> {cached_file}")
            cv_annot.to_parquet(cached_file)

    # Extract numeric values from SIFT and PolyPhen columns using regex
    print("Extracting numeric values from SIFT and PolyPhen columns")
    cv_annot['SIFT_score'] = cv_annot['SIFT'].str.extract(r'\(([\d.]+)\)').astype(float)
    cv_annot['PolyPhen_score'] = cv_annot['PolyPhen'].str.extract(r'\(([\d.]+)\)').astype(float)

    # Compute statistics from MaveDB_score column (comma-separated values)
    print("Computing statistics from MaveDB_score column")
    cv_annot['MaveDB_score'] = cv_annot['MaveDB_score'].astype(str)
    def safe_float(x):
        try:
            if x in [None, 'None', 'nan', 'NaN', '']:
                return np.nan
            return float(x)
        except Exception:
            return np.nan

    cv_annot['MaveDB_score_mean'] = cv_annot['MaveDB_score'].str.split(',').apply(
        lambda x: np.nanmean([safe_float(i) for i in x]) if isinstance(x, list) else np.nan
    )
    cv_annot['MaveDB_score_abs_mean'] = cv_annot['MaveDB_score'].str.split(',').apply(
        lambda x: np.nanmean([abs(safe_float(i)) for i in x]) if isinstance(x, list) else np.nan
    )
    cv_annot['MaveDB_score_min'] = cv_annot['MaveDB_score'].str.split(',').apply(
        lambda x: np.nanmin([safe_float(i) for i in x]) if isinstance(x, list) else np.nan
    )
    cv_annot['MaveDB_score_max'] = cv_annot['MaveDB_score'].str.split(',').apply(
        lambda x: np.nanmax([safe_float(i) for i in x]) if isinstance(x, list) else np.nan
    )


    # Create a new column using string operations in a more efficient way
    print("Creating chrom and chromStart columns")
    cv_annot["chrom"] = "chr" + cv_annot["Location"].str.split(":").str[0]
    cv_annot["chromStart"] = cv_annot["Location"].str.split(":").str[1].str.split("-").str[0]  
    
    print("Adding variant name column")
    cv_annot = utils.add_variant_name(cv_annot,  
                                      chrom_col='chrom',
                                        start_col='chromStart',
                                        end_col=None,
                                        ref_col='REF_ALLELE',
                                        alt_col='Allele',
                                        alias='site')
    
   
    #   Add mutant column: ref_aa + position + alt_aa
    if "Amino_acids" in cv_annot.columns:
        print("Adding mutant column")
        cv_annot.loc[:, "mutant"] = (
            cv_annot["Amino_acids"].str.split("/").str[0]
            + cv_annot["Protein_position"].astype(str)
            + cv_annot["Amino_acids"].str.split("/").str[1]
        ) 
    
    # Add ENST column: transcript ID without version
    print("Adding ENST column")
    cv_annot.loc[:, "ENST"] = cv_annot["Feature"].str.split(".").str[0]
    
    print(cv_annot.shape)
    return cv_annot

def _robust_to_numeric(series, col_name=None, verbose=False):
    """
    Robustly convert a pandas Series to numeric, handling lists, arrays, and string representations.
    
    Parameters
    ----------
    series : pd.Series
        Series to convert to numeric
    col_name : str, optional
        Name of the column for error messages
    verbose : bool, optional
        If True, print diagnostic information about conversion
    
    Returns
    -------
    pd.Series
        Series converted to float, with unconvertible values set to NaN
    """
    import ast
    
    # First, handle list/array cases
    def convert_value(x):
        # Handle NaN/None
        if pd.isna(x):
            return np.nan
        
        # Handle list/array types
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) > 0:
                try:
                    return float(x[0])
                except (ValueError, TypeError, IndexError):
                    return np.nan
            return np.nan
        
        # Handle string representations of lists/arrays
        if isinstance(x, str):
            x_stripped = x.strip()
            if x_stripped.startswith('[') and x_stripped.endswith(']'):
                try:
                    arr = np.array(ast.literal_eval(x))
                    if len(arr) > 0:
                        return float(arr[0])
                    return np.nan
                except (ValueError, SyntaxError, TypeError):
                    pass
        
        # Try direct conversion
        try:
            return float(x)
        except (ValueError, TypeError):
            return np.nan
    
    # Apply conversion
    converted = series.apply(convert_value)
    
    # Use pd.to_numeric as a fallback for any remaining non-numeric values
    # This handles edge cases like strings that look numeric
    converted = pd.to_numeric(converted, errors='coerce')
    
    if verbose and col_name:
        original_count = series.notna().sum()
        converted_count = converted.notna().sum()
        dropped = original_count - converted_count
        if dropped > 0:
            print(f"  {col_name}: Converted {converted_count}/{original_count} values ({dropped} dropped)")
    
    return converted

def run_correlation_analysis(vep_annot, 
                             group_col="is_ref",
                             vep_col="VEP",
                             ANNOT_COLS=ANNOT_COLS,
                             ref_vs_all=False,
                             method="spearman",
                             transform=None,
                             transform_kwargs={},
                             leave=False,
                             verbose=True):
    """
    Perform Spearman correlation analysis between VEP scores and annotation columns.

    For each annotation column in ANNOT_COLS, this function computes the Spearman correlation
    coefficient (rho) between the 'VEP' column and the annotation column, separately for
    reference ('is_ref' == True) and non-reference ('is_ref' == False) variants. The absolute
    difference in correlation coefficients between non-reference and reference is also computed.

    Parameters
    ----------
    vep_annot : pd.DataFrame
        DataFrame containing VEP scores, annotation columns, and an 'is_ref' boolean column.
    group_col : str, optional
        Name of the column containing the group to split the data into. Default is "is_ref".
    vep_col : str, optional
        Name of the column containing the VEP scores. Default is "VEP".
    ANNOT_COLS : list, optional
        List of annotation columns to include in the analysis. Default is ANNOT_COLS.
    ref_vs_all : bool, optional
        If False (default), compares the difference in correlation between reference VEPs and non-reference (personalized) VEPs.
        If True, compares the difference in correlation between reference VEPs and all VEPs.
    verbose : bool, optional
        If True, prints a summary of the correlation results. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing the Spearman correlation coefficients for each annotation column,
        including the reference, non-reference, and their absolute difference.

    Notes
    -----
    - The function skips annotation columns not present in the input DataFrame.
    - Only rows with non-missing values in both the annotation column and 'VEP' are used.
    - Set the 'plot' variable to True to enable plotting (currently disabled).
    """
    if method == "spearman":
        from scipy.stats import spearmanr as corr
    elif method == "pearson":
        from scipy.stats import pearsonr as corr
    else:
        raise ValueError(f"Invalid method: {method}")
    
    import warnings
 
    r2_results = []
    skipped_cols = []
    conversion_stats = []
    
    if group_col not in vep_annot.columns:
        raise ValueError(f"Group column {group_col} not found in vep_annot dataframe")
    
    # Track initial data size
    initial_rows = len(vep_annot)
    if verbose:
        print(f"Starting correlation analysis with {initial_rows} rows")
    
    for col in tqdm(ANNOT_COLS, 
                    desc="Calculating Spearman r", 
                    leave=leave):
        # print(col)
        

        if col not in vep_annot.columns:
            skipped_cols.append((col, "not in dataframe"))
            if verbose:
                warnings.warn(f"Skipping {col} because it is not in the vep_annot dataframe")
            continue

        vep_annot_sub = vep_annot.dropna(subset=[col, vep_col]) 

        if vep_annot_sub.empty:
            skipped_cols.append((col, "no non-missing values"))
            if verbose:
                warnings.warn(f"Skipping {col} because it has no non-missing values")
            continue
    
        # Track original count before conversion
        original_count = len(vep_annot_sub)
        
        # Apply robust conversion to VEP and annotation columns
        # This handles lists, arrays, and string representations gracefully
        vep_annot_sub = vep_annot_sub.copy()  # Avoid SettingWithCopyWarning
        vep_annot_sub.loc[:, vep_col] = _robust_to_numeric(
            vep_annot_sub[vep_col], 
            col_name=f"{vep_col}", 
            verbose=verbose
        )
        vep_annot_sub.loc[:, col] = _robust_to_numeric(
            vep_annot_sub[col], 
            col_name=f"{col}", 
            verbose=verbose
        )
        
        # Drop rows where conversion failed (resulted in NaN)
        vep_annot_sub = vep_annot_sub.dropna(subset=[col, vep_col])
        dropped_count = original_count - len(vep_annot_sub)
        
        if dropped_count > 0 and verbose:
            warnings.warn(
                f"{col}: Dropped {dropped_count}/{original_count} rows after conversion "
                f"({100*dropped_count/original_count:.1f}%)"
            )
        
        if vep_annot_sub.empty:
            skipped_cols.append((col, "all values dropped during conversion"))
            if verbose:
                warnings.warn(f"Skipping {col} because all values were dropped during conversion")
            continue
        
        # Track conversion stats
        conversion_stats.append({
            'annotation': col,
            'rows_before_conversion': original_count,
            'rows_after_conversion': len(vep_annot_sub),
            'rows_dropped': dropped_count,
            'pct_dropped': 100 * dropped_count / original_count if original_count > 0 else 0
        })

        if transform is not None:
            vep_annot_sub.loc[:, vep_col] = vep_annot_sub[vep_col].transform(transform, **transform_kwargs)
            vep_annot_sub.loc[:, col] = vep_annot_sub[col].transform(transform, **transform_kwargs) 

        # Calculate Spearman r for each facet
        try:
            ref_n = vep_annot_sub[vep_annot_sub[group_col]].shape[0]
            ref_r, ref_p = corr(
                vep_annot_sub[vep_annot_sub[group_col]][vep_col],
                vep_annot_sub[vep_annot_sub[group_col]][col]
            )
        except Exception as e:
            skipped_cols.append((col, f"error calculating ref_r: {str(e)}"))
            if verbose:
                warnings.warn(f"Error calculating ref_r for {col}: {e}")
            continue

        # Calculate Spearman r for all VEPs
        try:
            if ref_vs_all:
                nonref_n = vep_annot_sub.shape[0]
                nonref_r, nonref_p = corr(
                    vep_annot_sub[vep_col],
                    vep_annot_sub[col]
                )
            # Calculate Spearman r for non-reference VEPs
            else:
                nonref_n = vep_annot_sub[~vep_annot_sub[group_col]].shape[0]
                nonref_r, nonref_p = corr(
                    vep_annot_sub[~vep_annot_sub[group_col]][vep_col],
                    vep_annot_sub[~vep_annot_sub[group_col]][col]
                )
        except Exception as e:
            skipped_cols.append((col, f"error calculating nonref_r: {str(e)}"))
            if verbose:
                warnings.warn(f"Error calculating nonref_r for {col}: {e}")
            continue

        # Compute the difference in Rho
        r_diff = nonref_r - ref_r 
        rabs_diff = abs(nonref_r) - abs(ref_r)
        r2_diff =  nonref_r**2 - ref_r**2 

        r2_results.append({
            'annotation': col,
            'ref_r': ref_r,
            'nonref_r': nonref_r,
            'r_diff': r_diff,
            'rabs_diff': rabs_diff,
            'r2_diff': r2_diff,
            "ref_n": ref_n,
            "nonref_n": nonref_n,
            "ref_p": ref_p,
            "nonref_p": nonref_p
        }) 

    # Convert results to DataFrame for easy viewing
    r2_df = pd.DataFrame(r2_results)


    # Calculate FDR (q-values) for non-reference p-values using Benjamini-Hochberg
    from statsmodels.stats.multitest import multipletests
    r2_df["ref_fdr"] = multipletests(r2_df["ref_p"], method="fdr_bh")[1]
    r2_df["nonref_fdr"] = multipletests(r2_df["nonref_p"], method="fdr_bh")[1]

    # Calculate scores that weight the correlation by the p-value, then calclate the difference between groups
    r2_df["combined_diff"] = (r2_df["nonref_r"] *(1-r2_df["nonref_p"])) - (r2_df["ref_r"] *(1-r2_df["ref_p"]))
    r2_df["combined_abs_diff"] = (r2_df["nonref_r"].abs() *(1-r2_df["nonref_p"])) - (r2_df["ref_r"].abs() *(1-r2_df["ref_p"]))
    r2_df["combined2_diff"] = (r2_df["nonref_r"] *(1-r2_df["nonref_p"])).pow(2) - (r2_df["ref_r"] *(1-r2_df["ref_p"])).pow(2)

    
    if verbose:
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total annotations processed: {len(r2_results)}/{len(ANNOT_COLS)}")
        print(f"Successfully analyzed: {len(r2_results)}")
        print(f"Skipped: {len(skipped_cols)}")
        
        if skipped_cols:
            print("\nSkipped annotations:")
            skip_reasons = {}
            for col, reason in skipped_cols:
                if reason not in skip_reasons:
                    skip_reasons[reason] = []
                skip_reasons[reason].append(col)
            for reason, cols in skip_reasons.items():
                print(f"  {reason}: {len(cols)} annotations")
                if len(cols) <= 10:
                    print(f"    {', '.join(cols)}")
                else:
                    print(f"    {', '.join(cols[:10])} ... and {len(cols)-10} more")
        
        if conversion_stats:
            conv_df = pd.DataFrame(conversion_stats)
            avg_dropped = conv_df['pct_dropped'].mean()
            max_dropped = conv_df['pct_dropped'].max()
            print(f"\nConversion statistics:")
            print(f"  Average rows dropped per annotation: {avg_dropped:.1f}%")
            print(f"  Maximum rows dropped: {max_dropped:.1f}%")
            if max_dropped > 50:
                worst = conv_df.loc[conv_df['pct_dropped'].idxmax()]
                print(f"  Worst case: {worst['annotation']} ({worst['pct_dropped']:.1f}% dropped)")
        
        print("\nR2 Results Summary:")
        print(r2_df.sort_values('r_diff', ascending=False))
        print("="*80)
    
    return r2_df


def plot_correlation_analysis(
    r2_df,
    x_var="r2_diff",
    y_var="annotation",
    figsize=(6, 7),
    ylabel="Annotation",
    xlabel="Difference in Correlation Between Non-ref vs. Ref\n"
           r"($\rho_{\text{non-ref}} \text{ }  \Delta \text{ } \rho_{\text{ref}}$)",
    legend_title="non-ref\ncorrelation\n"
                 r"$(|ρ_{\text{non-ref}}|)$",
    title=None,
    add_summary_subtitle=True,
    win_rate_on_newline=True,
    hue="nonref_r_abs",
    palette="flare_r",
    # Filtering options
    min_n=None,
    max_p=None,
    min_diff=None,
    annotations=None,
    # Faceting option
    facet_col=None,
    facet_col_wrap=2,
    facet_sharex=True,
    facet_sharey=True,
    facet_height=None,
    facet_aspect=1.2,
    facet_legend="auto",
    # New argument for category column
    show_category_column=False,
    category_dict=None,
    category_palette=None,
    category_legend_title="Category",
    category_legend_loc='center left',
    category_legend_bbox_to_anchor=(0.8, 0.25),
    category_legend_fontsize='small',
    category_box_linewidth=1,
    show_category_legend=True,
    legend_loc=None,
    legend_bbox_to_anchor=None,
    legend_fontsize='small',
    ticklabel_fontsize='small',
    title_ha='left',
    title_x=None,
    flip_axes=False,
    yticklabel_rotation=0,
    yticklabel_pad=None,
    yticklabel_ha='right',
    barplot_kwargs={},
    draw_positive_negative_divider=True,
    divider_line_kwargs={},
    title_fontsize='medium',
):
    """
    Plot the correlation analysis results from r2_df as a barplot.

    Allows a wide variety of customization for annotation ordering, color, filtering, axes,
    and category highlights.

    Parameters
    ----------
    r2_df : pd.DataFrame
        Table of correlation analysis results (see output of `analyze_correlation_by_annotation`).
    x_var : str, default "r2_diff"
        Column name for x-axis (by default, difference in r^2/correlation between non-reference and reference).
    y_var : str, default "annotation"
        Column name for y-axis, by default annotation names.
    figsize : tuple, default (6,7)
        Figure size.
    ylabel, xlabel : str
        Axis labels.
    legend_title : str
        Legend title for bar color coding.
    title : str or None
        Figure title. If None and add_summary_subtitle=True, displays "Non-ref win rate: N%".
    add_summary_subtitle : bool
        If True, append the non-ref win rate to the title (fraction with x_var > 0).
    win_rate_on_newline : bool, default True
        If True, the win rate text is displayed on a new line after the title.
        If False, the win rate text is displayed on the same line as the title, separated by a space.
    hue : str, default "nonref_r_abs"
        r2_df column to use for hue/color scale across bars.
    palette : str or dict
        Seaborn color palette, by default "flare_r" (yellow→cyan→purple).
    min_n : int, optional
        Require at least min_n reference and non-reference variants per annotation.
    max_p : float, optional
        Restrict to annotations with both ref_p and nonref_p less than max_p.
    min_diff : float, optional
        Restrict to annotations with abs(x_var) > min_diff.
    annotations : list, optional
        Restrict to only these annotation names (strings or list of strings).
    facet_col : str, optional
        If not None, will facet the plot into columns by this column name.
    facet_col_wrap : int
        Maximum number of facets in a row.
    facet_sharex, facet_sharey : bool
        Whether to share axis limits across facets.
    facet_height : float
        Height of each facet in inches (default scales by number of facets).
    facet_aspect : float
        Aspect ratio (width/height) of each facet.
    facet_legend : str or bool
        Facet legend argument passed through to seaborn.catplot.
    show_category_column : bool
        If True, draws colored boxes indicating category beside the annotation labels.
        Categories are determined by `category_dict` mapping from annotation to category name.
    category_dict : dict or None
        Mapping from annotation names to category. If None, falls back to global ANNOT_DICT.
    category_palette : dict or None
        Mapping from category name to RGB/hex color, or None to use default colors.
    category_legend_title : str
        Title for category legend.
    category_legend_loc : str
        Location for category legend if not faceted and axes not flipped.
    category_legend_bbox_to_anchor : tuple
        Anchor for category legend.
    category_legend_fontsize : int or str, default 'small'
        Font size for category legend labels. E.g. "small", "medium", 10, etc.
    category_box_linewidth : float, default 1
        Line width (in points) for the border around category annotation boxes.
    show_category_legend : bool, default True
        If True, displays a legend showing the category colors. If False, the category
        boxes are still shown but no legend is displayed.
    legend_loc : str, optional
        Location for the non-ref correlation legend. If None, uses default locations
        ('upper right' when flip_axes=True, 'lower right' when flip_axes=False).
    legend_bbox_to_anchor : tuple, optional
        Anchor for the non-ref correlation legend. If None, uses default positioning.
        Example: (1.0, 1.0) for top-right corner.
    legend_fontsize : int or str, default 'small'
        Font size for non-ref correlation legend labels. E.g. "small", "medium", 10, etc.
    ticklabel_fontsize : int or str
        Size of tick labels. E.g. "small", 12, etc.
    title_ha : str, default 'left'
        Horizontal alignment for the plot title. Options: 'left', 'center', 'right'.
    title_x : float, optional
        Horizontal position of the title. If None (default), aligns with the left edge
        of the plot area (accounting for y-tick labels). For manual control, use:
        0.0 = left edge of axes, 0.5 = center, 1.0 = right edge.
    flip_axes : bool
        If True, swap x and y axes.
    yticklabel_rotation : float
        Rotation angle of annotation labels (applies to x-labels if flip_axes=True).
    yticklabel_pad : float, optional
        Horizontal padding/offset for y-axis tick labels when flip_axes=False and 
        show_category_column=True. Negative values move labels leftwards. 
        Default is -0.05. Larger negative values (e.g., -0.15) provide more spacing 
        from the category rectangles.
    yticklabel_ha : str, default 'right'
        Horizontal alignment/justification for tick labels. Options: 'left', 'center', 'right'.
        Default is 'right' for right-justified labels.
    barplot_kwargs : dict, optional
        Keyword arguments passed through to seaborn.barplot.
    draw_positive_negative_divider : bool, default True
        If True, draw a horizontal line between annotations where values transition from positive to negative.
        The line is positioned on the y-axis between the two annotations whose x-values cross from positive to negative.
    divider_line_kwargs : dict, optional
        Keyword arguments passed to matplotlib's axhline for the divider line.
        Default is {'color': 'black', 'linestyle': '--', 'linewidth': 1, 'alpha': 0.5}.
    title_fontsize : str or float, default 'medium'
        Font size for the plot title. Can be a string (e.g., 'small', 'medium', 'large', 'x-large')
        or a numeric value.
        
    Returns
    -------
    dict
        Dictionary with keys:
            - "fig" (matplotlib Figure)
            - "axes" (matplotlib Axes or tuple)
            - "data" (filtered DataFrame used for plot)
            - "facet" (seaborn FacetGrid, if applicable)
    """
    import src.utils as utils
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Store original variable names for numeric operations
    original_x_var = x_var
    original_y_var = y_var
    
    # Handle axis flipping
    if flip_axes:
        x_var, y_var = y_var, x_var
        xlabel, ylabel = ylabel, xlabel
        figsize = (figsize[1], figsize[0])

    # Use global ANNOT_DICT if not provided
    global ANNOT_DICT
    if category_dict is None:
        category_dict = ANNOT_DICT

    r2_df = r2_df.copy()

    if min_n is not None:
        r2_df = r2_df.loc[(r2_df["ref_n"] > min_n)
                          & (r2_df["nonref_n"] > min_n)]

    if max_p is not None:
        r2_df = r2_df.loc[(r2_df["ref_p"] < max_p)
                          & (r2_df["nonref_p"] < max_p)]

    if min_diff is not None:
        # Use original_x_var for numeric operations
        r2_df = r2_df.loc[(r2_df[original_x_var].abs() > min_diff)]

    if annotations is not None:
        annotations = utils.as_list(annotations)
        r2_df = r2_df.loc[r2_df["annotation"].isin(annotations)]

    r2_df.dropna(inplace=True)
    # Sort by the numeric variable for proper bar ordering
    # When not flipped: x_var is numeric, when flipped: y_var is numeric
    sort_var = x_var if not flip_axes else y_var
    r2_df.sort_values(sort_var, ascending=False, inplace=True)
    r2_df.loc[:, "nonref_r_abs"] = r2_df["nonref_r"].abs()

    # --- Consistent category palette setup ---
    # Always determine all possible categories from category_dict (including "other")
    all_possible_categories = list(sorted(set(category_dict.values()) | {"other"}))
    if show_category_column:
        r2_df["category"] = r2_df["annotation"].map(category_dict)
        r2_df["category"] = r2_df["category"].fillna("other")
        if category_palette is None:
            # Use a consistent palette for all possible categories, not just those present in the data
            palette_colors = sns.color_palette("tab10", n_colors=len(all_possible_categories))
            category_palette = dict(zip(all_possible_categories, palette_colors))
        else:
            # If user provides a palette, ensure it covers all possible categories
            # Avoid item assignment on string
            if isinstance(category_palette, dict):
                for cat in all_possible_categories:
                    if cat not in category_palette:
                        # Assign a default color if missing
                        category_palette = dict(category_palette)  # Make a copy if not already
                        category_palette[cat] = sns.color_palette("tab10", n_colors=len(all_possible_categories))[all_possible_categories.index(cat)]
            else:
                # If category_palette is not a dict, ignore and use default
                palette_colors = sns.color_palette("tab10", n_colors=len(all_possible_categories))
                category_palette = dict(zip(all_possible_categories, palette_colors))

    if add_summary_subtitle:
        # Use original_x_var for numeric operations
        nonref_win_pct = (r2_df[original_x_var] > 0).sum() / len(r2_df.loc[r2_df[original_x_var] != 0])
        latex_nonref = r"$VEP_{\text{nonref}}$"
        win_rate_text = f"{latex_nonref} win rate: {nonref_win_pct:.1%}"
        if title is None:
            title = win_rate_text
        else:
            separator = "\n" if win_rate_on_newline else " "
            title = f"{title}{separator}{win_rate_text}"

    # Initialize divider_line_kwargs with defaults if empty
    if not divider_line_kwargs:
        divider_line_kwargs = {'color': 'black', 'linestyle': '--', 'linewidth': 1, 'alpha': 0.5}
    
    # Helper function to find and draw the positive/negative transition line
    def draw_transition_divider(ax, r2_df, x_var, y_var, flip_axes):
        """Draw a horizontal line between annotations where values transition from positive to negative."""
        if not draw_positive_negative_divider or x_var not in r2_df.columns:
            return
        
        # Find where values transition from positive to negative in the sorted data
        transition_idx = None
        for i in range(len(r2_df) - 1):
            val1 = r2_df.iloc[i][x_var]
            val2 = r2_df.iloc[i + 1][x_var]
            if val1 > 0 and val2 < 0:
                # Transition found between row i and i+1
                transition_idx = i
                break
        
        if transition_idx is None:
            return  # No transition found
        
        # Get the actual tick positions from the plot
        if flip_axes:
            tick_positions = ax.get_xticks()
        else:
            tick_positions = ax.get_yticks()
        
        if len(tick_positions) == 0:
            return
        
        # The data is sorted descending, so row 0 (highest value) is first
        # In seaborn barplot, first row appears at the TOP (highest y-position when not flipped)
        # So tick_positions[0] corresponds to the LAST row (bottom), tick_positions[-1] to FIRST row (top)
        # We need to reverse the index mapping
        
        n_ticks = len(tick_positions)
        n_rows = len(r2_df)
        
        if n_ticks != n_rows:
            # If tick count doesn't match, use axis limits
            if flip_axes:
                axis_lim = ax.get_xlim()
                if n_rows > 1:
                    spacing = (axis_lim[1] - axis_lim[0]) / (n_rows - 1)
                    # transition_idx is from top, so position from left
                    transition_pos = axis_lim[0] + (transition_idx + 0.5) * spacing
                else:
                    transition_pos = axis_lim[0] + 0.5
                ax.axvline(transition_pos, **divider_line_kwargs)
            else:
                axis_lim = ax.get_ylim()
                if n_rows > 1:
                    spacing = (axis_lim[1] - axis_lim[0]) / (n_rows - 1)
                    # transition_idx is from top, so position from top (axis_lim[1])
                    transition_pos = axis_lim[1] - (transition_idx + 0.5) * spacing
                else:
                    transition_pos = axis_lim[1] - 0.5
                ax.axhline(transition_pos, **divider_line_kwargs)
        else:
            # Use actual tick positions
            # tick_positions from get_yticks() are ordered from bottom to top
            # Our data is sorted descending (row 0 = highest value = top)
            # So: tick_positions[0] = bottom = row (n_rows-1), tick_positions[-1] = top = row 0
            # transition_idx is the row index where we transition (0 = top, n-1 = bottom)
            # We want the position between row transition_idx (above) and row transition_idx+1 (below)
            
            # Map data index to tick index: row i in data -> tick_positions[n_rows - 1 - i]
            tick_idx_above = n_rows - 1 - transition_idx      # Row transition_idx (above transition)
            tick_idx_below = n_rows - 1 - (transition_idx + 1)  # Row transition_idx+1 (below transition)
            
            if tick_idx_below >= 0 and tick_idx_above < n_ticks:
                pos_above = tick_positions[tick_idx_above]  # Position of row transition_idx
                pos_below = tick_positions[tick_idx_below]  # Position of row transition_idx+1
                transition_pos = (pos_above + pos_below) / 2.0
            elif tick_idx_above < n_ticks:
                # Edge case: transition at the very bottom
                transition_pos = tick_positions[0] - 0.5
            else:
                # Edge case: transition at the very top
                transition_pos = tick_positions[-1] + 0.5
            
            if flip_axes:
                ax.axvline(transition_pos, **divider_line_kwargs)
            else:
                ax.axhline(transition_pos, **divider_line_kwargs)

    # Facet by a column if requested
    if facet_col is not None:
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Set facet height if not provided
        if facet_height is None:
            # Try to scale height by number of y categories per facet
            n_facets = r2_df[facet_col].nunique()
            n_y = r2_df[y_var].nunique()
            facet_height = max(figsize[1] / n_facets, 3)

        g = sns.catplot(
            data=r2_df,
            kind="bar",
            x=x_var,
            y=y_var,
            hue=hue,
            palette=palette,
            col=facet_col,
            col_wrap=facet_col_wrap,
            sharex=facet_sharex,
            sharey=facet_sharey,
            height=facet_height,
            aspect=facet_aspect,
            legend=facet_legend,
        )
        g.set_axis_labels(xlabel, ylabel)
        g.set_titles(col_template="{col_name}")
        if title is not None:
            plt.subplots_adjust(top=0.85)
            # Align title with left edge of plot area (leftmost axes)
            if title_x is None:
                # Get the leftmost axes position in figure coordinates
                axes = g.axes.flatten() if hasattr(g.axes, 'flatten') else [g.axes]
                leftmost_pos = min(ax.get_position().x0 for ax in axes)
                title_x_pos = leftmost_pos
            else:
                title_x_pos = title_x
            g.fig.suptitle(title, ha=title_ha, x=title_x_pos, fontsize=title_fontsize)
        # Replace underscores with spaces in y-tick labels for each facet
        # Use user-specified alignment (default 'right')
        for ax in g.axes.flatten():
            if flip_axes:
                # When flipped, x-tick labels use specified alignment
                labels = [label.get_text().replace('_af', '_AF').replace('_', ' ') for label in ax.get_xticklabels()]
                ax.set_xticklabels(labels, rotation=yticklabel_rotation)
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment(yticklabel_ha)
            else:
                # When not flipped, y-tick labels use specified alignment
                labels = [label.get_text().replace('_af', '_AF').replace('_', ' ') for label in ax.get_yticklabels()]
                ax.set_yticklabels(labels, rotation=yticklabel_rotation)
                for label in ax.get_yticklabels():
                    label.set_horizontalalignment(yticklabel_ha)
            # Set tick label font size
            ax.tick_params(axis='both', labelsize=ticklabel_fontsize)
            # Draw horizontal divider line between annotations where values transition from positive to negative
            draw_transition_divider(ax, r2_df, x_var, y_var, flip_axes)
            # Set legend title and fontsize
        if legend_title is not None:
            g._legend.set_title(legend_title)
            # Set fontsize for legend labels
            for text in g._legend.get_texts():
                text.set_fontsize(legend_fontsize)
        # Add category column if requested (not supported for facet for now)
        if show_category_column:
            import warnings
            warnings.warn("Category column not currently supported for faceted plots.")
        return {'fig': g.fig, 'axes': g.axes, 'data': r2_df, 'facet': g}
    else:
        if show_category_column:
            # --- Custom plotting with category column ---
            import matplotlib.patches as mpatches

            # Prepare data for plotting
            y_labels = r2_df[y_var].tolist()
            y_pos = range(len(y_labels))
            # Map annotation to category and color
            # Categories are always from "annotation" column, which should match
            # y_var when not flipped, or x_var when flipped
            categories = r2_df["category"].tolist()
            # Use the consistent palette for all possible categories
            cat_color_map = {cat: category_palette[cat] for cat in all_possible_categories}
            cat_colors = [cat_color_map[cat] for cat in categories]

            # Set up figure with two axes: one for category, one for barplot
            fig = plt.figure(figsize=figsize)
            # Gridspec: adjust based on flip_axes
            from matplotlib.gridspec import GridSpec
            if flip_axes:
                # Category at bottom when axes are flipped
                gs = GridSpec(2, 1, height_ratios=[0.95, 0.05], hspace=0.05)
                ax_bar = fig.add_subplot(gs[0, 0])
                ax_cat = fig.add_subplot(gs[1, 0], sharex=ax_bar)
            else:
                # Category on left (default)
                gs = GridSpec(1, 2, width_ratios=[0.05, 0.95], wspace=0.05)
                ax_cat = fig.add_subplot(gs[0, 0])
                ax_bar = fig.add_subplot(gs[0, 1], sharey=ax_cat)

            # Draw category boxes
            if flip_axes:
                # Category boxes at bottom (horizontal)
                x_labels = r2_df[x_var].tolist()
                x_pos = range(len(x_labels))
                for i, (cat, color) in enumerate(zip(categories, cat_colors)):
                    ax_cat.add_patch(
                        mpatches.Rectangle(
                            (i - 0.4, 0.01), 0.8, 1, color=color, ec='black', linewidth=category_box_linewidth
                        )
                    )
                ax_cat.set_xlim(-0.5, len(x_labels) - 0.5)
                ax_cat.set_ylim(0, 1)
                ax_cat.set_yticks([])
                ax_cat.set_xticks(x_pos)
                ax_cat.set_xticklabels([])
                ax_cat.tick_params(bottom=False, labelbottom=False, top=False)
                ax_cat.set_frame_on(False)
            else:
                # Category boxes on left (vertical)
                for i, (cat, color) in enumerate(zip(categories, cat_colors)):
                    ax_cat.add_patch(
                        mpatches.Rectangle(
                            (0.01, i - 0.4), 1, 0.8, color=color, ec='black', linewidth=category_box_linewidth
                        )
                    )
                ax_cat.set_ylim(-0.5, len(y_labels) - 0.5)
                ax_cat.set_xlim(0, 1)
                ax_cat.set_xticks([])
                ax_cat.set_yticks(y_pos)
                ax_cat.set_yticklabels([])
                ax_cat.tick_params(left=False, labelleft=False, right=False)
                ax_cat.set_frame_on(False)

            # Draw barplot
            sns.barplot(
                data=r2_df,
                x=x_var,
                y=y_var,
                hue=hue,
                palette=palette,
                ax=ax_bar,
                **barplot_kwargs,
            )
            ax_bar.set_ylabel(ylabel)
            ax_bar.set_xlabel(xlabel)
            if title is not None:
                # Align title with left edge of plot area
                # x=0 in axes coordinates aligns with left edge of axes
                title_x_pos = 0.0 if title_x is None else title_x
                ax_bar.set_title(title, ha=title_ha, x=title_x_pos, fontsize=title_fontsize)

            # Handle tick labels and legend based on flip_axes
            if flip_axes:
                # Move x-tick labels to the bottom axis (category)
                ax_bar.set_xticklabels([])
                ax_bar.tick_params(bottom=False, labelbottom=False)
                # Set legend position
                handles, labels = ax_bar.get_legend_handles_labels()
                if legend_title is not None:
                    # Reverse handles and labels so highest values are on top
                    legend_kwargs = {
                        'handles': handles[::-1],
                        'labels': labels[::-1],
                        'title': legend_title,
                        'frameon': False,
                        'fontsize': legend_fontsize,
                    }
                    if legend_loc is not None:
                        legend_kwargs['loc'] = legend_loc
                    else:
                        legend_kwargs['loc'] = 'upper right'
                    if legend_bbox_to_anchor is not None:
                        legend_kwargs['bbox_to_anchor'] = legend_bbox_to_anchor
                    ax_bar.legend(**legend_kwargs)
                # Replace underscores with spaces in x-tick labels
                # Use user-specified alignment (default 'right')
                labels = [label.get_text().replace('_', ' ').replace(' af', ' AF') for label in ax_bar.get_xticklabels()]
                ax_bar.set_xticklabels(labels, rotation=yticklabel_rotation)
                for label in ax_bar.get_xticklabels():
                    label.set_horizontalalignment(yticklabel_ha)
                # Add x-tick labels below the category boxes
                for i, label in enumerate(x_labels):
                    ax_cat.text(i, -0.05, label.replace('_', ' '), va='top', ha=yticklabel_ha, fontsize=ticklabel_fontsize, rotation=yticklabel_rotation)
                # Adjust subplot spacing for bottom category
                plt.subplots_adjust(bottom=0.22, top=0.98, hspace=0.02)
            else:
                # Move y-tick labels to the left axis (category)
                ax_bar.set_yticklabels([])
                ax_bar.tick_params(left=False, labelleft=False)
                # Set legend position
                handles, labels = ax_bar.get_legend_handles_labels()
                if legend_title is not None:
                    # Reverse handles and labels so highest values are on top
                    legend_kwargs = {
                        'handles': handles[::-1],
                        'labels': labels[::-1],
                        'title': legend_title,
                        'frameon': False,
                        'fontsize': legend_fontsize,
                    }
                    if legend_loc is not None:
                        legend_kwargs['loc'] = legend_loc
                    else:
                        legend_kwargs['loc'] = 'lower right'
                    if legend_bbox_to_anchor is not None:
                        legend_kwargs['bbox_to_anchor'] = legend_bbox_to_anchor
                    ax_bar.legend(**legend_kwargs)
                # Replace underscores with spaces in y-tick labels
                # Use user-specified alignment (default 'right')
                labels = [label.get_text().replace('_', ' ').replace(' af', ' AF') for label in ax_bar.get_yticklabels()]
                ax_bar.set_yticklabels(labels, rotation=yticklabel_rotation)
                for label in ax_bar.get_yticklabels():
                    label.set_horizontalalignment(yticklabel_ha)
                # Add y-tick labels to the left of the category boxes
                # Use yticklabel_pad if provided, otherwise default to -0.05
                label_x_pos = yticklabel_pad if yticklabel_pad is not None else -0.05
                for i, label in enumerate(y_labels):
                    ax_cat.text(label_x_pos, i, label.replace('_', ' '), va='center', ha=yticklabel_ha, fontsize=ticklabel_fontsize, rotation=yticklabel_rotation)
                # Adjust subplot spacing for left category
                plt.subplots_adjust(left=0.22, right=0.98, wspace=0.02)
                # Move the y-axis label further to the left to avoid overlap with annotation names
                ax_bar.yaxis.set_label_coords(-0.75, 0.5)

            # Add category legend, sorted alphabetically (if requested)
            if show_category_legend:
                sorted_categories = sorted(all_possible_categories)
                cat_legend_handles = [
                    mpatches.Patch(color=cat_color_map[cat], label=cat)
                    for cat in sorted_categories
                ]
                # Place category legend - adjust position based on flip_axes
                if flip_axes:
                    fig.legend(
                        handles=cat_legend_handles,
                        title=category_legend_title,
                        loc='lower left',
                        bbox_to_anchor=(0.2, 0.15),
                        frameon=False,
                        fontsize=category_legend_fontsize
                    )
                else:
                    fig.legend(
                        handles=cat_legend_handles,
                        title=category_legend_title,
                        loc=category_legend_loc,
                        bbox_to_anchor=category_legend_bbox_to_anchor,
                        frameon=False,
                        fontsize=category_legend_fontsize
                    )
            # Set tick label font size for bar axis
            ax_bar.tick_params(axis='both', labelsize=ticklabel_fontsize)

            # Draw horizontal divider line between annotations where values transition from positive to negative
            draw_transition_divider(ax_bar, r2_df, x_var, y_var, flip_axes)

            # Remove top and right spines (margin lines) for both axes
            ax_cat.spines['top'].set_visible(False)
            ax_cat.spines['right'].set_visible(False)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)

            return {'fig': fig, 'axes': (ax_cat, ax_bar), 'data': r2_df}
        else:
            plt.figure(figsize=figsize)
            g = sns.barplot(
                data=r2_df,
                x=x_var,
                y=y_var,
                hue=hue,
                palette=palette
            )
            # Set legend position
            if legend_title is not None:
                # Get handles and labels and reverse them so highest values are on top
                handles, labels = plt.gca().get_legend_handles_labels()
                legend_kwargs = {
                    'handles': handles[::-1],
                    'labels': labels[::-1],
                    'title': legend_title,
                    'fontsize': legend_fontsize,
                }
                if legend_loc is not None:
                    legend_kwargs['loc'] = legend_loc
                if legend_bbox_to_anchor is not None:
                    legend_kwargs['bbox_to_anchor'] = legend_bbox_to_anchor
                plt.legend(**legend_kwargs)
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            if title is not None:
                # Align title with left edge of plot area
                # x=0 in axes coordinates aligns with left edge of axes
                title_x_pos = 0.0 if title_x is None else title_x
                plt.title(title, ha=title_ha, x=title_x_pos, fontsize=title_fontsize)
            # Replace underscores with spaces in tick labels
            # Use user-specified alignment (default 'right')
            ax = plt.gca()
            if flip_axes:
                # When flipped, x-tick labels use specified alignment
                labels = [label.get_text().replace('_', ' ') for label in ax.get_xticklabels()]
                ax.set_xticklabels(labels, rotation=yticklabel_rotation)
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment(yticklabel_ha)
            else:
                # When not flipped, y-tick labels use specified alignment
                labels = [label.get_text().replace('_', ' ') for label in ax.get_yticklabels()]
                ax.set_yticklabels(labels, rotation=yticklabel_rotation)
                for label in ax.get_yticklabels():
                    label.set_horizontalalignment(yticklabel_ha)
            # Set tick label font size
            plt.gca().tick_params(axis='both', labelsize=ticklabel_fontsize)
            
            # Draw horizontal divider line between annotations where values transition from positive to negative
            ax = plt.gca()
            draw_transition_divider(ax, r2_df, x_var, y_var, flip_axes)
            
            # Remove top and right spines (margin lines) for both axes
            # Fix: g.axes is a numpy array of axes, not a string, so we should not assign to g.axes[0] as a string
            # Instead, check if g.axes is an array or a single axis
            axes = g.axes if hasattr(g, "axes") else [g]
            if hasattr(axes, "__iter__"):
                for ax in axes:
                    if hasattr(ax, "spines"):
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
            else:
                if hasattr(axes, "spines"):
                    axes.spines['top'].set_visible(False)
                    axes.spines['right'].set_visible(False)
            
            return {'fig': g.figure, 'axes': g.axes, 'data': r2_df}


def plot_vep_vs_af_grid(df,
                         af_cols,
                         vep_cols_per_af=None,
                         hue="CLNSIG_simple",
                         logy=False,
                         logx=True,
                         sharex=False,
                         sharey=True,
                         drop_vep_zeros=False,
                         palette="Set2",
                         height=4,
                         aspect=1.25,
                         fit_line=True,
                         alpha=0.6,
                         xlim=None,
                         ylim=(0, 1),
                         fit_model="logistic",
                         epsilon=1e-6,
                         show_ci=True,
                         show_plot=True,
                         invert_xaxis=False,    
                         hspace=0.3,
                         wspace=0.2,
                         minmax_scale_vep=False,
                         show_xlabels=True,
                         scatter_kwargs={},
                         rasterize_points=False,
                         supertitles=None,
                         supertitle_fontsize="large",
                         supertitle_fontweight="bold",
                         supertitle_y=.03,
                         generate_plot=True,
                         ):
    """
    Create a grid plot of VEP vs AF for multiple AF columns and VEP columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing VEP and AF data
    af_cols : list
        List of AF column names to plot (one row per AF column)
    vep_cols_per_af : dict or None
        Dictionary mapping AF column names to lists of VEP columns.
        If None, automatically determines VEP columns:
        - For "AF_coalesced": uses ["VEP_nonref", "VEP_mean", "VEP_ref"]
        - For others: uses ["VEP_" + af_col.split("_")[0], "VEP_mean", "VEP_ref"]
    hue : str
        Column name for color coding (default: "CLNSIG_simple")
    logy : bool
        Whether to use log scale for y-axis (default: False)
    logx : bool
        Whether to use log scale for x-axis (default: True)
    sharex : bool
        Whether to share x-axis across columns (default: False)
    sharey : bool
        Whether to share y-axis across rows (default: True)
    drop_vep_zeros : bool
        Whether to drop rows where VEP value is zero (default: False)
    palette : str
        Color palette name (default: "Set2")
    height : float
        Height of each subplot (default: 4)
    aspect : float
        Aspect ratio of each subplot (default: 1.25)
    fit_line : bool
        Whether to fit and plot a curve (default: True)
    alpha : float
        Transparency of scatter points (default: 0.6)
    xlim : tuple or None
        X-axis limits (default: None)
    ylim : tuple
        Y-axis limits (default: (0, 1))
    fit_model : str
        Model for curve fitting (default: "logistic")
        fit_model options:
        - "logistic": Logistic (sigmoid) curve fit
        - "exp": Exponential growth curve fit
        - "exp_decay": Exponential decay curve fit
        - "exp_decay_floor": Exponential decay with floor value
        - "poly2": Quadratic polynomial regression
        - "linear": Linear regression line
        - "power_law_inverse": Power law inverse: y = a / (x^b + c)
        - "hyperbolic": Hyperbolic: y = a / (1 + b*x)
        - "inverse": Inverse: y = a / (x + b)
        - "linear_log": Linear in log space (use when logy=True): y = a + b*x
        - "power_law_log": Power law in log space: y = a - b*log10(x + c)
        - "power_law_both_log": Power law when both axes log-transformed: y = a + b*x
        - "exp_decay_log_log": Exponential decay in log-log space
        - "power_law_offset": Modified power law: y = a - b*(x + c)^d
        - "sigmoid_log": Sigmoid in log space: y = a / (1 + exp(-b*(x - c)))
        
        Recommended combinations:
        - logx=False, logy=True: try "linear", "power_law_log", "sigmoid_log"
        - logx=True, logy=True: try "power_law_both_log", "linear", "exp_decay_log_log"
        - logx=False, logy=False: try "exp_decay_floor", "power_law_inverse", "hyperbolic"
        - logx=True, logy=False: try "linear", "exp_decay", "poly2"
    epsilon : float
        Small value to add before taking log10 to avoid log(0) (default: 1e-6)
    show_ci : bool
        Whether to show confidence intervals (default: True)
    show_plot : bool
        Whether to display the plot (default: True)
    hspace : float
        Height spacing between subplots (default: 0.3)
    wspace : float
        Width spacing between subplots (default: 0.2)
    invert_xaxis : bool
        Whether to invert the x-axis (default: False)
    minmax_scale_vep : bool
        Whether to minmax scale VEP scores before log transformation (default: False).
        Useful when VEP scores have large negative values. Scales values to [0, 1] range
        before applying log10, which helps with negative values: (VEP - min) / (max - min).
        Only applies when logx=True.
    show_xlabels : bool
        Whether to show x-axis labels (titles) on subplots (default: True).
        If False, x-axis labels will be omitted from all subplots.
    rasterize_points : bool
        Whether to rasterize only the scatter plot points (default: False).
        When True, points are rasterized while keeping axes, labels, and other elements as vectors.
        Useful for reducing file size of plots with many points.
    supertitles : list or None
        List of strings to display as supertitles above each column (default: None).
        Length should match the number of columns. If provided, these will be centered
        above each column of subplots.
    generate_plot : bool
        Whether to generate the plot (default: True). If False, only statistics
        are calculated and returned. When False, fig and axes will be None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object (None if generate_plot=False)
    axes : numpy.ndarray or None
        Array of axes objects (None if generate_plot=False)
    stats_df : pd.DataFrame
        DataFrame containing regression statistics for each subplot, including:
        - af_col: AF column name
        - vep_col: VEP column name
        - n_variants: Number of unique variants
        - n_points: Number of data points
        - fit_model: Model used for fitting
        - r2: R-squared value
        - adj_r2: Adjusted R-squared value
        - p_value: P-value from F-test
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - n_fit_points: Number of points used in fit
        - n_params: Number of parameters in the model
        - params: List of fitted parameter values
        - param_std_err: List of standard errors for each parameter
        - param_ci_lower: List of lower bounds (95% CI) for each parameter
        - param_ci_upper: List of upper bounds (95% CI) for each parameter
        - logx: Whether x-axis was log-transformed
        - logy: Whether y-axis was log-transformed
        - minmax_scale_vep: Whether VEP was minmax scaled before log transform
        - supertitle: Column supertitle (if provided)
    """
    df = df.copy()

    # only take the unique ones, but keep order
    seen = set()
    af_cols = [x for x in af_cols if not (x in seen or seen.add(x))]
    # Determine VEP columns for each AF column
    if vep_cols_per_af is None:
        vep_cols_per_af = {}
        valid_af_cols = []
        
        # Known superpop acronyms that map to VEP columns
        known_superpops = {'AFR', 'AMR', 'EAS', 'EUR', 'SAS', 'ASJ', 'FIN', 'NFE', 'MID', 'AMI', 'POPMAX'}
        
        for af_col in af_cols:
            parts = af_col.split("_")
            
            # Must end with _AF
            if not af_col.endswith("_AF") and af_col != "AF":
                print(f"Skipping {af_col}: does not end with _AF")
                continue
            
            # Handle simple "AF" (coalesced)
            if af_col == "AF":
                vep_cols_per_af[af_col] = ["VEP_nonref", "VEP_mean", "VEP_ref"]
                valid_af_cols.append(af_col)
                continue
            
            # Check if af_col is present in the DataFrame columns
            if af_col not in df.columns:
                print(f"Skipping {af_col}: not found in DataFrame")
                continue
            
            # For columns ending with _AF, check if second-to-last part is a known superpop
            if len(parts) >= 2 and parts[-1] == "AF":
                potential_superpop = parts[-2]
                
                # Special handling for POPMAX - treat as coalesced (use nonref/mean/ref)
                if potential_superpop == "POPMAX":
                    vep_cols_per_af[af_col] = ["VEP_nonref", "VEP_mean", "VEP_ref"]
                    valid_af_cols.append(af_col)
                    continue
                
                # Check if it's a known superpop acronym
                if potential_superpop in known_superpops:
                    # This is a superpop-specific column
                    vep_col = f"VEP_{potential_superpop}"
                    
                    # Check if the corresponding VEP column exists in the dataframe
                    if vep_col in df.columns:
                        vep_cols_per_af[af_col] = [vep_col, "VEP_mean", "VEP_ref"]
                        valid_af_cols.append(af_col)
                    else:
                        # Skip this AF column if no matching VEP column exists
                        print(f"Skipping {af_col}: no corresponding {vep_col} column found")
                        continue
                else:
                    # This is a coalesced column (ends with _AF but second-to-last part is not a known superpop)
                    # Examples: AF_coalesced, gnomADe_AF, 1000Gp3_AF, TOPMed_frz8_AF, 
                    #          gnomAD2.1.1_exomes_controls_AF, ALFA_Total_AF, etc.
                    vep_cols_per_af[af_col] = ["VEP_nonref", "VEP_mean", "VEP_ref"]
                    valid_af_cols.append(af_col)
            else:
                # Fallback: try to extract superpop from first part (for formats like AFR_AF)
                if len(parts) == 2 and parts[0] in known_superpops:
                    # Special handling for POPMAX - treat as coalesced (use nonref/mean/ref)
                    if parts[0] == "POPMAX":
                        vep_cols_per_af[af_col] = ["VEP_nonref", "VEP_mean", "VEP_ref"]
                        valid_af_cols.append(af_col)
                    else:
                        vep_col = f"VEP_{parts[0]}"
                        if vep_col in df.columns:
                            vep_cols_per_af[af_col] = [vep_col, "VEP_mean", "VEP_ref"]
                            valid_af_cols.append(af_col)
                        else:
                            print(f"Skipping {af_col}: no corresponding {vep_col} column found")
                            continue
                else:
                    # Unknown format - skip
                    print(f"Skipping {af_col}: unknown format")
                    continue
        
        # Update af_cols to only include valid ones
        af_cols = valid_af_cols
    
    # Check if we have any valid AF columns
    if len(af_cols) == 0:
        raise ValueError("No valid AF columns found with corresponding VEP columns")
    
    # Get all unique VEP columns to determine grid width
    all_vep_cols = set()
    for vep_cols in vep_cols_per_af.values():
        all_vep_cols.update(vep_cols)
    all_vep_cols = sorted(list(all_vep_cols))
    
    # Determine grid dimensions
    n_rows = len(af_cols)
    n_cols = max(len(vep_cols) for vep_cols in vep_cols_per_af.values())
    
    # Create figure and axes only if plotting is enabled
    if generate_plot:
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(height*aspect*n_cols, height*n_rows),
                                sharex=sharex, sharey=sharey)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
    else:
        fig = None
        axes = None
    
    # Get color palette
    if hue == "Super Population":
        cmap = utils.get_superpop_palette()
    elif hue == "CLNSIG_simple":
        cmap = utils.get_clinsig_palette()
    else:
        cmap = utils.make_palette(df[hue].unique(), palette=palette)
    
    # Handle categorical ordering
    if hue is not None and hue == "CLNSIG_simple":
        hue_order = list(utils.get_clinsig_palette().keys())
        hue_order.reverse()
        df.loc[:, "CLNSIG_simple"] = pd.Categorical(df["CLNSIG_simple"], categories=hue_order, ordered=True)
        df = df.sort_values(by="CLNSIG_simple")
    
    # Initialize list to store regression statistics
    regression_stats = []
    
    # Plot each combination
    for row_idx, af_col in enumerate(af_cols):
        vep_cols = vep_cols_per_af[af_col]
        
        for col_idx, vep_col_item in enumerate(vep_cols):
            if generate_plot:
                ax_current = axes[row_idx, col_idx]
            else:
                ax_current = None
            
            # Filter data for this combination
            df_subset = df.dropna(subset=[vep_col_item, af_col]).copy()
            
            # Drop zero VEP values if requested
            if drop_vep_zeros:
                df_subset = df_subset[df_subset[vep_col_item] != 0]
            
            if len(df_subset) == 0:
                ax_current.set_title(f"{vep_col_item}\n(No data)")
                continue
            
            # Create log-transformed columns if needed
            # For VEP: optionally minmax scale before logging (useful for large negative values)
            # Convert to numeric first to handle string dtypes
            vep_values = pd.to_numeric(df_subset[vep_col_item], errors='coerce')
            
            if logx and minmax_scale_vep:
                # Minmax scale VEP to [0, 1] range before logging
                vep_min = vep_values.min()
                vep_max = vep_values.max()
                vep_range = vep_max - vep_min
                if vep_range > 0:
                    vep_scaled = (vep_values - vep_min) / vep_range
                    df_subset.loc[:, "VEP_log"] = np.log10(vep_scaled + epsilon)
                else:
                    # If all values are the same, just use epsilon
                    df_subset.loc[:, "VEP_log"] = np.log10(epsilon)
            elif logx:
                df_subset.loc[:, "VEP_log"] = np.log10(vep_values + epsilon)
            else:
                # Not using logx, so VEP_log won't be used, but create it anyway to avoid errors
                df_subset.loc[:, "VEP_log"] = vep_values
            
            # For AF: always use standard log transformation
            # Convert to numeric first to handle string dtypes
            af_values = pd.to_numeric(df_subset[af_col], errors='coerce')
            if logy:
                df_subset.loc[:, "AF_log"] = np.log10(af_values + epsilon)
            else:
                df_subset.loc[:, "AF_log"] = af_values
            
            # Determine x and y variables
            if logy:
                y_var = "AF_log"
            else:
                y_var = af_col
            if logx:
                x_var = "VEP_log"
            else:
                x_var = vep_col_item
            
            # Filter out inf and NaN values before plotting and curve fitting
            # This is especially important after log transformations
            # Convert to numpy arrays to ensure numeric types for isfinite check
            x_values = np.asarray(df_subset[x_var], dtype=float)
            y_values = np.asarray(df_subset[y_var], dtype=float)
            valid_mask = (
                np.isfinite(x_values) & 
                np.isfinite(y_values)
            )
            df_subset_clean = df_subset[valid_mask].copy()
            
            if len(df_subset_clean) == 0:
                if generate_plot:
                    ax_current.set_title(f"{vep_col_item}\n(No valid data after filtering)")
                # Still record stats even if no valid data
                regression_stats.append({
                    'af_col': af_col,
                    'vep_col': vep_col_item,
                    'n_variants': df_subset_clean['site'].nunique() if len(df_subset_clean) > 0 else 0,
                    'n_points': len(df_subset_clean),
                    'fit_model': None,
                    'r2': np.nan,
                    'adj_r2': np.nan,
                    'p_value': np.nan,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'n_fit_points': np.nan,
                    'n_params': np.nan,
                    'params': None,
                    'param_std_err': None,
                    'param_ci_lower': None,
                    'param_ci_upper': None,
                    'logx': logx,
                    'logy': logy,
                    'minmax_scale_vep': minmax_scale_vep if logx else False
                })
                if supertitles is not None and col_idx < len(supertitles):
                    regression_stats[-1]['supertitle'] = supertitles[col_idx]
                continue
            
            # Add scatter plot (only if plotting)
            if generate_plot:
                sns.scatterplot(data=df_subset_clean,
                               x=x_var,
                               y=y_var,
                               hue=hue,
                               palette=cmap,
                               alpha=alpha,
                               ax=ax_current,
                               **scatter_kwargs)
                
                # Rasterize points if requested (only affects scatter points, not other elements)
                if rasterize_points:
                    for collection in ax_current.collections:
                        collection.set_rasterized(True)
            
            # Add regression line and calculate statistics
            fit_stats = {}
            if fit_line:
                # Convert to numpy arrays to ensure numeric types for curve fitting
                x_fit = np.asarray(df_subset_clean[x_var], dtype=float)
                y_fit = np.asarray(df_subset_clean[y_var], dtype=float)
                if generate_plot:
                    result = fit_curve_and_plot(x_fit, y_fit, ax_current, fit_model, alpha, show_ci)
                else:
                    # Calculate statistics without plotting
                    result = fit_curve_and_plot_no_plot(x_fit, y_fit, fit_model)
                if result is not None:
                    fit_stats = result
            
            # Store regression statistics
            stat_dict = {
                'af_col': af_col,
                'vep_col': vep_col_item,
                'n_variants': df_subset_clean['site'].nunique(),
                'n_points': len(df_subset_clean),
                'fit_model': fit_model if fit_line else None,
                'r2': fit_stats.get('r2', np.nan),
                'adj_r2': fit_stats.get('adj_r2', np.nan),
                'p_value': fit_stats.get('p_value', np.nan),
                'rmse': fit_stats.get('rmse', np.nan),
                'mae': fit_stats.get('mae', np.nan),
                'n_fit_points': fit_stats.get('n', np.nan),
                'n_params': fit_stats.get('n_params', np.nan),
                'params': fit_stats.get('params', None),
                'param_std_err': fit_stats.get('param_std_err', None),
                'param_ci_lower': fit_stats.get('param_ci_lower', None),
                'param_ci_upper': fit_stats.get('param_ci_upper', None),
                'logx': logx,
                'logy': logy,
                'minmax_scale_vep': minmax_scale_vep if logx else False
            }
            # Add supertitle if provided
            if supertitles is not None and col_idx < len(supertitles):
                stat_dict['supertitle'] = supertitles[col_idx]
            regression_stats.append(stat_dict)
            
            # All plotting code below (only execute if generate_plot is True)
            if generate_plot:
                # Set axis limits
                if xlim is not None:
                    ax_current.set_xlim(xlim)
                if ylim is not None:
                    ax_current.set_yticks(np.linspace(ylim[0], ylim[1], 5))
                
                # Format column names for LaTeX using to_latex_subscript
                af_label = to_latex_subscript(af_col)
                vep_label = to_latex_subscript(vep_col_item)
                
                # Set title with af_col vs vep_col format (LaTeX formatted)
                ax_current.set_title(f"{af_label} vs. {vep_label} (variants={df_subset_clean['site'].nunique()})")
                
                # Set y-axis label with LaTeX formatting
                # Extract LaTeX content from to_latex_subscript (removes $ delimiters if present)
                af_label_ylab_full = to_latex_subscript(af_col)
                # Remove $ delimiters if present to avoid nested math mode
                if af_label_ylab_full.startswith('$') and af_label_ylab_full.endswith('$'):
                    af_label_ylab_content = af_label_ylab_full[1:-1]
                else:
                    af_label_ylab_content = af_label_ylab_full
                
                if logy:
                    ax_current.set_ylabel(f"$log_{{10}}({af_label_ylab_content})$")
                else:
                    ax_current.set_ylabel(f"${af_label_ylab_content}$")
                
                # Create x-axis label based on VEP column name
                # If vep_col contains underscore, subscript the part after underscore
                if "_" in vep_col_item:
                    parts = vep_col_item.split("_", 1)
                    vep_label = f"{parts[0]}_{{{parts[1]}}}"
                else:
                    vep_label = vep_col_item
                
                if show_xlabels:
                    if logx:
                        if minmax_scale_vep:
                            ax_current.set_xlabel(f"$log_{{10}}(\\text{{minmax}}({vep_label}))$")
                        else:
                            ax_current.set_xlabel(f"$log_{{10}}({vep_label})$")
                    else:
                        ax_current.set_xlabel(f"${vep_label}$")
                
                # Ensure x-axis labels and ticks are visible for all subplots
                ax_current.tick_params(labelbottom=True, bottom=True)
                ax_current.xaxis.set_visible(True)
                # Explicitly show/hide the xlabel based on show_xlabels
                if ax_current.xaxis.label:
                    ax_current.xaxis.label.set_visible(show_xlabels)
                
                # Remove legend from all subplots except first one in first row
                if row_idx == 0 and col_idx == 0:
                    if hue is not None:
                        handles, labels = ax_current.get_legend_handles_labels()
                        # Set fixed marker size for legend handles before creating legend
                        # This ensures legend markers remain visible even when scatter points are small
                        for handle in handles:
                            # Check if this is a PathCollection (scatter plot handle)
                            if hasattr(handle, 'get_sizes'):
                                sizes = handle.get_sizes()
                                if sizes is not None and len(sizes) > 0:
                                    # Set a fixed size that's independent of scatter point size
                                    handle.set_sizes([36])  # Fixed size for legend markers
                            # Also check for _sizes attribute (alternative way sizes are stored)
                            if hasattr(handle, '_sizes') and handle._sizes is not None:
                                if len(handle._sizes) > 0:
                                    handle._sizes = [36]
                        legend = ax_current.legend(handles, labels, loc='lower left', markerscale=1.0)
                else:
                    if ax_current.get_legend() is not None:
                        ax_current.get_legend().remove()
    
    # Final plotting adjustments (only if plotting is enabled)
    if generate_plot:
        # Only show x-tick labels for the last row, but keep x-axis titles for all (if show_xlabels=True)
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                ax.xaxis.set_visible(True)
                if ax.xaxis.label:
                    ax.xaxis.label.set_visible(show_xlabels)
                # Only show tick labels on the last row
                if row_idx == n_rows - 1:
                    ax.tick_params(labelbottom=True, bottom=True)
                else:
                    ax.tick_params(labelbottom=False, bottom=True)
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        
        # Add supertitles above each column if provided
        if supertitles is not None:
            if len(supertitles) != n_cols:
                raise ValueError(f"Number of supertitles ({len(supertitles)}) must match number of columns ({n_cols})")
            
            # Get the bounding box of the top row of subplots to position supertitles
            for col_idx, supertitle in enumerate(supertitles):
                # Get the top-left subplot in this column
                top_ax = axes[0, col_idx]
                # Get the position of this subplot in figure coordinates
                bbox = top_ax.get_position()
                # Position supertitle at the top center of the column
                # Use the top of the subplot plus a small offset
                fig.text(bbox.x0 + bbox.width / 2, bbox.y1 + supertitle_y, supertitle,
                        ha='center', va='bottom', fontsize=supertitle_fontsize, weight=supertitle_fontweight)
        
        # Apply x-axis inversion as the VERY LAST step
        # This ensures it's not overridden by any layout adjustments
        if invert_xaxis:
            # Explicitly reverse the x-axis limits to invert the axis
            # This method is more reliable than invert_xaxis() when sharex=True
            for ax in axes.flat:
                if ax is not None:
                    # Get current limits
                    xmin, xmax = ax.get_xlim()
                    # Set reversed limits to invert the axis
                    # This makes positive values appear on the left, negative on the right
                    ax.set_xlim(xmax, xmin)
                    # Also set the inversion flag to ensure it persists
                    ax.xaxis.set_inverted(True)
        if show_plot:
            plt.show()
    
    # Create dataframe with regression statistics
    stats_df = pd.DataFrame(regression_stats)
    
    return fig, axes, stats_df


def compute_vep_superpopulation_aggregates(vep_df, vep_col="VEP"):
    """
    Aggregate VEP scores by superpopulation and reference group for each variant site.

    This function calculates mean VEP scores per superpopulation (excluding REF samples)
    for each variant, the overall non-REF mean VEP per site, and the REF-specific mean VEP.
    All results are merged into a single DataFrame indexed by 'site'.

    NOTE: Onl works for Flashzoi and SpliceAI VEP results. 
    ESM results must be computed separately.

    Parameters
    ----------
    vep_df : pd.DataFrame
        DataFrame containing at least columns 'sample', 'site', and 'VEP'.
        Should include multiple rows per site-sample pair. 

    Returns
    -------
    vep_superpop_agg : pd.DataFrame
        DataFrame, one row per variant site. Columns include:
        - VEP_<superpop> : mean VEP per superpopulation (excluding REF)
        - VEP_mean      : mean VEP across all non-REF samples per site
        - VEP_nonref    : mean VEP across all non-REF samples per site (identical to VEP_mean)
        - VEP_ref       : mean VEP for reference samples per site (if present)
        'site' is the key/index for identification.
    """
    import src.onekg as og
    meta = og.get_sample_metadata()

    # Ensure indicator column for REF
    if "is_ref" not in vep_df.columns:
        vep_df["is_ref"] = vep_df["sample"] == "REF"

    # List superpopulations
    superpop_list = sorted(meta["superpopulation"].unique())
    superpop_dfs = []
    for sp in tqdm(superpop_list, desc="Computing superpopulation means"):
        # Get samples for this superpopulation
        samples_in_sp = meta.loc[meta["superpopulation"] == sp, "sample"]
        mask = (vep_df["sample"].isin(samples_in_sp)) & (vep_df["sample"] != "REF")
        # Skip if no samples available for this superpopulation
        if len(vep_df.loc[mask]) == 0:
            continue
        group = (
            vep_df.loc[mask]
            .groupby("site", observed=True)[vep_col]
            .mean()
            .reset_index()
            .rename(columns={vep_col: f"VEP_{sp}"})
        )
        superpop_dfs.append(group)

    # Merge all superpops into one df (each superpop as own column)
    from functools import reduce
    if len(superpop_dfs) > 0:
        vep_superpop_agg = reduce(lambda left, right: pd.merge(left, right, on="site", how="outer"), superpop_dfs)
    else:
        # No data case
        vep_superpop_agg = pd.DataFrame(columns=["site"])

    # Mean VEP for all non-REF samples, per site
    print("Computing mean VEP per variant across all (non-REF) samples")
    vep_nonref = (
        vep_df.loc[vep_df["sample"] != "REF"]
        .groupby("site", observed=True)[vep_col]
        .mean()
        .reset_index()
        .rename(columns={vep_col: "VEP_nonref"})
    )

    # Mean VEP per site across non-REF (for this application, this is same as above)
    print("Computing mean VEP per variant across all (non-REF) samples")
    vep_mean = (
        vep_nonref.groupby("site", observed=True)["VEP_nonref"]
        .mean()
        .reset_index()
        .rename(columns={"VEP_nonref": "VEP_mean"})
    )

    # Mean VEP for reference samples, per site
    print("Computing mean VEP for reference samples per variant")
    vep_ref = (
        vep_df.loc[vep_df["sample"] == "REF"]
        .groupby("site", observed=True)[vep_col]
        .mean()
        .reset_index()
        .rename(columns={vep_col: "VEP_ref"})
    )

    # Merge all into one table
    vep_superpop_agg = (
        vep_superpop_agg
        .merge(vep_mean, on="site", how="outer")
        .merge(vep_nonref, on="site", how="outer")
        .merge(vep_ref, on="site", how="outer")
    )

    print("vep_superpop_agg.shape:", vep_superpop_agg.shape)
    return vep_superpop_agg

def to_latex_subscript(s):
    """Convert underscores to LaTeX subscript (for display in matplotlib/seaborn).
    
    If string ends with '_af' or '_AF', format as 'AF_{\text{firstpart}}'.
    """
    if s.lower().endswith('_af'):
        base = s[:-3]
        return rf"$AF_{{\text{{{base}}}}}$"
    elif '_' in s:
        base, sub = s.split('_', 1)
        return rf"${base}_{{\text{{{sub}}}}}$"
    else:
        return s


# Exponential decay
def exp_func(x, a, b):
    return a * np.exp(b * x)

 # Exponential decay: y = a * exp(-b * x)
def exp_decay_func(x, a, b):
    return a * np.exp(-b * x)

# Logistic function
def logistic_func(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# Quadratic polynomial
def poly2_func(x, a, b, c):
    return a * x**2 + b * x + c

def linear_func(x, a, b):
    return a * x + b

# Power law (inverse power): y = a / (x^b + c)
def power_law_inverse_func(x, a, b, c):
    return a / (x**b + c)

# Hyperbolic: y = a / (1 + b*x)
def hyperbolic_func(x, a, b):
    return a / (1 + b * x)

# Exponential decay with floor: y = floor + (a - floor) * exp(-b*x)
def exp_decay_floor_func(x, a, b, floor):
    return floor + (a - floor) * np.exp(-b * x)

# Inverse: y = a / (x + b)
def inverse_func(x, a, b):
    return a / (x + b)

# Linear decay in log space (for when y is log-transformed)
# If AF = a * exp(-b * VEP), then log10(AF) = log10(a) - b * VEP / ln(10)
# This is linear: y = intercept + slope * x
# But we'll use this for more general linear fits in log space
def linear_log_func(x, a, b):
    return a + b * x

# Power law in log space: if y = log10(AF) and AF = a * VEP^(-b)
# then y = log10(a) - b * log10(VEP)
# But since VEP is not log-transformed here, we use: y = a - b * log10(VEP + c)
def power_law_log_func(x, a, b, c):
    return a - b * np.log10(x + c)

# When both x and y are log-transformed: log10(AF) vs log10(VEP)
# If AF = a * VEP^b, then log10(AF) = log10(a) + b * log10(VEP)
def power_law_both_log_func(x, a, b):
    return a + b * x  # x is already log10(VEP), so this is linear in log-log space

# Exponential decay in log-log space: for log10(AF) vs log10(VEP)
# y = a - b * exp(-c * x) where x is log10(VEP)
def exp_decay_log_log_func(x, a, b, c):
    return a - b * np.exp(-c * x)

# Modified power law with offset: y = a - b * (x + c)^d
# Works well when x is not log-transformed but shows power-law-like behavior
def power_law_offset_func(x, a, b, c, d):
    return a - b * (x + c)**d

# Sigmoid in log space: y = a / (1 + exp(-b * (x - c)))
# For log-transformed y when relationship is sigmoidal
def sigmoid_log_func(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))


def fit_curve_and_plot(x, y, ax, fit_model="logistic", alpha=0.6, show_ci=True):
    """Helper function to fit curve and plot it with confidence intervals"""
    from scipy import stats
    from scipy.optimize import curve_fit
    
    # Ensure inputs are numpy arrays with numeric types
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Filter out any remaining invalid values
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    if len(x) == 0:
        return  # No valid data to fit
    
    # Fit the selected model
    if fit_model == "logistic":
        popt, pcov = curve_fit(logistic_func, x, y, maxfev=10000)
        y_pred = logistic_func(x, *popt)
    elif fit_model == "exp":
        popt, pcov = curve_fit(exp_func, x, y, maxfev=10000)
        y_pred = exp_func(x, *popt)
    elif fit_model == "exp_decay": 
        popt, pcov = curve_fit(exp_decay_func, x, y, maxfev=10000)
        y_pred = exp_decay_func(x, *popt)
    elif fit_model == "poly2":
        popt, pcov = curve_fit(poly2_func, x, y, maxfev=10000)
        y_pred = poly2_func(x, *popt)
    elif fit_model == "linear":
        popt, pcov = curve_fit(linear_func, x, y, maxfev=10000)
        y_pred = linear_func(x, *popt)
    elif fit_model == "power_law_inverse":
        # Initialize with reasonable guesses
        p0 = [np.max(y), 1.0, 0.1]
        popt, pcov = curve_fit(power_law_inverse_func, x, y, p0=p0, maxfev=10000)
        y_pred = power_law_inverse_func(x, *popt)
    elif fit_model == "hyperbolic":
        # Initialize with reasonable guesses
        p0 = [np.max(y), 1.0]
        popt, pcov = curve_fit(hyperbolic_func, x, y, p0=p0, maxfev=10000)
        y_pred = hyperbolic_func(x, *popt)
    elif fit_model == "exp_decay_floor":
        # Initialize with reasonable guesses
        p0 = [np.max(y), 1.0, np.min(y)]
        popt, pcov = curve_fit(exp_decay_floor_func, x, y, p0=p0, maxfev=10000)
        y_pred = exp_decay_floor_func(x, *popt)
    elif fit_model == "inverse":
        # Initialize with reasonable guesses
        p0 = [np.max(y) * np.mean(x), np.mean(x)]
        popt, pcov = curve_fit(inverse_func, x, y, p0=p0, maxfev=10000)
        y_pred = inverse_func(x, *popt)
    elif fit_model == "linear_log":
        # Linear model for log-transformed y (when AF decays exponentially with VEP)
        popt, pcov = curve_fit(linear_log_func, x, y, maxfev=10000)
        y_pred = linear_log_func(x, *popt)
    elif fit_model == "power_law_log":
        # Power law in log space: y = a - b * log10(x + c)
        p0 = [np.max(y), 1.0, 0.01]
        popt, pcov = curve_fit(power_law_log_func, x, y, p0=p0, maxfev=10000)
        y_pred = power_law_log_func(x, *popt)
    elif fit_model == "power_law_both_log":
        # Power law when both x and y are log-transformed: y = a + b*x
        popt, pcov = curve_fit(power_law_both_log_func, x, y, maxfev=10000)
        y_pred = power_law_both_log_func(x, *popt)
    elif fit_model == "exp_decay_log_log":
        # Exponential decay in log-log space
        p0 = [np.max(y), np.max(y) - np.min(y), 1.0]
        popt, pcov = curve_fit(exp_decay_log_log_func, x, y, p0=p0, maxfev=10000)
        y_pred = exp_decay_log_log_func(x, *popt)
    elif fit_model == "power_law_offset":
        # Modified power law with offset: y = a - b * (x + c)^d
        p0 = [np.max(y), 1.0, 0.01, 0.5]
        popt, pcov = curve_fit(power_law_offset_func, x, y, p0=p0, maxfev=10000)
        y_pred = power_law_offset_func(x, *popt)
    elif fit_model == "sigmoid_log":
        # Sigmoid in log space
        p0 = [np.max(y), 1.0, np.mean(x)]
        popt, pcov = curve_fit(sigmoid_log_func, x, y, p0=p0, maxfev=10000)
        y_pred = sigmoid_log_func(x, *popt)
    else:
        raise ValueError(f"Invalid fit model: {fit_model}")

    # Calculate R²
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calculate p-value using F-test
    n = len(x)
    p = len(popt)
    f_stat = (ss_tot - ss_res) / p / (ss_res / (n - p))
    p_value = 1 - stats.f.cdf(f_stat, p, n - p)
    
    # Calculate parameter standard errors (needed for both CI plotting and statistics)
    perr = np.sqrt(np.diag(pcov))
    
    # Store axis limits based on data before plotting CI (to prevent CI from affecting scale)
    if show_ci:
        xlim_data = ax.get_xlim()
        ylim_data = ax.get_ylim()
    
    # Plot curve and confidence intervals
    x_line = np.linspace(x.min(), x.max(), 100)
    if fit_model == "logistic":
        y_line = logistic_func(x_line, *popt)
        if show_ci:
            y_upper = logistic_func(x_line, *(popt + 1.96 * perr))
            y_lower = logistic_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "exp":
        y_line = exp_func(x_line, *popt)
        if show_ci:
            y_upper = exp_func(x_line, *(popt + 1.96 * perr))
            y_lower = exp_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "exp_decay": 
        y_line = exp_decay_func(x_line, *popt)
        if show_ci:
            y_upper = exp_decay_func(x_line, *(popt + 1.96 * perr))
            y_lower = exp_decay_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "poly2":
        y_line = poly2_func(x_line, *popt)
        if show_ci:
            y_upper = poly2_func(x_line, *(popt + 1.96 * perr))
            y_lower = poly2_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "linear":
        y_line = linear_func(x_line, *popt)
        if show_ci:
            y_upper = linear_func(x_line, *(popt + 1.96 * perr))
            y_lower = linear_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "power_law_inverse":
        y_line = power_law_inverse_func(x_line, *popt)
        if show_ci:
            y_upper = power_law_inverse_func(x_line, *(popt + 1.96 * perr))
            y_lower = power_law_inverse_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "hyperbolic":
        y_line = hyperbolic_func(x_line, *popt)
        if show_ci:
            y_upper = hyperbolic_func(x_line, *(popt + 1.96 * perr))
            y_lower = hyperbolic_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "exp_decay_floor":
        y_line = exp_decay_floor_func(x_line, *popt)
        if show_ci:
            y_upper = exp_decay_floor_func(x_line, *(popt + 1.96 * perr))
            y_lower = exp_decay_floor_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "inverse":
        y_line = inverse_func(x_line, *popt)
        if show_ci:
            y_upper = inverse_func(x_line, *(popt + 1.96 * perr))
            y_lower = inverse_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "linear_log":
        y_line = linear_log_func(x_line, *popt)
        if show_ci:
            y_upper = linear_log_func(x_line, *(popt + 1.96 * perr))
            y_lower = linear_log_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "power_law_log":
        y_line = power_law_log_func(x_line, *popt)
        if show_ci:
            y_upper = power_law_log_func(x_line, *(popt + 1.96 * perr))
            y_lower = power_law_log_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "power_law_both_log":
        y_line = power_law_both_log_func(x_line, *popt)
        if show_ci:
            y_upper = power_law_both_log_func(x_line, *(popt + 1.96 * perr))
            y_lower = power_law_both_log_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "exp_decay_log_log":
        y_line = exp_decay_log_log_func(x_line, *popt)
        if show_ci:
            y_upper = exp_decay_log_log_func(x_line, *(popt + 1.96 * perr))
            y_lower = exp_decay_log_log_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "power_law_offset":
        y_line = power_law_offset_func(x_line, *popt)
        if show_ci:
            y_upper = power_law_offset_func(x_line, *(popt + 1.96 * perr))
            y_lower = power_law_offset_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "sigmoid_log":
        y_line = sigmoid_log_func(x_line, *popt)
        if show_ci:
            y_upper = sigmoid_log_func(x_line, *(popt + 1.96 * perr))
            y_lower = sigmoid_log_func(x_line, *(popt - 1.96 * perr))
    if show_ci:
        ax.fill_between(x_line, y_lower, y_upper, color='black', alpha=0.1, clip_on=True)
        # Restore axis limits to data-based limits (not CI-based)
        ax.set_xlim(xlim_data)
        ax.set_ylim(ylim_data)
    ax.plot(x_line, y_line, color='black', alpha=alpha)
    
    # Add R² and p-value as text
    ax.text(0.95, 0.95, f'R² = {r2:.3f}\np = {p_value:.2e}', 
           transform=ax.transAxes, 
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Calculate additional statistics
    # 95% confidence intervals for parameters (assuming normal distribution)
    param_ci_lower = popt - 1.96 * perr
    param_ci_upper = popt + 1.96 * perr
    
    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(residuals))
    
    # Calculate adjusted R²
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else np.nan
    
    # Return dictionary with all statistics
    stats_dict = {
        'r2': r2,
        'adj_r2': adj_r2,
        'p_value': p_value,
        'rmse': rmse,
        'mae': mae,
        'n': n,
        'n_params': p,
        'params': popt.tolist(),  # Convert to list for DataFrame storage
        'param_std_err': perr.tolist(),
        'param_ci_lower': param_ci_lower.tolist(),
        'param_ci_upper': param_ci_upper.tolist(),
    }
    
    return stats_dict

def fit_curve_and_plot_no_plot(x, y, fit_model="logistic"):
    """Calculate fit statistics without plotting"""
    from scipy import stats
    from scipy.optimize import curve_fit
    
    # Ensure inputs are numpy arrays with numeric types
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Filter out any remaining invalid values
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    if len(x) == 0:
        return None
    
    # Fit the selected model (same logic as fit_curve_and_plot)
    if fit_model == "logistic":
        popt, pcov = curve_fit(logistic_func, x, y, maxfev=10000)
        y_pred = logistic_func(x, *popt)
    elif fit_model == "exp":
        popt, pcov = curve_fit(exp_func, x, y, maxfev=10000)
        y_pred = exp_func(x, *popt)
    elif fit_model == "exp_decay": 
        popt, pcov = curve_fit(exp_decay_func, x, y, maxfev=10000)
        y_pred = exp_decay_func(x, *popt)
    elif fit_model == "poly2":
        popt, pcov = curve_fit(poly2_func, x, y, maxfev=10000)
        y_pred = poly2_func(x, *popt)
    elif fit_model == "linear":
        popt, pcov = curve_fit(linear_func, x, y, maxfev=10000)
        y_pred = linear_func(x, *popt)
    elif fit_model == "power_law_inverse":
        p0 = [np.max(y), 1.0, 0.1]
        popt, pcov = curve_fit(power_law_inverse_func, x, y, p0=p0, maxfev=10000)
        y_pred = power_law_inverse_func(x, *popt)
    elif fit_model == "hyperbolic":
        p0 = [np.max(y), 1.0]
        popt, pcov = curve_fit(hyperbolic_func, x, y, p0=p0, maxfev=10000)
        y_pred = hyperbolic_func(x, *popt)
    elif fit_model == "exp_decay_floor":
        p0 = [np.max(y), 1.0, np.min(y)]
        popt, pcov = curve_fit(exp_decay_floor_func, x, y, p0=p0, maxfev=10000)
        y_pred = exp_decay_floor_func(x, *popt)
    elif fit_model == "inverse":
        p0 = [np.max(y) * np.mean(x), np.mean(x)]
        popt, pcov = curve_fit(inverse_func, x, y, p0=p0, maxfev=10000)
        y_pred = inverse_func(x, *popt)
    elif fit_model == "linear_log":
        popt, pcov = curve_fit(linear_log_func, x, y, maxfev=10000)
        y_pred = linear_log_func(x, *popt)
    elif fit_model == "power_law_log":
        p0 = [np.max(y), 1.0, 0.01]
        popt, pcov = curve_fit(power_law_log_func, x, y, p0=p0, maxfev=10000)
        y_pred = power_law_log_func(x, *popt)
    elif fit_model == "power_law_both_log":
        popt, pcov = curve_fit(power_law_both_log_func, x, y, maxfev=10000)
        y_pred = power_law_both_log_func(x, *popt)
    elif fit_model == "exp_decay_log_log":
        p0 = [np.max(y), np.max(y) - np.min(y), 1.0]
        popt, pcov = curve_fit(exp_decay_log_log_func, x, y, p0=p0, maxfev=10000)
        y_pred = exp_decay_log_log_func(x, *popt)
    elif fit_model == "power_law_offset":
        p0 = [np.max(y), 1.0, 0.01, 0.5]
        popt, pcov = curve_fit(power_law_offset_func, x, y, p0=p0, maxfev=10000)
        y_pred = power_law_offset_func(x, *popt)
    elif fit_model == "sigmoid_log":
        p0 = [np.max(y), 1.0, np.mean(x)]
        popt, pcov = curve_fit(sigmoid_log_func, x, y, p0=p0, maxfev=10000)
        y_pred = sigmoid_log_func(x, *popt)
    else:
        raise ValueError(f"Invalid fit model: {fit_model}")
    
    # Calculate R²
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calculate p-value using F-test
    n = len(x)
    p = len(popt)
    f_stat = (ss_tot - ss_res) / p / (ss_res / (n - p))
    p_value = 1 - stats.f.cdf(f_stat, p, n - p)
    
    # Calculate parameter standard errors
    perr = np.sqrt(np.diag(pcov))
    
    # Calculate additional statistics
    param_ci_lower = popt - 1.96 * perr
    param_ci_upper = popt + 1.96 * perr
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else np.nan
    
    # Return dictionary with all statistics
    stats_dict = {
        'r2': r2,
        'adj_r2': adj_r2,
        'p_value': p_value,
        'rmse': rmse,
        'mae': mae,
        'n': n,
        'n_params': p,
        'params': popt.tolist(),
        'param_std_err': perr.tolist(),
        'param_ci_lower': param_ci_lower.tolist(),
        'param_ci_upper': param_ci_upper.tolist(),
    }
    
    return stats_dict

def plot_vep_vs_af(df,
                   vep_col="VEP",
                   af_col="AF",
                   col=None,
                   hue="CLNSIG_simple",
                   logy=False,
                   logx=False,
                   sharex=True, 
                   sharey=True,
                   drop_vep_zeros=False,
                   palette="Set2",
                   height=4, 
                   fit_line=True,
                   alpha=0.6,
                   aspect=1,
                   xlim=None,
                   ylim=(0, 1),
                   fit_model="logistic",
                   ax=None,
                   col_wrap=3,
                   x_label=r"$log_{10}$(VEP)",
                   epsilon=1e-6,
                   show_ci=True,
                   show_plot=True
                   ): 

    # Create a figure with subplots for each CLNSIG_simple category

    df = df.copy()
    
    # Check if vep_col is a list
    if isinstance(vep_col, list):
        # Create subplots for each VEP column
        n_cols = len(vep_col)
        fig, axes = plt.subplots(1, n_cols, figsize=(height*aspect*n_cols, height), sharey=sharey)
        if n_cols == 1:
            axes = [axes]
        
        # Plot each VEP column
        for i, vep_col_item in enumerate(vep_col):
            ax_current = axes[i]
            
            # Filter data for this VEP column
            df_subset = df.dropna(subset=[vep_col_item, af_col]).copy()

            # Drop zero VEP values for this subplot if requested
            if drop_vep_zeros:
                df_subset = df_subset[df_subset[vep_col_item] != 0]

            if len(df_subset) == 0:
                ax_current.set_title(f"{vep_col_item}\n(No data)")
                continue
            
            df_subset.loc[:, "VEP_log"] = np.log10(df_subset[vep_col_item] + epsilon)
            df_subset.loc[:, "AF_log"] = np.log10(df_subset[af_col] + epsilon)

            # Handle categorical ordering
            if col is not None and col == "Super Population":
                col_order = ["REF"]+[x for x in df_subset["Super Population"].unique().tolist() if x!="REF"]
                df_subset.loc[:, "Super Population"] = pd.Categorical(df_subset["Super Population"], categories=col_order, ordered=True)
                df_subset = df_subset.sort_values(by="Super Population")
                
            if hue is not None and hue == "CLNSIG_simple":
                hue_order = list(utils.get_clinsig_palette().keys())
                hue_order.reverse()
                df_subset.loc[:, "CLNSIG_simple"] = pd.Categorical(df_subset["CLNSIG_simple"], categories=hue_order, ordered=True)
                df_subset = df_subset.sort_values(by="CLNSIG_simple")

            if logy:
                y_var = "AF_log"
            else:
                y_var = af_col
            if logx:
                x_var = "VEP_log"
            else:
                x_var = vep_col_item

            # Get appropriate color palette
            if hue=="Super Population":
                cmap = utils.get_superpop_palette()
            elif hue=="CLNSIG_simple":
                cmap = utils.get_clinsig_palette()
            else:
                cmap = utils.make_palette(df_subset[hue].unique(), palette=palette)

            # Add scatter plot
            sns.scatterplot(data=df_subset,
                           x=x_var,
                           y=y_var, 
                           hue=hue,
                           palette=cmap,
                           alpha=alpha,
                           ax=ax_current)

            # Add regression line and calculate statistics
            if fit_line:
                fit_curve_and_plot(df_subset[x_var], df_subset[y_var], ax_current, fit_model, alpha, show_ci)
                
            if xlim is not None:
                ax_current.set_xlim(xlim)
            if ylim is not None:
                # Set y-ticks to show only the range we want to display
                ax_current.set_yticks(np.linspace(ylim[0], ylim[1], 5))
            
            # Set title with af_col name
            ax_current.set_title(f"{af_col}, Variants: {df_subset['site'].nunique()}")
            
            # Create x-axis label based on VEP column name
            # If vep_col contains underscore, subscript the part after underscore
            if "_" in vep_col_item:
                parts = vep_col_item.split("_", 1)
                vep_label = f"{parts[0]}_{{{parts[1]}}}"
            else:
                vep_label = vep_col_item
            
            if logx:
                ax_current.set_xlabel(f"$log_{{10}}({vep_label})$")
            else:
                ax_current.set_xlabel(f"${vep_label}$")
        
        # Add legend in lower left of first subplot only
        if hue is not None:
            handles, labels = axes[0].get_legend_handles_labels()
            axes[0].legend(handles, labels, loc='lower left')
            for ax_item in axes[1:]:  # Remove legend from other subplots
                ax_item.get_legend().remove()
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        return fig, axes

    ####### Original single vep_col behavior #######
    df.dropna(subset=[vep_col, af_col], inplace=True)
    df.loc[:, "VEP_log"] = np.log10(df[vep_col] + epsilon)
    df.loc[:, "AF_log"] = np.log10(df[af_col] + epsilon)

    if col is not None and col == "Super Population":
        col_order = ["REF"]+[x for x in df["Super Population"].unique().tolist() if x!="REF"]
        df.loc[:, "Super Population"] = pd.Categorical(df["Super Population"], categories=col_order, ordered=True)
        df = df.sort_values(by="Super Population")
    else:
        col_order = None
    if hue is not None and hue == "CLNSIG_simple":
        hue_order = list(utils.get_clinsig_palette().keys())
        hue_order.reverse()
        df.loc[:, "CLNSIG_simple"] = pd.Categorical(df["CLNSIG_simple"], categories=hue_order, ordered=True)
        df = df.sort_values(by="CLNSIG_simple")
    else:
        hue_order = None

    if logy:
        y_var = "AF_log"
    else:
        y_var = af_col
    if logx:
        x_var = "VEP_log"
    else:
        x_var = vep_col

    # Get appropriate color palette
    if hue=="Super Population":
        cmap = utils.get_superpop_palette()
    elif hue=="CLNSIG_simple":
        cmap = utils.get_clinsig_palette()
    else:
        cmap = utils.make_palette(df[hue].unique(), palette=palette)

    # Create figure based on whether faceting is requested
    if col is None:
        plt.figure(figsize=(height*aspect, height))
        if ax is None:
            ax = plt.gca()
        
        # Add scatter plot
        sns.scatterplot(data=df,
                       x=x_var,
                       y=y_var, 
                       hue=hue,
                       palette=cmap,
                       alpha=alpha,
                       ax=ax)

        # Add regression line and calculate statistics
        if fit_line:
            fit_curve_and_plot(df[x_var], df[y_var], ax, fit_model, alpha, show_ci)
            
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            # Set y-ticks to show only the range we want to display
            ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))
            # Don't set ylim to allow points to be visible outside the tick range
        plt.title(f"{af_col}, Variants: {df['site'].nunique()}")
        if x_label is not None:
            ax.set_xlabel(x_label)
        else:
            # Format vep_col name with subscript if it contains underscore
            if "_" in vep_col:
                parts = vep_col.split("_", 1)
                vep_label = f"{parts[0]}_{{{parts[1]}}}"
            else:
                vep_label = vep_col
            if logx:
                ax.set_xlabel(f"$log_{{10}}({vep_label})$")
            else:
                ax.set_xlabel(f"${vep_label}$")
        
    else:
        g = sns.FacetGrid(df, 
                        col=col,
                        sharex=sharex, 
                        sharey=sharey,
                        col_wrap=col_wrap, 
                        height=height,
                        aspect=aspect,
                        col_order=col_order,
                        margin_titles=True)

        # Add scatter plot
        g.map_dataframe(sns.scatterplot, 
                        x=x_var,
                        y=y_var, 
                        hue=hue,
                        palette=cmap,
                        alpha=alpha)

        # Add curve fit to each facet
        if fit_line:
            def fit_and_plot_facet(data, **kwargs):
                ax = plt.gca()
                fit_curve_and_plot(data[x_var], data[y_var], ax, fit_model, alpha, show_ci)
                if ylim is not None:
                    # Set y-ticks to show only the range we want to display
                    ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))
            
            g.map_dataframe(fit_and_plot_facet)

        g.add_legend()
        g.fig.suptitle(f"{af_col}, Variants: {df['site'].nunique()}", y=1.02)
        if xlim is not None:
            g.set(xlim=xlim)
        if ylim is not None:
            # Set y-ticks for all facets
            for ax_item in g.axes.flat:
                ax_item.set_yticks(np.linspace(ylim[0], ylim[1], 5))
        if x_label is not None:
            g.set_xlabels(x_label)
        else:
            # Format vep_col name with subscript if it contains underscore
            if "_" in vep_col:
                parts = vep_col.split("_", 1)
                vep_label = f"{parts[0]}_{{{parts[1]}}}"
            else:
                vep_label = vep_col
            if logx:
                g.set_xlabels(f"$log_{{10}}({vep_label})$")
            else:
                g.set_xlabels(f"${vep_label}$")
       
    if show_plot:
        plt.show()
    return fig, g

    
# Radar (spider) plot of R2 by AF dataset and VEP dataset,
# with AF dataset family color ring ONLY for each dataset's contiguous section

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re
from itertools import groupby

def plot_vep_vs_af_radar(
    stats_df,
    value_col="r2",
    af_col="af_col",
    vep_col="supertitle",
    palette="Accent",
    af_dataset_names=None,
    af_dataset_palette=None,
    outer_ring_palette=None,
    superpop_palette=utils.get_superpop_palette(), 
    fig_filename=None,
    fig_save_kwargs=utils.FIG_SAVE_KWARGS,
    figsize=4,
    title="VEP as a Predictor of Allele Frequency\n($R^2$, higher is better)",
    legend1_title="VEP Dataset",
    legend2_title="AF Dataset Family",
    r2_tick_count=6,
    legend1_loc=(-.4, 1.15),
    legend2_loc=(-.4, 0.7),
    legend_fontsize='small',
    legend_title_fontsize='small',
    tight_layout_kwargs=None,
    show=True,
    family_ring_thickness=None,
    label_ring_padding=None,
    linewidth=2,
    group_superpop_as_all=False,
    grid_kwargs=None,
    radial_grid_kwargs=None,
    title_pad=35,
    family_ring_alpha=1.0,
    reverse_area_order=False,
    return_mapping=False
):
    """
    Generate a radar plot comparing R^2 by AF dataset and VEP dataset.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame containing statistics with columns for value_col, af_col, and vep_col
    value_col : str
        Column name for the values to plot (default: "r2")
    af_col : str
        Column name for AF dataset identifiers (default: "af_col")
    vep_col : str
        Column name for VEP dataset identifiers (default: "supertitle")
    palette : str
        Name of color palette to use for AF datasets. Can be a seaborn palette name
        (e.g., "Accent", "Set2", "tab10") or a matplotlib colormap name (e.g., "viridis", "plasma").
        Colors will be sampled from this palette. Default: "Accent"
    af_dataset_names : list or None
        List of AF dataset names (default: None, uses predefined list)
    af_dataset_palette : dict or None
        Dictionary mapping AF dataset names to colors (default: None)
    outer_ring_palette : list or None
        Color palette for VEP dataset lines (default: None, uses ["gold", "goldenrod", "gray"])
    superpop_palette : dict
        Palette for superpopulation colors (default: from utils.get_superpop_palette())
    fig_filename : str or None
        Filename to save figure (default: None)
    fig_save_kwargs : dict or None
        Keyword arguments for saving figure (default: None)
    figsize : float
        Figure size (default: 4)
    title : str
        Plot title (default: "VEP as a Predictor of Allele Frequency\n($R^2$, higher is better)")
    legend1_title : str
        Title for VEP dataset legend (default: "VEP Dataset")
    legend2_title : str
        Title for AF dataset family legend (default: "AF Dataset Family")
    r2_tick_count : int
        Number of R^2 tick marks (default: 6)
    legend1_loc : tuple
        Location for VEP legend (default: (-.4, 1.15))
    legend2_loc : tuple
        Location for AF family legend (default: (-.4, 0.7))
    legend_fontsize : str or float
        Font size for legend entries (default: 'small')
    legend_title_fontsize : str or float
        Font size for legend titles (default: 'small')
    tight_layout_kwargs : dict or None
        Keyword arguments for tight_layout (default: None)
    show : bool
        Whether to display the plot (default: True)
    family_ring_thickness : float, tuple, or None
        Controls the thickness of the outer family ring. Options:
        - None (default): Uses default thickness of 0.01 * r2_max_tick
        - Single float: Thickness as fraction of r2_max_tick (e.g., 0.02 for double thickness)
        - Tuple of 2 floats: (inner_multiplier, outer_multiplier) relative to r2_max_tick
          (e.g., (1.04, 1.06) for a thicker ring starting at 1.04 and ending at 1.06)
    label_ring_padding : float or None
        Controls the padding/gap between AF labels and the family ring, as a fraction
        of r2_max_tick. The label position is calculated as: ring_inner - (r2_max_tick * padding).
        - None (default): Uses default padding of 0.01 * r2_max_tick
        - Single float: Padding as fraction of r2_max_tick (e.g., 0.02 for double padding,
          0.005 for half padding)
    linewidth : float
        Line width (thickness) for the area plot lines (default: 2)
    group_superpop_as_all : bool
        If True, all AF columns that match keys in superpop_palette will be grouped
        under a new group called "All" (default: False)
    grid_kwargs : dict or None
        Keyword arguments for the radar plot concentric grid lines (circles). If None, uses default:
        {'linestyle': ':', 'color': 'gray', 'alpha': 0.6}. Can override any grid
        parameters like linestyle, color, alpha, linewidth, etc.
    radial_grid_kwargs : dict or None
        Keyword arguments for the radial grid lines (lines along the radius). If None, uses default:
        {'linestyle': '-', 'color': 'gray', 'alpha': 0.3}. Can override any grid
        parameters like linestyle, color, alpha, linewidth, etc.
    title_pad : float
        Padding (in points) for the title's vertical position (default: 35).
        Increase to move title higher, decrease to move it lower.
    family_ring_alpha : float
        Alpha (transparency) value for the family ring segments (default: 1.0).
        Range: 0.0 (fully transparent) to 1.0 (fully opaque).
    reverse_area_order : bool
        If True, reverse the draw order of the area plots (last dataset drawn first)
        so later datasets appear underneath earlier ones. Default: False.
    return_mapping : bool
        If True, also return a DataFrame describing how AF columns were processed
        (columns: af_col, af_group, af_short, af_color). Default: False.
    """
    # === AF dataset palette ===
    # Sample colors from the specified palette
    import warnings
    import matplotlib.colors as mcolors
    
    if af_dataset_names is None:
        af_dataset_names = [
            'AF', 'gnomADg', 'gnomADe', '1000Gp3', 'ALFA', 'TOPMed', 'gnomAD2.1.1_exomes', 'gnomAD4.1_joint'
        ]
    
    # Get colors from palette - try seaborn first, then matplotlib colormap
    n_colors_needed = len(af_dataset_names)
    palette_found = False
    try:
        # Try seaborn palette first (handles both seaborn palettes and matplotlib colormaps)
        palette_colors = sns.color_palette(palette, n_colors=n_colors_needed)
        # Convert to hex if needed
        if hasattr(palette_colors, 'as_hex'):
            accent_colors = palette_colors.as_hex()
        else:
            # Convert RGB tuples to hex
            accent_colors = [mcolors.to_hex(c) for c in palette_colors]
        palette_found = True
    except (ValueError, KeyError):
        # Fallback to matplotlib colormap
        try:
            cmap = plt.get_cmap(palette)
            # Sample evenly spaced colors from the colormap
            if n_colors_needed == 1:
                colors_rgba = [cmap(0.5)]  # Use middle of colormap for single color
            else:
                colors_rgba = [cmap(i / (n_colors_needed - 1)) for i in range(n_colors_needed)]
            accent_colors = [mcolors.to_hex(c) for c in colors_rgba]
            palette_found = True
        except ValueError:
            # Final fallback to Accent
            warnings.warn(
                f"Palette '{palette}' not found in seaborn or matplotlib. "
                f"Falling back to 'Accent' palette.",
                UserWarning
            )
            accent_colors = sns.color_palette("Accent", n_colors=n_colors_needed).as_hex()
    
    if af_dataset_palette is None:
        af_dataset_palette = {
            name: accent_colors[i % len(accent_colors)]
            for i, name in enumerate(af_dataset_names)
        }
    # Extend palette with superpop mapping, if given (expects a dict)
    if superpop_palette is not None:
        af_dataset_palette.update(superpop_palette)

    # === Label helpers ===
    def extract_dataset(af_col_val):
        # Treat only AF-prefixed columns (e.g., AF_coalesced, AF_global) as the "AF" dataset.
        # AFR_* and other superpopulation prefixes will be handled below via palette matching.
        if af_col_val.startswith("AF_"):
            return "AF"
        for ds in af_dataset_palette:
            if af_col_val.startswith(ds):
                return ds
        return af_col_val.split("_")[0]

    def short_af_label(col):
        rm_leading = re.sub(r"^(gnomADe|gnomADg|ALFA|1000Gp3|gnomAD2\.1\.1_exomes|gnomAD4\.1_joint|TOPMed)_?", "", col)
        rm_tail = re.sub(r"_AF$", "", rm_leading)
        return rm_tail.replace("_", " ")

    all_af_cols = stats_df[af_col].unique().tolist()
    print(f"Total number of unique AF labels before processing: {len(all_af_cols)}")
    
    af_col_to_dataset = {col_: extract_dataset(col_) for col_ in all_af_cols}
    
    # Group superpop keys into "All" if requested
    if group_superpop_as_all and superpop_palette is not None:
        superpop_keys = set(superpop_palette.keys())
        # Include global AF dataset in the aggregated "All" group as requested.
        superpop_keys.add("AF")
        # Update mapping: if extracted dataset is in superpop_palette, change to "All"
        af_col_to_dataset = {
            col_: "All" if dataset in superpop_keys else dataset
            for col_, dataset in af_col_to_dataset.items()
        }
        # Add "All" to palette if it doesn't exist
        if "All" not in af_dataset_palette:
            # Use black color for "All" group
            af_dataset_palette["All"] = "#000000"  # Black color
    
    af_col_to_color = {col_: af_dataset_palette.get(af_col_to_dataset[col_], "#cccccc") for col_ in all_af_cols}

    stats_df = stats_df.copy()  # Safe to mutate for plotting
    stats_df["af_group"] = stats_df[af_col].map(af_col_to_dataset)
    stats_df["af_color"] = stats_df[af_col].map(af_col_to_color)
    stats_df["af_short"] = stats_df[af_col].apply(short_af_label)

    cat_df = stats_df.drop_duplicates(af_col)[[af_col, "af_group", "af_short", "af_color"]]
    cat_df = cat_df.sort_values(["af_group", "af_short"])
    all_categories = cat_df[af_col].tolist()
    short_labels = cat_df["af_short"].tolist()
    outer_colors = cat_df["af_color"].tolist()
    
    print(f"Total number of AF labels after processing (for plotting): {len(all_categories)}")

    pivot_df = stats_df.pivot_table(
        index=vep_col,
        columns=af_col,
        values=value_col
    )
    pivot_df = pivot_df.reindex(columns=all_categories)
    N = len(all_categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    r2_max = np.nanmax(pivot_df.values)
    # Use actual max value instead of rounding up
    r2_max_tick = r2_max
    r2_tick_vals = np.linspace(0, r2_max_tick, r2_tick_count)

    if outer_ring_palette is None:
        colors = ["gold", "goldenrod", "gray"]
    else:
        colors = outer_ring_palette

    vep_labels = pivot_df.index.tolist()
    label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(vep_labels)}

    fig, ax = plt.subplots(figsize=(figsize, figsize), subplot_kw=dict(polar=True))

    draw_order = vep_labels[:]
    if reverse_area_order:
        draw_order = list(reversed(draw_order))

    for idx, label in enumerate(draw_order):
        row = pivot_df.loc[label]
        values = [row[cat] if cat in row else np.nan for cat in all_categories]
        values += values[:1]
        color = label_to_color[label]
        ax.plot(angles, values, linewidth=linewidth, label=label, color=color)
        ax.fill(angles, values, alpha=0.23, color=color)

    ax.set_xticks(angles[:-1])
    
    # Set radial grid (theta/x-axis grid) with customizable kwargs
    if radial_grid_kwargs is None:
        radial_grid_kwargs = {'linestyle': '-', 'color': 'gray', 'alpha': 0.3}
    ax.xaxis.grid(True, **radial_grid_kwargs)

    # === Calculate ring positions first ===
    # Control family ring thickness
    if family_ring_thickness is None:
        # Default behavior: thickness = 0.01 * r2_max_tick
        ring_r_outer = r2_max_tick*1.05
        ring_r_inner = r2_max_tick*1.04
    else:
        # family_ring_thickness can be:
        # - A single float: thickness as fraction of r2_max_tick (e.g., 0.01)
        # - A tuple/list of 2 floats: (inner_multiplier, outer_multiplier) relative to r2_max_tick
        if isinstance(family_ring_thickness, (tuple, list)) and len(family_ring_thickness) == 2:
            inner_mult, outer_mult = family_ring_thickness
            ring_r_inner = r2_max_tick * inner_mult
            ring_r_outer = r2_max_tick * outer_mult
        else:
            # Single value: thickness as fraction of r2_max_tick
            thickness = float(family_ring_thickness)
            ring_r_inner = r2_max_tick * 1.04  # Keep default inner position
            ring_r_outer = ring_r_inner + (r2_max_tick * thickness)

    # === Calculate label position based on ring and padding ===
    # Control padding between labels and ring
    if label_ring_padding is None:
        # Default padding: 0.01 * r2_max_tick (gap between 1.03 and 1.04)
        label_padding = 0.01
    else:
        # label_ring_padding is a fraction of r2_max_tick
        label_padding = float(label_ring_padding)
    
    label_r = ring_r_inner - (r2_max_tick * label_padding)

    # === Draw labels ===
    for i, (short_lab, angle) in enumerate(zip(short_labels, angles)):
        angle_deg = np.degrees(angle)
        ha = 'left'
        rotation = angle_deg
        if 90 < angle_deg < 270:
            rotation = angle_deg + 180
            ha = 'right'
        ax.text(
            angle, label_r, short_lab, size="small",
            horizontalalignment=ha, verticalalignment='center',
            rotation=rotation, rotation_mode='anchor'
        )

    ax.set_xticklabels([])

    # === Draw outer colored ring ONLY covering each family section ===

    # Find contiguous runs for each af_group in the list of af_col
    family_blocks = []
    cur_family = None
    run_start = 0
    for idx, family in enumerate(cat_df["af_group"]):
        if family != cur_family:
            if cur_family is not None:
                family_blocks.append((cur_family, run_start, idx-1))
            cur_family = family
            run_start = idx
    family_blocks.append((cur_family, run_start, len(cat_df["af_group"])-1))

    # Draw colored ring segments for each block
    for fam, i_start, i_end in family_blocks:
        color = af_dataset_palette.get(fam, "#cccccc")
        theta1 = angles[i_start]
        theta2 = angles[i_end+1] if (i_end+1) < len(angles) else angles[0]
        if theta2 < theta1:
            # Wrap-around case for last family
            span = (theta1, 2*np.pi)
            ax.bar(
                x=(span[0]+2*np.pi)/2, height=ring_r_outer - ring_r_inner, width=2*np.pi - span[0],
                bottom=ring_r_inner, color=color, align='center', linewidth=0, alpha=family_ring_alpha
            )
            span2 = (0, theta2)
            if theta2 > 0:
                ax.bar(
                    x=theta2/2, height=ring_r_outer - ring_r_inner, width=theta2,
                    bottom=ring_r_inner, color=color, align='center', linewidth=0, alpha=family_ring_alpha
                )
        else:
            mid_angle = (theta1 + theta2) / 2
            width = theta2 - theta1
            if width > 0:
                ax.bar(
                    x=mid_angle, height=ring_r_outer - ring_r_inner, width=width,
                    bottom=ring_r_inner, color=color, align='center', linewidth=0, alpha=family_ring_alpha
                )

    ax.set_ylim(0, ring_r_outer)
    ax.set_yticks(r2_tick_vals)
    ax.set_yticklabels([f"{y:.2f}" for y in r2_tick_vals], fontsize="small", color='gray')
    
    # Set grid with customizable kwargs
    if grid_kwargs is None:
        grid_kwargs = {'linestyle': ':', 'color': 'gray', 'alpha': 0.6}
    ax.yaxis.grid(True, **grid_kwargs)

    ax.set_title(
        title,
        pad=title_pad,
    )

    legend_handles_vep = [Patch(color=label_to_color[lbl], label=lbl) for lbl in vep_labels]
    legend_handles_af = []
    added_groups = set()
    for col_val, family in zip(all_categories, cat_df["af_group"]):
        if family not in added_groups:
            legend_handles_af.append(Patch(color=af_dataset_palette.get(family, "#cccccc"), label=family, alpha=family_ring_alpha))
            added_groups.add(family)

    # VEP legend (supertitle legend)
    legend_vep = ax.legend(
        handles=legend_handles_vep,
        loc='upper left', bbox_to_anchor=legend1_loc,
        title=legend1_title,
        fontsize=legend_fontsize, title_fontsize=legend_title_fontsize
    )
    ax.add_artist(legend_vep)  # Keep first legend

    # AF group legend (further down)
    legend_af = ax.legend(
        handles=legend_handles_af,
        loc='upper left', bbox_to_anchor=legend2_loc,
        title=legend2_title,
        fontsize=legend_fontsize, title_fontsize=legend_title_fontsize
    )

    if tight_layout_kwargs is None:
        tight_layout_kwargs = {}
    plt.tight_layout(**tight_layout_kwargs)
    if show:
        plt.show()
    if fig_filename is not None:
        fig.savefig(
            fig_filename,
            **(fig_save_kwargs or {})
        )
    if return_mapping:
        mapping_df = cat_df[["af_col", "af_group", "af_short", "af_color"]].reset_index(drop=True)
        return fig, ax, mapping_df
    return fig, ax

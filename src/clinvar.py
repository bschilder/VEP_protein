import os
import pooch
import polars as pl
import pandas as pd

import src.utils as utils


## Split clinvar.vcf.gx into one file per chromosome
# https://www.biostars.org/p/9506417/ 
# %%bash
# vcf_in=$HOME/.cache/pooch/clinvar.vcf.gz
# vcf_out_stem=$HOME/.cache/pooch/clinvar_by_chrom
# mkdir -p $vcf_out_stem
# bcftools index -s ${vcf_in} | cut -f 1 | while read C; do bcftools view -O z -o ${vcf_out_stem}/chr${C}.vcf.gz ${vcf_in} "${C}" ; done


# Detailed informaiton on each VCF field:
# https://ftp.ncbi.nlm.nih.gov/pub/clinvar/README_VCF.txt
INFO_COLS_SELECT = [
    "AF_ESP", "AF_EXAC", "AF_TGP", "ALLELEID", "CLNDISDB", "CLNDN",
    "CLNHGVS", "CLNREVSTAT", "CLNSIG", "CLNVC", "CLNVCSO", "CLNSIGCONF",
    "GENEINFO","MC", "ORIGIN", "RS",
    "ONC", "ONCDN", "ONCDISDB", "ONCREVSTAT", "ONCCONF",
    "SCI", "SCIDN", "SCIDISDB", "SCIREVSTAT"
]

def download_vcf(vcf_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"):
    """
    Download the latest ClinVar VCF file and index file.

    Args:
        vcf_url (str): The URL of the ClinVar VCF file.

    Returns:
        dict: A dictionary containing the path to the VCF file and the path to the index file.
    
    Example:
        download_clinvar()
    """
    vcf_file = pooch.retrieve(vcf_url,
                              fname=os.path.basename(vcf_url),
                              known_hash="19cf6d08cecbd4bae1c09c711a2a31478fc8194a18073c7a86b06583111171b4",
                              progressbar=True)
    idx_file = pooch.retrieve(vcf_url+".tbi",
                              fname=os.path.basename(vcf_url)+".tbi",
                              known_hash="90fd8754c61bc0442c86e295d4d6b7fdbac3ffbb6273d4ead8690e10a2682abf",
                              progressbar=True)
    return {"vcf": vcf_file, "idx": idx_file}

def vcf_to_df(
    vcf_file=None,
    attrs=["CHROM", "POS", "ID", "REF", "ALT"],
    filter=None,  # e.g., lambda v: "UTR" in v.INFO.get("MC", "")
    contig=None,
    info=INFO_COLS_SELECT,
    extract_ids=True,
    progress=True,
    cache=os.path.join(pooch.os_cache("pooch"), "clinvar.parquet"),
    force=False
):
    """
    Convert a ClinVar VCF file to a Polars DataFrame with selected annotations.

    This function reads a ClinVar VCF file, extracts specified attributes and INFO fields,
    processes key annotation columns, and returns a Polars DataFrame. It supports caching
    to avoid repeated parsing and can extract additional ID columns if requested.

    Parameters
    ----------
    vcf_file : str or None, optional
        Path to the ClinVar VCF file. If None, downloads the latest ClinVar VCF.
    attrs : list of str, default=["CHROM", "POS", "ID", "REF", "ALT"]
        List of VCF record attributes to extract as columns.
    filter : callable or None, optional
        Optional filter function to select VCF records (e.g., lambda v: ...).
    contig : str or None, optional
        If specified, only extract records from this contig/chromosome.
    info : list of str, default=INFO_COLS_SELECT
        List of INFO field keys to extract from the VCF.
    extract_ids : bool, default=True
        Whether to extract additional ID columns using _extract_id_cols.
    progress : bool, default=True
        Whether to display a progress bar during parsing.
    cache : str, default=os.path.join(pooch.os_cache("pooch"),"clinvar.parquet")
        Path to cache the resulting DataFrame as a parquet file.
    force : bool, default=False
        If True, force re-parsing even if cache exists.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame containing the extracted VCF records and annotations.

    Notes
    -----
    - The function will cache the parsed DataFrame to disk for faster future access.
    - The "MC" (Molecular Consequence) field is split into "MC_id" and "MC_term".
    - The "CLNREVSTAT" field is mapped to a numeric "CLNREVSTAT_score" for review status.
    - If `extract_ids` is True, additional ID columns are extracted using _extract_id_cols.
    """
    from genoray import VCF

    if vcf_file is None:
        vcf_file = download_vcf()["vcf"]

    if os.path.exists(cache) and not force:
        print(f"Reading from {cache}")
        return pl.read_parquet(cache)

    vcf = VCF(vcf_file, filter=filter)

    # Extract record attributes and INFO fields
    vcf_df = vcf.get_record_info(
        contig=contig,
        attrs=attrs,
        info=info,
        progress=progress
    )

    # Parse Molecular Consequence (MC) into SO ID and term
    vcf_df = vcf_df.with_columns(
        vcf_df["MC"].str.split("|").list.first().alias("MC_id"),
        vcf_df["MC"].str.split("|").list.last().alias("MC_term")
    )

    # Map review status to numeric score
    review_score_map = {
        "practice_guideline": 4,
        "reviewed_by_expert_panel": 3,
        "criteria_provided,_multiple_submitters,_no_conflicts": 2,
        "criteria_provided,_conflicting_classifications": 1,
        "criteria_provided,_single_submitter": 1,
        "no_assertion_criteria_provided": 0,
        "no_classification_provided": 0,
        "no_classification_for_the_single_variant": 0,
        "no_classifications_from_unflagged_records": 0
    }
    vcf_df = vcf_df.with_columns(
        pl.col("CLNREVSTAT").replace(review_score_map).alias("CLNREVSTAT_score").cast(pl.Int8)
    )

    # Optionally extract additional ID columns
    if extract_ids:
        vcf_df = _extract_id_cols(vcf_df)

    print(f"Caching to {cache}")
    vcf_df.write_parquet(cache)

    return vcf_df


def simplify_annotations(bed,
                         maps=None,
                         verbose=True):
    """
    Simplify the annotations in the ClinVar DataFrame.

    Args:
        bed (pl.DataFrame): The ClinVar DataFrame.
        maps (list): A list of dictionaries, each containing the input column, output column, and map.
            e.g. [{"input_col": "CLNSIG", "output_col": "CLNSIG_simple", 
                    "map": {"Benign":"benign", 
                            "Likely_benign":"likely_benign", 
                            "Benign/Likely_benign":"likely_benign", 
                            "Pathogenic/Likely_pathogenic":"likely_path",
                              "Pathogenic":"path", 
                              "Likely_pathogenic":"likely_path",
                              ...
                              }]
        verbose (bool): Whether to print the number of genes in the filtered DataFrame.
    Returns:
        pl.DataFrame: The simplified ClinVar DataFrame.
    """
    if maps is None:
        if verbose:
            print("Using default maps.")
        maps = [ 
                {"input_col": "CLNSIG",
                "output_col": "CLNSIG_simple",
                "map": {
                    'Benign':"benign",
                    'Likely_benign':"likely_benign",
                    'Benign/Likely_benign':"likely_benign",
                    'Pathogenic/Likely_pathogenic':"likely_path",
                    'Pathogenic':"path",
                    'Likely_pathogenic':"likely_path",
                    'Pathogenic/Likely_pathogenic/Pathogenic,_low_penetrance':"likely_path",
                    'Benign|other':"benign",
                    'Benign|confers_sensitivity':"benign",
                    'Benign|Affects|association|other': "benign",
                    'confers_sensitivity': "other",
                    'no_classification_for_the_single_variant': "other",
                    'Likely_pathogenic|other': "likely_path",
                    'Pathogenic/Likely_risk_allele': "likely_path",
                    'Benign|other': "benign",
                    'Benign': "benign",
                    'Benign/Likely_benign|risk_factor': "likely_benign",
                    'Conflicting_classifications_of_pathogenicity|drug_response': "conflicting",
                    'Pathogenic|association': "path",
                    'Uncertain_significance': "other",
                    'Pathogenic|confers_sensitivity': "path",
                    'Likely_benign|other': "likely_benign",
                    'Affects': "other",
                    'Likely_risk_allele': "likely_path",
                    'Pathogenic|association|protective': "path",
                    'Pathogenic|drug_response': "path",
                    'protective|risk_factor': "other",
                    'Pathogenic|risk_factor': "path",
                    'protective': "other",
                    'Conflicting_classifications_of_pathogenicity|drug_response|other': "conflicting",
                    'Benign/Likely_benign|drug_response|other': "likely_benign",
                    'Conflicting_classifications_of_pathogenicity|Affects': "conflicting",
                    'Likely_pathogenic|Affects': "likely_path",
                    'Benign/Likely_benign': "likely_benign",
                    'Likely_benign|drug_response': "likely_benign",
                    'Likely_benign|drug_response|other': "likely_benign",
                    'Conflicting_classifications_of_pathogenicity': "conflicting",
                    'Likely_benign': "likely_benign",
                    'Benign/Likely_benign|other|risk_factor': "likely_benign",
                    'association|risk_factor': "other",
                    'Uncertain_significance|association': "other",
                    'Benign/Likely_benign|drug_response': "likely_benign",
                    'Conflicting_classifications_of_pathogenicity|other|risk_factor': "conflicting",
                    'Benign/Likely_benign|association': "likely_benign",
                    'Likely_benign|Affects|association': "likely_benign",
                    'Pathogenic|other': "path",
                    'Uncertain_risk_allele': "other",
                    'Benign|confers_sensitivity': "benign",
                    'Benign|association': "benign",
                    'Likely_pathogenic|association': "likely_path",
                    'not_provided': "other",
                    'Pathogenic|Affects': "path",
                    'Pathogenic/Likely_pathogenic|risk_factor': "likely_path",
                    'drug_response': "other",
                    'Conflicting_classifications_of_pathogenicity|protective': "conflicting",
                    'Likely_pathogenic/Likely_risk_allele': "likely_path",
                    'Benign|Affects': "benign",
                    'confers_sensitivity|other': "other",
                    'association_not_found': "other",
                    'other': "other",
                    # None: "other",
                    'Pathogenic/Likely_pathogenic|other': "likely_path",
                    'Benign|risk_factor': "benign",
                    'Likely_pathogenic|risk_factor': "likely_path",
                    'Pathogenic/Likely_pathogenic/Pathogenic,_low_penetrance': "likely_path",
                    'Uncertain_risk_allele|risk_factor': "other",
                    'Likely_benign|risk_factor': "likely_benign",
                    'Uncertain_significance|drug_response': "other",
                    'association|drug_response|risk_factor': "other",
                    'Conflicting_classifications_of_pathogenicity|association|risk_factor': "conflicting",
                    'Likely_pathogenic,_low_penetrance': "likely_path",
                    'risk_factor': "other",
                    'association': "other",
                    'no_classifications_from_unflagged_records': "other",
                    'Uncertain_significance|Affects': "other",
                    'Uncertain_significance|other': "other",
                    'Pathogenic|protective': "conflicting",
                    'Uncertain_significance/Uncertain_risk_allele': "other",
                    'Uncertain_significance|risk_factor': "other",
                    'Conflicting_classifications_of_pathogenicity|other': "conflicting",
                    'Conflicting_classifications_of_pathogenicity|association': "conflicting",
                    'Pathogenic/Likely_pathogenic/Pathogenic,_low_penetrance|risk_factor': "likely_path",
                    'Pathogenic/Likely_pathogenic/Likely_risk_allele': "likely_path",
                    'Benign/Likely_benign|other': "likely_benign",
                    'Benign|protective': "benign",
                    'Benign|drug_response': "benign",
                    'Pathogenic/Pathogenic,_low_penetrance|other|risk_factor': "path",
                    'Likely_benign|association': "likely_benign",
                    'Pathogenic/Likely_pathogenic': "likely_path",
                    'drug_response|other': "other",
                    'Conflicting_classifications_of_pathogenicity|risk_factor': "conflicting",
                    'drug_response|risk_factor': "other",
                    'Pathogenic/Pathogenic,_low_penetrance|other': "path",
                    'Likely_pathogenic': "likely_path",
                    'other|risk_factor': "other",
                    'Pathogenic': "path",
                    'Likely_pathogenic|drug_response': "likely_path",
                    'Pathogenic/Likely_pathogenic|association': "likely_path",
                    'Likely_pathogenic|protective': "likely_path"
                    }},
            
                {"input_col": "CLNSIG_simple",
                "output_col": "CLNSIG_super_simple",
                "map": {
                    'benign':"benign",
                    'likely_benign':"benign",
                    'pathogenic':"path",
                    'likely_pathogenic':"path",
                    'conflicting':"conflicting",
                    'other':"other"
                    }},
        ]

    was_pandas = False
    if isinstance(bed, pd.DataFrame):
        bed = pl.DataFrame(bed)
        was_pandas = True
    
    if verbose:
        print("Simplifying annotations.")
    for map in maps:
        if map["input_col"] in bed.columns and map["output_col"] not in bed.columns:
            bed = bed.with_columns(pl.col(map["input_col"]).replace_strict(map["map"], default=pl.lit("other")).alias(map["output_col"]))
     
    if "GENEINFO" in bed.columns and "GENE" not in bed.columns:
        bed = bed.with_columns(pl.col("GENEINFO").str.split(":").list.first().alias("GENE"))

    if was_pandas:
        bed = bed.to_pandas()
    return bed 

def _explode_col(df, col):
    if df[col].dtype.__str__()=='List(String)':
        df = df.explode(col)
    return df

def df_to_bed(vcf_df,  
              save_path=None,
              extract_ids=True,
              variant_name_alias="name",
              extra_cols=[],
              simplify=True):
    """Convert VCF DataFrame to BED format.
    
    Args:
        vcf_df: Polars DataFrame in VCF format
        save_path: Optional path to save BED file
        
    Returns:
        DataFrame in BED format
    """

    vcf_df = _explode_col(vcf_df, "ALT")

    bed = vcf_df.rename({
        'CHROM': 'chrom',
        'POS': 'chromStart',
    }).with_columns([
        # pl.lit(None).alias('strand'),
        (pl.col('chromStart') + pl.col('REF').str.len_chars()).alias('chromEnd')
    ])
    if "CLNREVSTAT_score" in vcf_df.columns:
        bed = bed.with_columns(pl.col("CLNREVSTAT_score").alias("score"))
    
    # Add variant name using the function instead of method
    if variant_name_alias is not None:
        bed = utils.add_variant_name(bed, alias=variant_name_alias)
    
    select_cols = [
        # required columns
        'chrom',
        'chromStart',
        'chromEnd',
        # optional columns
        variant_name_alias,
        "score",
        # "strand",
        # extra columns
        "REF",
        "ALT",
        'MC_id',
        'MC_term',
        *INFO_COLS_SELECT,
        "CLNREVSTAT_score"
        
        # "MC", "ORIGIN", "RS"
    ] + extra_cols
    select_cols = [col for col in select_cols if col in bed.columns]
    bed = bed.select(select_cols).filter(pl.col('chrom').str.contains('^[0-9]+$|^X$|^Y$'))

    bed = bed.with_columns(pl.col('ALT').fill_null("")).drop_nulls(subset=['ALT'])
    bed = bed.with_columns(pl.col('REF').fill_null("")).drop_nulls(subset=['REF'])

    if extract_ids:
        bed = _extract_id_cols(bed) 

    if simplify:
        bed = simplify_annotations(bed)

    if save_path:
        bed.to_pandas().to_csv(save_path, sep='\t', index=False)
    return bed


def df_to_sites(vcf_df):

    sites = vcf_df.with_columns([
        # pl.lit(None).alias('strand'),
        (pl.col('POS') + pl.col('REF').str.len_chars()).alias('POS_END'),
        (pl.col('ALT').list.join(',').alias('ALT')),
    ])

    # Add variant name using the function instead of method
    sites = utils.add_variant_name(sites,
                                    chrom_col='CHROM',
                                    start_col='POS',
                                    end_col='POS_END')
    
    sites = sites.select([
        # required columns
        'CHROM',
        'POS',
        'POS_END',
        # optional columns
        "name",
        # extra columns
        "REF",
        "ALT",
        'MC_id',
        'MC_term',
        *INFO_COLS_SELECT,
        "CLNREVSTAT_score"
        # "MC", "ORIGIN", "RS"
    ]).filter(pl.col('CHROM').str.contains('^[0-9]+$|^X$|^Y$'))

    sites = sites.with_columns(pl.col('ALT').fill_null("")).drop_nulls(subset=['ALT'])
    sites = sites.with_columns(pl.col('REF').fill_null("")).drop_nulls(subset=['REF'])

    return sites


def bed_to_sites(bed):
    """Convert BED DataFrame to sites format.
    
    Args:
        bed_df: Polars DataFrame in BED format
        save_path: Optional path to save sites file

    Returns:
        DataFrame in sites format
    """
    sites =  bed.rename({
        'chrom': 'CHROM',
        'chromStart': 'POS',
        'chromEnd': 'POS_END'
    })

    sites = sites.select([
        'CHROM',
        'POS',
        'REF',
        'ALT',
        *[col for col in sites.columns if col not in ['CHROM', 'POS', 'REF', 'ALT']]
    ])
    return sites

def _extract_id_cols(df,
                     search_terms=["MONDO", "OMIM", "Orphanet", "MedGen", "MeSH"],
                     add_counts=True,
                     verbose=True):
    if verbose:
        print("Extracting ID columns.")
    for id_type in search_terms:
        if id_type not in df.columns:
            df = df.with_columns(
                pl.col("CLNDISDB").str.extract_all(f'({id_type}:[^,|]+)').alias(id_type)
            )
        if add_counts:
            if f"{id_type}_n" not in df.columns:
                df = df.with_columns(
                    pl.col(id_type).list.len().alias(f"{id_type}_n")
                )
    
    return df


def read_bed(path, 
             schema_overrides=None,
             separator='\t',
             simplify=True,
             extract_ids=True,
             as_pandas=False,
             **kwargs):
    """
    Read a BED file created by the clinvar submodule for use with GenVarloader.

    Args:
        path (str): The path to the BED file.
        schema_overrides (dict): A dictionary of column names and their data types.
        separator (str): The separator used in the BED file.
        simplify (bool): Whether to simplify the annotations.
        extract_ids (bool): Whether to extract the ID columns.
        **kwargs: Additional arguments to pass to the pl.read_csv function.

    Returns:
        pl.DataFrame: The BED file as a Polars DataFrame.
    
    Example:
        >>> bed = cv.read_bed("data/UTR/clinvar_utr_snv.bed.gz", simplify=True)
        >>> bed
    """

    if schema_overrides is None:
        schema_overrides = {
            'chrom': pl.Utf8,
            'chromStart': pl.Int64,
            'chromEnd': pl.Int64,
            'score': pl.Float64
        }
    # Import bed file
    bed = pl.read_csv(
        path,
        schema_overrides=schema_overrides,
        separator=separator,
        **kwargs
    ).drop_nulls(subset=['ALT'])

    bed = bed.with_columns(
        pl.col("name").alias("site")
    )

    if extract_ids:
        bed = _extract_id_cols(bed)

    if simplify:
        bed = simplify_annotations(bed)

    if as_pandas:
        bed = bed.to_pandas()

    return bed

def filter_df(vcf_df,
              filters = {},
              verbose=True):
    """
    Filter the VCF DataFrame based on the provided filters.
    
    Args:
        vcf_df (pl.DataFrame): The input VCF DataFrame.
        filters (dict): A dictionary of filters to apply to the DataFrame.
        verbose (bool): Whether to print the number of genes in the filtered DataFrame.
    Returns:
        pl.DataFrame: The filtered VCF DataFrame.

    Example:
        >>> vcf_df = vcf_to_df()
        >>> filters = {
                "MC_term": ["5_prime_UTR_variant","3_prime_UTR_variant"],
                "CLNREVSTAT_score": 2,
                "CLNVC": "single_nucleotide_variant",
                "CLNSIG": ["benign","pathogenic","Benign","Pathogenic"]
            }
        >>> filter_df(vcf_df, filters)
    """

    # Build filter conditions dynamically based on provided filters
    filter_conditions = []
    for key, value in filters.items():
        if key not in vcf_df.columns:
            if verbose:
                print(f"Column {key} not found in DataFrame. Skipping filter.")
            continue
        if isinstance(value, list):
            filter_conditions.append(pl.col(key).str.contains("|".join(value)))
        else:
            filter_conditions.append(pl.col(key) >= value if key == "CLNREVSTAT_score" else pl.col(key) == value)

    # Combine all conditions with &
    combined_condition = filter_conditions[0]
    for condition in filter_conditions[1:]:
        combined_condition = combined_condition & condition

    cv_df = (vcf_df
        .filter(combined_condition)
        # Convert POS column to integer
        .with_columns(pl.col("POS").cast(pl.Int64))
        .drop_nulls(subset=['CLNDN'])
    )

    if verbose:
        print(f"Filtered DataFrame shape: {cv_df.shape}")
        print(f"Variant count: {cv_df.shape[0]}")
        print(f"Gene count: {cv_df['GENEINFO'].unique().len()}")

    return cv_df


def count_sites_per_gene(vcf_df=None,
                         groupby_cols=["MONDO","GENEINFO"],
                         sort=True):
    """
    Count the number of sites per gene in the VCF DataFrame.
    """
    if vcf_df is None:
        vcf_df = vcf_to_df()

    cv_df = utils.add_variant_name(df_to_bed(vcf_df))

    vpd = cv_df.explode("MONDO").group_by(groupby_cols).agg(pl.col("name").n_unique()).sort("MONDO").to_pandas().rename(columns={"name":"sites"})
    vpd = vpd.loc[vpd["MONDO"].notnull()]
    vpd.loc[:,"MONDO"]  = vpd["MONDO"].str.replace("MONDO:MONDO:","MONDO:")
    if sort:
        vpd.sort_values("sites", ascending=False)

    return vpd
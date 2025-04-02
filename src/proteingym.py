import os
import pooch
import pandas as pd
from tqdm import tqdm
import glob

import src.utils as utils
import src.ensembl_rest as er

##### Usage example #####
## import sys
## sys.path.append('ProteinGym') # Make functions available
##
## import proteingym.utils.download as pgd # Import functions
## pgd.set_cache("my/local/dir/") # Set cache directory
## print(pgs.PROTEINGYM_CACHE) # Get cache directory
## resources_df = pgd.get_resources_df() # Get resources DataFrame
## pg_resources = pgd.download_resources(resources_df.iloc[:2]) # Download resources
## pgd.count_resources(pg_resources) # Count resources

# ProteinGym version
PROTEINGYM_VERSION = 'v1.1'

# Cache directory for ProteinGym data
PROTEINGYM_CACHE = pooch.os_cache('ProteinGym')

def set_cache(cache):
    """Set the cache directory for ProteinGym data.
    
    Args:
        cache (str): Path to the directory where ProteinGym data will be cached.
            
    Example:
        >>> # Set custom cache directory
        >>> set_cache("my/local/dir/")
    """
    global PROTEINGYM_CACHE
    PROTEINGYM_CACHE = cache

def get_resources_df(version = PROTEINGYM_VERSION,
                     cache = PROTEINGYM_CACHE,
                     force = False):
    """Get DataFrame containing ProteinGym resource metadata and download URLs.
    
    Args:
        version (str): Version of ProteinGym resources to use. Defaults to 'v1.1'.
        cache (str): Path to save the resources DataFrame. Defaults to cache directory.
        force (bool): Whether to force regeneration of DataFrame even if file exists.
            Defaults to False.
            
    Returns:
        pd.DataFrame: DataFrame containing resource metadata including filenames, sizes,
            download URLs and SHA256 hashes.

    Example:
        >>> # Get resources DataFrame
        >>> resources_df = get_resources_df()
        >>> resources_df.head()
    """
    # Create full save path
    save_path = os.path.join(cache, 'ProteinGym_data_urls.tsv')

    # Copied from here: https://github.com/OATML-Markslab/ProteinGym?tab=readme-ov-file#resources
    file_text = """Data	Size (unzipped)	Filename	Raw	Hash
    DMS benchmark - Substitutions	1.0GB	DMS_ProteinGym_substitutions.zip	False	3a83766254ac9ac9984ec25cb73c6e010ea4418f5e35f143933e6b6e6473b921
    DMS benchmark - Indels	200MB	DMS_ProteinGym_indels.zip	False	5c5c7446a8c8f89534dfa87e546d2f9c00590d19aa5ce4c01d271abc7c962f74
    Zero-shot DMS Model scores - Substitutions	31GB	zero_shot_substitutions_scores.zip	False	22df5c0f47e8278b39d0c1a51518e20d674b5109e136578bbede660af2bd7ecd
    Zero-shot DMS Model scores - Indels	5.2GB	zero_shot_indels_scores.zip	False	957dc5d0d3e4163f56b3d45b865150a44fcd8ea9e2cf172e9c3fbbac2e344d81
    Supervised DMS Model performance - Substitutions	2.7MB	DMS_supervised_substitutions_scores.zip	False	8167ff7eee01e748a7820034940847f888532cb2c942bc9ae18e413f77bce2cb
    Supervised DMS Model performance - Indels	0.9MB	DMS_supervised_indels_scores.zip	False	3cf375bc9ae80b878e6c55ddeade2ef5f2895d479e4d414872d205007351bf15
    Multiple Sequence Alignments (MSAs) for DMS assays	5.2GB	DMS_msa_files.zip	False	f8c894f0f113f5f49f2945c512b73f488bdf582097dff04658fbb703d92fe34d
    Redundancy-based sequence weights for DMS assays	200MB	DMS_msa_weights.zip	False	2f36a2a7882b264142eca273255da659fc8640249234edf934ffef364a585084
    Predicted 3D structures from inverse-folding models	84MB	ProteinGym_AF2_structures.zip	False	c78f5ff60cf59104fe19b8318c5647587aad033ee832e051d0efec8e137c423a
    Clinical benchmark - Substitutions	123MB	clinical_ProteinGym_substitutions.zip	False	afe711af49365bc1ee220a5d212c570a4d9bc35e6960d19a93a0d1ed4ce37be4
    Clinical benchmark - Indels	2.8MB	clinical_ProteinGym_indels.zip	False	644192ef474998346ff760c3b3d6d0d731aebf79ce3c5057e3f2748c687128d6
    Clinical MSAs	17.8GB	clinical_msa_files.zip	False	9f55b0792419f0f7f0d64f39f5345bb1510db5e02fb7a85347db3b0d2f8b3531
    Clinical MSA weights	250MB	clinical_msa_weights.zip	False	564bbef2a6f22e544fc88ea49a31f1d1e585ad663e17d4d1e5f78f06a412fa49
    Clinical Model scores - Substitutions	0.9GB	zero_shot_clinical_substitutions_scores.zip	False	8bd9bbfe2a686974072f28c10cb1e0418f37c44a1fddf6e6b820f06b5f4b6515
    Clinical Model scores - Indels	0.7GB	zero_shot_clinical_indels_scores.zip	False	1834dfe2a43e34529eea77c1dbe7b0503153578455b7b146856b31268ee17aa7
    CV folds - Substitutions - Singles	50M	cv_folds_singles_substitutions.zip	False	920f0be936233b96b5052cd23679e42355cfd2b4e6f45b4f571eb79c0b2f9c35
    CV folds - Substitutions - Multiples	81M	cv_folds_multiples_substitutions.zip	False	4f1453ee8ccf2d38f23ae43f97fc7f962e54e5f10390711b59f6929538dd25f9
    CV folds - Indels	19MB	cv_folds_indels.zip	False	b3f123321b499b470da03ddd3530241502851152f9a98775ecd6b508ae9c856d
    DMS benchmark: Substitutions (raw)	500MB	substitutions_raw_DMS.zip	True	6d83b16585de2b71b67ae1985193b9eec2e01804784286c515ff276b5372e412
    DMS benchmark: Indels (raw)	450MB	indels_raw_DMS.zip	True	93c21d4cdc09755428e417e330fdf7b3bf16705f125b23df208648b3ca5595a0
    Clinical benchmark: Substitutions (raw)	58MB	substitutions_raw_clinical.zip	True	caa461bd2e0c58501131e7c1ad9d26c118c67704efe1b67c7ff7ca1d72ae7275
    Clinical benchmark: Indels (raw)	12.4MB	indels_raw_clinical.zip	True	f9eb7232657ab5732eda8dcb922bf17b228eae212ca794e753ba73a017f40a8d
    """
    if not os.path.exists(save_path) or force:
        from io import StringIO
        df = pd.read_csv(StringIO(file_text), sep='\t') 
        df['URL'] = df['Filename'].apply(lambda x: f"https://marks.hms.harvard.edu/proteingym/ProteinGym_{version}/{x}")
        df['Version'] = version
        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, sep='\t')
    else:
        df = pd.read_csv(save_path, sep='\t')

    # Add an extra row with some share raw data
    new_row = pd.DataFrame({
        'Data': 'dbNSFP/Ensembl VEP Annotations (raw)',
        'Size (unzipped)': '981.5MB',
        'Filename': 'all_models_deduplicated_scores_clinvar_proteingym_20230611.GRCh38_filter_isoform_clean.csv',
        'Raw': True,
        'Hash': 'c7570c0c5c6c92a2ae2646aec9bd990a62fbc21f512d5acc81d9027fb9c338bf',
        'URL': 'https://marks.hms.harvard.edu/proteingym/other/all_models_deduplicated_scores_clinvar_proteingym_20230611.GRCh38_filter_isoform_clean.csv',
        'Version': version
    }, index=[len(df)])
    df = pd.concat([df, new_row], ignore_index=True)
    # Sort by Data column
    return df

def download_resources(resources_df = None,
                       cache = PROTEINGYM_CACHE,
                       include_raw = False, 
                       remove_zip = False, 
                       redecompress = False, 
                       error = True,
                       progressbar = 1,
                       verbose = False):
    """Download ProteinGym resources and optionally extract them.
    
    Args:
        resources_df (pd.DataFrame): DataFrame containing resource metadata from get_resources_df().
            Defaults to downloading all resources by calling get_resources_df().
            To download specific resources, subset the resources_df first and feed it into this function.
        cache (str): Directory to save downloaded files. Defaults to cache directory.
        include_raw (bool): Whether to include raw data files. Defaults to False.
        remove_zip (bool): Whether to remove zip files after extraction. Defaults to False.
        redecompress (bool): Whether to re-extract zip files. Defaults to False.
        error (bool): Whether to raise errors on download failures. Defaults to True.
        progressbar (int): Level of progress bar detail (0=none, 1=overall, 2=per-file).
            Defaults to 1.
            
    Returns:
        dict: Dictionary mapping resource names to lists of extracted file paths.
        
    Example:
        >>> # Download resources
        >>> pg_resources = download_resources()
        >>> # Count resources
        >>> count_resources(pg_resources)
    """
    if resources_df is None:
        resources_df = get_resources_df()
    if not include_raw:
        resources_df = resources_df[~resources_df['Raw']]
        
    def _rm_zip(row, fpath, verbose=False):
        zipped_name = os.path.join(fpath, os.path.basename(row['URL'])+".zip")
        if os.path.exists(zipped_name):
            if verbose:
                print(f"Removing compressed file: {zipped_name}")
            os.remove(zipped_name)

    file_dict = {}
    for i, row in tqdm(resources_df.iterrows(), 
                       total=len(resources_df), 
                       desc='Gathering resources', 
                       disable=progressbar<1): 
        try:
            unzipped_name = os.path.basename(row['URL']).removesuffix('.zip')
            if os.path.basename(row['URL']).endswith('.zip'):
                processor = pooch.Unzip(extract_dir=unzipped_name)
            else:
                processor = None
            if row['Filename'].endswith('.zip'):
                if verbose:
                    print("Skipping re-extraction of zip files")
                if os.path.exists(unzipped_name) and not redecompress:
                    file_dict[unzipped_name] = glob.glob(os.path.join(unzipped_name, '**/*'), recursive=True)
                    print(f"Skipping {unzipped_name} because it already exists")
                    if remove_zip:
                        _rm_zip(row, cache, verbose=verbose)
                    continue
            file_dict[unzipped_name] = pooch.retrieve(url=row['URL'], 
                                                        fname=os.path.basename(row['URL']),
                                                        known_hash=None if pd.isna(row['Hash']) else row['Hash'], 
                                                        path=cache,
                                                        progressbar=progressbar>1,
                                                        processor=processor) 
            if remove_zip:
                _rm_zip(row, cache, verbose=verbose)
        except Exception as e:
            if error:
                raise e
            else:
                print(f"Error downloading {row['Filename']}: {e}")
    return file_dict

def merge_csvs(pg_resources, 
               key,
               max_files=None):
    """Concatenate CSV files from a ProteinGym resource into a single DataFrame.
    
    Args:
        pg_resources (dict): Dictionary of resources from download_resources().
        fkey (str): Key of resource to concatenate. Defaults to clinical substitutions.
        max_files (int, optional): Maximum number of files to read. Defaults to None (all files).
        
    Returns:
        pd.DataFrame: Concatenated DataFrame with source file column added.
        
    Example:
        >>> # Download resources
        >>> pg_resources = download_resources()
        >>> # Concatenate all clinical substitution files
        >>> df = merge_csvs(pg_resources, key='clinical_ProteinGym_substitutions')
        >>> # Or concatenate first 5 files only
        >>> df_subset = merge_csvs(pg_resources, key='clinical_ProteinGym_substitutions', max_files=5)
    """
    clinical_subs_dfs = []
    for csv_path in tqdm(pg_resources[key][:max_files],
                        desc="Reading files in " + key):
        df = pd.read_csv(csv_path, index_col=0)
        # Add source file name as column
        df['source_file'] = os.path.basename(csv_path)
        clinical_subs_dfs.append(df)
    return pd.concat(clinical_subs_dfs, ignore_index=True)


def count_resources(pg_resources):
    """Count the number of resources in a ProteinGym resource dictionary.
    
    Args:
        pg_resources (dict): Dictionary of resources from download_resources().
        
    Returns:
        int: Number of resources in the dictionary.

    Example:
        >>> # Download resources
        >>> pg_resources = download_resources()
        >>> # Count resources
        >>> count_resources(pg_resources)
    """
    return {k:f"{len(v)} file(s)" for k,v in pg_resources.items()}

def merge_resources(keys=['clinical_ProteinGym_substitutions.zip', 
                          'clinical_ProteinGym_indels.zip'],
                    redecompress = False,
                    map_ids = True,
                    rows_per_id = None,
                    force = False,
                    verbose = False): 

    resources_df = get_resources_df()
    pg_resources = download_resources(
        resources_df.loc[resources_df['Filename'].isin(keys)],
        remove_zip = False, 
        redecompress = redecompress,
        error=False)
    proteins_df = pd.DataFrame(
    list(set([
        f
        for key in pg_resources.keys()
        for f in pg_resources[key]
    ])),
        columns=['source_file']
    )
    proteins_df['source_type'] = proteins_df['source_file'].apply(lambda x: os.path.basename(os.path.dirname(x)))
    proteins_df.insert(0, 'protein', proteins_df['source_file'].str.split(os.sep).str[-1].str.replace('.csv', '', regex=False))
    proteins_df.insert(1, 'RefSeq peptide ID', proteins_df['protein'].str.split('.').str[0]) 
    proteins_df['experiment_id'] = proteins_df['source_type'] + '/' + proteins_df['protein']

    # Map RefSeq IDs to equivalent Ensembl/RefSeq IDs
    ## WARNING: Massive expands the number of rows in df due to many:many mappings
    if map_ids:
        # Using gprofiler
        proteins_df = map_resources(df=proteins_df, 
                                    force=force, 
                                    verbose=verbose,
                                    rows_per_id=rows_per_id)

    return proteins_df

def map_resources(df, 
                  input_col='protein',
                  select_cols = ['protein','genename',
                                 'Ensembl_geneid','Ensembl_transcriptid','Ensembl_proteinid',
                                #  'Feature',
                                #  'Uniprot_acc','Uniprot_entry','Uniprot_acc(HGNC/Uniprot)','Uniprot_id(HGNC/Uniprot)'
                                 ],
                  target_namespace=['ENSP','ENST','REFSEQ_PEPTIDE'],
                  method=['proteingym','ensembl_rest','gprofiler'],
                  rows_per_id = None,
                  force = False,
                  verbose = True): 
    """
    Map ProteinGym resources to target namespaces.
    
    Args:
        df (pd.DataFrame): DataFrame containing gene/protein/transcript IDs.
        input_col (str): Column containing gene/protein/transcript IDs to be mapped.
        target_namespace (list): List of target namespaces to map to. Only used if method is 'gprofiler'.
        method (str): Method to use for mapping. Either 'gprofiler' or 'ensembl_rest'.
        force (bool): Whether to force mapping even if files exist.

    Returns:
        pd.DataFrame: DataFrame with mapped IDs.
        
    Example:
        >>> # Download resources
        >>> pg_resources = download_resources()
        >>> # Merge resources
        >>> proteins_df = merge_resources()
        >>> # Map IDs
        >>> proteins_df = map_resources(proteins_df)
    """
    df = df.copy()
    method = utils.one_only(method)
    input_col = utils.one_only(input_col)
    
    if method == 'gprofiler':
        # Using gprofiler
        for tn in target_namespace:
            import src.gprofiler as gp
            df = gp.map_ids(df,
                            on_left=input_col,
                            target_namespace=tn, 
                            rows_per_id=rows_per_id,
                            force=force, 
                            verbose=verbose)
            
    elif method == 'ensembl_rest':

        # Using Ensembl REST API
        map_dict = er.xref_external(ids=df[input_col], 
                                    force=force, 
                                    verbose=verbose)
        col_key = {'ENSG':'gene','ENSP':'translation','ENST':'transcript'}
        for col, key in col_key.items():
            map_k_v = {k:v[key] for k,v in map_dict.items()}
            df[col] = df[input_col].map(map_k_v)

    elif method == 'proteingym':

        # Using Proteingym
        df = map_proteingym_ids(df,
                                select_cols=select_cols,
                                rows_per_id=rows_per_id,
                                force=force, 
                                verbose=verbose)
        rename_cols = {'Ensembl_geneid':'ENSG',
                       'Ensembl_proteinid':'ENSP',
                       'Ensembl_transcriptid':'ENST'}
        # drop existing columns that are being renamed
        df = df.drop(columns=rename_cols.values(), errors='ignore')
        df.rename(columns=rename_cols, inplace=True)
    else:
        raise ValueError(f"Invalid method: {method}") 
    
    # Return the selected columns
    return df


def map_proteingym_ids(df=None, 
                        input_col='protein',
                        select_cols = ['protein','Feature','genename','Gene',
                                        'Ensembl_geneid','Ensembl_transcriptid','Ensembl_proteinid',
                                        'Uniprot_acc','Uniprot_entry','Uniprot_acc(HGNC/Uniprot)','Uniprot_id(HGNC/Uniprot)',
                                        'Entrez_gene_id','CCDS_id','Refseq_id','ucsc_id'
                                        ],
                        return_map = False,
                        rows_per_id = None,
                        cache = PROTEINGYM_CACHE,
                        force = False,
                        verbose = True):
    """
    Map ProteinGym resources to target namespaces.

    Args:
        df (pd.DataFrame): DataFrame containing gene/protein/transcript IDs.
        input_col (str): Column containing gene/protein/transcript IDs to be mapped.
        select_cols (list): List of columns to select from the mapping file.
        return_map (bool): Whether to return the mapping file.
        force (bool): Whether to force mapping even if files exist.
        verbose (bool): Whether to print verbose output.

    Returns:
        pd.DataFrame: DataFrame with mapped IDs.
        
    Example:
        >>> # Download resources
        >>> pg_resources = download_resources()
        >>> # Merge resources
        >>> proteins_df = merge_resources()
    """


    # Check if cols already exist
    if not return_map:
        if all(col in df.columns for col in select_cols) and not force:
            if verbose:
                print(f"All {len(select_cols)} columns already exist in DataFrame.")
            return df

    if verbose:
        print("Mapping Proteingym IDs")
    # Set the save path
    checksum = utils.as_checksum(";".join(utils.process_ids(select_cols)))
    save_path = os.path.join(cache, f'proteingym_id_map_{checksum}.csv.gz')
    if os.path.exists(save_path):
        # Load the mapping file
        if verbose:
            print(f"Loading mapping file from {save_path}")
        pg_annot = pd.read_csv(save_path, index_col=0)
    else:
        # Download the mapping file
        resources_df = get_resources_df()
        resources_df = resources_df.loc[resources_df['Data'] == 'dbNSFP/Ensembl VEP Annotations (raw)']
        pg_resources = download_resources(resources_df = resources_df, include_raw=True) 
        pg_annot = pd.read_csv(pg_resources['all_models_deduplicated_scores_clinvar_proteingym_20230611.GRCh38_filter_isoform_clean.csv'], index_col=0)
        pg_annot = pg_annot[select_cols].drop_duplicates()

        if 'genename' in pg_annot.columns:
            pg_annot['genename'] = pg_annot['genename'].fillna('').str.split(';').apply(lambda x: ';'.join(set(x)))
            pg_annot = pg_annot[select_cols].drop_duplicates()
        
        # Save the mapping file
        if verbose:
            print(f"Saving mapping file to {save_path}") 
        pg_annot.to_csv(save_path)
    
    # If rows_per_id is not None, take the first rows_per_id rows for each input_col
    if rows_per_id is not None:
        if verbose:
            print(f"Taking the first {rows_per_id} rows for each {input_col}")
        pg_annot = pg_annot.groupby(input_col).head(rows_per_id)

    

    # Return the mapping file
    if return_map:
        return pg_annot
    
    # Merge the mapping file with the input DataFrame
    assert input_col in df.columns, f"Input column {input_col} not found in DataFrame"
    assert input_col in pg_annot.columns, f"Input column {input_col} not found in mapping file"
    assert set(select_cols).issubset(pg_annot.columns), f"Output columns {select_cols} not found in mapping file"
    df = df.merge(pg_annot, on=input_col, how='left')
    return df

from cProfile import label
import os
import glob
import pandas as pd
import numpy as np
import argparse
import pathlib 
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Literal
import pooch

# Local imports
import src.config as config
import src.utils as utils
import src.haplosaurus as hs
import src.proteingym as pg
import src.biopython as bp
import src.vep_pipeline as vp
import src.onekg as og

def get_models_palette(palette="husl"):
    models = vp.list_models()
    palette = sns.color_palette(palette=palette, 
                                n_colors=len(models))
    return dict(zip(models, palette))


def _get_default_save_dir(save_dir):
    if save_dir is None:
        save_dir = os.path.join(config.DATA_DIR,"1KG","vep")
        print(f"No save_dir provided, using: {save_dir}")
    return save_dir


def list_all_models(save_dir=None):
    save_dir = _get_default_save_dir(save_dir)
    return [os.path.basename(d) for d in glob.glob(os.path.join(save_dir, "*"))]

def list_vep_files(save_dir = None,
                   scoring_strategy: Literal["wt-marginals", "masked-marginals", "pseudo-ppl"] = ["*"],
                   model_location: Literal["esm2_t33_650M_UR50D", "esm2_t33_650M_UR50S"] = "*",
                   save_format = "parquet",
                   as_df=False,
                   verbose=True
                   ):
    """List VEP files in the specified directory.
    
    Args:
        save_dir (str, optional): Directory to search for VEP files. If None, uses default directory.
        scoring_strategy (list): List of scoring strategies to search for. Options are "wt-marginals", 
            "masked-marginals", or "pseudo-ppl". 
            By default, all files ending in ".parquet" are returned (i.g. scoring_strategy=["*"]).
        save_format (str, optional): File format to search for. Defaults to "parquet".
        as_df (bool, optional): If True, returns results as a DataFrame with additional metadata columns.
            If False, returns list of file paths. Defaults to False.
        verbose (bool, optional): If True, prints summary statistics. Defaults to True.
    
    Returns:
        Union[pd.DataFrame, List[str]]: Either a DataFrame containing file paths and metadata, or a list of file paths.
    """
    
    save_dir = _get_default_save_dir(save_dir)
    
    all_files = []
    for ss in tqdm(scoring_strategy, 
                   desc="Finding VEP files"):
         all_files.extend(glob.glob(
            os.path.join(save_dir,model_location, "**", f"{ss}.{save_format}"), 
                         recursive=True))
    if len(all_files) == 0:
        raise FileNotFoundError(f"No {save_format} files found in {save_dir}")
    if as_df:
        df = pd.DataFrame({'file': all_files})
        df['protein'] = df['file'].str.split(os.sep).str[-4]
        df['haplotype'] = df['file'].str.split(os.sep).str[-3]
        df['variant_set'] = df['file'].str.split(os.sep).str[-2]
        df['model_location'] = df['file'].str.split(os.sep).str[-5]
        df['scoring_strategy'] = df['file'].str.split(os.sep).str[-1].str.split('.').str[0]
        if verbose:
            print(f"Found {len(df)} {save_format} files in {save_dir}")
            print("Unique values:")
            print('>> model_location:', df['model_location'].nunique())
            print('scoring_strategy:', df['scoring_strategy'].nunique())
            print('variant_set:', df['variant_set'].nunique())
            print('haplotype:', df['haplotype'].nunique())
            print('protein:', df['protein'].nunique())
        return df
    else:
        if verbose:
            print(f"Found {len(all_files)} {save_format} files in {save_dir}")
        return all_files

def merge_vep(save_dir = None,
              haplotypes = None,
              vep_files = None,
              save_format = "parquet",
              scoring_strategy = ["wt-marginals", "masked-marginals", "pseudo-ppl"],
              add_model_location=True,
              rename_model_location_col=True,
              add_variant_set=True,
              add_filename=False,
              add_metadata=True,
              target_namespace=None,
              col_map= {'mutant': 'mutant', 
                        'protein': 'protein'},
              max_files=None,
              verbose=True
              ):
    """Merge VEP results from multiple files into a single dataframe.
    
    Args:
        save_dir (str, optional): Directory containing the VEP results. If None, uses default directory.
        haplotypes (list, optional): List of haplotypes to include. If None, includes all haplotypes.
        vep_files (list or pd.DataFrame, optional): List of VEP files or DataFrame with file paths to process.
        save_format (str, optional): Format of saved files. Defaults to "parquet".
        scoring_strategy (list, optional): List of scoring strategies to include. Defaults to ["wt-marginals", "masked-marginals", "pseudo-ppl"].
        add_model_location (bool, optional): Whether to add model location column. Defaults to True.
        rename_model_location_col (bool, optional): Whether to rename model location column. Defaults to True.
        add_variant_set (bool, optional): Whether to add variant set column. Defaults to True.
        add_filename (bool, optional): Whether to add filename column. Defaults to False.
        add_metadata (bool, optional): Whether to add metadata columns. Defaults to True.
        target_namespace (str, optional): Target namespace for column mapping. Defaults to None.
        col_map (dict, optional): Dictionary mapping input columns to output columns. Defaults to {'mutant': 'mutant', 'protein': 'protein'}.
        max_files (int, optional): Maximum number of files to process. Defaults to None.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        pd.DataFrame: Merged dataframe containing all VEP results with specified columns and metadata.
    """
    save_dir = _get_default_save_dir(save_dir)

    if rename_model_location_col and not add_model_location:
        print("`add_model_location` must be set to True if `rename_model_location_col` is True.\nSetting `add_model_location` to True.")
        add_model_location = True

    save_dir = os.path.expanduser(save_dir)
    if isinstance(scoring_strategy, str):
        scoring_strategy = [scoring_strategy]

    scoring_strategy = [ss for ss in scoring_strategy if ss ]


    # Create empty list to store dataframes
    dfs = []
    for ss in scoring_strategy:
        
        # Find all csv.gz files recursively
        if vep_files is None:
            all_files = list_vep_files(save_dir=save_dir, 
                                        scoring_strategy=ss, 
                                        save_format=save_format)
        else:
            if isinstance(vep_files, pd.DataFrame):
                if vep_files.empty:
                    raise ValueError("`vep_files` is empty")
                else:
                    all_files = vep_files.loc[vep_files['scoring_strategy']==ss].file.unique().tolist()
            elif isinstance(vep_files, list):
                all_files = vep_files
            else:
                raise ValueError(f"Invalid type for `vep_files`: {type(vep_files)}")
        
        # Limit the number of files
        if max_files is not None:
            all_files = all_files[:max_files]
        
        # Print the number of files
        if verbose:
            print("Found", len(all_files), ss, save_format, "files") 
        
        # Skip if no files are found
        if len(all_files) == 0:
            continue
        # Read each file and append to list
        for filename in tqdm(all_files, desc="Reading files"):
            try: 
                # Read the CSV
                if save_format == "parquet":
                    df = pd.read_parquet(filename)
                else:
                    df = pd.read_csv(filename, index_col=[0,1])
                
                df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
                df.insert(0, 'haplotype', os.path.dirname(filename).split(os.sep)[-2])
                df['scoring_strategy'] = os.path.basename(filename).split('.')[0]  
                df['is_ref'] = df['haplotype'].str.endswith('REF')
                if add_model_location:
                    model_location = os.path.dirname(filename).split(os.sep)[-4]
                    df['model_location'] = model_location

                    # Rename model location column
                    if rename_model_location_col:
                        df.rename(columns={model_location: 'VEP'}, inplace=True)

                        # Hot fix for column naming bug
                        hotfix_cols = [col for col in df.columns if "('ProteinBertModel(" in col]
                        if len(hotfix_cols)>0:
                            df.rename(columns=dict(zip(hotfix_cols, ["VEP"]*len(hotfix_cols))), inplace=True)

                if add_variant_set:
                    df['variant_set'] = os.path.dirname(filename).split(os.sep)[-1]
                if add_filename:
                    df['filename'] = filename 
                dfs.append(df) 
            except Exception as e:
                if verbose:
                    print(f"Error reading {filename}: {str(e)}")
                continue
        
    # Concatenate all dataframes
    if dfs:
        if verbose:
            print("Concatenating VEP dataframes")
        vep_df = pd.concat(dfs, ignore_index=True)
        if 'ENSP' not in vep_df.columns:
            vep_df['ENSP'] = vep_df['haplotype'].str.split(":").str[0]
        
        # Map protein IDs to ENST and HGNC
        ## Via GProfiler
        if target_namespace is not None:
            import src.gprofiler as gp
            vep_df = gp.map_ids(vep_df, 
                                rows_per_id=1,
                                target_namespace=target_namespace)
        ## Via Haplosaurus
        else: 
            tx_id_map = hs.get_txid_map(haplotypes, invert=True)
            vep_df['ENST'] = vep_df['ENSP'].map(tx_id_map)
        
        # Check if reformatted mutant is in haplotype string
        vep_df['mutant_in_haplotype'] = vep_df.apply(lambda x: _reformat_mutant(x[col_map['mutant']]) in x['haplotype'], axis=1)

        # Add site column to get unique identifiers for each variant
        vep_df['site'] = vep_df['protein'] + ":" + vep_df['mutant'] 

        assert len(vep_df)>0, "No VEP data found"
        if add_metadata:
            if all(col in vep_df.columns for col in [col_map['protein'], col_map['mutant']]):
                # Gather additional PGD metadata
                resources_df = pg.get_resources_df()
                pgd_resources = pg.download_resources(resources_df.loc[resources_df['Filename'].isin(['substitutions_raw_clinical.zip', 'indels_raw_clinical.zip'])], 
                                                    include_raw=True)
                assert len(pgd_resources)>0, "No PGD metadata found"

                pgd_subs_raw = pd.read_csv(pgd_resources['substitutions_raw_clinical'][0], low_memory=False)
                # Annotate with PGD variants
                vep_df = vep_df.merge(pgd_subs_raw.groupby([col_map['protein'], 
                                                            col_map['mutant']]
                                                            ).head(1), # Drop duplicate entries for the same protein-mutant pair
                                      on=[col_map['protein'], col_map['mutant']], 
                                      how='left')
                assert len(vep_df)>0, "No PGD metadata found"
            else:
                print("Cannot `add_metadata`: No 'protein' or 'mutant' columns found in VEP dataframe.")
        
        if verbose:
            print(f"VEP dataframe shape: {vep_df.shape}")
        return vep_df
    else:
        print("No files were successfully read")
        return None 
    


def _reformat_mutant(mutant):
    """
    Reformat a mutant string to the format "poswt>mt"
    Example:
       _reformat_mutant("S223I") in "ENSP00000261556:223S>I,565S>N"
       # True
    """
    wt, pos, mt = mutant[0], int(mutant[1:-1]), mutant[-1]
    return f"{pos}{wt}>{mt}" 

def report_vep(vep_df,
               haplotype_col='haplotype',
               clinsig_col='clinsig'):
    """
    Report on the VEP dataframe
    """
    
    # Add extra columns
    vep_df['protein_sequence_length'] = vep_df['protein_sequence'].map(bp.preprocess_sequence).str.len()
    vep_df['mutated_sequence_length'] = vep_df['mutated_sequence'].map(bp.preprocess_sequence).str.len()

    clinsig_counts = vep_df['clinsig'].value_counts().to_frame(name='clinsig_count')
    seq_check_df = pd.merge(vep_df.groupby(clinsig_col).apply(lambda x: sum(x['protein_sequence_length'] != x['mutated_sequence_length'])).to_frame(name='mismatched_length').reset_index(),
                            vep_df.groupby(clinsig_col).apply(lambda x: sum(x['protein_sequence'] == x['mutated_sequence'])).to_frame(name='identical_sequences').reset_index(),
                            on=clinsig_col)

    # Get variant counts
    mutant_in_haplotype = vep_df.groupby(['mutant_in_haplotype',clinsig_col])['mutant'].nunique().to_frame(name='variant_count').reset_index()
    mutant_in_haplotype = mutant_in_haplotype.merge(mutant_in_haplotype.groupby(clinsig_col).sum().rename(columns={'variant_count':f'variant_count_by_{clinsig_col}'}).reset_index().drop(columns=['mutant_in_haplotype']),
            on=clinsig_col,
            how='left').fillna(0) 
    mutant_in_haplotype['variant_proportion'] = mutant_in_haplotype['variant_count'] / mutant_in_haplotype[f'variant_count_by_{clinsig_col}']
    # Get haplotype counts
    haplotype_counts = vep_df.groupby(['mutant_in_haplotype',clinsig_col])[haplotype_col].nunique().to_frame(name='haplotype_count').reset_index()
    haplotype_counts = haplotype_counts.merge(haplotype_counts.groupby(clinsig_col).sum().rename(columns={'haplotype_count':f'haplotype_count_by_{clinsig_col}'}).reset_index().drop(columns=['mutant_in_haplotype']),
            on=clinsig_col,
            how='left').fillna(0)
    haplotype_counts['haplotype_proportion'] = haplotype_counts['haplotype_count'] / haplotype_counts[f'haplotype_count_by_{clinsig_col}']
    # Merge variant and haplotype counts
    mutant_in_haplotype = mutant_in_haplotype.merge(haplotype_counts, 
                                                    on=['mutant_in_haplotype',clinsig_col], 
                                                    how='left').fillna(0)
    return seq_check_df, clinsig_counts, mutant_in_haplotype




def get_seq_names(seq_reps):
    return [seq_tuple[0] for seq_tuple in seq_reps]

def get_distances(seq_reps,
                  method="manhattan",
                  fill_diagonal=True):
    # Compute pairwise cosine distances between all sequence representations
    if method=="cosine":
        from sklearn.metrics.pairwise import cosine_distances as dist_func
    elif method=="manhattan":
        from sklearn.metrics.pairwise import manhattan_distances as dist_func
    
    # Extract just the tensors from the tuples and convert to numpy arrays
    seq_tensors = [seq_tuple[1].numpy() for seq_tuple in seq_reps]
    # Stack into 2D array
    seq_reps_array = np.vstack(seq_tensors) 
    # Compute distances
    distances = dist_func(seq_reps_array)  
    # Fill the diagonal with NAs
    if fill_diagonal:
        np.fill_diagonal(distances, np.nan) 
    return distances 

def get_paired_distances(seq_reps,  
                         group1, 
                         group2, 
                         distances=None,
                         split="_", 
                         tx_id_sep=":",
                         by_edits=False,
                         as_df=True,
                         verbose=True
                        ):
    """
    Extract distances between two groups of sequences for matching transcript prefixes.
    
    Args:
        seq_names: List of sequence names
        distances: Distance matrix between all sequences
        group1_idx: List of indices for first group
        group2_idx: List of indices for second group
        
    Returns:
        Subset of distance matrix containing only distances between matching transcripts
    """
    import numpy as np

    def _get_paired_distances_i(seq_reps,
                                distances,
                                group1, 
                                group2,
                                split="_",
                                tx_id_sep=":",
                                as_df=True):
        
        seq_names = get_seq_names(seq_reps)
        group1_idx = [i for i,x in enumerate(seq_names) if x.find(group1)>0]
        group2_idx = [i for i,x in enumerate(seq_names) if x.find(group2)>0]
        # Get transcript prefixes for each group
        group1_prefixes = [seq_names[i].split(split)[0] for i in group1_idx]
        group2_prefixes = [seq_names[i].split(split)[0] for i in group2_idx]

        # Get indices where prefixes match
        matching_group1_idx = [i for i,x in enumerate(group1_idx) 
                            if seq_names[group1_idx[i]].split(split)[0] == group2_prefixes[i]]
        matching_group2_idx = [i for i,x in enumerate(group2_idx)
                            if seq_names[group2_idx[i]].split(split)[0] == group1_prefixes[i]]

        idx_final = ([group1_idx[i] for i in matching_group1_idx],
                     [group2_idx[i] for i in matching_group2_idx])
        # Extract distance matrix subset
        distances_subset = distances[np.ix_(*idx_final)] 
        if as_df:
            df = pd.DataFrame(
                {'sample1':[seq_names[i] for i in idx_final[0]],
                'sample2':[seq_names[i] for i in idx_final[1]],
                'comparison':f"{group1.strip('_')}_vs_{group2.strip('_')}",
                'distance':np.diag(distances_subset)}
                )
            # Add a column for the haplotype
            df['haplotype'] = df['sample1'].str.split('_').str[0]
            # Add a column for the sample type
            df['group1'] = df['sample1'].str.split('_').str[1]
            df['group2'] = df['sample2'].str.split('_').str[1]
            # Variants
            df['variants'] = df['group2'].str.split(':').str[1]
            # Add a column for the protein_id
            df['protein_id'] = df['sample1'].str.split(':').str[0]
            # Add a column for the number of edits
            df['sample1_edits'] = utils.count_edits(df['sample1'], 
                                                    tx_id_sep=tx_id_sep)
            df['sample2_edits'] = utils.count_edits(df['sample2'], 
                                                    tx_id_sep=tx_id_sep)
            df['max_edits'] = df[['sample1_edits', 'sample2_edits']].max(axis=1)
            return df
        else:
            return distances_subset
    
    if by_edits:
        edits = utils.count_variants(get_seq_names(seq_reps), 
                               tx_id_sep=tx_id_sep)
        distances_subset_dict = {}
        seq_reps_ref = [seq_reps[i] for i,x in enumerate(edits) if x==0]
        # Reference sequence is 0 edits, always want to include it to compare to
        for n_edits in set(edits) - set([0]):
            if verbose:
                print(f"Processing {n_edits} edits") 
            seq_reps_i = [seq_reps[i] for i,x in enumerate(edits) if x==n_edits]
            if len(seq_reps_i)==0:
                continue
            else:
                seq_reps_i = utils.as_list(seq_reps_ref) + seq_reps_i
            if verbose:
                print(f"Found {len(seq_reps_i)} sequences")
            distances_i = get_distances(seq_reps_i)
            distances_subset_dict[n_edits] = _get_paired_distances_i(
                seq_reps=seq_reps_i,
                distances=distances_i,
                group1=group1, 
                group2=group2,
                split=split,
                as_df=as_df
                ) 
            if as_df:
                distances_subset_dict[n_edits]['edits_group'] = n_edits

        if as_df:
            return pd.concat([distances_subset_dict[i] for i in distances_subset_dict.keys()], axis=0)
        else:
            return distances_subset_dict
    else:
        # Group by edit distance too
        if distances is None:
            distances = get_distances(seq_reps)
        return _get_paired_distances_i(seq_reps=seq_reps,
                                       distances=distances,
                                       group1=group1, 
                                       group2=group2,
                                       split=split,
                                       as_df=as_df
                                       )             
    
def plot_paired_distances(dist_df,
                          x='max_edits',
                          y='distance',
                          hue='comparison',
                          **kwargs):

    # Create a seaborn violin plot of the distances
    from scipy import stats
    # Perform t-test between WT vs Benign and WT vs Pathogenic distances
    wt_benign = dist_df[dist_df['comparison']=='WT_vs_Benign']['distance']
    wt_pathogenic = dist_df[dist_df['comparison']=='WT_vs_Pathogenic']['distance']
    t_stat, p_value = stats.ttest_ind(wt_benign, wt_pathogenic)

    # Create violin plot
    sns.violinplot(data=dist_df,
                   x=x,
                   y=y,
                   hue=hue,
                   linewidth=.5,  # Remove outline
                   saturation=1.0,  # Full color saturation
                   **kwargs)
    plt.title(f't-test p-value: {p_value:.2e}')
    plt.show()

def batches_to_df(batches):
    df = pd.DataFrame()
    for tx_id, batch in tqdm(batches.items()):
        # add a new row to df for each tx_id
        wt_idx = [i for i, x in enumerate(batch) if x[0].endswith("_WT")]
        pathogenic_idx = [i for i, x in enumerate(batch) if "_Pathogenic" in x[0]]
        benign_idx = [i for i, x in enumerate(batch) if "_Benign" in x[0]]
        
        for i in wt_idx:
            wt_seq = batch[wt_idx[i]][1]
            wt_name = batch[wt_idx[i]][0]
            base_name = wt_name.split("_")[0]
            pathogenic_seq = batch[pathogenic_idx[i]][1]
            pathogenic_name = batch[pathogenic_idx[i]][0]
            benign_seq = batch[benign_idx[i]][1]
            benign_name = batch[benign_idx[i]][0]
            assert pathogenic_name.split("_")[0] == base_name
            assert benign_name.split("_")[0] == base_name
            row = pd.DataFrame({
                'tx_id': tx_id,
                'name': base_name,
                'WT_seq': wt_seq,
                'Pathogenic_seq': pathogenic_seq,
                'Pathogenic_variants': pathogenic_name.split("_")[1].split(":")[1],
                'Benign_seq': benign_seq,
                'Benign_variants': benign_name.split("_")[1].split(":")[1]
            }, 
            index=[i]
            )
        df = pd.concat([df, row], ignore_index=True, axis=0)
    return df



def plot_vep_violin(vep_df, 
                    model_location = None, 
                    max_models = 1,
                    scoring_strategy = None,
                    clinsig_col='clinsig',  
                    max_proteins = None,             
                    bar_width = 0.5,
                    violin_alpha = 0.25,
                    point_size = 3,
                    point_alpha = 0.1,
                    palette = utils.get_clinsig_palette(),
                    add_connections = False, 
                    connection_alpha = 0.3,
                    connection_linewidth = 1,
                    title_y = 1,
                    freq_filters = {'freq_1000GENOMES:phase_3:ALL':None},
                    figsize=[6, 8],
                    add_side_labels = False,
                    add_stats = False,
                    save_path = None,
                    verbose=True
                    ):
    from scipy import stats  

    vep_df = vep_df.copy()
    y_col = 'VEP'

    # Filter by model location
    vep_df = _filter_vep_df(vep_df, verbose=verbose)        
    # Filter by model location
    if model_location is not None:
        model_location = utils.as_list(model_location)
    else:
        model_location = vep_df['model_location'].unique()

    # Filter by max models
    if max_models is not None:
        model_location = model_location[:max_models]
    vep_df = vep_df.loc[vep_df['model_location'].isin(model_location)]

    # Filter by scoring strategy
    if scoring_strategy is not None:
        scoring_strategy = utils.as_list(scoring_strategy)
        vep_df = vep_df[vep_df['scoring_strategy'].isin(scoring_strategy)]
    # Sort by scoring strategy
    vep_df = utils.sort_by_reverse_string(vep_df, 'scoring_strategy')

    # Filter by max proteins
    if max_proteins is not None:
        vep_df = vep_df.loc[vep_df['protein'].isin(vep_df['protein'].unique()[:max_proteins])]
    
    # Get unique categories
    categories = sorted(vep_df[clinsig_col].unique())
    category_positions = {cat: i for i, cat in enumerate(categories)}
    
    # Create figure with subplots for each protein
    proteins = vep_df['protein'].unique()
    figsize[1] = figsize[1]*len(proteins)
    fig, axes = plt.subplots(len(proteins), 1, figsize=figsize)
    if len(proteins) == 1:
        axes = [axes]

    for ax, protein in zip(axes, proteins):
        # Get data for this protein
        protein_data = vep_df[vep_df['protein'] == protein].sort_values(by=clinsig_col)
        protein_data = hs.filter_haplotype_freqs(protein_data, freq_filters)
        
        # Create violin plots for each mutant, grouped by clinsig category
        for mutant in protein_data[~protein_data['is_ref']]['mutant'].unique():
            mutant_data = protein_data[~protein_data['is_ref'] & (protein_data['mutant'] == mutant)]
            
            # Plot violin for each category
            for category in categories:
                cat_data = mutant_data[mutant_data[clinsig_col] == category]
                if not cat_data.empty:
                    sns.violinplot(data=cat_data,
                                x=clinsig_col, y=y_col,
                                ax=ax, color=palette[category], alpha=violin_alpha)
        
        # Add individual points
        sns.stripplot(data=protein_data[~protein_data['is_ref']], 
                    x=clinsig_col, y=y_col,
                    ax=ax, color='black', alpha=point_alpha, size=point_size)
        
        # Plot horizontal lines for ref samples by category
        ref = protein_data[protein_data['is_ref']]
        
        if not ref.empty:
            # Store y-positions for label placement
            y_positions = {cat: [] for cat in categories}
            
            for category in categories:
                ref_cat = ref[ref[clinsig_col] == category]
                if not ref_cat.empty:
                    for _, ref_row in ref_cat.iterrows():
                        ref_score = ref_row[y_col]
                        ref_mutation = ref_row['mutant']
                        
                        # Only compare against scores for the same mutation and category
                        cat_scores = protein_data[
                            ~protein_data['is_ref'] & 
                            (protein_data[clinsig_col] == category) &
                            (protein_data['mutant'] == ref_mutation)
                        ][y_col]
                        
                        percentile = int(100 * (cat_scores < ref_score).mean())
                        
                        # Draw horizontal reference line
                        ax.hlines(y=ref_score, 
                                xmin=category_positions[category] - bar_width, 
                                xmax=category_positions[category] + bar_width,
                                color=palette[category], linestyle='--')
                        
                        y_positions[category].append((ref_score, f'{category} {ref_mutation}\n(Percentile={percentile})'))
            
            if add_side_labels:
                # Improved label positioning with dynamic spacing
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                
                def sort_positions(positions):
                    return sorted(positions, key=lambda x: x[0])
                
                def adjust_positions(positions, y_range):
                    if not positions:
                        return []
                    positions = sort_positions(positions)
                    min_gap = y_range * 0.1  # Dynamic gap based on plot range
                    adjusted = [positions[0]]
                    for i in range(1, len(positions)):
                        prev_y = adjusted[-1][0]
                        curr_y = positions[i][0]
                        if curr_y - prev_y < min_gap:
                            curr_y = prev_y + min_gap
                        adjusted.append((curr_y, positions[i][1]))
                    return adjusted
                
                # Add labels with connecting lines for each category
                for category in categories:
                    adjusted_positions = adjust_positions(y_positions[category], y_range)
                    original_positions = sort_positions(y_positions[category])
                    
                    if category_positions[category] == 0:  # First category
                        x_text = -1
                        ha = 'right'
                        x_start, x_end = -1, category_positions[category] - bar_width
                    else:  # Last category
                        x_text = len(categories)
                        ha = 'left'
                        x_start, x_end = category_positions[category] + bar_width, len(categories)
                    
                    for i, (y_pos, label) in enumerate(adjusted_positions):
                        ax.plot([x_start, x_end], [original_positions[i][0], y_pos],
                            color=palette[category], linestyle=':', alpha=0.6)
                        ax.text(x_text, y_pos, label, ha=ha, va='center',
                            color=palette[category])
        
        # Add connecting lines between haplotypes
        if add_connections:
            non_ref_data = protein_data[~protein_data['is_ref']]
            haplotypes = non_ref_data['haplotype'].unique()
            
            for hap in haplotypes:
                hap_data = non_ref_data[non_ref_data['haplotype'] == hap]
                if len(hap_data) > 1:  # Only connect if we have multiple variants
                    # Connect all pairs of categories
                    for i, cat1 in enumerate(categories[:-1]):
                        for cat2 in categories[i+1:]:
                            scores1 = hap_data[hap_data[clinsig_col] == cat1][y_col].values
                            scores2 = hap_data[hap_data[clinsig_col] == cat2][y_col].values
                            
                            for score1 in scores1:
                                for score2 in scores2:
                                    ax.plot([i, i+1], [score1, score2],
                                        color='gray', 
                                        alpha=connection_alpha, 
                                        linestyle='--', 
                                        linewidth=connection_linewidth)
        
        # Add statistical test
        if add_stats:
            # Perform one-way ANOVA if more than 2 categories, t-test if exactly 2
            category_vals = [protein_data[~protein_data['is_ref'] & (protein_data[clinsig_col] == cat)][y_col] 
                            for cat in categories]
            
            if len(categories) == 2:
                stat, pval = stats.ttest_ind(*category_vals)
            else:
                stat, pval = stats.f_oneway(*category_vals)
            
            y_max = max(val.max() for val in category_vals)
            y_min = min(val.min() for val in category_vals)
            y_range = y_max - y_min
            y_bracket = y_max + 0.03 * y_range
            
            if pval < 0.001:
                pval_text = 'p < 0.001'
            else:
                pval_text = f'p = {pval:.3f}'
                
            ax.text(len(categories)/2 - 0.5, y_bracket+0.5*y_range*0.03, pval_text, ha='center', va='bottom')
            
            # Draw significance bracket
            ax.plot([0, len(categories)-1], [y_bracket, y_bracket], 'k:', linewidth=1, alpha=0.8)
            ax.plot([0, 0], [y_bracket-30, y_bracket], 'k:', linewidth=1, alpha=0.8)
            ax.plot([len(categories)-1, len(categories)-1], [y_bracket-30, y_bracket], 'k:', linewidth=1, alpha=0.8)
        
        # Get summary of unique mutants per clinical significance
        mutant_summary_str = _summarise_mutants(vep_df, clinsig_col)
        title_label1 = _summarise_title(vep_df, label_cols=['model_location'])
        title_label2 = _summarise_title(vep_df)  
        
        ax.set_title(f'{title_label1}\n{title_label2}\n{mutant_summary_str}', 
                     y=title_y)
        ax.set_ylabel(f'{", ".join(vep_df["scoring_strategy"].unique())}')
        ax.set_xlabel('Clinical classification')
        
        # Adjust plot margins
        if add_side_labels:
            ax.set_xlim(-2, len(categories)+1)
            ax.set_ylim(y_min - 0.05*y_range, y_max + 0.15*y_range)

    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path)
        
    plt.show()

def _summarise_mutants(vep_df,
                       clinsig_col='clinsig'):
    
    vep_df['protein_mutant'] = vep_df['protein'] + '_' + vep_df['mutant']
    mutant_summary = vep_df.groupby(clinsig_col)['protein_mutant'].nunique() 
    mutant_summary_str = ', '.join([f"{k}: {v}" for k,v in mutant_summary.items()]) 
    return mutant_summary_str

def _summarise_title(vep_df,
                     label_cols = ['protein', 'ENST', 'HGNC', 'haplotype']):
    
    labels = {}
    for col in label_cols:
        if col not in vep_df.columns:
            label_cols.remove(col)
            continue
        if vep_df[col].nunique() > 1:
            labels[col] = f"{col}: {vep_df[col].nunique()}"
        else:
            labels[col] = f"{col}: {vep_df[col].tolist()[0]}"
    return ', '.join([labels[col] for col in label_cols if col in labels])

def _get_model_location(vep_df):
    if 'model_location' in vep_df.columns:
        return vep_df['model_location'].unique()[0]
    else:
        return None
    
def _filter_vep_df(vep_df,
                   verbose=True):
    """
    Filter the VEP dataframe to ensure that the model location is not None
    """
    rows_before = len(vep_df) 
    vep_df = vep_df.loc[vep_df['VEP'].notna()]
    rows_after = len(vep_df)
    if verbose and rows_before != rows_after:
        print(f"Filtered {((rows_before-rows_after)/rows_before)*100:.1f}% of rows with NAs in col 'VEP'")
    return vep_df

def add_legend(g, 
                include=None,
                palette=utils.get_clinsig_palette(),
                 loc='upper center',
                 bbox_to_anchor=(0.5, 1.05),
                 top=0.9,
                 ncol=None):
    
    if include is not None:
        labels = [label for label in palette if label in include]
    else:
        labels = list(palette.keys())
    handles = [plt.Rectangle((0,0),1,1, color=palette[label]) for label in labels]
    if ncol is None:    
        ncol = len(labels)
    g.figure.legend(handles, labels, 
                 loc=loc,
                 bbox_to_anchor=bbox_to_anchor,
                 ncol=ncol) 
    g.figure.subplots_adjust(top=top)  # Adjust spacing for titles 

def plot_vep_density(vep_df, 
                     clinsig_col = 'clinsig',
                     palette = utils.get_clinsig_palette(),
                     plot_func = sns.kdeplot,
                     model_location = None,
                     alpha=.8,
                     height=3,
                     aspect=1.5,
                     title_y=1,
                     legend_y=0.9, 
                     col='scoring_strategy',
                     row='model_location',
                     sharex=True,
                     sharey=False,
                     flip_xaxis=False,
                     verbose=True,
                     save_path=None,
                     facetgrid_kwargs={},
                     **kwargs): 
    # Get filtered data
    vep_df = vep_df.copy()

    # Filter by model location
    if model_location is not None:
        model_location = utils.as_list(model_location)
        vep_df = vep_df.loc[vep_df['model_location'].isin(model_location)]
    else:
        model_location = _get_model_location(vep_df)

    # Sort by scoring strategy
    vep_df = vep_df.sort_values(by='model_location')
    vep_df = utils.sort_by_reverse_string(vep_df, 
                                          column='scoring_strategy', 
                                          extra_sort_cols=[clinsig_col],
                                          ascending=[True, True])

    vep_df = _filter_vep_df(vep_df, verbose=verbose) 
    # Create facet grid
    g = sns.FacetGrid(data=vep_df, 
                    col=col,
                    row=row,
                    height=height,
                    aspect=aspect,
                    sharex=sharex,
                    sharey=sharey, 
                    margin_titles=True,
                    **facetgrid_kwargs)

    # Plot KDE
    g.map_dataframe(plot_func, 
                    x='VEP',
                    hue=clinsig_col,
                    fill=True,
                    palette=palette, 
                    alpha=alpha,
                    legend=True,
                    **kwargs)
    
    # Get summary of unique mutants per clinical significance
    mutant_summary_str = _summarise_mutants(vep_df, clinsig_col)
    final_label = _summarise_title(vep_df) 
    # Add title with protein, haplotype and mutant counts 
    g.figure.suptitle(f'{final_label}\n{mutant_summary_str}', y=title_y)

    # Add legend 
    add_legend(g, palette=palette,  top=legend_y, include=vep_df[clinsig_col].unique(),
                   loc='lower center') 
    rm_subplot_prefixes(g)

    if flip_xaxis:
        for ax in g.axes.flat:
            ax.invert_xaxis()

    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, **utils.FIG_SAVE_KWARGS)
        
    plt.show()
    # Add vertical lines for REF haplotypes
    # for ax in g.axes.flat:
    #     ref_data = esm_vep[(esm_vep['protein'] == np_id) & 
    #                        (esm_vep['is_ref'] == True)]
    #     for _, row in ref_data.iterrows():
    #         ax.axvline(x=row['esm1v_t33_650M_UR90S_1'], 
    #                   color='black', 
    #                   linestyle='--', 
    #                   alpha=0.5)
    return {'fig': g.fig, 'axes': g.axes, 'data': vep_df}

def rm_subplot_prefixes(g):
    g.set_titles(row_template='{row_name}', 
                 col_template='{col_name}')  # Only show model name without prefix
    

def compute_representativeness_stats(
    vep_df,
    groupby_cols=['model_location', 'protein', 'clinsig', 'mutant', 'scoring_strategy'],
    y='VEP_percentile',
    is_ref=True
):
    """
    Compute statistics summarizing how representative the reference VEP is of each variant-specific VEP distribution.

    This function calculates the percentile rank, mean, and standard deviation of the 'VEP' column
    within groups defined by `groupby_cols`. If the specified percentile column (`y`) does not exist,
    it will be created. The function also ensures that 'VEP_mean' and 'VEP_std' columns are present,
    computing them if necessary.

    Parameters
    ----------
    vep_df : pd.DataFrame
        Input DataFrame containing at least a 'VEP' column and columns specified in `groupby_cols`.
    groupby_cols : list of str, optional
        Columns to group by when computing statistics (default:
        ['model_location','protein','clinsig','mutant','scoring_strategy']).
    y : str, optional
        Name of the percentile column to compute if not present (default: 'VEP_percentile').
    is_ref : bool, optional
        Whether to return only the rows that correspond to the reference haplotypes (default: True).

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns for percentile rank (`y`), 'VEP_mean', and 'VEP_std'.
    """
    print(f"Computing representativeness stats for {y}")
    # Compute percentile rank if not present
    if y not in vep_df.columns:
        vep_df[y] = vep_df.groupby(groupby_cols, observed=True)['VEP'].rank(pct=True) * 100
    # Compute standard deviation if not present
    if 'VEP_std' not in vep_df.columns:
        vep_df['VEP_std'] = vep_df.groupby(groupby_cols, observed=True)['VEP'].transform('std')
    # Compute mean if not present
    if 'VEP_mean' not in vep_df.columns:
        vep_df['VEP_mean'] = vep_df.groupby(groupby_cols, observed=True)['VEP'].transform('mean')

    if "VEP_mean_diff" not in vep_df.columns:
        vep_df["VEP_mean_diff"] = vep_df["VEP"] - vep_df["VEP_mean"]

    if "VEP_ref_diff" not in vep_df.columns:
        # Avoid InvalidIndexError by merging instead of using .map with a non-unique index
        ref_vep = vep_df.loc[vep_df["is_ref"], groupby_cols + ["VEP"]].rename(columns={"VEP": "VEP_REF"})
        vep_df = vep_df.merge(ref_vep, on=groupby_cols, how="left")
        vep_df["VEP_ref_diff"] = vep_df["VEP"] - vep_df["VEP_REF"]

     # Filter for REF haplotypes
    vep_df = vep_df.copy()
    if is_ref:
        pct_df = vep_df.loc[vep_df['is_ref']==True]
    else:
        pct_df = vep_df

    return pct_df


def plot_vep_ref_representativeness(vep_df, 
            groupby_cols=['model_location','protein','clinsig','mutant','scoring_strategy'],
            title=r"VEP$_{ref}$ Representativeness",
            figsize=(5, 5), 
            xlabel="Clinical Significance",
            ylabel=r"VEP$_{ref}$ Percentile",   
            clinsig_col="clinsig", 
            palette = utils.get_clinsig_palette(), 
            hline_x=50, 
            reverse_percentiles=False):
    
    if "is_ref" not in vep_df.columns:
        vep_df["is_ref"] = vep_df["sample"]=="REF"
    data = compute_representativeness_stats(vep_df, groupby_cols=groupby_cols)
    
    # Reverse percentiles to ensure larger VEP_percentile values are benign, smaller are pathogenic
    if reverse_percentiles:
        data["VEP_percentile"] = 100 - data["VEP_percentile"]
    # n_bins=10
    # vep_binned, bin_edges = pd.qcut(data['VEP_mean'], q=n_bins, labels=False, retbins=True, duplicates='drop')
    # data = data.copy()
    # data['VEP_binned'] = vep_binned
 
    data = utils.sort_by_clinsig(data, clinsig_col=clinsig_col)
    

    fig, ax = plt.subplots(figsize=figsize)
    g = sns.violinplot(data=data, x=clinsig_col, y="VEP_percentile", hue=clinsig_col,
                       cut=0,
                       palette=palette,
                       legend=False,
                       ax=ax)
    g.set_xlabel(xlabel)
    g.set_ylabel(ylabel)
    g.set_title(title)

    if hline_x is not None:
        g.axhline(hline_x, color='grey', linestyle='--', linewidth=1)

    return {"fig":g.figure, 'axes':g.axes , "data":data}



def _filter_palette(palette, labels):
    return {k:v for k,v in palette.items() if k in labels}


def compute_bpratio(b,p):
    return b/(b+p)

def plot_vep_bpratios(vep_df, 
                        suptitle='Benign/Pathogenic Ratios \n(Higher means greater relative difference\nbetween Benign vs. Pathogenic VEPs)',
                        groupby_cols =['model_location', 'haplotype', 'scoring_strategy','clinsig'],
                        x='model_location',
                        y=['benign_path_ratio','benign_path_ratio_strict'][0],
                        hue='model_location',
                        y_label='VEP_Benign /\nVEP_Benign + VEP_Pathogenic',
                        row="scoring_strategy",
                        func=sns.violinplot,
                        palette = None,
                        height=4,
                        aspect=.9,
                        sharex=True,
                        sharey=True,
                        return_df=True,
                        add_legend=True,
                        **kwargs):

    y = utils.one_only(y) 
    
    if palette is None:
        palette = get_models_palette()

    # Get filtered data
    vep_pbratios = vep_df.groupby(groupby_cols)['VEP'].mean().reset_index().pivot(
            index=['model_location', 'haplotype', 'scoring_strategy'],
            columns='clinsig',
            values='VEP'
        ).reset_index()
 
    vep_pbratios['benign_path_ratio_strict'] = compute_bpratio(b=vep_pbratios['benign'],
                                                               p=vep_pbratios['path'])
    vep_pbratios['benign_path_ratio'] = compute_bpratio(b=vep_pbratios[['benign','likely_benign']].mean(axis=1), 
                                                        p=vep_pbratios[['path','likely_path']].mean(axis=1))
    
 
    # Sort by scoring strategy
    vep_pbratios = utils.sort_by_reverse_string(vep_pbratios, 
                                            column='scoring_strategy', 
                                            extra_sort_cols=['model_location'],
                                            ascending=[False, True])

    g = sns.FacetGrid(data=vep_pbratios, row=row, height=height, aspect=aspect, sharex=sharex, sharey=sharey)
    g.map_dataframe(func, x=x, y=y, hue=hue, palette=palette, **kwargs)
    # if log_scale:
    #     g.set(yscale="log")

    # Rotate x-axis labels for better readability
    g.set_xticklabels(rotation=45, ha='right')

    # Remove subplot titles and add margin titles
    g.figure.suptitle(suptitle)  # Remove overall title if any
    g.set_ylabels(y_label)

    rm_subplot_prefixes(g)

    if add_legend:
        labels = vep_pbratios[x].unique()
        palette = _filter_palette(palette, labels)
        add_legend(g, palette=palette, ncol=1, loc='upper right', bbox_to_anchor=(1.7, 1))

    plt.tight_layout()
    plt.show()

    if return_df:
        return vep_pbratios
    
def add_haplotype_sequence(vep_df,
                           haplotypes: Dict[str, Dict] = None, 
                           encode_haplotype_name_threshold: int = 10,
                           add_missing_ref: bool = True,
                           add_consensus: bool = False,
                           force: bool = False,
                           verbose: bool = True):
    """
    Add a column to the dataframe containing the haplotype sequence.
    Args:
        vep_df (pd.DataFrame): The dataframe to add the column to.
        haplotypes (Dict[str, Dict]): The haplotypes to use.
        force (bool): Whether to force the addition of the column.

    Returns:
        pd.DataFrame: The dataframe with the new column.
    """
    
    if 'haplotype_sequence' not in vep_df.columns or force:
        if haplotypes is None:
            haplotypes = hs.get_haplotypes(verbose=verbose)
        
        if verbose:
            print("Getting haplotype sequences")

        hap_seqs_flattened = hs.get_haplotype_seqs(
            haplotypes=haplotypes,
            encode_haplotype_name_threshold=encode_haplotype_name_threshold,
            add_haplotype_names=3, 
            add_missing_ref=add_missing_ref,
            add_consensus=add_consensus
            ) 
        
        if verbose:
            print("Adding 'haplotype_sequence' column")
        vep_df["haplotype_sequence"] = vep_df["haplotype"].map(hap_seqs_flattened)
    if verbose:
        print("Adding 'haplotype_sequence_len' column")

    # Add reference sequence length column
    if 'reference_sequence_len' not in vep_df.columns and 'protein_sequence' in vep_df.columns:
        vep_df['reference_sequence_len'] = vep_df['protein_sequence'].apply(len)

    # Add haplotype sequence length column
    vep_df['haplotype_sequence_len'] = vep_df['haplotype_sequence'].apply(lambda x: len(bp.preprocess_sequence(x)) if pd.notna(x) else np.nan)
    vep_df['haplotype_sequence_len_pct'] = vep_df['haplotype_sequence_len'] / vep_df['reference_sequence_len']

    return vep_df

def add_mutant_out_of_frame(vep_df,
                             haplotypes: Dict[str, Dict] = None, 
                             protein_position_col: str = 'Protein_position',
                             encode_haplotype_name_threshold: int = 10,
                             force: bool = False,
                             verbose: bool = True):
    """
    Add a column to the dataframe indicating whether the mutant is out of frame.
    i.e. if the mutation position within the amino acid sequence is greater than the haplotype sequence length.

    Args:
        vep_df (pd.DataFrame): The dataframe to add the column to.
        haplotypes (Dict[str, Dict]): The haplotypes to use.
        protein_position_col (str): The column name of the protein position.
        force (bool): Whether to force the addition of the column.

    Returns:
        pd.DataFrame: The dataframe with the new column.
    """
    assert protein_position_col in vep_df.columns, f"Column {protein_position_col} not found in dataframe"
    
    if 'mutant_out_of_frame' not in vep_df.columns or force:
        if verbose:
            print("Adding 'haplotype_sequence' column")
        vep_df = add_haplotype_sequence(vep_df=vep_df, 
                                        haplotypes=haplotypes, 
                                        encode_haplotype_name_threshold=encode_haplotype_name_threshold,
                                        force=force,
                                        verbose=verbose)

    if verbose:
        print("Adding 'mutant_out_of_frame' column")
    vep_df['mutant_out_of_frame'] = vep_df[protein_position_col] > vep_df['haplotype_sequence_len']
    return vep_df

def encode_labels(vec, 
                  binarize: bool = False,
                  true_substring: str = "path",
                  verbose: bool = True):
    """
    Encode labels within a column.
    """
    if binarize:
        if verbose:
            print("Binarizing pathogenic vs benign")
        return np.array([true_substring in yy_.lower() for yy_ in vec]).astype(int)
    else:
        if verbose:
            print(f"Encoding {len(set(vec))} multiclass labels")
        from sklearn.preprocessing import LabelEncoder
        # Encode multiclass labels for clinical significance
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(vec)

def compute_precision_recall(vep_df,
                             groupby_cols: List[str] = ['model_location', 'scoring_strategy',
                                                        'protein',
                                                         'is_ref'],
                             binarize: bool = True,
                             agg_cols: List[str] = ['protein'],
                             x='VEP',
                             y='DMS_bin_score',
                             verbose: bool = True):
    """
    Compute precision and recall for a given scoring strategy and model location.
    """
    # Train logistic regression models to predict DMS binary scores
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score  


    groupby_cols = utils.as_list(groupby_cols)

    vep_df = utils.sort_by_reverse_string(vep_df, 
                                            column='scoring_strategy', 
                                            extra_sort_cols=['model_location'],
                                            ascending=[False, True])
    
    # Get unique combinations of model_location and scoring_strategy
    model_combos = vep_df.groupby(groupby_cols, observed=True).size().reset_index()[groupby_cols]

    if binarize:
        if verbose:
            print("Binarizing pathogenic vs benign labels")
    # Initialize results dictionary
    results = []
    # Train separate model for each combination
    for _, combo in tqdm(model_combos.iterrows(), 
                         total=len(model_combos),
                         desc="Computing precision and recall"):
        
        # Build filter conditions for all groupby columns
        filter_conditions = [(vep_df[col] == combo[col]) for col in groupby_cols]
        combined_filter = pd.concat(filter_conditions, axis=1).all(axis=1)
        
        # Filter data for this combination
        sub_df = vep_df.loc[combined_filter,:].copy()
        sub_df.dropna(subset=[x,y], inplace=True)
        
        # Skip if there are less than 2 samples or less than 2 unique values in x or y
        if len(sub_df) < 2 or sub_df[x].nunique() < 2 or sub_df[y].nunique() < 2:
            continue
            
        # Prepare X and y
        X = sub_df[x].values.reshape(-1, 1)
        y_ = sub_df[y].values

        # Binarize pathogenic vs benign
        y_ = encode_labels(y_, 
                           binarize=binarize,
                           verbose=verbose>1)

        # Train model
        clf = LogisticRegression(random_state=42)
        clf.fit(X, y_)
        
        # Get predictions
        y_pred = clf.predict(X)
        y_pred_proba = clf.predict_proba(X)[:,1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_, y_pred)
        auc = roc_auc_score(y_, y_pred_proba)
        
        # Create results dictionary with all groupby columns
        result_dict = {
            'accuracy': accuracy,
            'auc': auc,
            'coef': clf.coef_[0][0],
            'intercept': clf.intercept_[0],
            'n_samples': len(sub_df)
        }
        # Add groupby columns to results
        for col in groupby_cols:
            result_dict[col] = combo[col]
            
        results.append(result_dict)

    vep_pr = pd.DataFrame(results)
    
    # Aggregate results by agg column
    if agg_cols is not None:
        agg_cols = utils.as_list(agg_cols)
        remaining_cols = list(set(groupby_cols) - set(agg_cols))
        vep_pr = vep_pr.groupby(remaining_cols, observed=True).agg({'accuracy':'mean','auc': 'mean','coef':'mean','intercept':'mean','n_samples':'sum'}).reset_index()

    # Add back in extra cols
    if 'protein_sequence_len' not in vep_pr.columns and 'protein_sequence' in vep_pr.columns:
        seq_lens = {seq:len(seq) for seq in vep_pr['protein_sequence']}
        vep_pr['protein_sequence_len'] = vep_pr['protein_sequence'].map(seq_lens)

    if 'edits_scaled' not in vep_pr.columns and ('edits' in vep_pr.columns and 'protein_sequence_len' in vep_pr.columns):
        vep_pr['edits_scaled'] = vep_pr['edits'] / vep_pr['protein_sequence_len']
            
    if 'edits_clipped' not in vep_pr.columns and 'edits' in vep_pr.columns:
        vep_pr['edits_clipped'] = vep_pr['edits'].clip(upper=20)

    # Convert results to dataframe
    return vep_pr


def plot_precision_recall(vep_pr,
                          x='model_location',
                          y='auc',
                          style='is_ref',
                          hue='scoring_strategy',
                          size='n_samples',
                          row='scoring_strategy',   
                          alpha=0.7,
                          figsize=[8, 6],
                          markers=None,
                          plot_types=['scatter_AUCROC','bar_AUCROC','bar_coef'],
                          save_path=None,
                          **kwargs):
    """
    Plot precision and recall for a given scoring strategy and model location.

    Args:
        vep_pr (pd.DataFrame): The dataframe containing the precision and recall results.
        x (str): The column name of the x-axis.
        y (str): The column name of the y-axis.

    Returns:
        None
    
    Example:
    >>> vep_pr = compute_precision_recall(vep_df, x='VEP', y='DMS_bin_score')
    >>> plot_precision_recall(vep_pr, x='VEP', y='DMS_bin_score')
    """
    if markers is None and style is not None:
        markers = utils.get_marker_map(max(vep_pr[style]), 
                                       is_ref_marker=style == 'is_ref')
    
    if style is not None:
        if style not in vep_pr.columns:
            style = None

    vep_pr = utils.sort_by_reverse_string(vep_pr, 
                                        column='scoring_strategy', 
                                        extra_sort_cols=['model_location','is_ref'],
                                        ascending=[False, True, False])
    # Plot results
    if 'scatter_AUCROC' in plot_types:
        plt.figure(figsize=figsize)
        sns.scatterplot(data=vep_pr, x=x, y=y, size=size, 
                        hue=hue, 
                    alpha=alpha, 
                    style=style,
                    markers=markers,
                    **kwargs)
        plt.xticks(rotation=45, ha='right')
        plt.title('Model Performance by Location and Scoring Strategy')
        plt.ylabel('AUC-ROC')
        plt.xlabel('Model Location')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.splitext(save_path)[0] + "_scatter_AUCROC.png")
            
        plt.show()

    if 'bar_AUCROC' in plot_types:
        # Create faceted bar plots using FacetGrid
        g = sns.FacetGrid(data=vep_pr, 
                          row=row,
                          height=figsize[1], 
                          aspect=figsize[0]/figsize[1])
        g.map_dataframe(sns.barplot, 
                        x=x, 
                        y=y,
                        hue=style,
                        palette='dark:#1f77b4')
        
        # Rotate x-axis labels for better readability
        g.set_xticklabels(rotation=45, ha='right')
        
        # Customize the subplot titles and labels
        rm_subplot_prefixes(g)
        g.set_ylabels('AUC-ROC') 
        # Add legend
        g.add_legend(bbox_to_anchor=(0.05, 1), loc='upper left', title=style)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.splitext(save_path)[0] + "_bar_AUCROC.png")
        plt.show()
    
    if 'bar_coef' in plot_types:
        # Plot coefficient values
        plt.figure(figsize=figsize) 
        sns.barplot(data=vep_pr, x=x, y='coef',
                    hue=hue)
        plt.xticks(rotation=45, ha='right')
        plt.title('Model Coefficients by Location and Scoring Strategy')
        plt.ylabel('Coefficient Value')
        plt.xlabel('Model Location')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.splitext(save_path)[0] + "_bar_coef.png")
        plt.show()

def plot_auprc(vep_df,
               x='VEP',
               y='DMS_bin_score',
               alpha=0.7,
               figsize=(10, 6),
               xlim=None,
               binarize=True,
               palette=None,
               verbose: bool = True):
    """
    Plot the AUPRC for a given scoring strategy and model location.

    Args:
        vep_df (pd.DataFrame): The dataframe containing the VEP data.
        alpha (float): The alpha value for the plot.
        figsize (tuple): The size of the figure.
        reverse_recall (bool): Whether to reverse the recall axis.
        palette (dict): The palette to use for the plot.

    Returns:
        pd.DataFrame: The dataframe containing the precision and recall results.
    """
    # Create precision-recall curves using matplotlib
    from sklearn.metrics import precision_recall_curve, average_precision_score

    if palette is None:
        palette = get_models_palette()
        
    vep_df = utils.sort_by_reverse_string(vep_df, 
                                            column='scoring_strategy', 
                                            extra_sort_cols=['model_location'],
                                            ascending=[False, True])
    
    # Create subplots for each scoring strategy
    unique_models = vep_df['model_location'].unique()
    unique_strategies = vep_df['scoring_strategy'].unique()
    n_strategies = len(unique_strategies)

    fig, axes = plt.subplots(n_strategies, 1, figsize=(figsize[0], figsize[1]*n_strategies))
    if n_strategies == 1:
        axes = [axes] 

    results_df = pd.DataFrame()
    for i, ss in tqdm(enumerate(unique_strategies), 
                            total=len(unique_strategies),
                            desc="Computing AUPRC"):
        for j, model in enumerate(unique_models):
            subset = vep_df.loc[(vep_df['scoring_strategy'] == ss) & 
                            (vep_df['model_location'] == model)].copy().dropna(subset=[x])
            
            y_true = encode_labels(subset[y], 
                                   binarize=binarize,
                                   verbose=verbose>1)
            y_score = subset[x]
            
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            avg_precision = average_precision_score(y_true, y_score) 
            
            # Store results
            res_df = pd.DataFrame({'precision':precision,
                                   'recall':recall})
            res_df.loc[:, 'avg_precision'] = avg_precision
            res_df.loc[:, 'model_location'] = model
            res_df.loc[:, 'scoring_strategy'] = ss 
            results_df = pd.concat([results_df, res_df])
            
            # Plot precision-recall curve
            axes[i].plot(recall, precision, color=palette[model], 
                        label=f'{model} (AP = {avg_precision:.3f})',
                        alpha=alpha)
            if xlim is not None:
                axes[i].set_xlim(xlim[0], xlim[1])  # Reverse x-axis
        
        axes[i].set_xlabel('Recall')
        axes[i].set_ylabel('Precision')
        axes[i].set_title(f'Precision-Recall Curve - {ss}')
        axes[i].grid(True)
        
        # Customize legend
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    return results_df


def plot_auc_by_edits(vep_pr,
                      x="edits_scaled",
                      y="auc",
                      row="model_location",
                      col="scoring_strategy", 
                      style="edits",
                      markers=None,
                      height=3, 
                      aspect=1,
                      alpha=0.7,
                      add_smooth=True,
                      frac=0.6666666666666666, 
                      it=3,
                      save_path=None,
                      **kwargs):
    # Create scatter plot of AUC-ROC scores by number of edits with fitted curves
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import statsmodels.api as sm
    
    vep_pr = vep_pr.copy()

    if markers is None and style in vep_pr.columns:
        markers = utils.get_marker_map(max(vep_pr[style]), 
                                       is_ref_marker=style == 'is_ref')

    if style is not None:
        if style not in vep_pr.columns:
            style = None
    
    # Fit the LOESS curve
    lowess = sm.nonparametric.lowess
    # smoothed_data = lowess(y, x, frac=0.3) # frac is the fraction of data points used for smoothing


    # Suppress seaborn markers warning
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        # Create facet grid
        g = sns.FacetGrid(vep_pr, 
                            row=row, col=col, 
                            height=height, aspect=aspect,
                            margin_titles=True) 
        
        # Plot scatter points
        g.map_dataframe(sns.scatterplot, 
                        x=x,
                        y=y, 
                        style=style,
                        markers=markers,
                        alpha=alpha,
                        **kwargs)
        if add_smooth: 
            # Fit and plot curves for each subplot
            for ax in g.axes.flat:
                # Get data only from current subplot
                data = [c.get_offsets() for c in ax.collections]
                if not data:
                    continue
                data = np.concatenate(data)
                x_, y_ = data[:, 0], data[:, 1]
                
                try:
                    # Fit LOWESS curve
                    smoothed = lowess(y_, x_, frac=frac, it=it)  # frac is the fraction of data points used for smoothing
                    x_smooth, y_smooth = smoothed[:, 0], smoothed[:, 1]
                    
                    # Plot smoothed curve
                    ax.plot(x_smooth, y_smooth, 'k--', alpha=alpha, zorder=-1)  # Set zorder to -1 to put line under points
                except:
                    pass # Skip curve fitting if it fails

        rm_subplot_prefixes(g)

        g.add_legend()
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
            
        plt.show()

def csv_to_parquet(save_dir: str = os.path.join(config.DATA_DIR,"1KG","vep"), 
                   pattern: str = "*.csv.gz",
                   compression: str = "brotli", 
                   delete_csv: bool = True,
                   **kwargs):
    """Convert CSV files to Parquet format.

    Recursively finds all CSV files matching the pattern in save_dir and converts them
    to Parquet format with the specified compression.

    Args:
        save_dir (str): Directory containing CSV files to convert. Defaults to VEP data directory.
        pattern (str): File pattern to match CSV files. Defaults to "*.csv.gz".
        compression (str): Compression algorithm to use for Parquet files. Defaults to "brotli".
        delete_csv (bool): Whether to delete original CSV files after conversion. Defaults to True.
        **kwargs: Additional arguments passed to pandas.to_parquet().

    Example:
        >>> csv_to_parquet(save_dir="data/", pattern="*.csv.gz", compression="snappy")
    """
    # Find all csv.gz files recursively 
    csv_files = glob.glob(os.path.join(save_dir, "**", pattern), recursive=True)

    # Convert each csv.gz file to parquet
    for csv_file in tqdm(csv_files, desc="Converting files"):
        try:
            # Create equivalent parquet path 
            parquet_file = csv_file.replace(pattern.replace("*",""), '.parquet')
            
            # Skip if parquet already exists
            if os.path.exists(parquet_file):
                continue
                
            # Read CSV and write to parquet
            df = pd.read_csv(csv_file)
            df.to_parquet(parquet_file, 
                          compression=compression,
                          **kwargs)
            
            # Optionally remove original csv.gz file to save space
            if delete_csv:
                os.remove(csv_file)
            
        except Exception as e:
            print(f"Error converting {csv_file}: {str(e)}")

def parquet_to_csv(save_dir: str = os.path.join(config.DATA_DIR,"1KG","vep"),
                   pattern: str = "*.parquet", 
                   compression: str = "gzip",
                   delete_parquet: bool = True,
                   **kwargs):
    """Convert Parquet files to CSV format.

    Recursively finds all Parquet files matching the pattern in save_dir and converts them
    to CSV format with the specified compression.

    Args:
        save_dir (str): Directory containing Parquet files to convert. Defaults to VEP data directory.
        pattern (str): File pattern to match Parquet files. Defaults to "*.parquet".
        compression (str): Compression algorithm to use for CSV files. Defaults to "gzip".
        delete_parquet (bool): Whether to delete original Parquet files after conversion. Defaults to True.
        **kwargs: Additional arguments passed to pandas.to_csv().

    Example:
        >>> parquet_to_csv(save_dir="data/", pattern="*.parquet", compression="gzip")
    """
    # Find all parquet files recursively
    parquet_files = glob.glob(os.path.join(save_dir, "**", pattern), recursive=True)

    # Convert each parquet file to csv
    for parquet_file in tqdm(parquet_files, desc="Converting files"):
        try:
            # Create equivalent csv path
            csv_file = parquet_file.replace('.parquet', '.csv.gz')
            
            # Skip if csv already exists 
            if os.path.exists(csv_file):
                continue

            # Read parquet and write to csv
            df = pd.read_parquet(parquet_file)
            df.to_csv(csv_file,
                     compression=compression,
                     **kwargs)

            # Optionally remove original parquet file
            if delete_parquet:
                os.remove(parquet_file)

        except Exception as e:
            print(f"Error converting {parquet_file}: {str(e)}")




def recompress_parquet(save_dir: str = os.path.join(config.DATA_DIR,"1KG","vep"),
                      pattern: str = "*.parquet",
                      compression: str = "brotli",
                      **kwargs):
    """Recompress all parquet files with snappy compression.
    
    Args:
        save_dir (str): Directory containing parquet files
        pattern (str): File pattern to match
        compression (str): Compression algorithm to use (default: snappy). See `pandas.to_parquet` for more options.
    """
    parquet_files = glob.glob(os.path.join(save_dir, "**", pattern), recursive=True)

    for parquet_file in tqdm(parquet_files, desc="Recompressing files"):
        try:
            # Read parquet file
            df = pd.read_parquet(parquet_file)
            
            # Save with snappy compression
            df.to_parquet(parquet_file, 
                          compression=compression,
                          **kwargs)
            
        except Exception as e:
            print(f"Error recompressing {parquet_file}: {str(e)}")

def plot_population_vep_violin_weighted(df, 
                                        mutant=None, 
                                        min_freq=0, 
                                        top_pop_col = 'top_superpopulation',
                                        freq_col = 'frequency',     
                                        log_fold_change=False,
                                        palette=None,
                                        sharex=True,
                                        sharey=False): 
    """
    Create weighted violin plots for VEP scores across different populations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    mutant : str, default="M1775R"
        The mutant to filter for
    min_freq : float, default=0.005
        Minimum frequency threshold for filtering
    log_fold_change : bool, default=False
        If True, display y-axis as log-fold change relative to the mean within each facet
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the violin plots
    """
    # From: https://stackoverflow.com/a/36904316
    import weighted # pip install wquantiles
    from matplotlib.cbook import violin_stats
    import statsmodels.api as sm

    if palette is None:
        palette = utils.get_superpop_palette()

    # Create a copy of the data with cleaned superpop labels
    plot_data = df.dropna(subset=[top_pop_col,'VEP']).copy().sort_values(by=top_pop_col)
    if mutant is not None:
        plot_data = plot_data.loc[plot_data['mutant']==mutant]

    # Filter out haplotypes that don't have a frequency > min_freq in their top superpopulation
    if freq_col in plot_data.columns:
        plot_data = plot_data.loc[plot_data[freq_col] > min_freq]
    else:
        freq_col = 'frequency'
        plot_data = plot_data.loc[plot_data.apply(lambda row: row[row[top_pop_col]] > min_freq if row[top_pop_col] in row.index else False, axis=1)].copy()
        # Remove the prefix from top_superpopulation for plotting
        plot_data.loc[:, 'Top Superpopulation'] = plot_data[top_pop_col].str.replace('freq_1000GENOMES:phase_3:', '')
    
    plot_data.dropna(subset=[freq_col], inplace=True)

    def vdensity_with_weights(weights):
        ''' Outer function allows inner function access to weights. Matplotlib
        needs function to take in data and coords, so this seems like only way
        to 'pass' custom density function a set of weights '''
        
        def vdensity(data, coords):
            ''' Custom matplotlib weighted violin stats function '''
            # Using weights from closure, get KDE from statsmodels
            weighted_cost = sm.nonparametric.KDEUnivariate(data)
            weighted_cost.fit(fft=False, weights=weights)

            # Return y-values for graph of KDE by evaluating on coords
            return weighted_cost.evaluate(coords)
        return vdensity

    def custom_violin_stats(data, weights):
        # Get weighted median and mean (using weighted module for median)
        median = weighted.quantile_1D(data, weights, 0.5)
        mean, sumw = np.ma.average(data, weights=list(weights), returned=True)
        
        # Use matplotlib violin_stats, which expects a function that takes in data and coords
        # which we get from closure above
        results = violin_stats(data, vdensity_with_weights(weights))
        
        # Update result dictionary with our updated info
        results[0][u"mean"] = mean
        results[0][u"median"] = median
        
        return results

    ### Create violin plots for each superpopulation, faceted by model and scoring strategy
    # Get unique superpopulations
    superpops = plot_data[top_pop_col].unique()
    # Get unique models and strategies for faceting
    models = plot_data['model_location'].unique()
    strategies = plot_data['scoring_strategy'].unique()

    # Create figure with subplots
    fig, axes = plt.subplots(nrows=len(strategies), 
                             ncols=len(models), 
                            figsize=(3*len(models), 4*len(strategies)), 
                            sharex=sharex, 
                            sharey=sharey)

    # Iterate through each model, strategy, and superpopulation
    for i, model in enumerate(models):
        for j, strategy in enumerate(strategies):
            # Get the appropriate subplot
            if len(models) == 1 and len(strategies) == 1:
                ax = axes
            elif len(models) == 1:
                ax = axes[j]
            elif len(strategies) == 1:
                ax = axes[i]
            else:
                ax = axes[i, j]
                
            # Filter data for this model and strategy
            subset = plot_data[(plot_data['model_location'] == model) & 
                            (plot_data['scoring_strategy'] == strategy)].copy()
            
            # Calculate facet mean if using log fold change
            if log_fold_change:
                facet_mean = subset['VEP'].mean()
                if facet_mean == 0:
                    facet_mean = 1e-10  # Avoid division by zero
            
            # Create violin plots for each superpopulation
            positions = []
            for k, superpop in enumerate(superpops):
                # Filter data for this superpopulation
                superpop_data = subset[subset[top_pop_col] == superpop]
                if len(superpop_data) > 0:
                    # Transform data if using log fold change
                    if log_fold_change:
                        plot_values = np.log2(superpop_data['VEP'] / facet_mean)
                    else:
                        plot_values = superpop_data['VEP']
                    
                    # Calculate violin stats with weights
                    vpstats = custom_violin_stats(plot_values.to_numpy(),
                                                superpop_data[freq_col].to_numpy())
                    # Plot violin
                    vplot = ax.violin(vpstats, [k], 
                                    vert=True, 
                                    showmeans=True, 
                                    showextrema=True, 
                                    showmedians=False,  # Don't show default medians
                                    )
                    
                    # Add dotted line for median
                    median = vpstats[0]['median']
                    ax.hlines(median, k-0.2, k+0.2, colors='black', linestyles='dotted', linewidth=1, alpha=0.75)
                    
                    # Set edge color
                    for pc in vplot['bodies']:
                        pc.set_edgecolor('black')
                    positions.append(k)
            
            # Set title and labels
            ax.set_title(f"{model}\n{strategy}")
            ax.set_xticks(positions)
            ax.set_xticklabels([p.split(':')[-1] for p in superpops], rotation=45)
            # Apply color palette to violin plots
            for pc, superpop in zip([pc for pc in vplot['bodies']], [p for p in superpops if p in subset[top_pop_col].values]):
                pc.set_facecolor(palette[superpop.split(':')[-1]])
                pc.set_alpha(0.8)
            if j == 0:  # Only add y-label on leftmost plots
                if log_fold_change:
                    ax.set_ylabel("log2(VEP Score / Facet Mean)")
                else:
                    ax.set_ylabel("VEP Score")
            if i == len(models) - 1:  # Only add x-label on bottom plots
                ax.set_xlabel("superpopulation")

    plt.tight_layout()
    return fig


def plot_population_vep_violin_unweighted(df,
                                          mutant=None, 
                                          min_freq=0.005, 
                                          top_pop_col = 'top_superpopulation',
                                          freq_col = 'frequency',
                                          palette=None):
    import seaborn as sns
    import matplotlib.pyplot as plt

    if palette is None:
        palette = utils.get_superpop_palette()

    # Create a copy of the data with cleaned superpop labels
    plot_data = df.dropna(subset=[top_pop_col,'VEP']).copy().sort_values(by=top_pop_col)
    if mutant is not None:
        plot_data = plot_data.loc[plot_data['mutant']==mutant]

    # Filter out haplotypes that don't have a frequency >0.01 in their top superpopulation
    plot_data = plot_data.loc[plot_data.apply(lambda row: row[row[top_pop_col]] > min_freq if row[top_pop_col] in row.index else False, axis=1)]

    # Remove the prefix from top_superpopulation for plotting
    plot_data['Top Superpopulation'] = plot_data[top_pop_col].str.replace('freq_1000GENOMES:phase_3:', '')


    g = sns.FacetGrid(data=plot_data, 
                    col="model_location", 
                    row="scoring_strategy",
                    sharex=True,
                    sharey=False,
                    margin_titles=True,
                    height=4, 
                    aspect=1.2)
    g.map_dataframe(sns.violinplot, 
                    x="Top Superpopulation", 
                    y="VEP", 
                    hue="Top Superpopulation",
                    palette=palette
                    )
    g.set_axis_labels("superpopulation", "VEP Score")
    g.set_titles(col_template="{col_name}")

    plt.tight_layout()


def create_vep_frequency_df(df,
                            keep_cols = ['haplotype','model_location','scoring_strategy','mutant','VEP'],
                            pop_col = "Population",
                            drop_na = True):
    """
    Create a dataframe of VEP scores and frequencies for each superpopulation.
    """
    # Get frequencies for every population (or superpopulation), not just the top one per haplotype
    import src.onekg as onekg
    new_dat = []
    
    superpops = onekg.get_sample_metadata()[pop_col].unique().tolist()
    for p in ["freq_1000GENOMES:phase_3:"+str(p) for p in superpops]:
        df_tmp = df.loc[:,keep_cols+[p]].rename(columns={p: 'frequency'})
        df_tmp[pop_col] = p.split(':')[-1]
        new_dat.append(df_tmp)
    new_dat = pd.concat(new_dat, axis=0)
    print(new_dat.shape)

    if drop_na:
        new_dat.dropna(subset=['frequency'], inplace=True)

    return new_dat
    
    
def compute_vep_mmr(df,
                    lambda_param=0.5,
                    k=10):
    # Implement Maximal Marginal Relevance (MMR) to select diverse mutants
    # MMR balances relevance (high VEP score differences) with diversity
    import numpy as np
    def compute_mmr(scores, lambda_param=0.5, k=10):
        """
        Compute Maximal Marginal Relevance to select diverse mutants
        
        Args:
            scores: DataFrame with mutants as index and VEP scores
            lambda_param: Balance between relevance and diversity (0-1)
            k: Number of mutants to select
            
        Returns:
            List of selected mutants
        """
        # Start with an empty set of selected mutants
        selected = []
        
        # Convert to numpy for faster computation
        mutants = scores.index.tolist()
        score_matrix = scores.values
        
        # Relevance is the variance of scores across populations
        relevance = np.var(score_matrix, axis=1)
        
        # Normalize relevance scores
        if np.max(relevance) > 0:
            relevance = relevance / np.max(relevance)
        
        while len(selected) < min(k, len(mutants)):
            # Compute MMR score for each candidate
            mmr_scores = []
            
            for i, mutant in enumerate(mutants):
                if mutant in selected:
                    mmr_scores.append(-np.inf)  # Already selected
                    continue
                    
                # Relevance component
                rel_score = relevance[i]
                
                # Diversity component (maximum similarity to already selected)
                div_score = 0
                if selected:
                    # Calculate similarity as correlation between score patterns
                    similarities = []
                    for sel_mutant in selected:
                        sel_idx = mutants.index(sel_mutant)
                        # Use correlation as similarity measure
                        sim = np.corrcoef(score_matrix[i], score_matrix[sel_idx])[0, 1]
                        # Handle NaN values that might occur
                        if np.isnan(sim):
                            sim = 0
                        similarities.append(sim)
                    div_score = max(similarities) if similarities else 0
                
                # MMR score combines relevance and diversity
                mmr_score = lambda_param * rel_score - (1 - lambda_param) * div_score
                mmr_scores.append(mmr_score)
            
            # Select the mutant with highest MMR score
            next_idx = np.argmax(mmr_scores)
            selected.append(mutants[next_idx])
        
        return selected
    # ---- end of compute_mmr function ---- #


    # Group by model and scoring strategy to compute MMR for each group
    mmr_results = {}

    # Get unique combinations of model_location and scoring_strategy
    model_strategy_combos = df.loc[df['mutant_in_haplotype']==False].groupby(['model_location', 'scoring_strategy'], observed=True).size().reset_index()[['model_location', 'scoring_strategy']]

    for _, row in tqdm(model_strategy_combos.iterrows(),
                       total=len(model_strategy_combos),
                       desc="Computing MMR for each model and strategy"):
        model = row['model_location']
        strategy = row['scoring_strategy']
        
        # Filter data for this model and strategy
        filtered_data = df.loc[(df['mutant_in_haplotype']==False) & 
                            (df['model_location']==model) & 
                            (df['scoring_strategy']==strategy)]
        
        # Pivot to get populations as columns
        if 'top_superpopulation' in filtered_data.columns and not filtered_data['top_superpopulation'].isna().all():
            pop_scores = filtered_data.pivot_table(
                index='mutant', 
                columns='top_superpopulation', 
                values='VEP',
                aggfunc='mean'
            )
            
            # Only proceed if we have enough data
            if not pop_scores.empty and pop_scores.shape[1] > 1:
                # Fill NaN values with the mean of the row
                pop_scores = pop_scores.fillna(pop_scores.mean(axis=1))
                
                # Compute MMR
                selected_mutants = compute_mmr(pop_scores, 
                                               lambda_param=lambda_param,
                                                k=k)
                mmr_results[(model, strategy)] = selected_mutants

    # Store the results in a dataframe
    mmr_results_df = pd.DataFrame(columns=['model', 'strategy', 'rank', 'mutant'])

    # Populate the dataframe with results
    for (model, strategy), mutants in mmr_results.items():
        for i, mutant in enumerate(mutants, 1):
            new_row = pd.DataFrame({
                'model': [model],
                'strategy': [strategy],
                'rank': [i],
                'mutant': [mutant]
            })
            mmr_results_df = pd.concat([mmr_results_df, new_row], ignore_index=True)

    # Compute mean rank for each mutant
    mmr_results_df['mean_rank'] = mmr_results_df.groupby('mutant', observed=True)['rank'].transform('mean')

    # Sort by mean rank
    mmr_results_df = mmr_results_df.sort_values('mean_rank')
    return mmr_results_df


def plot_vep_by_superpop(
    vep_df,
    within_site_var=None,
    i=None,
    vep_col="VEP",
    variant_col="mutant",
    haps_to_samples=None,
    unique_haplotypes: bool = True,
    remove_zeros: bool = False,
    figsize=(10, 8),
    hue_top: str = "mutant",
    hue: str = "mutant",
    hue_bottom: str = "mutant",
    legend_loc: str = "upper left",
    binwidth_scaler: float = 200,
    add_clinsig_labels: bool = True,
    palette: str = "Set2"
):
    """
    Plot the distribution of Variant Effect Prediction (VEP) scores by super population.

    This function visualizes the distribution of VEP scores for variants across different super populations,
    with options to facet by variant, clinical significance, or super population. It supports plotting
    distributions for unique haplotypes or for samples, and can highlight reference alleles and annotate
    with clinical significance labels.

    Parameters
    ----------
    vep_df : pd.DataFrame
        DataFrame containing VEP scores and variant information.
    within_site_var : pd.DataFrame
        DataFrame with within-site variant information (used for selecting a specific variant/protein).
    i : int or None, default=0
        Index of the variant/protein to plot. If None, plot all.
    vep_col : str, default="VEP"
        Column name for VEP scores.
    variant_col : str, default="mutant"
        Column name for variant identifier.
    haps_to_samples : pd.DataFrame or None, optional
        DataFrame mapping haplotypes to samples. Required if unique_haplotypes is False.
    unique_haplotypes : bool, default=True
        If True, plot distributions of unique haplotypes; otherwise, plot by samples.
    remove_zeros : bool, default=False
        If True, remove rows where VEP score is zero.
    figsize : tuple, default=(10, 8)
        Figure size for the plot.
    hue_top : str, default="mutant"
        Variable to use for coloring the top histogram.
    hue : str, default="mutant"
        Variable to use for coloring the faceted histograms.
    hue_bottom : str, default="mutant"
        Variable to use for coloring the bottom summary histogram.
    legend_loc : str, default="upper left"
        Location of the legend in the plots.
    binwidth_scaler : float, default=200
        Factor to scale the bin width for histograms.
    add_clinsig_labels : bool, default=True
        If True, add clinical significance labels to reference lines.
    palette : str, default="Set2"
        Color palette to use for variants.

    Returns
    -------
    plot_df : pd.DataFrame
        The DataFrame used for plotting (after filtering and processing).

    Notes
    -----
    - Requires seaborn as sns, matplotlib.pyplot as plt, and utility functions from `utils` and `hs`.
    - The function creates a multi-panel figure showing the distribution of VEP scores for all populations,
      for each super population, and summary histograms colored by the specified hue.
    - Reference alleles are indicated with dashed lines and optional labels.
    - The function returns the DataFrame used for plotting, which may be useful for further analysis.
    """
    if i is not None and within_site_var is not None:
        # If i is a range, convert to list
        if isinstance(i, range):
            i = list(i)
        # Allow i to be a single integer or a list of integers
        if isinstance(i, int):
            indices = [i]
        elif isinstance(i, (list, tuple, set, np.ndarray)):
            indices = list(i)
        else:
            raise ValueError("Parameter 'i' must be an int or a list/array of ints.")

        # Ensure indices are within valid range
        n_rows = within_site_var.drop_duplicates(subset=[vep_col]).shape[0]
        indices = [idx for idx in indices if 0 <= idx < n_rows]
        if not indices:
            raise IndexError("All provided indices are out of range.")

        rows_selected = within_site_var.drop_duplicates(subset=[vep_col]).iloc[indices]
        # If only one row is selected, keep as DataFrame for consistency
        if isinstance(rows_selected, pd.Series):
            rows_selected = rows_selected.to_frame().T

        # Merge on both variant_col and protein
        plot_df = vep_df.merge(
            rows_selected[[variant_col, "protein"]],
            on=[variant_col, "protein"],
            how="inner"
        )
    else:
        plot_df = vep_df.copy()

    if plot_df.empty:
        raise ValueError("No rows remain after filtering.")
    else:
        print("plot_df.shape:",plot_df.shape)

    print("Adding haplotype frequencies.")
    if "protein" in plot_df.columns:
        plot_df = hs.add_haplotype_freqs(plot_df, protein_id_col="protein")
    else:
        plot_df = hs.add_haplotype_freqs(plot_df)

    if remove_zeros:
        print("Removing zeros.")
        plot_df = plot_df.loc[plot_df[vep_col] != 0]

    multi_mutant = plot_df[variant_col].nunique() > 1
    mutant_palette = utils.make_palette(plot_df[variant_col].unique(), palette=palette)

    # Determine color palettes for each hue
    if hue == "clinsig":
        cmap = utils.get_clinsig_palette()
    elif hue == "superpopulation":
        cmap = utils.get_superpop_palette()
    elif hue == variant_col:
        cmap = mutant_palette
    else:
        raise ValueError(f"Invalid hue: {hue}")

    if hue_top == "clinsig":
        cmap_top = utils.get_clinsig_palette()
    elif hue_top == "superpopulation":
        cmap_top = utils.get_superpop_palette()
    elif hue_top == variant_col:
        cmap_top = mutant_palette
    else:
        raise ValueError(f"Invalid hue_top: {hue_top}")

    if hue_bottom == "clinsig":
        cmap_bottom = utils.get_clinsig_palette()
    elif hue_bottom == "superpopulation":
        cmap_bottom = utils.get_superpop_palette()
    elif hue_bottom == variant_col:
        cmap_bottom = mutant_palette
    else:
        raise ValueError(f"Invalid hue_bottom: {hue_bottom}")

    def ylabeler(hue):
        """Return a human-readable label for the y-axis based on the hue variable."""
        if hue == "mutant":
            return "Variant"
        elif hue == "clinsig":
            return "ClinSig"
        elif hue == "superpopulation":
            return "Superpop"
        else:
            return hue

    # Prepare data for plotting: by unique haplotypes or by samples
    if unique_haplotypes:
        plot_df["superpopulation"] = plot_df["top_superpopulation"].str.split(":").str[-1]
    else:
        if haps_to_samples is None:
            raise ValueError("haps_to_samples must be provided if unique_haplotypes is False")
        plot_df = haps_to_samples.merge(plot_df, on="haplotype", how="inner")

    # Calculate global min and max for consistent x-axis limits
    x_min = plot_df[vep_col].min()
    x_max = plot_df[vep_col].max()
    binwidth = (x_max - x_min) / binwidth_scaler

    # Get unique super populations (excluding REF)
    super_pops = plot_df.loc[
        plot_df["is_ref"] == False
    ].dropna(subset=["superpopulation"])["superpopulation"].unique()
    n_pops = len(super_pops)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=n_pops + 3,
        ncols=1,
        height_ratios=[1.5] + [1] + ([1] * (n_pops + 1)),
        hspace=0.1,
        top=0.9
    )

    # Add histogram with all data in first subplot
    ax0 = fig.add_subplot(gs[0])

    # Add REF lines and labels
    ref_rows = plot_df.loc[plot_df["is_ref"] == True].drop_duplicates(subset=[variant_col])
    label_map = {"path": "P", "benign": "B", "likely_path": "LP", "likely_benign": "LB"}

    # Create a new subplot for labels above the histogram
    ax_labels = fig.add_subplot(gs[0])
    ax_labels.set_axis_off()  # Hide the axis

    for _, row in ref_rows.iterrows():
        ref_value = row[vep_col]
        ax0.axvline(
            x=ref_value,
            color=mutant_palette[row[variant_col]],
            linestyle='--',
            label=row[variant_col]
        )
        label = (
            row[variant_col] + " (" + label_map.get(row["clinsig"], row["clinsig"]) + ")"
            if add_clinsig_labels else row[variant_col]
        )
        # Position text in the label subplot above the histogram
        ax_labels.text(
            ref_value - 0.01,
            ax0.get_ylim()[1],
            label,
            color=mutant_palette[row[variant_col]],
            rotation=90,
            va='bottom',
            ha='right',
            transform=ax0.transData
        )

    # Main histogram showing each mutant's distribution
    if multi_mutant:
        sns.histplot(
            data=plot_df,
            x="VEP",
            hue=hue_top,
            multiple="stack",
            legend=False,
            ax=ax0,
            palette=cmap_top,
            binwidth=binwidth
        )
    else:
        sns.histplot(
            plot_df,
            x=vep_col,
            binwidth=binwidth,
            color="white",
            ax=ax0
        )
    ax0.set_xlabel(None)
    ax0.set_xticklabels([])  # Remove x-tick labels

    # Adjust legend location
    if legend_loc == "upper left":
        ax0.text(0.02, 0.95, "All Populations", transform=ax0.transAxes, ha='left', va='top')
    else:
        ax0.text(0.98, 0.95, "All Populations", transform=ax0.transAxes, ha='right', va='top')
    ax0.set_xlim(x_min, x_max)

    if not multi_mutant:
        # Add mean and median lines
        ax0.axvline(
            x=plot_df[vep_col].mean(),
            color='grey',
            linestyle=':',
            label='Mean',
            linewidth=1
        )
        ax0.text(
            plot_df[vep_col].mean() - 0.01,
            ax0.get_ylim()[1],
            'Mean',
            color='grey',
            rotation=90,
            va='top',
            ha='right'
        )
        ax0.axvline(
            x=plot_df[vep_col].median(),
            color='grey',
            linestyle='--',
            label='Median',
            linewidth=1
        )
        ax0.text(
            plot_df[vep_col].median() - 0.01,
            ax0.get_ylim()[1],
            'Median',
            color='grey',
            rotation=90,
            va='top',
            ha='right'
        )

    # Add summary histogram with all superpopulations, colored by mutant
    ax1 = fig.add_subplot(gs[1])
    sns.histplot(
        plot_df.loc[plot_df["superpopulation"] != "REF"],
        x=vep_col,
        binwidth=binwidth * 4,
        hue=hue_top,
        palette=cmap_top,
        legend=False,
        multiple="fill",
        ax=ax1
    )
    ax1.set_xlabel(f"Variant Effect Prediction ({vep_col})")
    ax1.set_ylabel(f"Proportion\nby {ylabeler(hue_top)}")
    ax1.set_xlim(x_min, x_max)

    # Add faceted histograms for each superpopulation
    for idx, pop in enumerate(sorted(super_pops)):
        ax = fig.add_subplot(gs[idx + 2])
        sns.histplot(
            plot_df.loc[plot_df["superpopulation"] == pop],
            x=vep_col,
            binwidth=binwidth,
            hue=hue,
            palette=cmap,
            legend=True if hue == "superpopulation" else False,
            ax=ax
        )
        if hue == variant_col:
            ax.text(0.02, 0.95, pop, transform=ax.transAxes, ha='left', va='top')
        else:
            ax.legend(title="Superpop", loc=legend_loc, labels=[pop])
        ax.set_title(None)
        ax.set_xlabel(None)
        ax.set_xticklabels([])  # Remove x-tick labels
        ax.set_xlim(x_min, x_max)

    # Add summary histogram with all superpopulations (bottom panel)
    ax1 = fig.add_subplot(gs[-1])
    sns.histplot(
        plot_df.loc[plot_df["superpopulation"] != "REF"],
        x=vep_col,
        binwidth=binwidth * 4,
        hue=hue_bottom,
        palette=cmap_bottom,
        legend=False,
        multiple="fill",
        ax=ax1
    )
    ax1.set_xlabel(f"Variant Effect Prediction ({vep_col})")
    if hue_bottom == variant_col:
        ax1.set_ylabel("Proportion\nby Variant")
    elif hue_bottom == "clinsig":
        ax1.set_ylabel("Proportion\nby ClinSig")
    elif hue_bottom == "superpopulation":
        ax1.set_ylabel("Proportion\nby Superpop")
    else:
        ax1.set_ylabel(f"Proportion\nby {hue_bottom}")
    ax1.set_xlim(x_min, x_max)

    # Add figure title
    if i is not None:
        plt.suptitle(
            f"Distribution of VEP scores by Super Population"
            f"\n• Haplotypes: {plot_df['haplotype'].nunique()}"
            f"\n• Proteins (Genes): "
            f"{plot_df['protein'].iloc[0] if plot_df['protein'].nunique() == 1 else plot_df['protein'].nunique()} "
            f"({plot_df['GENEINFO'].iloc[0].split(':')[0] if plot_df['GENEINFO'].nunique() == 1 else plot_df['GENEINFO'].nunique()})"
            f"\n• Disease: {rows_selected['MONDO_label'].str.replace('_',' ').unique()}"
            f"\n• Review Status: {plot_df['CLNREVSTAT'].iloc[0].replace('_',' ')}",
            y=1.03, x=0.125, ha='left'
        )
    else:
        plt.suptitle(
            f"Distribution of VEP scores by Super Population"
            f"\n• Haplotypes: {plot_df['haplotype'].nunique()}"
            f"\n• Variants: {plot_df.groupby('clinsig', observed=True)['mutant'].nunique().to_dict()}"
            f"\n• Proteins (Genes): "
            f"{plot_df['protein'].iloc[0] if plot_df['protein'].nunique() == 1 else plot_df['protein'].nunique()} "
            f"({plot_df['GENEINFO'].iloc[0].split(':')[0] if plot_df['GENEINFO'].nunique() == 1 else plot_df['GENEINFO'].nunique()})"
            f"\n• Diseases: {plot_df['CLNDN'].iloc[0] if plot_df['CLNDN'].nunique() == 1 else plot_df['CLNDN'].nunique()}",
            y=1.03, x=0.125, ha='left'
        )
    plt.tight_layout()

    return plot_df

def extract_id_cols(df,
                    input_col="CLNDISDB",
                    search_terms=["MONDO","OMIM","Orphanet","MedGen","MeSH"],
                     add_counts=True,
                     sep="|",
                     verbose=True):
    """
    Extract ID columns from the input column using regex.
    Args:
        df: DataFrame
        input_col: Column to extract IDs from
        search_terms: List of search terms to extract
        add_counts: Whether to add count columns
        sep: Separator for the input column
        verbose: Whether to print verbose output
    """
    search_terms = utils.as_list(search_terms)
    
    if verbose:
        print(f"Extracting {len(search_terms)} ID column(s).")
    
    # Create regex pattern once
    pattern = '|'.join(f'({term}:[^,{sep}]+)' for term in search_terms)
    
    # Get unique values and their indices
    unique_values = df[input_col].unique()
    unique_indices = {val: idx for idx, val in enumerate(unique_values)}
    
    # Extract IDs only from unique values
    extracted = pd.Series(unique_values).str.extractall(pattern)
    
    # Process results for each term
    for i, term in enumerate(search_terms):
        if verbose:
            print(f"Extracting {term} IDs.")
        # Get results for unique values
        unique_results = extracted[i].groupby(level=0, observed=True).agg(list)
        # Map back to original dataframe
        df.loc[:,term] = df[input_col].map(lambda x: unique_results.get(unique_indices[x], []))
        if verbose:
            print(f"Adding {term} count column.")
        if add_counts:
            df.loc[:,f'{term}_n'] = df[term].str.len()
    
    return df


def filter_top_mutants(vep_df, 
                       within_site_var, 
                       search_terms, 
                       search_col=["CLNDN","MONDO_label"],
                       top_n=5,
                       mutants_per_group=None):
    if search_terms is None:
        sub_df = vep_df.copy()
        within_site_var_sub = within_site_var.copy()
    else:
        sub_df = vep_df.loc[vep_df[search_col[0]].str.lower().str.contains("|".join(search_terms))]
        within_site_var_sub = within_site_var.loc[within_site_var[search_col[1]].str.lower().str.contains("|".join(search_terms))]

    if mutants_per_group is not None:
        top_mutants = within_site_var_sub.groupby(["clinsig"], observed=True).apply(lambda x: x.head(mutants_per_group))["mutant"]
    else:
        top_mutants = within_site_var_sub["mutant"].unique()[:top_n]
    print(top_mutants.shape[0],"top mutants selected")
    sub_df=  sub_df.loc[sub_df["mutant"].isin(top_mutants)]
    return sub_df


def get_mondo_within_site_var(vep_df,
                              groupby_cols= ["model_location",
                                             "clinsig","protein",
                                             "ENSP","GENEINFO",
                                             "RS",   
                                             "mutant",
                                             "scoring_strategy"],
                            input_col="CLNDISDB",
                            search_terms=["MONDO"]
                            ):
    import src.owlready2 as OWL

    group_col = "MONDO"
    split_col = group_col+"_split"
    label_col = group_col+"_label"

    if split_col not in vep_df.columns or group_col not in vep_df.columns:
        vep_df = extract_id_cols(vep_df,
                                input_col=input_col,
                                search_terms=search_terms)
        vep_df.loc[:,split_col] = vep_df.loc[:,group_col].apply(lambda x: [m.replace("MONDO:MONDO:", "MONDO:") for m in x] if isinstance(x, list) else [])
        vep_df.loc[:,group_col] = vep_df.loc[:,split_col].str.join("|")


    if split_col not in vep_df.columns:
        vep_df.loc[:,split_col] = vep_df.loc[:,group_col].str.split("|")


    groupby_cols = [group_col,split_col] + groupby_cols
    within_site_var = vep_df.explode(split_col).reset_index(drop=True).groupby(groupby_cols, observed=True).agg({"VEP":"var",  "haplotype":"nunique"} ).reset_index().sort_values(by="VEP", ascending=False)
    
    onto = OWL.get_onto_mondo()
    id_map = OWL.get_id_map(onto)
    within_site_var.loc[:,label_col] = within_site_var.loc[:,split_col].map(id_map)


    within_site_var_mean = within_site_var.groupby(["model_location",split_col,label_col], observed=True).agg({"VEP":"mean", 
                                                                                                "mutant":"nunique", 
                                                                                                "haplotype":"unique"}
                                                                                                ).sort_values(by="VEP", ascending=False).reset_index()
    within_site_var_mean.loc[:, "haplotype"] = within_site_var_mean.loc[:, "haplotype"].apply(lambda x: x[0] if len(x) > 0 else None)
    
    return within_site_var, within_site_var_mean


def plot_top_mondo(within_site_var_mean,
                   vep_col="VEP",
                   label_col="MONDO_label",
                   id_col="MONDO_split",
                   hue="mutant",
                   palette="plasma",
                   max_label_length=70,
                   top_n=20,
                   figsize=(7, 8)):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import textwrap

    plot_dat = within_site_var_mean.drop_duplicates(subset=[vep_col]).head(top_n)
    
    plot_dat['disease_label'] = plot_dat.apply(lambda row: textwrap.shorten(row[label_col], width=max_label_length, placeholder="...") + f" ({row[id_col]})", axis=1)
    plt.figure(figsize=figsize) 
    ax = sns.barplot(data=plot_dat,
                y="disease_label", 
                x=vep_col,
                palette=palette,
                hue=hue)
    plt.legend(title="Unique variants", loc="lower right")
    plt.xlabel("Within-variant variance (N unique variants)")
    plt.ylabel("Disease (MONDO ID)")
    
    # Add site count annotations 
    for i, row in enumerate(plot_dat.itertuples()):
        ax.text(getattr(row, vep_col) + plot_dat[vep_col].max()*0.01, i, f"({row.site})", 
                va='center', ha='left')

def weighted_variance(data, weights):
    """
    Calculates the weighted variance of a dataset.

    Args:
      data: A list or NumPy array of data values.
      weights: A list or NumPy array of weights corresponding to the data values.

    Returns:
      The weighted variance of the data.
    """
    data = np.array(data)
    weights = np.array(weights)

    if len(data) != len(weights):
        raise ValueError("Data and weights must have the same length.")
    
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")

    weighted_mean = np.average(data, weights=weights)
    variance = np.average((data - weighted_mean)**2, weights=weights)
    
    return variance
 
def plot_vep_dms_correlation(df, 
                             x="VEP",
                             y="DMS_score",
                             row="model_location",
                             col="scoring_strategy",
                             hue="is_ref",
                             size="mutant",
                             logx=False,
                             logy=False,
                             gene_col="gene",
                             cmap=None,
                             ax=None, 
                             figsize=(6, 6)):
    """
    Plot correlation between VEP and DMS scores with regression line and statistics.
    
    Args:
        data (pd.DataFrame): DataFrame containing VEP and DMS scores
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure
        figsize (tuple): Figure size (width, height) - only used if ax is None
    """
    from scipy import stats

    df = df.copy()
    if size is not None:
        if size not in df.columns:
            size = None
    if cmap is None:
        cmap = utils.make_palette(df[hue].unique().tolist(),
                                     palette="Set2")
    
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    if logx:
        df[x] = np.log10(df[x])
    if logy:
        df[y] = np.log10(df[y])
    
    # Create FacetGrid for model_location
    g = sns.FacetGrid(df, row=row, col=col, height=4, aspect=1.5)
    
    # Map scatterplot to each facet
    g.map_dataframe(sns.scatterplot,
                   x=x,
                   y=y,
                   hue=hue,
                   alpha=0.5,
                   size=size,
                   palette=cmap)
    
    # Map regression line to each facet
    g.map_dataframe(sns.regplot,
                   x=x,
                   y=y,
                   scatter=False,
                   line_kws={'color': 'red'})

    # Add titles and labels
    gene_list = df[gene_col].str.split(':').str[0].unique()
    variant_counts = df.mutant.nunique() if pd.api.types.is_string_dtype(df.mutant) else df.mutant.sum()
    g.fig.suptitle(f"Correlation between VEP and DMS Scores\
              \nGene: {gene_list[0] if len(gene_list) == 1 else len(gene_list)}\
              \nVariants: {variant_counts}", y=1.02)
    
    g.set_axis_labels('VEP Score', 'DMS Score')

    # Calculate and add statistics for each facet
    for (row_val, col_val), facet_data in df.groupby([row, col], observed=True):
        # Get the corresponding axes
        row_idx = list(df[row].unique()).index(row_val)
        col_idx = list(df[col].unique()).index(col_val)
        ax = g.axes[row_idx, col_idx]
        
        # Calculate statistics for this facet  
        clean_data = facet_data.dropna(subset=[x, y])
        slope, intercept, r_value, p_value, std_err = stats.linregress(clean_data[x], clean_data[y])
        
        # Calculate R²
        y_pred = slope * clean_data[x] + intercept
        residuals = clean_data[y] - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((clean_data[y] - np.mean(clean_data[y])) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        stats_text = f'Rho: {r_value:.3f}\nR²: {r2:.3f}\np-value: {p_value:.2e}'
        ax.text(0.05, 0.95, stats_text,
               transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.8),
               verticalalignment='top')

    if ax is None:
        plt.tight_layout()
        plt.show()




def plot_dr_with_kde_topo(
        dr_df, 
        x_col="dim1",
        y_col="dim2",
        hue_col="superpopulation",
        sort=True,
        symbol_col=None, #"Sex",
        point_opacity=0.95,
        point_size=3,
        color_continuous_scale="Viridis",
        # KDE background
        add_kde=False, 
        kde_bw_method='scott', 
        kde_n=20, 
        kde_levels=50,  # More levels for a topographic effect
        border_pad_frac=0.15,  # Increase border padding to show full islands

        # New param: how many dotted lines to draw (1=every, 2=every other, etc)
        contour_line_step=4,

        # Shadow effect
        add_shadow=True,
        shadow_color="black",
        shadow_offset = 0.0, # adjust for best effect
        shadow_opacity = 0.8,
        shadow_size_increase = 3,  # how much larger than the main marker

        # Plot params
        plot_bgcolor="white",   
        paper_bgcolor="white",  
        height=600,
        width=800,

        # New argument to control gridlines
        show_grid=True,
        hover_data=None,

        # New argument for cluster labeling
        cluster_col=None,
        scatter_kwargs={},
        point_outline_kwargs=dict(width=0),
    ): 
    """
    Plots a dimensionality reduction (DR) scatter plot with optional KDE topographic background.

    Parameters
    ----------
    dr_df : pd.DataFrame
        DataFrame containing the DR coordinates and metadata.
    x_col : str, default="dim1"
        Column name for the x-axis (first DR dimension).
    y_col : str, default="dim2"
        Column name for the y-axis (second DR dimension).
    hue_col : str, default="superpopulation"
        Column name for coloring points by group.
    sort : str or bool, default=None
        Column name to sort by, or True to sort by hue_col.
    symbol_col : str or None, default=None
        Column name for symbolizing points by group.
    add_kde : bool, default=False
        Whether to add a KDE-based topographic background.
    kde_bw_method : str or float, default='scott'
        Bandwidth method for KDE ('scott', 'silverman', or float).
    kde_n : int, default=20
        Number of grid points per axis for KDE evaluation.
    kde_levels : int, default=30
        Number of contour/heatmap levels for the KDE background.
    border_pad_frac : float, default=0.15
        Fractional padding to add to plot borders for KDE background.
    contour_line_step : int, default=1
        Draw every Nth contour line for the topographic effect.
    add_shadow : bool, default=True
        Whether to add a shadow effect behind points.
    shadow_color : str, default="black"
        Color of the shadow markers.
    shadow_offset : float, default=0.0
        Offset for the shadow effect.
    shadow_opacity : float, default=0.3
        Opacity of the shadow markers.
    shadow_size_increase : float, default=3
        Size increase for shadow markers relative to main markers.
    plot_bgcolor : str, default="white"
        Background color of the plot area.
    paper_bgcolor : str, default="white"
        Background color of the entire figure.
    height : int, default=600
        Height of the figure in pixels.
    width : int, default=800
        Width of the figure in pixels.
    show_grid : bool, default=True
        Whether to show gridlines on the plot.
    hover_data : list of str, default=None
        Columns to include in the hover data.
    cluster_col : str or None, default=None
        Column name for cluster labels to annotate each cluster (one label per cluster).
    scatter_kwargs : dict
        Additional keyword arguments passed to px.scatter.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        The generated Plotly figure.
    """
 
    # Import libraries
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    from scipy.stats import gaussian_kde

    # --- Begin: Mark mean REF hue_col value on color bar ---
    # We'll do this after the scatter is created, but need to compute it now
    # Find REF points
    if "sample" in dr_df.columns:
        ref_points = dr_df[dr_df["sample"] == "REF"]
    else:
        ref_points = dr_df[dr_df["haplotype"].str.endswith(":REF")]
    # Only compute if hue_col is numeric
    ref_hue_mean = None
    hue_is_numeric = False
    if not ref_points.empty and hue_col in dr_df.columns:
        try:
            # Try to convert to float to check if numeric
            _ = dr_df[hue_col].astype(float)
            hue_is_numeric = True
        except Exception:
            hue_is_numeric = False
        if hue_is_numeric:
            ref_hue_mean = ref_points[hue_col].astype(float).mean()
    # --- End: Mark mean REF hue_col value on color bar ---

    hover_data = [x for x in hover_data if x in dr_df.columns]
    palette = utils.get_superpop_palette()
    palette["N/A"] = "grey"
    
    if sort is True:
        dr_df = dr_df.sort_values(by=hue_col, ascending=True)
    if isinstance(sort, str):
        dr_df = dr_df.sort_values(by=sort, ascending=True)
    
    fig = go.Figure()

    # Optionally add KDE background
    if add_kde: 
        x = dr_df[x_col].values
        y = dr_df[y_col].values
        if len(x) > 1:
            # Compute KDE
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy, bw_method=kde_bw_method)
            # Compute the full plot range for background fill
            all_x = dr_df[x_col].values
            all_y = dr_df[y_col].values
            # Increase padding to extend the borders and show full islands
            xpad = (all_x.max() - all_x.min()) * border_pad_frac
            ypad = (all_y.max() - all_y.min()) * border_pad_frac
            xgrid = np.linspace(all_x.min() - xpad, all_x.max() + xpad, kde_n)
            ygrid = np.linspace(all_y.min() - ypad, all_y.max() + ypad, kde_n)
            xx, yy = np.meshgrid(xgrid, ygrid)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            # Discretize the KDE into "levels" for a topographic effect
            if kde_levels is not None and kde_levels > 1:
                zmin, zmax = zz.min(), zz.max()
                levels = np.linspace(zmin, zmax, kde_levels + 1)
                zz_digitized = np.digitize(zz, levels, right=True)
                # For contour lines, we want the actual level values
                zz_levels = levels[zz_digitized]
            else:
                zz_levels = zz

            # Add as heatmap (background "elevation")
            fig.add_trace(go.Heatmap(
                x=xgrid,
                y=ygrid,
                z=zz_levels,
                colorscale=utils.topo_colorscale,
                opacity=0.9,
                showscale=False,
                hoverinfo='skip',
                zsmooth='best'
            ))

            # Add contour lines for topographic effect, with control over how many lines to draw
            if kde_levels is not None and kde_levels > 1:
                zmin, zmax = zz.min(), zz.max()
                all_levels = np.linspace(zmin, zmax, kde_levels + 1)
                # Only draw every Nth contour line
                contour_levels = all_levels[::contour_line_step]
                # If the last level is not included, add it to ensure the outermost contour is drawn
                if contour_levels[-1] != all_levels[-1]:
                    contour_levels = np.append(contour_levels, all_levels[-1])
                # Plot each contour line individually for full control
                for i, level in enumerate(contour_levels):
                    # Skip the first level (lowest) if you don't want a line at the very bottom
                    if i == 0:
                        continue
                    fig.add_trace(go.Contour(
                        x=xgrid,
                        y=ygrid,
                        z=zz,
                        contours=dict(
                            start=level,
                            end=level,
                            size=0,
                            coloring='none',
                            showlines=True
                        ),
                        line=dict(
                            color='black',
                            dash='dot',
                            width=1
                        ),
                        showscale=False,
                        hoverinfo='skip',
                        opacity=0.5,
                        showlegend=False
                    ))
            else:
                # Fallback: draw all contours as before
                fig.add_trace(go.Contour(
                    x=xgrid,
                    y=ygrid,
                    z=zz,
                    contours=dict(
                        start=zmin,
                        end=zmax,
                        size=(zmax-zmin)/kde_levels if kde_levels else 1,
                        coloring='none',
                        showlines=True
                    ),
                    line=dict(
                        color='black',
                        dash='dot',
                        width=1
                    ),
                    showscale=False,
                    hoverinfo='skip',
                    opacity=0.5,
                    showlegend=False  # Remove contour (dotted lines) from legend
                ))

    # Add scatter points with shadow effect
    # We'll add a "shadow" marker for each point, slightly offset and with a blurred, semi-transparent black color.
    # Then add the main points on top.

    # Create scatter plot with Plotly Express (for color mapping and legend)
    scatter = px.scatter(
        dr_df,
        x=x_col,
        y=y_col,
        color=hue_col,
        symbol=symbol_col,
        color_discrete_map=palette,
        color_continuous_scale=color_continuous_scale,
        opacity=point_opacity, 
        hover_data=hover_data,
        width=width,
        height=height,
        **scatter_kwargs
    )
    # Decrease point size by setting marker size in the scatter plot
    if point_size is not None and "size" not in scatter_kwargs.keys():
        scatter.update_traces(marker=dict(size=point_size))

    # Remove marker outline from scatter points
    for trace in scatter.data:
        if hasattr(trace, "marker") and trace.marker is not None:
            trace.marker.line = point_outline_kwargs

    # Add shadow traces first (one per color group)
    if add_shadow:
        for trace in scatter.data:
            # Get the points for this trace
            x_shadow = [v + shadow_offset for v in trace.x]
            y_shadow = [v - shadow_offset for v in trace.y]

            # Handle marker size: could be scalar or array (if size is a column in df)
            marker_size = trace.marker.size if trace.marker.size is not None else 12
            if hasattr(marker_size, "__len__") and not isinstance(marker_size, str):
                # marker_size is an array-like (e.g., list, np.ndarray, pd.Series)
                shadow_marker_size = [s + shadow_size_increase for s in marker_size]
            else:
                # marker_size is a scalar
                shadow_marker_size = marker_size + shadow_size_increase

            # Add shadow trace (underneath)
            fig.add_trace(
                go.Scatter(
                    x=x_shadow,
                    y=y_shadow,
                    mode="markers",
                    marker=dict(
                        size=shadow_marker_size,
                        opacity=shadow_opacity,
                        color=shadow_color,
                        line=dict(width=0),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Add main scatter traces, but make points bigger in legend only
    for trace in scatter.data:
        # Add the trace to the figure
        fig.add_trace(trace)
 
    # Add a black diamond outline around the REF point
    # (ref_points already computed above)
    if not ref_points.empty:
        fig.add_scatter(
            x=ref_points[x_col],
            y=ref_points[y_col],
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=18,
                color="rgba(0,0,0,0)",  # transparent fill
                line=dict(
                    color="white",
                    width=3
                )
            ),
            showlegend=False,
            hoverinfo="skip"
        )

     # Add cluster labels if requested
    # Add a parameter to control the cluster label offset
    cluster_label_offset = 0.5  # You can move this to the function signature if you want it user-configurable
 
    if cluster_col is not None and cluster_col in dr_df.columns:
        # For each cluster, pick a representative point (e.g., the centroid)
        cluster_groups = dr_df.groupby(cluster_col, observed=True)
        cluster_label_traces = []
        for cluster_id, group in cluster_groups:
            if cluster_id in ['-1', -1]:
                continue
            # Use the mean as the label position
            x_label = group[x_col].mean()
            y_label = group[y_col].mean()
            # Offset the label by a fixed amount so it's not right on top of the cluster
            x_offset = cluster_label_offset
            y_offset = cluster_label_offset

            # Add a black shadow text (slightly offset)
            cluster_label_traces.append(
                go.Scatter(
                    x=[x_label + x_offset + 0.01],  # offset for shadow
                    y=[y_label + y_offset - 0.01],
                    mode="text",
                    text=[str(cluster_id)],
                    textposition="middle center",
                    textfont=dict(
                        size=18,
                        color="black",
                        family="Roboto Mono, monospace",
                    ),
                    showlegend=False,
                    hoverinfo="skip"
                )
            )
            # Add the main white label on top, also offset
            cluster_label_traces.append(
                go.Scatter(
                    x=[x_label + x_offset],
                    y=[y_label + y_offset],
                    mode="text",
                    text=[str(cluster_id)],
                    textposition="middle center",
                    textfont=dict(
                        size=18,
                        color="white",
                        family="Roboto Mono, monospace",
                    ),
                    showlegend=False,
                    hoverinfo="skip"
                )
            )
        # Add all label traces at the end so they're above all other layers
        for trace in cluster_label_traces:
            fig.add_trace(trace)

    # Set axis ranges to match the KDE background, with extra padding to show full islands
    if add_kde and 'xgrid' in locals() and 'ygrid' in locals() and len(dr_df[x_col].values) > 1:
        fig.update_xaxes(range=[xgrid[0], xgrid[-1]])
        fig.update_yaxes(range=[ygrid[0], ygrid[-1]])
    else:
        # Even if not using KDE, extend the axis limits to show full islands
        all_x = dr_df[x_col].values
        all_y = dr_df[y_col].values
        xpad = (all_x.max() - all_x.min()) * border_pad_frac
        ypad = (all_y.max() - all_y.min()) * border_pad_frac
        fig.update_xaxes(range=[all_x.min() - xpad, all_x.max() + xpad])
        fig.update_yaxes(range=[all_y.min() - ypad, all_y.max() + ypad])

    # Add a subtle background and grid to mimic a map
    fig.update_layout(
        width=width,
        height=height,
        xaxis_title=None,
        yaxis_title=None,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        
        xaxis=dict(
            showgrid=show_grid,
            gridcolor="#bdbdbd" if show_grid else None,  # medium gray (grid lines)
            zeroline=False,
            showticklabels=False,
            title=None
        ),
        yaxis=dict(
            showgrid=show_grid,
            gridcolor="#bdbdbd" if show_grid else None,  # medium gray (grid lines)
            zeroline=False,
            showticklabels=False,
            title=None
        ),
        font=dict(
            family="Roboto Mono, monospace",
            size=14,
            color="#222"  # very dark gray (almost black, font)
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # --- Begin: Add marker to color bar for mean REF hue_col value ---
    # Only if hue_col is numeric and ref_hue_mean is not None
    if hue_is_numeric and ref_hue_mean is not None:
        # Find the colorbar trace (should be the scatter trace with coloraxis)
        # We'll add a dummy invisible scatter for the colorbar marker
        # But first, find the min/max of the color scale
        hue_vals = dr_df[hue_col].astype(float)
        cmin = hue_vals.min()
        cmax = hue_vals.max()
        # Normalize the mean REF value to [0,1] for colorbar position
        ref_norm = (ref_hue_mean - cmin) / (cmax - cmin) if cmax > cmin else 0.5
        # Add a colorbar marker using an invisible scatter with a custom colorbar
        # We'll use a single point at (None, None) so it doesn't show on the plot
        # and set the colorbar with a marker at the mean REF value
        # Only add a dummy invisible scatter to show the colorbar at the correct range;
        # annotation will handle the REF label, so we don't need tickvals/ticktext
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                color=[ref_hue_mean],
                colorscale=color_continuous_scale,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title=None,
                    thickness=18,
                    outlinewidth=1,
                    ticks="outside",
                    ticklen=8,
                    tickcolor="white",
                    tickfont=dict(color="white", size=12),
                    lenmode="fraction",
                    len=0.8,
                ),
                showscale=True,
                size=0.1,  # invisible
            ),
            showlegend=False,
            hoverinfo="skip"
        ))
        # Add annotation for vertical centering of the label
        # The colorbar is placed at x=1.02 by default, so we offset x accordingly
        # The y position is ref_norm * 0.8 + 0.1 to match colorbar's len and position (len=0.8, starts at 0.1)
        fig.add_annotation(
            x=1.04,  # slightly to the right of the colorbar
            y=ref_norm ,
            xref="paper",
            yref="paper",
            text=f"<b>REF</b>",
            showarrow=False,
            font=dict(color="black", size=14),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=0,
            borderpad=2,
        )
        # Add a horizontal line on the colorbar at the REF mean value
        # (Plotly does not support this natively, but the above tick is a good visual marker)
    # --- End: Add marker to color bar for mean REF hue_col value ---

    fig.show()



def merge_vep_and_samples(vep_df, 
                          haps_to_samples,
                          vep_cols = ["VEP","VEP_norm"],
                          tx_id_col = ["ENST","ENST_haplosaurus"],
                          site_col = "site",
                          ploid_col = "ploid",
                          haplotype_col = "haplotype",
                          protein_col = "protein", 
                          extra_cols = [],
                          check_ploid = True,
                          verbose=True):
    import numpy as np

    # Check if the number of rows is as expected
    # Each sample has exactly 2 haplotypes, and each haplotype should be tested against each site
    if check_ploid:
        haps2 = haps_to_samples.loc[haps_to_samples['ploid_count']==2]
        vep_df_filtered = vep_df.loc[~vep_df[tx_id_col[0]].isin(haps_to_samples.loc[haps_to_samples['ploid_count']!=2]['ENST_haplosaurus'].unique())]
    else:
        haps2 = haps_to_samples
        vep_df_filtered = vep_df
    
    # Expected rows = samples × 2 haplotypes per sample × sites
    expected_rows = haps2['sample'].nunique() * 2 * vep_df_filtered['site'].nunique()
    if verbose:
        print(f"Expected rows: {expected_rows}")
        print(f"Unique samples: {haps2['sample'].nunique()}")
        print(f"Unique sites after filtering: {vep_df_filtered['site'].nunique()}")
        print(f"Calculation: {haps2['sample'].nunique()} samples × 2 haplotypes × {vep_df_filtered['site'].nunique()} sites = {expected_rows}")

    # Fast merge, but ensure all relevant columns are preserved and join is correct
    vep_df_tmp = vep_df[[protein_col, haplotype_col,site_col,tx_id_col[0]]+vep_cols+extra_cols].copy()
    # Ensure 'haplotype' is string for merge
    vep_df_tmp["haplotype"] = vep_df_tmp["haplotype"].astype(str)
    haps_to_samples["haplotype"] = haps_to_samples["haplotype"].astype(str)
    

    # Filter once, no .copy() needed for merge
    if check_ploid:
        rm_transcripts = haps_to_samples.loc[haps_to_samples['ploid_count']!=2]['ENST_haplosaurus'].unique()
        if verbose:
            print(len(rm_transcripts),"transcripts removed due to ploidy != 2")
        vep_df_tmp = vep_df_tmp.loc[~vep_df_tmp[tx_id_col[0]].isin(rm_transcripts)]
        haps2 = haps_to_samples.loc[~haps_to_samples[tx_id_col[1]].isin(rm_transcripts)]
    else:
        haps2 = haps_to_samples

    haps2.drop_duplicates(inplace=True)
    vep_df_tmp.drop_duplicates(inplace=True)

    if verbose:
        print("Mapping haplotype codes")
        print(f"After deduplication - VEP_df_tmp shape: {vep_df_tmp.shape}")
        print(f"After deduplication - haps2 shape: {haps2.shape}")

    if verbose and verbose > 1:
        # Check for potential merge issues that could cause doubling
        print(f"Checking for merge issues that could cause doubling:")
        
        # Check if there are any columns that might be causing issues
        common_cols = set(vep_df_tmp.columns) & set(haps2.columns)
        print(f"Common columns between VEP and haps2: {common_cols}")
        
        # Check if haplotype column has any issues
        vep_haplotype_duplicates = vep_df_tmp['haplotype'].duplicated().sum()
        haps2_haplotype_duplicates = haps2['haplotype'].duplicated().sum()
        print(f"Duplicate haplotypes in VEP after dedup: {vep_haplotype_duplicates}")
        print(f"Duplicate haplotypes in haps2 after dedup: {haps2_haplotype_duplicates}")
        
        # Check for potential many-to-many relationships
        vep_haplotype_counts = vep_df_tmp['haplotype'].value_counts()
        haps2_haplotype_counts = haps2['haplotype'].value_counts()
        
        # Find haplotypes that appear multiple times in either dataset
        vep_multi = vep_haplotype_counts[vep_haplotype_counts > 1]
        haps2_multi = haps2_haplotype_counts[haps2_haplotype_counts > 1]
        
        print(f"Haplotypes appearing >1 time in VEP: {len(vep_multi)}")
        print(f"Haplotypes appearing >1 time in haps2: {len(haps2_multi)}")
        
        if len(vep_multi) > 0:
            print(f"Example VEP haplotype with multiple entries: {vep_multi.head(1)}")
        if len(haps2_multi) > 0:
            print(f"Example haps2 haplotype with multiple entries: {haps2_multi.head(1)}")
        print(f"VEP_df_tmp shape: {vep_df_tmp.shape}")
        print(f"haps2 shape: {haps2.shape}")
        print(f"Unique haplotypes in VEP: {vep_df_tmp['haplotype'].nunique()}")
        print(f"Unique haplotypes in haps2: {haps2['haplotype'].nunique()}")
        print(f"Common haplotypes: {len(set(vep_df_tmp['haplotype']) & set(haps2['haplotype']))}")
        
        # Check for potential issues
        print(f"Duplicate haplotypes in haps2: {haps2['haplotype'].duplicated().sum()}")
        print(f"Duplicate haplotypes in VEP: {vep_df_tmp['haplotype'].duplicated().sum()}")
        
        # Check haplotype distribution
        haplotype_counts = haps2['haplotype'].value_counts()
        print(f"Most common haplotype appears {haplotype_counts.max()} times")
        print(f"Average haplotype frequency: {haplotype_counts.mean():.2f}")
    
    # Only keep necessary columns from haps_to_samples to avoid duplicate columns
    merge_cols = [col for col in haps_to_samples.columns if col != "site"]  # avoid duplicate 'site'
    
    if verbose and verbose > 1:
        print(f"Merge columns: {merge_cols}")
        print(f"VEP_df_tmp columns: {list(vep_df_tmp.columns)}")
        print(f"haps2 columns: {list(haps2.columns)}")
        
        # Check for any potential merge issues
        vep_haplotype_counts = vep_df_tmp['haplotype'].value_counts()
        haps2_haplotype_counts = haps2['haplotype'].value_counts()
        
        print(f"Top 5 haplotypes in VEP data:")
        print(vep_haplotype_counts.head())
        print(f"Top 5 haplotypes in haps2 data:")
        print(haps2_haplotype_counts.head())
        
        # Check if there are any haplotypes that appear many times in haps2
        if haps2_haplotype_counts.max() > 100:
            print(f"WARNING: Some haplotypes appear very frequently in haps2!")
            print(f"Haplotypes appearing >100 times: {(haps2_haplotype_counts > 100).sum()}")
    
    vep_samples = vep_df_tmp.merge(
        haps2[merge_cols],
        on="haplotype",
        how="inner",
        # validate="many_to_many"
    )
    
    if verbose:
        print(f"After merge - vep_samples shape: {vep_samples.shape}")
        print(f"Expected shape: {expected_rows}")
        print(f"Ratio: {vep_samples.shape[0]/expected_rows:.2f}")
        
    if verbose and verbose > 1:
        # If we're getting close to 2x, let's investigate further
        if 1.8 < vep_samples.shape[0]/expected_rows < 2.2:
            print("WARNING: Getting close to 2x expected rows - investigating...")
            
            # Check if there are any haplotypes that appear exactly twice as much as expected
            haplotype_counts = vep_samples['haplotype'].value_counts()
            site_counts = vep_samples['site'].value_counts()
            
            # Check if some haplotypes are appearing twice as much as they should
            expected_per_haplotype = vep_samples['sample'].nunique() * vep_samples['site'].nunique()
            print(f"Expected rows per haplotype: {expected_per_haplotype}")
            print(f"Actual max rows per haplotype: {haplotype_counts.max()}")
            print(f"Actual min rows per haplotype: {haplotype_counts.min()}")
            
            # Check for any obvious patterns
            if haplotype_counts.max() > expected_per_haplotype * 1.5:
                print("Some haplotypes are appearing more than expected!")
                print(f"Haplotypes with >1.5x expected: {(haplotype_counts > expected_per_haplotype * 1.5).sum()}")

    ## Add site ploid col
    if ploid_col is not None and ploid_col in vep_samples.columns:
        site_ploid_col = site_col + "_" + ploid_col
        if site_ploid_col not in vep_samples.columns:
            vep_samples[site_ploid_col] = vep_samples[site_col].astype(str) + "_" + vep_samples[ploid_col].astype(str)
       
    if verbose:
        print(f"Final vep_samples shape: {vep_samples.shape}")
        print(f"Expected shape: {expected_rows}")
        print(f"Ratio actual/expected: {vep_samples.shape[0]/expected_rows:.2f}")
        
        for col in ['sample','haplotype','ploid']:
            if col in vep_samples.columns:
                print(f">> {col}(s): {vep_samples[col].nunique()}")
        
    if verbose and verbose > 1:
        # Check for potential issues in the final dataframe
        if vep_samples.shape[0] > expected_rows * 1.5:  # If more than 50% over expected
            print("WARNING: Final dataframe has many more rows than expected!")
            print("Checking for potential causes:")
            
            # Check if some haplotypes are creating too many combinations
            haplotype_sample_counts = vep_samples.groupby('haplotype', observed=True)['sample'].nunique()
            print(f"Haplotypes appearing in >10 samples: {(haplotype_sample_counts > 10).sum()}")
            if haplotype_sample_counts.max() > 50:
                print(f"Most frequent haplotype appears in {haplotype_sample_counts.max()} samples")
            
            # Check for duplicate combinations
            duplicate_check = vep_samples.groupby(['haplotype', 'site', 'sample'], observed=True).size()
            if duplicate_check.max() > 1:
                print(f"WARNING: Found duplicate haplotype-site-sample combinations!")
                print(f"Max duplicates: {duplicate_check.max()}")
    
    return vep_samples



def plot_top_diff_variants(
        vep_df,
        freq_df=None,
        figsize=(8, None),
        label_fontsize=10,
        show=True,
        x="VEP_mean_diff",
        is_ref_filter=None,
        abs_diff_threshold=1,
        x_label="VEP difference",
        y_label=None,
        title=  r"Variants where REF VEP is $\pm$1 VEP unit from the mean",
        max_rows=10,
        sample_head_and_tail=False,
        arrow_bottom_margin=0.25,
        arrow_text_padding=0.05,
        table_width_ratio=0.4,
        palatte = utils.get_clinsig_palette()
    ):
    """
    Plot variants where the REF was +/- 1 VEP unit from the mean (or other x column).

    Parameters
    ----------
    vep_df : pd.DataFrame
        DataFrame with variant effect predictions, must include columns:
        'is_ref', 'VEP_mean_diff', 'haplotype', 'GENEINFO', 'HGVSp', 'mutant',
        'RS', 'n_haplotypes', 'VEP', 'VEP_mean', 'mean_freq', 'clinsig'
    freq_df : pd.DataFrame, optional
        DataFrame with haplotype frequencies, must include 'haplotype' and *_freq columns.
    figsize : tuple, optional
        Figure size (width, height). If height is None, will be set to 2 * n rows.
    label_fontsize : int, optional
        Font size for annotation labels.
    show : bool, optional
        Whether to call plt.show().
         x : str, optional
         Column to use for the x-axis (default: "VEP_mean_diff").
     arrow_bottom_margin : float, optional
         Amount of whitespace between the arrows and the bottom of the plot (default: 0.25).
     arrow_text_padding : float, optional
         Amount of padding between the arrows and the text annotations below them (default: 0.05).
     table_width_ratio : float, optional
         Width ratio for the table relative to the barplot (default: 0.4).

     Returns
     -------
     matplotlib.axes.Axes
         The barplot axes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd


    diff_df = vep_df.copy()

    if "n_haplotypes" not in diff_df.columns:
        diff_df["n_haplotypes"] = diff_df.groupby(["model_location", "scoring_strategy", "mutant"])["haplotype"].transform("nunique")

    if is_ref_filter is not None:
        diff_df = diff_df.loc[diff_df["is_ref"] ==is_ref_filter]

    # Find variants where the REF was +/- 1 VEP unit from the mean
    # (If x is not "VEP_mean_diff", still filter by VEP_mean_diff for consistency)
    if abs_diff_threshold is not None:
        diff_df = diff_df.loc[((diff_df[x] > abs_diff_threshold) | (diff_df[x] < -abs_diff_threshold))]

    # Merge in mean_freq if freq_df is provided
    if freq_df is not None:
        freq_cols = [col for col in freq_df.columns if col.endswith("_freq") and col != "top_superpopulationulation_freq"]
        total_hap_freqs = freq_df.set_index("haplotype")[freq_cols].mean(axis=1).reset_index().rename(columns={0: "mean_freq"})
        diff_df = diff_df.merge(total_hap_freqs, on=["haplotype"], how="left")
    

    diff_df = utils.sort_by_clinsig(diff_df)
    diff_df[x+"_abs"] = diff_df[x].abs()
    diff_df.sort_values(by=x, ascending=False, inplace=True)

    if max_rows is not None:
        print(f"Sampling {max_rows} rows from {len(diff_df)}")
        if title is not None:
            title_tmp = f" ({min(max_rows, len(diff_df))} / {len(diff_df)} rows shown)"
            title += title_tmp #r"$\it{" + title_tmp.replace(" ", r"\ ") + r"}$"
        if sample_head_and_tail:
            diff_df = pd.concat([diff_df.head(max_rows//2), 
                                 diff_df.tail(max_rows//2)])
        else:
            diff_df = diff_df.head(max_rows)
    



    print(diff_df.groupby(["model_location", "scoring_strategy"], observed=True).agg({"mutant": pd.Series.nunique, "protein": pd.Series.nunique}))
    
    diff_df["Gene"] = diff_df["GENEINFO"].str.split(":").str[0]
    diff_df["label"] = (
        r"$\bf{HGVSp}$: " + r"$\bf{" + diff_df["HGVSp"].str.replace(r'([\\_{}$%#&^~])', r'\\\1', regex=True) + "}$" + "\n"
        + r"$\it{Mutation}$: " + diff_df["mutant"] + "\n"
        + r"$\it{Gene}$: " + diff_df["Gene"] + "\n"
        + r"$\it{RSID}$: " + diff_df["RS"].apply(lambda x: f"rs{x}" if pd.notna(x) else "N/A") + "\n"
        + r"$\it{Haplotypes}$: " + diff_df["n_haplotypes"].astype(str) + "\n"
        + (
            r"$\it{Haplotype\ ID}$: " + diff_df["haplotype"].str.split(":").str[1] + "\n"
            + r"$\it{VEP_{hap}}$: " + diff_df["VEP"].apply(lambda x: f"{x:.2f}") + "\n"
            if is_ref_filter is not True else ""
        )
        + r"$\it{VEP_{REF}}$: " + diff_df["VEP_REF"].apply(lambda x: f"{x:.2f}") + "\n"
        + r"$\it{VEP_{\text{mean}}}$: " + diff_df["VEP_mean"].apply(lambda x: f"{x:.2f}") + "\n"
        + r"$\it{Freq_{REF}}$: " + diff_df["mean_freq"].apply(lambda x: f"{x:.2e}")
    )
    # Make all labels unique by appending a unique index to each label
    diff_df["label"] = diff_df["label"] + " [#" + diff_df.reset_index().index.astype(str) + "]"
     
    n_rows = len(diff_df)
    if figsize[1] is None:
        fig_height = max(2, n_rows * .5)
        figsize = (figsize[0], fig_height)

    # Create figure with subplots for barplot and table
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=figsize)
    
    # Use width_ratios to control barplot vs table proportions
    # Adjust height_ratios to align table header with space above first bar
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, table_width_ratio], wspace=0.02, height_ratios=[1])
    
    # Create barplot subplot
    ax = fig.add_subplot(gs[0])

    # Add an extra row with value 0 to create space above the first data row
    # This will allow the table header to align with the space above the first bar
    extra_row = diff_df.iloc[0].copy()
    extra_row[x] = 0  # Set the x value to 0
    extra_row["label"] = "SPACER"  # Add a label for the spacer row
    
    # Create a new dataframe with the extra row at the top
    plot_df = pd.concat([pd.DataFrame([extra_row]), diff_df], ignore_index=True)
    # Ensure the extra row is at the top by setting categorical order for y
    # The first label is the spacer, followed by the real data labels in order
    y_order = plot_df["label"].tolist()
    y_order.reverse()
    # Set the 'label' column as a categorical with this order
    plot_df["label"] = pd.Categorical(plot_df["label"], categories=y_order, ordered=True)
    
    barplot = sns.barplot(
        plot_df,
        x=x,
        y="label",
        hue="clinsig",
        palette=palatte,
        ax=ax
    )
    ax.set_title(
        title,
        loc='left'  # Align title to the left
    )
    if x_label is not None:
        ax.set_xlabel(x_label)
    else:
        ax.set_xlabel(x.replace("_", " ").title())
    if y_label is not None:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(None)
    ax.legend(title="Clinical Significance")
    ax.yaxis.tick_right()  # Move y-axis tick labels to the right
    ax.yaxis.set_label_position("right")  # Move y-axis label to the right (if present)

    # Create table subplot
    table_ax = fig.add_subplot(gs[1])
    table_ax.axis('off')  # Hide the table subplot axes
    
    # Hide the spacer row label (first row) and show the real data labels
    ax.set_yticklabels([''] + [f"{row['GENEINFO'].split(':')[0]}:{row['mutant']}" 
                              for _, row in diff_df.iterrows()])
    
    # Adjust y-axis limits to account for the extra row
    ax.set_ylim(-0.5, len(plot_df) - 0.5)
    
    # Get the y-axis positions for perfect alignment
    y_positions = ax.get_yticks()
    
    # Create table data (without prefixes in cells, just the values)
    table_data = []
    for i, pos in enumerate(y_positions):
        # Skip the first row (spacer row) and get data from the original diff_df
        if i == 0:
            continue  # Skip spacer row
        row_data = diff_df.iloc[i-1] if (i-1) < len(diff_df) else {}
        
        # Build row data conditionally
        row = [
            row_data.get('GENEINFO', 'N/A').split(':')[0] if pd.notna(row_data.get('GENEINFO')) else 'N/A',
            row_data.get('mutant', 'N/A'),
            f"rs{row_data.get('RS', 'N/A')}" if pd.notna(row_data.get('RS')) else "N/A",
            str(row_data.get('n_haplotypes', 'N/A'))
        ]
        
        # Add VEP column conditionally
        if is_ref_filter is not True:
            row.append(f"{row_data.get('VEP', 'N/A'):.2f}" if pd.notna(row_data.get('VEP')) else "N/A")
        
        # Add remaining columns
        row.extend([
            f"{row_data.get('VEP_REF', 'N/A'):.2f}" if pd.notna(row_data.get('VEP_REF')) else "N/A",
            f"{row_data.get('VEP_mean', 'N/A'):.2f}" if pd.notna(row_data.get('VEP_mean')) else "N/A",
            f"{row_data.get('mean_freq', 'N/A'):.2e}" if pd.notna(row_data.get('mean_freq')) else "N/A"
        ])
        
        # Add haplotype ID conditionally
        if is_ref_filter is not True:
            hap_id = row_data.get('haplotype', 'N/A').split(':')[1] if pd.notna(row_data.get('haplotype')) else 'N/A'
            row.append(hap_id)
        
        table_data.append(row)
    
    # Create table using matplotlib.pyplot.table
    # Build column headers conditionally
    col_labels = [r'Gene', r'Mutation', r'RS', r'Haplotypes']
    
    # Add VEP column conditionally
    if is_ref_filter is not True:
        col_labels.append(r'VEP$_{hap}$')
    
    # Add remaining columns
    col_labels.extend([r'VEP$_{REF}$', r'VEP$_{mean}$', r'Freq$_{REF}$'])
    
    # Add haplotype ID column conditionally
    if is_ref_filter is not True:
        col_labels.append(r'Hap ID')
    
    table = table_ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]  # Fill the entire table subplot
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    
    # Use reasonable row heights for readability
    table.scale(1, 2)  # Adjust row heights to match barplot
    
    # Auto-adjust column widths to accommodate content
    table.auto_set_column_width(col=list(range(len(table_data[0]))))
    
    # Style the header row
    for i in range(len(table_data[0])):
        header_cell = table[(0, i)]
        header_cell.set_facecolor('#666666')  # Lighter grey background
        header_cell.set_text_props(weight='bold', color='white')
        header_cell.set_height(0.1)  # Make header row slightly taller
    
    # Style the data rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            # Alternate row colors for better readability
            if (i-1) % 2 == 0:  # Adjust for header row offset
                cell.set_facecolor('#F5F5F5')  # Light gray
            else:
                cell.set_facecolor('white')
            cell.set_text_props(weight='normal', color='black')
    
    # Add borders to the table
    for i in range(len(table_data) + 1):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            cell.set_edgecolor('black')
            cell.set_linewidth(0.5)
    
    # Remove y-axis labels since we now have a table
    ax.set_yticklabels([])
    ax.set_ylabel(None)

    # Add vertical grey dotted lines at -1 and 1 if x is VEP_mean_diff
    if x == "VEP_mean_diff":
        ax.axvline(0, color='grey', linestyle='-', linewidth=3, zorder=1)
        ax.axvline(-1, color='grey', linestyle=':', linewidth=1.5, zorder=1)
        ax.axvline(1, color='grey', linestyle=':', linewidth=1.5, zorder=1)

    # Add grid lines through the center of each bar
    # Skip the top-most grid line (last ytick) to avoid cluttering the spacer row area
    yticks = ax.get_yticks()
    for i, ytick in enumerate(yticks):
        if i < len(yticks) - 1:  # Skip the last (top-most) grid line
            ax.axhline(ytick, color='lightgray', linestyle='--', linewidth=0.7, zorder=0)

    # Add arrows and labels under the x-axis
    xmin, xmax = ax.get_xlim()
    arrow_y = -arrow_bottom_margin  # relative to axes fraction, below x-axis label
    label_y = -(arrow_bottom_margin + arrow_text_padding)  # further below for the text

    arrow_length = (xmax - xmin) * 0.35

    # Draw left arrow: from 0 to negative
    ax.annotate(
        '', xy=(0 - arrow_length, arrow_y), xytext=(0, arrow_y),
        xycoords=('data', 'axes fraction'), textcoords=('data', 'axes fraction'),
        arrowprops=dict(arrowstyle='-|>', color=palatte["path"], lw=3),
        annotation_clip=False
    )
    ax.text(
        0 - arrow_length/2, label_y, r"$\bf{VEP_{REF}\ overestimates}$" + "\n" + r"$\bf{pathogenicity}$",
        ha='right', va='top', color=palatte["path"], fontsize=label_fontsize,
        transform=ax.get_xaxis_transform()
    )

    # Draw right arrow: from 0 to positive
    ax.annotate(
        '', xy=(0 + arrow_length, arrow_y), xytext=(0, arrow_y),
        xycoords=('data', 'axes fraction'), textcoords=('data', 'axes fraction'),
        arrowprops=dict(arrowstyle='-|>', color=palatte["benign"], lw=2),
        annotation_clip=False
    )
    ax.text(
        0 + arrow_length/2, label_y, r"$\bf{VEP_{REF}\ underestimates}$" + "\n" + r"$\bf{pathogenicity}$",
        ha='left', va='top', color=palatte["benign"], fontsize=label_fontsize,
        transform=ax.get_xaxis_transform()
    )

    plt.ylabel(None)
    plt.tight_layout()
    if show:
        plt.show()
    return {"fig": fig.figure, "axes": fig, "data": diff_df}

def variant_count_by_source_barplot(
    urls={
        'substitutions': "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.1/clinical_ProteinGym_substitutions.zip",
        'splice_variants': "~/projects/VEP_DNA/data/splicing/splicevardb_x_clinvar_snv.csv",
        'utr_variants': "~/projects/VEP_DNA/data/UTR/clinvar_utr_snv.bed.gz"
    },
    show_plot=True,
    figsize=(10, 2),
    legend_loc='lower right',
    palette = utils.get_clinsig_palette(),
    x_label="Variant Type",
    y_label="Variants",
    width=0.9,
    linewidth=1,
    flip_axes=False
):
    """
    Plots a stacked bar plot of clinical variant counts by type and clinical significance.

    Parameters
    ----------
    urls : dict
        Dictionary of file paths/URLs for substitutions, splice_variants, and utr_variants.
    show_plot : bool
        Whether to display the plot.
    figsize : tuple
        Figure size.
    legend_loc : str
        Location of the legend.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    flip_axes : bool
        If True, flip the x and y axes (vertical bars instead of horizontal).
    """
    pg_subs_paths = pooch.retrieve(
        urls['substitutions'],
        known_hash=None,
        processor=pooch.Unzip(),
        progressbar=True
    )

    #### ProteinGym ####
    pg_subs = pd.concat([pd.read_csv(path, index_col=0) for path in tqdm(pg_subs_paths)])
    sub_counts = pg_subs.groupby("DMS_bin_score", observed=True).size().reset_index().rename(
        columns={"DMS_bin_score": "ClinSig", 0: "Count"}
    )
    sub_counts["Source"] = "ProteinGym"
    sub_counts["Variant Type"] = "Missense"

    #### Splicing ####
    splice_variants = pd.read_csv(urls['splice_variants'])
    splice_counts = splice_variants.groupby("CLNSIG_simplified", observed=True).size().reset_index().rename(
        columns={"CLNSIG_simplified": "ClinSig", 0: "Count"}
    )
    splice_counts["Source"] = "SpliceVarDB"
    splice_counts["Variant Type"] = "Splicing"

    #### UTR ####
    import src.clinvar as cv
    utr_variants = cv.read_bed(urls['utr_variants']).to_pandas()

    clnsig_map = {
        "likely_path": "pathogenic",
        "likely_pathogenic": "pathogenic",
        "pathogenic": "pathogenic",
        "path": "pathogenic",
        "likely_benign": "benign",
        "vus": "VUS",
        "VUS": "VUS",
        
    }
    utr_counts = utr_variants.replace({"CLNSIG_simple": clnsig_map}).groupby("CLNSIG_simple", observed=True).size().reset_index().rename(
        columns={"CLNSIG_simple": "ClinSig", 0: "Count"}
    )
    utr_counts["Source"] = "ClinVar"
    utr_counts["Variant Type"] = "UTR"

    #### Combine and plot ####
    variant_counts = pd.concat([splice_counts, utr_counts, sub_counts])
    variant_counts["ClinSig"] = variant_counts["ClinSig"].str.lower()
    # variant_counts = variant_counts.loc[variant_counts["ClinSig"].str.lower() != "vus"]
    variant_counts["Variant Type (Source)"] = variant_counts["Variant Type"] + " (" + variant_counts["Source"] + ")"
    print(variant_counts)
    # Create figure and axes with specified size
    fig, ax = plt.subplots(figsize=figsize)

    variant_counts["ClinSig"] = variant_counts["ClinSig"].str.replace("vus", "VUS")
    
    # Define the desired order: Missense, Splicing, UTR (top to bottom when flip_axes=False)
    variant_type_order = ["Missense", "Splicing", "UTR"]
    variant_counts["Variant Type"] = pd.Categorical(
        variant_counts["Variant Type"], 
        categories=variant_type_order, 
        ordered=True
    )
    variant_counts.sort_values(by="Variant Type", inplace=True)

    # Group data by Consequence (Source) and ClinSig
    grouped = variant_counts.groupby(['Variant Type', 'ClinSig'], sort=False, observed=True)['Count'].sum().unstack()
    
    # Ensure the index order matches the desired order (only reindex existing types)
    existing_types = [vt for vt in variant_type_order if vt in grouped.index]
    grouped = grouped.reindex(existing_types)

    # Create stacked bar plot, with option to flip axes
    if not flip_axes:
        grouped.plot(
            kind='barh',
            stacked=True,
            color=[palette[c] for c in grouped.columns],  # Ensure order matches columns
            width=width,
            ax=ax,
            
        )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # Invert y-axis so Missense is at top, UTR at bottom
        ax.invert_yaxis()
        # Format x-axis (counts) with k notation
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    else:
        grouped.plot(
            kind='bar',
            stacked=True,
            color=[palette[c] for c in grouped.columns],  # Ensure order matches columns
            width=width,
            ax=ax,
        )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # Format y-axis (counts) with k notation
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    
    # Set edge color and linewidth on all bar patches
    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(1)

    # Rotate x-axis tick labels for better readability
    if flip_axes:
        plt.xticks(rotation=0, ha='center')
    plt.title("Clinical Variants by Type")
    plt.legend(title="Clinical\nSignificance" if flip_axes else "Clinical Significance", loc=legend_loc)


    # Remove the top and right spines (margin lines) for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend = plt.legend(title="Clinical Significance", loc=legend_loc)
    # legend.get_frame().set_linewidth(0)
    plt.tight_layout()

    if show_plot:
        plt.show()
    return {'fig': fig, 'ax': ax, 'data': grouped}


import matplotlib.ticker as mticker


def plot_n_variants_histogram(
    df, 
    x="n_variants",
    title="WT Variants per Haplotype",
    x_label="WT Variants per Haplotype",
    y_label="Haplotypes",
    formatter=mticker.FuncFormatter(lambda x, pos: f"{x/1_000_000:.0f}M" if x >= 1_000_000 else (f"{int(x)}" if x > 0 else "0")),
    save_path=None, 
    bins=50,
    figsize=(10, 4),
    show_plot=True
):
    """
    Plot a histogram showing the distribution of the number of WT variants per haplotype.

    This function creates a histogram of the specified column (default: "n_variants") from the input DataFrame,
    displaying the number of haplotypes with a given number of WT variants. The median value is highlighted
    with a vertical dashed line. The y-axis is formatted to display counts in thousands (k) or millions (M)
    for readability.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data to plot.
    x : str, optional
        Column name in `df` representing the number of variants per haplotype (default: "n_variants").
    title : str, optional
        Title of the plot (default: "WT Variants per Haplotype").
    x_label : str, optional
        Label for the x-axis (default: "WT Variants per Haplotype").
    y_label : str, optional
        Label for the y-axis (default: "Number of Haplotypes").
    formatter : matplotlib.ticker.Formatter, optional
        Formatter for the y-axis ticks (default: formats to "k" or "M" as appropriate).
    save_path : str or None, optional
        If provided, the path to save the figure (default: None).
    bins : int, optional
        Number of bins for the histogram (default: 50).
    figsize : tuple, optional
        Size of the figure (default: (10, 4)).
    show_plot : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    dict
        Dictionary containing the figure, axes, and the input DataFrame:
        {'fig': fig, 'ax': ax, 'data': df}
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(
        df, 
        x=x, 
        bins=bins, 
        color="gray",
        # hue="cohort", multiple="stack", palette="gray"
    )
    median_val = df[x].median()
    plt.axvline(
        median_val, 
        color="goldenrod", 
        linestyle="--", 
        linewidth=2, 
        label=f"Median = {int(median_val)}"
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    legend = plt.legend(title="")
    # Remove the margin line (frame) around the legend
    if legend is not None:
        legend.get_frame().set_linewidth(0)
        legend.get_frame().set_edgecolor('none')

    # Set y-axis to thousands (k) or millions (M) instead of scientific notation
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)

    # Remove the top and right spines (margin lines) for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, **utils.FIG_SAVE_KWARGS)
    if show_plot:
        plt.show()
    return {'fig': fig, 'ax': ax, 'data': df}


# Pairwise tests shared by barplot annotation helpers (statannotations names).
_PAIRWISE_STATS_TESTS = frozenset(
    {"t-test_ind", "t-test_welch", "Mann-Whitney", "Mann-Whitney-ls"}
)


def _pairwise_stat_and_p(y1, y2, test_name):
    """Return (statistic, p) for two independent samples; aligns with statannotations names."""
    from scipy import stats as scipy_stats

    if test_name == "t-test_ind":
        stat, pval = scipy_stats.ttest_ind(
            y1, y2, equal_var=True, nan_policy="omit"
        )
    elif test_name == "t-test_welch":
        stat, pval = scipy_stats.ttest_ind(
            y1, y2, equal_var=False, nan_policy="omit"
        )
    elif test_name == "Mann-Whitney":
        res = scipy_stats.mannwhitneyu(y1, y2, alternative="two-sided")
        stat, pval = res.statistic, res.pvalue
    elif test_name == "Mann-Whitney-ls":
        res = scipy_stats.ranksums(y1, y2, alternative="two-sided")
        stat, pval = res.statistic, res.pvalue
    else:
        raise ValueError(test_name)
    return float(stat), float(pval)


def _validate_pairwise_stats_test(stats_test):
    if stats_test not in _PAIRWISE_STATS_TESTS:
        raise ValueError(
            f"stats_test={stats_test!r} is not supported. "
            f"Use one of: {sorted(_PAIRWISE_STATS_TESTS)}."
        )


def _p_bonferroni_if_applicable(p_raw, n_comp, comparisons_correction):
    """Bonferroni-adjusted p only when correction label is bonferroni; else NaN."""
    if comparisons_correction is None:
        return float("nan")
    if str(comparisons_correction).strip().lower() == "bonferroni":
        return min(1.0, float(p_raw) * n_comp)
    return float("nan")


def plot_variant_type_and_clinsig(
    df,
    x_var = "MC_term_label",
    y_var = "VEP",
    hue_var = "clinsig_simple",
    x_label = "Clinical Variant Type",
    y_label = r"VEP$_{mean}$",
    title = "VEP by Clinical Variant Type",
    legend_title = "Clinical Significance", 
    palette=utils.get_clinsig_palette(),
    figsize=(3, 5),
    legend=True,
    save_path=None,
    fig_kwargs=None,
    hide_xtick_labels=False,
    stats_test="Mann-Whitney",
):
    """
    Create and save a barplot of mean VEP by clinical variant type and significance with statistical annotation.
    Includes comparisons both within each clinical variant type (i.e., among clinsig groups for each MC_term)
    and across clinical variant types (i.e., among MC_terms for each clinsig group).

    Parameters
    ----------
    flashzoi_agg : pd.DataFrame
        DataFrame with columns 'MC_term', 'VEP', 'clinsig_simple'.
    save_path : str
        Output path for the saved figure.
    fig_kwargs : dict or None
        Additional kwargs for plt.savefig (merged with utils.FIG_SAVE_KWARGS).
    hide_xtick_labels : bool
        If True, hide x-axis tick labels.
    stats_test : str
        Pairwise test passed to ``statannotations`` and used to build ``stats`` (must match exactly).
        Default ``'Mann-Whitney'``. Also supported: ``'t-test_ind'`` (Student, equal variance),
        ``'t-test_welch'``, ``'Mann-Whitney-ls'`` (Wilcoxon rank-sum / ``scipy.stats.ranksums``).

    Returns
    -------
    dict
        ``fig``, ``ax``, ``data`` (working copy of ``df``), and ``stats`` (DataFrame): one row
        per pairwise comparison with ``test``, ``correction``, ``n_comparisons``, ``y_var``,
        ``x_var``, ``kind``, ``group1``, ``group2``, ``n1``, ``n2``, ``statistic``,
        ``rank_biserial_r`` (only for ``Mann-Whitney``: :math:`1 - 2U/(n_1 n_2)` from scipy's
        :math:`U`, else NaN), ``p`` (uncorrected), ``p_bonferroni``, and optional ``note``.
        Tests and Bonferroni correction match the plot annotations.
    """
    from statannotations.Annotator import Annotator

    _validate_pairwise_stats_test(stats_test)

    # Map MC_term for pretty labels
    df = df.copy()  # avoid modifying original

    # Sort by clinsig to preserve ordering
    df = utils.sort_by_clinsig(df, clinsig_col=hue_var)

    # Extract unique values preserving the order from the sorted dataframe
    # This respects the clinsig ordering from sort_by_clinsig
    if x_var == hue_var or x_var in ['clinsig', 'clinsig_simple']:
        # For clinsig columns, preserve order from sorted dataframe
        mc_terms = df[x_var].drop_duplicates().tolist()
    else:
        # For non-clinsig columns, sort normally
        mc_terms = sorted(df[x_var].unique())
    
    # For hue_var, preserve order from sorted dataframe (already sorted by clinsig)
    clinsig_simple_cats = df[hue_var].drop_duplicates().tolist()

    fig = plt.figure(figsize=figsize)
    
    # Handle case where x_var == hue_var (no separate grouping)
    if x_var == hue_var:
        ax = sns.barplot(
            data=df,
            x=x_var,
            y=y_var,
            palette=palette,
            order=mc_terms,
            legend=legend,
            edgecolor="black",
            linewidth=1,
        )
    else:
        ax = sns.barplot(
            data=df,
            x=x_var,
            y=y_var,
            hue=hue_var,
            palette=palette,
            order=mc_terms,
            hue_order=clinsig_simple_cats,
            legend=legend,
            edgecolor="black",
            linewidth=1,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if hide_xtick_labels:
        ax.set_xticklabels([])
    if legend:
        ax.legend(
            title=legend_title,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0)
        )

    sns.despine(ax=ax, top=True, right=True)

    # Handle case where x_var == hue_var (no separate grouping)
    if x_var == hue_var:
        # When x and hue are the same, generate simple pairs between categories
        pairs = [
            (cat1, cat2)
            for i, cat1 in enumerate(clinsig_simple_cats)
            for j, cat2 in enumerate(clinsig_simple_cats)
            if j > i
        ]
        
        annotator = Annotator(
            ax=ax,
            pairs=pairs,
            data=df,
            x=x_var,
            y=y_var,
            order=mc_terms,
        )
    else:
        # Prepare all pairwise comparisons within MC_term (between clinsigs)
        pairs_within = [
            ((mc_term, cat1), (mc_term, cat2))
            for mc_term in mc_terms
            for i, cat1 in enumerate(clinsig_simple_cats)
            for j, cat2 in enumerate(clinsig_simple_cats)
            if j > i
        ]

        # Prepare all pairwise comparisons across MC_terms (between variant types, per clinsig group)
        pairs_across = [
            ((mc1, cln), (mc2, cln))
            for cln in clinsig_simple_cats
            for i, mc1 in enumerate(mc_terms)
            for j, mc2 in enumerate(mc_terms)
            if j > i
        ]

        # Combine both within and across pairs
        pairs = pairs_within + pairs_across

        annotator = Annotator(
            ax=ax,
            pairs=pairs,
            data=df,
            x=x_var,
            y=y_var,
            hue=hue_var,
            order=mc_terms,
            hue_order=clinsig_simple_cats,
        )

    # Match Annotator: independent two-sample t-test + Bonferroni across all pairs
    x_hue_same = x_var == hue_var
    n_comp = len(pairs)

    def _yvals_and_labels_for_pair(pair):
        if x_hue_same:
            c1, c2 = pair
            y1 = df.loc[df[x_var] == c1, y_var].dropna().to_numpy(dtype=float)
            y2 = df.loc[df[x_var] == c2, y_var].dropna().to_numpy(dtype=float)
            return y1, y2, str(c1), str(c2), "between_x_categories"
        (xa, ha), (xb, hb) = pair
        y1 = df.loc[(df[x_var] == xa) & (df[hue_var] == ha), y_var].dropna().to_numpy(
            dtype=float
        )
        y2 = df.loc[(df[x_var] == xb) & (df[hue_var] == hb), y_var].dropna().to_numpy(
            dtype=float
        )
        lab1, lab2 = f"{xa}|{ha}", f"{xb}|{hb}"
        if xa == xb:
            kind = "within_x_between_hue"
        else:
            kind = "across_x_within_hue"
        return y1, y2, lab1, lab2, kind

    _correction = "bonferroni"
    comparisons = []
    for pair in pairs:
        y1, y2, lab1, lab2, kind = _yvals_and_labels_for_pair(pair)
        n1, n2 = len(y1), len(y2)
        row_meta = {
            "test": stats_test,
            "correction": _correction,
            "n_comparisons": n_comp,
            "y_var": y_var,
            "x_var": x_var,
        }
        if n1 < 2 or n2 < 2:
            comparisons.append(
                {
                    **row_meta,
                    "kind": kind,
                    "group1": lab1,
                    "group2": lab2,
                    "n1": n1,
                    "n2": n2,
                    "statistic": float("nan"),
                    "rank_biserial_r": float("nan"),
                    "p": float("nan"),
                    "p_bonferroni": float("nan"),
                    "note": "insufficient_data",
                }
            )
            continue
        stat, p_raw = _pairwise_stat_and_p(y1, y2, stats_test)
        # Rank-biserial r from Mann–Whitney U (same as src/analysis/attributions.py); not defined for other tests.
        if stats_test == "Mann-Whitney" and n1 > 0 and n2 > 0 and np.isfinite(stat):
            rank_biserial_r = 1.0 - (2.0 * stat) / (n1 * n2)
        else:
            rank_biserial_r = float("nan")
        comparisons.append(
            {
                **row_meta,
                "kind": kind,
                "group1": lab1,
                "group2": lab2,
                "n1": n1,
                "n2": n2,
                "statistic": float(stat),
                "rank_biserial_r": float(rank_biserial_r),
                "p": p_raw,
                "p_bonferroni": _p_bonferroni_if_applicable(
                    p_raw, n_comp, "bonferroni"
                ),
            }
        )

    _stats_cols = [
        "test",
        "correction",
        "n_comparisons",
        "y_var",
        "x_var",
        "kind",
        "group1",
        "group2",
        "n1",
        "n2",
        "statistic",
        "rank_biserial_r",
        "p",
        "p_bonferroni",
        "note",
    ]
    stats = pd.DataFrame(comparisons)
    for c in _stats_cols:
        if c not in stats.columns:
            stats[c] = np.nan
    stats = stats[_stats_cols]

    annotator.configure(
        test=stats_test,
        text_format='star',
        loc='inside',  # could also consider 'outside' but inside is default
        comparisons_correction="bonferroni",
    )
    annotator.apply_and_annotate()

    save_kwargs = dict(utils.FIG_SAVE_KWARGS)
    if fig_kwargs is not None:
        save_kwargs.update(fig_kwargs)
    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)

    plt.show()
    return {"fig": fig, "ax": ax, "data": df, "stats": stats}


def plot_score_type_by_clinsig(
    df,
    clinsig_col="clinsig_simple",
    effect_class_col="max_delta_score_type",
    value_col=None,
    count_col="count",
    label_fix_func=None,
    sort_func=None,
    palette=utils.get_clinsig_palette(),
    mutant_col="site",
    x_label="Predicted Variant Effect Class",
    y_label="Counts (Millions)",
    title="Predicted Variant Effect Class Counts",
    legend_title="Clinical Significance",
    legend_loc="upper center",
    legend_bbox_to_anchor=None,
    legend_frameon=False,
    show_legend=True,
    save_path=None,
    show=True,
    fig_kwargs=None,
    barplot_kwargs=None,
    millions_format=True,
    figsize=(5, 5),
    flip_axes=False,
    wrap_labels=False
):
    """
    More configurable plot for score/effect type by clinical significance.

    Args:
        df: DataFrame containing input data.
        clinsig_col: Column for clinical significance categories.
        effect_class_col: Column for score/effect type classes.
        value_col: Optional column to aggregate (if not None, use sum; else counts).
        count_col: Name for the count column in output DataFrame.
        label_fix_func: Function to fix clinsig labels (should return DataFrame, palette).
        sort_func: Function to sort DataFrame by clinsig; if None, uses utils.sort_by_clinsig.
        palette: Color palette or None.
        mutant_col: Used if label_fix_func is called (escaped if not used).
        x_label, y_label, title: Plot axes labels and title.
        legend_title: Legend title.
        legend_loc: Location for the legend.
        legend_bbox_to_anchor: Tuple for bbox_to_anchor (e.g., (1.05, 1) to place outside right).
        legend_frameon: Legend frame on/off.
        show_legend: If False, omit the legend.
        save_path: Output file (if not None).
        show: Whether to show the plot.
        fig_kwargs: Dict of extra kwargs for plt.figure.
        barplot_kwargs: Dict of extra kwargs for sns.barplot.
        millions_format: Whether to format y-axis in millions.
        figsize: Figure size.
        flip_axes: If True, swap x and y axes (horizontal bars).
        wrap_labels: If True, replace spaces with line breaks in variant effect class labels.
        return_data: Whether to return the aggregated data and ax.

    Returns:
        ax (matplotlib Axes), and optionally aggregated data if return_data is True.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns

    # Aggregate counts or values as desired
    groupby_cols = [clinsig_col, effect_class_col]
    if value_col is not None:
        agg_dat = df.groupby(groupby_cols, observed=True)[value_col].sum().reset_index(name=count_col)
    else:
        agg_dat = df.groupby(groupby_cols, observed=True).size().reset_index(name=count_col)

    # Prettify effect class labels
    if effect_class_col in agg_dat.columns:
        agg_dat[effect_class_col] = agg_dat[effect_class_col].astype(str).str.replace("_", " ")

    # Fix clinsig labels and palette if function supplied
    if label_fix_func is not None:
        agg_dat, palette_labeled = label_fix_func(
            agg_dat, clinsig_col=clinsig_col, mutant_col=mutant_col
        )
        hue_col = clinsig_col + "_label"
    else:
        palette_labeled = palette
        hue_col = clinsig_col

    # Sort if a function is provided; try utils.sort_by_clinsig as default
    if sort_func is not None:
        agg_dat = sort_func(agg_dat, clinsig_col=hue_col)
    else:
        try:
            agg_dat = utils.sort_by_clinsig(agg_dat, clinsig_col=hue_col)
        except Exception:
            pass

    plt.figure(figsize=figsize, **(fig_kwargs if fig_kwargs else {}))
    barplot_kwargs = barplot_kwargs or {}

    # Swap axes if flip_axes is True
    if flip_axes:
        barplot_x = count_col
        barplot_y = effect_class_col
        axis_x_label = y_label
        axis_y_label = x_label
    else:
        barplot_x = effect_class_col
        barplot_y = count_col
        axis_x_label = x_label
        axis_y_label = y_label

    ax = sns.barplot(
        data=agg_dat,
        x=barplot_x,
        y=barplot_y,
        hue=hue_col,
        palette=palette_labeled,
        edgecolor="black",
        linewidth=1,
        **barplot_kwargs,
    )

    plt.xlabel(axis_x_label)
    plt.ylabel(axis_y_label)
    
    # Rotate y-axis labels to 0 degrees when axes are flipped
    if flip_axes:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Wrap labels by replacing spaces with line breaks
    if wrap_labels:
        if flip_axes:
            # Labels are on y-axis when flipped
            labels = [label.get_text().replace(" ", "\n") for label in ax.get_yticklabels()]
            ax.set_yticklabels(labels)
        else:
            # Labels are on x-axis when not flipped
            labels = [label.get_text().replace(" ", "\n") for label in ax.get_xticklabels()]
            ax.set_xticklabels(labels)
    
    plt.title(title)
    if show_legend:
        legend_kwargs = {
            'title': legend_title,
            'frameon': legend_frameon,
            'loc': legend_loc,
        }
        if legend_bbox_to_anchor is not None:
            legend_kwargs['bbox_to_anchor'] = legend_bbox_to_anchor
        plt.legend(**legend_kwargs)
    else:
        # Remove legend if it exists
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    sns.despine(ax=ax, top=True, right=True)

    # Option for millions formatting - apply to the count axis (x when flipped, y when not)
    if millions_format:
        def millions_formatter(x, pos):
            if x >= 1e6:
                return f"{x*1e-6:.0f}M"
            elif x < 1e6:
                return f"{x:.0f}"
            else:
                return f"{x*1e-6:.0f}M"
        if flip_axes:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))
        else:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))

    if save_path is not None:
        save_kwargs = dict(utils.FIG_SAVE_KWARGS)
        if fig_kwargs is not None:
            save_kwargs.update(fig_kwargs)
        plt.savefig(save_path, **save_kwargs)
    if show:
        plt.show()

    return {'fig': ax.figure, 'ax': ax, 'data': agg_dat}


def plot_categorical_entropy(
    df,
    groupby_cat="clinsig_simple",
    subgroupby="site",
    categorical_col="max_delta_score_type",
    palette=utils.get_clinsig_palette(),
    figsize=(2.5, 5),
    label_dict=None,
    xlabel="Clinical Significance",
    ylabel="Entropy",
    title="Entropy of\nPredicted Variant Effect Class",
    show_legend=False,
    legend_loc=None,
    legend_bbox_to_anchor=None,
    sort_cat_func=utils.sort_by_clinsig,
    save_path=None,
    add_stats=True,
    stats_test="t-test_ind",
    stats_text_format='star',
    stats_loc='inside',
    stats_correction="bonferroni",
    plot_kwargs=None,
    bar_kwargs=None,
    errorbar_kwargs=None,
    despine_kwargs=None,
    show_xticklabels=True,
    errorbar_capsize=4,
    **kwargs
):
    """
    Plots Shannon entropy of a categorical column, grouped by a chosen category.

    How it works
    -------------
    For each combination of ``groupby_cat`` (e.g. clinical significance) and
    ``subgroupby`` (e.g. site), the function builds the empirical distribution
    of ``categorical_col`` (e.g. predicted variant effect class). It then
    computes the entropy of that distribution. High entropy means predictions
    are spread across many categories (more uncertainty/variety); low entropy
    means predictions concentrate on one or few categories (more agreement).
    The plot shows the mean entropy (and SEM) across the ``subgroupby`` units
    for each ``groupby_cat``, so you can compare how "spread out" predicted
    classes are across clinical significance groups.

    Mathematical formulation
    -----------------------
    For each group (c, s) with category c and subgroup s, let n_k be the count
    of rows in category k of ``categorical_col``. The empirical probabilities
    are p_k = n_k / N with N = sum_k n_k. The entropy (in bits) is:

        H(c, s) = - sum_{k=1}^{K} p_k * log2(p_k)

    with the convention 0*log(0) = 0. This is Shannon entropy in base 2.
    The bar plot displays, for each ``groupby_cat`` value c:

        mean_c = (1/|S_c|) * sum_{s in S_c} H(c, s)
        sem_c  = std(H(c, ·)) / sqrt(|S_c|)

    where S_c is the set of subgroup values for that category.

    Parameters:
    -----------
    df : DataFrame
        Input DataFrame.
    groupby_cat : str
        Name of column for main grouping (e.g. clinical significance label).
    subgroupby : str
        Name of column for sub-grouping (e.g. site/locus/position).
    categorical_col : str
        Column to calculate entropy over.
    palette : dict or None
        Optional, for bar colors.
    figsize : tuple
        Figure dimensions.
    label_dict : dict or None
        Optional, for relabeling categories on plot.
    xlabel, ylabel, title : str
        Plot labels/titles.
    show_legend : bool
        Whether to show legend.
    legend_loc, legend_bbox_to_anchor : as in matplotlib/seaborn.
    sort_cat_func : function or None
        Optional, give function(df, clinsig_col) to sort categories or None for no sorting.
    save_path : str or None
        Optional, file path for saving the plot.
    add_stats : bool
        Whether to add statistical annotation.
    stats_test : str
        Pairwise test for annotations and ``stats`` (must match statannotations).
        Default ``'t-test_ind'`` (Student's independent t-test). Same options as
        ``plot_variant_type_and_clinsig``.
    stats_text_format : str
        Format for stats text.
    stats_loc : str
        Location for annotation.
    stats_correction : str
        Multiple testing correction for statannotations. ``p_bonferroni`` in ``stats`` is
        filled only when this is ``'bonferroni'`` (case-insensitive); otherwise NaN.
    plot_kwargs, bar_kwargs, errorbar_kwargs, despine_kwargs : dict or None
        Additional arguments passed through.
    **kwargs : dict
        Passed to plt.savefig.

    Returns:
    --------
    dict
        Keys ``'fig'``, ``'ax'``, ``'data'`` (mean and sem per ``groupby_cat``), and
        ``'stats'`` (DataFrame of pairwise tests on raw ``entropy`` per subgroup, same
        columns as ``plot_variant_type_and_clinsig`` stats: ``test``, ``correction``,
        ``n_comparisons``, ``y_var``, ``x_var``, ``kind``, ``group1``, ``group2``, ``n1``,
        ``n2``, ``statistic``, ``rank_biserial_r`` (Mann-Whitney only), ``p``,
        ``p_bonferroni``, ``note``).
    """

    from scipy.stats import entropy
    from statannotations.Annotator import Annotator

    plot_kwargs = plot_kwargs or {}
    bar_kwargs = bar_kwargs or {}
    errorbar_kwargs = errorbar_kwargs or {}
    despine_kwargs = despine_kwargs or {}

    _validate_pairwise_stats_test(stats_test)

    # Calculate entropy of predicted class per group
    entropy_df = df.groupby([groupby_cat, subgroupby], observed=True).apply(
        lambda x: entropy(x[categorical_col].value_counts(normalize=True), base=2)
    ).reset_index(name="entropy")
    
    # Sort if required
    if sort_cat_func is not None:
        entropy_df = sort_cat_func(entropy_df, clinsig_col=groupby_cat)

    # Compute mean & sem per group
    agg_df = entropy_df.groupby(groupby_cat, observed=True, sort=False)["entropy"].agg(['mean', 'sem']).reset_index()
    if sort_cat_func is not None:
        agg_df = sort_cat_func(agg_df, clinsig_col=groupby_cat)

    # Palette and relabels
    if palette is None and hasattr(utils, "get_clinsig_palette"):
        palette = utils.get_clinsig_palette()
    if label_dict:
        agg_df[groupby_cat] = agg_df[groupby_cat].map(label_dict).fillna(agg_df[groupby_cat])
        entropy_df[groupby_cat] = entropy_df[groupby_cat].map(label_dict).fillna(entropy_df[groupby_cat])
        if palette: palette = {label_dict.get(k, k): v for k,v in palette.items()} 

    fig = plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=agg_df,
        x=groupby_cat,
        y="mean",
        hue=groupby_cat,
        legend=show_legend,
        edgecolor="black",
        linewidth=1,
        palette=palette,
        **bar_kwargs,
    )
    # Manual error bars to avoid category mismatch
    ax.errorbar(
        x=ax.get_xticks(),
        y=agg_df['mean'],
        yerr=agg_df['sem'],
        fmt='none',
        ecolor='black',
        capsize=errorbar_capsize,
        linewidth=1,
        zorder=10,
        **errorbar_kwargs
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if not show_xticklabels:
        ax.set_xticklabels([])
    if not show_legend:
        ax.get_legend().remove() if ax.get_legend() is not None else None
    elif show_legend and legend_loc is not None:
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)

    sns.despine(ax=ax, top=True, right=True, **despine_kwargs)

    cat_list = agg_df[groupby_cat].drop_duplicates().tolist()
    pairs = [
        (cat1, cat2)
        for i, cat1 in enumerate(cat_list)
        for j, cat2 in enumerate(cat_list)
        if j > i
    ]
    n_comp = len(pairs)
    _entropy_y = "entropy"
    comparisons = []
    for c1, c2 in pairs:
        y1 = entropy_df.loc[entropy_df[groupby_cat] == c1, _entropy_y].dropna().to_numpy(
            dtype=float
        )
        y2 = entropy_df.loc[entropy_df[groupby_cat] == c2, _entropy_y].dropna().to_numpy(
            dtype=float
        )
        n1, n2 = len(y1), len(y2)
        row_meta = {
            "test": stats_test,
            "correction": stats_correction,
            "n_comparisons": n_comp,
            "y_var": _entropy_y,
            "x_var": groupby_cat,
        }
        if n1 < 2 or n2 < 2:
            comparisons.append(
                {
                    **row_meta,
                    "kind": "between_x_categories",
                    "group1": str(c1),
                    "group2": str(c2),
                    "n1": n1,
                    "n2": n2,
                    "statistic": float("nan"),
                    "rank_biserial_r": float("nan"),
                    "p": float("nan"),
                    "p_bonferroni": float("nan"),
                    "note": "insufficient_data",
                }
            )
            continue
        stat, p_raw = _pairwise_stat_and_p(y1, y2, stats_test)
        if stats_test == "Mann-Whitney" and n1 > 0 and n2 > 0 and np.isfinite(stat):
            rank_biserial_r = 1.0 - (2.0 * stat) / (n1 * n2)
        else:
            rank_biserial_r = float("nan")
        comparisons.append(
            {
                **row_meta,
                "kind": "between_x_categories",
                "group1": str(c1),
                "group2": str(c2),
                "n1": n1,
                "n2": n2,
                "statistic": float(stat),
                "rank_biserial_r": float(rank_biserial_r),
                "p": p_raw,
                "p_bonferroni": _p_bonferroni_if_applicable(
                    p_raw, n_comp, stats_correction
                ),
            }
        )

    _entropy_stats_cols = [
        "test",
        "correction",
        "n_comparisons",
        "y_var",
        "x_var",
        "kind",
        "group1",
        "group2",
        "n1",
        "n2",
        "statistic",
        "rank_biserial_r",
        "p",
        "p_bonferroni",
        "note",
    ]
    stats = pd.DataFrame(comparisons)
    for c in _entropy_stats_cols:
        if c not in stats.columns:
            stats[c] = np.nan
    stats = stats[_entropy_stats_cols]

    if add_stats and n_comp > 0:
        annotator = Annotator(
            ax=ax,
            pairs=pairs,
            data=entropy_df,
            x=groupby_cat,
            y=_entropy_y,
            order=cat_list,
        )
        annotator.configure(
            test=stats_test,
            text_format=stats_text_format,
            loc=stats_loc,
            comparisons_correction=stats_correction,
        )
        annotator.apply_and_annotate()

    if save_path is not None:
        plt.savefig(save_path, **getattr(utils, "FIG_SAVE_KWARGS", {}), **kwargs)
    plt.show()

    return {"fig": fig, "ax": ax, "data": agg_df, "stats": stats}

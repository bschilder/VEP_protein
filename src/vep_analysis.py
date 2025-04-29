import os
import glob
import pandas as pd
import numpy as np
import argparse
import pathlib 
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List
# Local imports
import src.config as config
import src.utils as utils
import src.haplosaurus as hs
import src.proteingym as pg
import src.biopython as bp
import src.vep_pipeline as vp

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

def list_vep_files(save_dir = None,
                   scoring_strategy = ["wt-marginals", "masked-marginals", "pseudo-ppl"],
                    save_format = "parquet",
                    as_df=False,
                    verbose=True
                   ):
    
    save_dir = _get_default_save_dir(save_dir)
    
    all_files = []
    for ss in tqdm(scoring_strategy, 
                   desc="Finding VEP files"):
         all_files.extend(glob.glob(
            os.path.join(save_dir, "**", f"{ss}.{save_format}"), 
                         recursive=True))
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
    """
    Merge VEP results from multiple files into a single dataframe.
    
    Args:
        save_dir (str): Directory containing the VEP results
        scoring_strategy (str or list): Scoring strategy to use
        add_model_location (bool): Whether to add the model location to the dataframe
        add_filename (bool): Whether to add the filename to the dataframe

    Returns:
        merged_df (pd.DataFrame): Merged dataframe containing all VEP results
    """
    
    save_dir = _get_default_save_dir(save_dir)

    if rename_model_location_col and not add_model_location:
        print("`add_model_location` must be set to True if `rename_model_location_col` is True.\nSetting `add_model_location` to True.")
        add_model_location = True

    save_dir = os.path.expanduser(save_dir)
    if isinstance(scoring_strategy, str):
        scoring_strategy = [scoring_strategy]
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
        assert len(vep_df)>0, "No VEP data found"
        if add_metadata:
            if all(col in vep_df.columns for col in [col_map['protein'], col_map['mutant']]):
                # Gather additional PGD metadata
                resources_df = pg.get_resources_df()
                pgd_resources = pg.download_resources(resources_df.loc[resources_df['Filename'].isin(['substitutions_raw_clinical.zip', 'indels_raw_clinical.zip'])], 
                                                    include_raw=True)
                assert len(pgd_resources)>0, "No PGD metadata found"

                pgd_subs_raw = pd.read_csv(pgd_resources['substitutions_raw_clinical'][0])
                # Annotate with PGD variants
                vep_df = vep_df.merge(pgd_subs_raw.groupby([col_map['protein'], col_map['mutant']]).head(1),
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

def _add_legend(g, palette=utils.get_clinsig_palette(), loc='upper center',
                 bbox_to_anchor=(0.5, 1.05),
                 top=0.9,
                 ncol=None):
    
    handles = [plt.Rectangle((0,0),1,1, color=palette[label]) for label in palette]
    labels = list(palette.keys())
    if ncol is None:    
        ncol = len(palette)
    g.figure.legend(handles, labels, 
                 loc=loc,
                 bbox_to_anchor=bbox_to_anchor,
                 ncol=ncol) 
    g.figure.subplots_adjust(top=top)  # Adjust spacing for titles 

def plot_vep_density(vep_df, 
                     clinsig_col = 'clinsig',
                     palette = utils.get_clinsig_palette(),
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
                     verbose=True,
                     save_path=None,
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
    vep_df = utils.sort_by_reverse_string(vep_df, 
                                          column='model_location', 
                                          extra_sort_cols=['scoring_strategy', clinsig_col],
                                          ascending=[False, True, True])

    vep_df = _filter_vep_df(vep_df, verbose=verbose) 
    # Create facet grid
    g = sns.FacetGrid(data=vep_df, 
                    col=col,
                    row=row,
                    height=height,
                    aspect=aspect,
                    sharex=sharex,
                    sharey=sharey, 
                    margin_titles=True)

    # Plot KDE
    g.map_dataframe(sns.kdeplot, 
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
    _add_legend(g, palette=palette,  top=legend_y) 
    _rm_subplot_prefixes(g)

    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        
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

def _rm_subplot_prefixes(g):
    g.set_titles(row_template='{row_name}', 
                 col_template='{col_name}')  # Only show model name without prefix
    
def plot_vep_variance(vep_df,
                      groupby_cols = ['model_location','protein','clinsig','mutant','scoring_strategy'],
                      x='clinsig',
                      y='VEP_variance',
                      hue='clinsig',
                      row='model_location',
                      col='scoring_strategy',
                      func=sns.boxplot,
                      palette = utils.get_clinsig_palette(),
                      height=3,
                      aspect=.9,
                      sharex=True,
                      sharey=True,
                      return_df=False,
                      save_path=None,
                      **kwargs):
    # Get filtered data
    vep_df = vep_df.copy()
    vep_variance = vep_df.groupby(groupby_cols)['VEP'].var().reset_index().rename(columns={'VEP':'VEP_variance'})

    # Sort by scoring strategy
    vep_variance = utils.sort_by_reverse_string(vep_variance, 
                                                column='scoring_strategy', 
                                                extra_sort_cols=['model_location','clinsig'],
                                                ascending=[False, True, True])

    g = sns.FacetGrid(data=vep_variance, 
                     row=row,
                     col=col,
                     height=height,
                     aspect=aspect,
                     sharex=sharex,
                     sharey=sharey,
                     margin_titles=True
                     )  

    g.map_dataframe(func,
                    x=x,
                    y=y,
                    hue=hue,
                    palette=palette,
                    **kwargs)

    # Rotate x-axis labels for better readability
    g.set_xticklabels(rotation=45, ha='right')
    
    # Remove subplot titles and add margin titles
    g.figure.suptitle("")  # Remove overall title if any

    _rm_subplot_prefixes(g)
    _add_legend(g, palette=palette) 

    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()

    if return_df:
        return vep_variance


def plot_vep_percentiles(vep_df,
                        is_ref=True,
                        suptitle=None,
                        groupby_cols = ['model_location','protein','clinsig','mutant','scoring_strategy'],
                        x='clinsig',
                        y='VEP_percentile',
                        hue='clinsig',
                        row='model_location',
                        col='scoring_strategy',
                        func=sns.violinplot,
                        palette = utils.get_clinsig_palette(),
                        height=3,
                        aspect=.9,
                        title_y=1,
                        sharex=True,
                        sharey=True,
                        return_df=False,
                        save_path=None,
                        **kwargs):

    if suptitle is None and is_ref:
        suptitle = 'Reference Representativeness'
    elif suptitle is None and not is_ref:
        suptitle = y.replace('_', ' ')

    # Get filtered data
    if y not in vep_df.columns:
        vep_df[y] = vep_df.groupby(groupby_cols)['VEP'].rank(pct=True)*100
    vep_df = vep_df.copy()

    # Filter for REF haplotypes
    if is_ref:
        vep_df = vep_df.loc[vep_df['is_ref']==True]

    # Sort by scoring strategy
    vep_df = utils.sort_by_reverse_string(vep_df, 
                                                column='scoring_strategy', 
                                                extra_sort_cols=['model_location','clinsig'],
                                                ascending=[False, True, True])

    g = sns.FacetGrid(data=vep_df, col=col, row=row, height=height, aspect=aspect, margin_titles=True, sharex=sharex, sharey=sharey)
    g.map_dataframe(func, x=x, y=y, hue=hue, palette=palette, **kwargs)

    # Rotate x-axis labels for better readability
    g.set_xticklabels(rotation=45, ha='right')

    # Remove subplot titles and add margin titles
    g.figure.suptitle(suptitle, y=title_y)  # Remove overall title if any

    _rm_subplot_prefixes(g)

    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path)
        
    plt.show()

    if return_df:
        return vep_df


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

    _rm_subplot_prefixes(g)

    if add_legend:
        labels = vep_pbratios[x].unique()
        palette = _filter_palette(palette, labels)
        _add_legend(g, palette=palette, ncol=1, loc='upper right', bbox_to_anchor=(1.7, 1))

    plt.tight_layout()
    plt.show()

    if return_df:
        return vep_pbratios
    
def add_haplotype_sequence(vep_df,
                           haplotypes: Dict[str, Dict] = None, 
                           encode_haplotype_name_threshold: int = 10,
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
            add_haplotype_names=3) 
        
        if verbose:
            print("Adding 'haplotype_sequence' column")
        vep_df["haplotype_sequence"] = vep_df["haplotype"].map(hap_seqs_flattened)
    if verbose:
        print("Adding 'haplotype_sequence_len' column")
    vep_df['haplotype_sequence_len'] = vep_df['haplotype_sequence'].apply(lambda x: len(x) if pd.notna(x) else np.nan)
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
    model_combos = vep_df.groupby(groupby_cols).size().reset_index()[groupby_cols]

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
        vep_pr = vep_pr.groupby(remaining_cols).agg({'accuracy':'mean','auc': 'mean','coef':'mean','intercept':'mean','n_samples':'sum'}).reset_index()

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
        _rm_subplot_prefixes(g)
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

        _rm_subplot_prefixes(g)

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
                                        top_pop_col = 'top_superpop',
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
        # Remove the prefix from top_superpop for plotting
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
                ax.set_xlabel("Super Population")

    plt.tight_layout()
    return fig


def plot_population_vep_violin_unweighted(df,
                                          mutant=None, 
                                          min_freq=0.005, 
                                          top_pop_col = 'top_superpop',
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

    # Remove the prefix from top_superpop for plotting
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
    g.set_axis_labels("Super Population", "VEP Score")
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
    model_strategy_combos = df.loc[df['mutant_in_haplotype']==False].groupby(['model_location', 'scoring_strategy']).size().reset_index()[['model_location', 'scoring_strategy']]

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
        if 'top_superpop' in filtered_data.columns and not filtered_data['top_superpop'].isna().all():
            pop_scores = filtered_data.pivot_table(
                index='mutant', 
                columns='top_superpop', 
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
    mmr_results_df['mean_rank'] = mmr_results_df.groupby('mutant')['rank'].transform('mean')

    # Sort by mean rank
    mmr_results_df = mmr_results_df.sort_values('mean_rank')
    return mmr_results_df
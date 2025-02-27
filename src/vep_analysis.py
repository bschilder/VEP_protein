import os
import pandas as pd
import numpy as np
import argparse
import pathlib 
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Local imports
import src.config as config
import src.utils as utils
import src.haplosaurus as hs
import src.gprofiler as gp
import src.proteingym as pg
import src.biopython as bp



def merge_vep(save_dir = None,
              scoring_strategy = ["wt-marginals", "masked-marginals", "pseudo-ppl"],
              add_model_location=True,
              rename_model_location_col=True,
              add_variant_set=True,
              add_filename=False,
              add_metadata=True,
              target_namespace='ENST',
              col_map= {'mutant': 'mutant', 
                        'protein': 'protein'},
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
    import pandas as pd
    import glob
    import os
    from tqdm import tqdm
    if save_dir is None:
        save_dir = os.path.join(config.DATA_DIR,"1KG","vep")
        print(f"No save_dir provided, using: {save_dir}")

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
        all_files = glob.glob(
            os.path.join(save_dir, "**", f"{ss}.csv.gz"), 
                         recursive=True)
        if verbose:
            print("Found", len(all_files), ss, "files") 
        if len(all_files) == 0:
            continue
        # Read each file and append to list
        for filename in tqdm(all_files, desc="Reading files"):
            try: 
                # Read the CSV
                df = pd.read_csv(filename, index_col=[0,1])
                df.insert(0, 'haplotype', os.path.dirname(filename).split(os.sep)[-2])
                df['scoring_strategy'] = os.path.basename(filename).split('.')[0]  
                df['is_ref'] = df['haplotype'].str.endswith('REF')
                if add_model_location:
                    model_location = os.path.dirname(filename).split(os.sep)[-4]
                    df['model_location'] = model_location
                    if rename_model_location_col:
                        df.rename(columns={model_location: 'VEP'}, inplace=True)
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
        vep_df = pd.concat(dfs, ignore_index=True)
        if 'ENSP' not in vep_df.columns:
            vep_df['ENSP'] = vep_df['haplotype'].str.split(":").str[0]
        # Map protein IDs to ENST and HGNC
        if target_namespace is not None:
            vep_df = gp.map_ids(vep_df, 
                                rows_per_id=1,
                                target_namespace=target_namespace)
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
            df['sample1_edits'] = utils.count_variants(df['sample1'], 
                                                 tx_id_sep=tx_id_sep)
            df['sample2_edits'] = utils.count_variants(df['sample2'], 
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
                    point_size = 4,
                    point_alpha = 0.5,
                    palette = utils.get_clinsig_palette(),
                    add_connections = False, 
                    connection_alpha = 0.3,
                    connection_linewidth = 1,
                    title_y = 1,
                    freq_filters = {'freq_1000GENOMES:phase_3:ALL':None},
                    figsize=[6, 8],
                    add_side_labels = False,
                    add_stats = False,
                    verbose=True
                    ):
    from scipy import stats  

    vep_df = vep_df.copy()
    y_col = 'VEP'

    # Filter by scoring strategy
    if scoring_strategy is not None:
        scoring_strategy = utils.as_list(scoring_strategy)
        vep_df = vep_df[vep_df['scoring_strategy'].isin(scoring_strategy)]
    # Sort by scoring strategy
    vep_df = utils.sort_by_reverse_string(vep_df, 'scoring_strategy')

    # Filter by max proteins
    if max_proteins is not None:
        vep_df = vep_df.loc[vep_df['protein'].isin(vep_df['protein'].unique()[:max_proteins])]
    
    # Filter by model location
    if model_location is not None:
        model_location = utils.as_list(model_location)
    else:
        model_location = vep_df['model_location'].unique()
    if max_models is not None:
        model_location = model_location[:max_models]
    vep_df = vep_df.loc[vep_df['model_location'].isin(model_location)]
    
    vep_df = _filter_vep_df(vep_df, verbose=verbose) 
     
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
        final_label = _summarise_title(vep_df)  
        
        ax.set_title(f'{final_label}\n{mutant_summary_str}', 
                     y=title_y)
        ax.set_ylabel(f'{", ".join(vep_df["scoring_strategy"].unique())}')
        ax.set_xlabel('Clinical classification')
        
        # Adjust plot margins
        if add_side_labels:
            ax.set_xlim(-2, len(categories)+1)
            ax.set_ylim(y_min - 0.05*y_range, y_max + 0.15*y_range)

    plt.tight_layout()
    plt.show()

def _summarise_mutants(vep_df,
                       clinsig_col='clinsig'):
    mutant_summary = vep_df.groupby(clinsig_col)['mutant'].nunique() 
    mutant_summary_str = ', '.join([f"{k}: {v}" for k,v in mutant_summary.items()]) 
    return mutant_summary_str

def _summarise_title(vep_df,
                     label_cols = ['protein', 'ENST', 'HGNC', 'haplotype']):
    
    labels = {}
    for col in label_cols:
        if col not in vep_df.columns:
            continue
        if vep_df[col].nunique() > 1:
            labels[col] = f"{col}: {vep_df[col].nunique()}"
        else:
            labels[col] = f"{col}: {vep_df[col].iloc[0]}"
    return ', '.join([labels[col] for col in label_cols])

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
    if verbose:
        print(f"Filtered {((rows_before-rows_after)/rows_before)*100:.1f}% of rows with NAs in col 'VEP'")
    return vep_df


def plot_vep_density(vep_df, 
                     clinsig_col = 'clinsig',
                     model_location = None,
                     alpha=.7,
                     figsize=[4,4],
                     verbose=True,
                     **kwargs): 
    # Get filtered data
    vep_df = vep_df.copy()
    if model_location is not None:
        model_location = utils.as_list(model_location)
        vep_df = vep_df.loc[vep_df['model_location'].isin(model_location)]
    else:
        model_location = _get_model_location(vep_df)
    # Sort by scoring strategy
    vep_df = utils.sort_by_reverse_string(vep_df, 
                                          column='scoring_strategy', 
                                          extra_sort_cols=['clinsig'],
                                          ascending=[False, True])
    n_scoring_strategies = vep_df['scoring_strategy'].nunique()

    vep_df = _filter_vep_df(vep_df, verbose=verbose) 
    # Create facet grid
    g = sns.FacetGrid(data=vep_df, 
                    col='scoring_strategy',
                    row='model_location',
                    height=figsize[0],
                    aspect=(figsize[1]*n_scoring_strategies)/figsize[0],
                    sharex=False,
                    sharey=False)

    # Plot KDE
    g.map_dataframe(sns.kdeplot, 
                    x='VEP',
                    hue=clinsig_col,
                    fill=True,
                    palette=utils.get_clinsig_palette(), 
                    alpha=alpha,
                    legend=True,
                    **kwargs)
    
    # Get summary of unique mutants per clinical significance
    mutant_summary_str = _summarise_mutants(vep_df, clinsig_col)
    final_label = _summarise_title(vep_df) 
    # Add title with protein, haplotype and mutant counts 
    g.fig.suptitle(f'{final_label}\n{mutant_summary_str}', y=1.1)


    # Add vertical lines for REF haplotypes
    # for ax in g.axes.flat:
    #     ref_data = esm_vep[(esm_vep['protein'] == np_id) & 
    #                        (esm_vep['is_ref'] == True)]
    #     for _, row in ref_data.iterrows():
    #         ax.axvline(x=row['esm1v_t33_650M_UR90S_1'], 
    #                   color='black', 
    #                   linestyle='--', 
    #                   alpha=0.5)


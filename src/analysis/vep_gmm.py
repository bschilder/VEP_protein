import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import seaborn as sns
from tqdm import tqdm
import pandas as pd

import src.utils as utils
import src.haplosaurus as hs


def train_vep_gmm(vep_df, 
                    clinsig_col="clinsig",
                    vep_col="VEP",
                    clinsig_map={"path": ["path", "pathogenic", "likely_path", "likely_pathogenic"],
                                 "benign": ["benign", "likely_benign"]},
                    groupby_cols=['model_location', 'protein', 'scoring_strategy'],
                    min_variants_per_group=10,
                    plot=True):
    """
    Train Gaussian Mixture Models to find decision boundaries between pathogenic and benign variants
    
    Parameters:
    -----------
    vep_df : pd.DataFrame
        DataFrame containing VEP data
    groupby_cols : list
        List of columns to group by
    plot : bool
        Whether to plot the results
    Returns:
    --------
    pd.DataFrame
        DataFrame containing GMM results
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    # Group data by model, protein, and scoring strategy
    groupby_cols = [x for x in groupby_cols if x in vep_df.columns]
    model_groups = vep_df.groupby(groupby_cols, observed=True)

    all_clinsig_values = [item for sublist in clinsig_map.values() for item in sublist]

    # Store results
    gmm_results = []

    # Process each group
    for groups, group_data in tqdm(model_groups,
                                    desc="Training GMMs",
                                    total=len(model_groups)):
        # Filter for variants with clear clinical significance (benign/likely_benign vs path/likely_path)
        filtered_data = group_data[group_data[clinsig_col].isin(all_clinsig_values)].copy()
        
        # Skip if we don't have enough data or if we don't have both classes
        if len(filtered_data) < min_variants_per_group:
            continue
        
        # Create binary labels (0 for benign, 1 for pathogenic)
        filtered_data.loc[:, 'binary_label'] = filtered_data[clinsig_col].apply(
            lambda x: 1 if x in clinsig_map["path"] else 0
        )
        
        # Check if we have both classes
        if filtered_data['binary_label'].nunique() < 2:
            continue
        
        # Prepare data for GMM - drop NaN values to avoid ValueError
        filtered_data_no_nan = filtered_data.dropna(subset=[vep_col])
        
        # Skip if we don't have enough data after dropping NaNs
        if len(filtered_data_no_nan) < min_variants_per_group:
            continue
        
        X = filtered_data_no_nan[vep_col].values.reshape(-1, 1)
        y_true = filtered_data_no_nan['binary_label'].values
        
        # Train GMM with 2 components
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(X)
            
            # Get predicted cluster assignments
            y_pred_cluster = gmm.predict(X)
            
            # Determine which cluster corresponds to pathogenic variants
            # We'll check which cluster has a higher proportion of pathogenic variants
            cluster_0_path_ratio = np.mean(y_true[y_pred_cluster == 0])
            cluster_1_path_ratio = np.mean(y_true[y_pred_cluster == 1])
            
            # Map clusters to binary labels (0 for benign, 1 for pathogenic)
            if cluster_0_path_ratio > cluster_1_path_ratio:
                y_pred = y_pred_cluster
            else:
                y_pred = 1 - y_pred_cluster
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate AUC if possible
            try:
                # Get probabilities for the pathogenic class
                probs = gmm.predict_proba(X)
                # Determine which column corresponds to the pathogenic class
                if cluster_0_path_ratio > cluster_1_path_ratio:
                    path_prob_col = 0
                else:
                    path_prob_col = 1
                auc = roc_auc_score(y_true, probs[:, path_prob_col])
            except:
                auc = np.nan
            
            # Find the decision boundary
            means = gmm.means_.flatten()
            variances = gmm.covariances_.flatten()
            weights = gmm.weights_
            
            # Store results
            gmm_results.append({
                **dict(zip(groupby_cols, groups)),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'benign_count': sum(y_true == 0),
                'path_count': sum(y_true == 1),
                'mean_0': means[0],
                'mean_1': means[1],
                'var_0': variances[0],
                'var_1': variances[1],
                'weight_0': weights[0],
                'weight_1': weights[1]
            })
        except Exception as e:
            print(f"Error processing {groups}: {str(e)}")
            continue

    # Convert to DataFrame
    gmm_df = pd.DataFrame(gmm_results)

    if plot:
        try:
            plot_vep_gmm(gmm_df, vep_df)
        except Exception as e:
            print(f"Error plotting {groups}: {str(e)}")
            

    return gmm_df


def _get_example_data(gmm_df, vep_df):
    if len(gmm_df) > 0:
       # Get the model with the highest accuracy
        best_model_idx = gmm_df['accuracy'].idxmax()
        best_model = gmm_df.loc[best_model_idx]
        
        # Filter data for this model
        example_data = vep_df[(vep_df['protein'] == best_model['protein']) & 
                            (vep_df['model_location'] == best_model['model']) &
                            (vep_df['scoring_strategy'] == best_model['scoring_strategy']) &
                            (vep_df['clinsig'].isin(['benign', 'likely_benign', 'path', 'likely_path']))]
        
        # Drop NaN values
        example_data = example_data.dropna(subset=['VEP'])
        
        # Create binary labels
        example_data['binary_label'] = example_data['clinsig'].apply(
            lambda x: 1 if x in ['path', 'likely_path'] else 0
        )

        return best_model, example_data
    else:
        print("No valid GMM models could be created. Check your data for sufficient non-NaN values.")


def plot_vep_gmm(gmm_df, vep_df):
    # Display summary
    if len(gmm_df) > 0:
        print(f"Total model-protein-scoring combinations with GMM models: {len(gmm_df)}")
        print(f"Average accuracy: {gmm_df['accuracy'].mean():.4f}")
        print(f"Average AUC: {gmm_df['auc'].mean():.4f}")

        # Display top performing models
        # print("\nTop 10 models by accuracy:")
        # print(gmm_df.sort_values('accuracy', ascending=False).head(10))

        # Visualize an example
        if len(gmm_df) > 0:
            best_model, example_data = _get_example_data(gmm_df, vep_df)
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Plot histograms
            sns.histplot(data=example_data, 
                         x='VEP', 
                         hue='binary_label', 
                         palette={np.int64(0): 'blue', np.int64(1): 'red'}, 
                         bins=30, 
                         element='step', 
                         common_norm=False, 
                         stat='density')
            
            # Plot GMM distributions
            x = np.linspace(example_data['VEP'].min() - 1, example_data['VEP'].max() + 1, 1000)
            
            # Function to calculate Gaussian PDF
            def gaussian_pdf(x, mean, var, weight):
                return weight * np.exp(-(x - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)
            
            # Plot each Gaussian component
            plt.plot(x, gaussian_pdf(x, best_model['mean_0'], best_model['var_0'], best_model['weight_0']), 
                    'k--', linewidth=2, label='GMM Component 1')
            plt.plot(x, gaussian_pdf(x, best_model['mean_1'], best_model['var_1'], best_model['weight_1']), 
                    'k-.', linewidth=2, label='GMM Component 2')
            
            # Add title and labels
            plt.title(f"GMM for {best_model['protein']} ({example_data['mutant'].nunique()} variants)\nModel: {best_model['model']}, Accuracy: {best_model['accuracy']:.4f}, AUC: {best_model['auc']:.4f}")
            plt.xlabel("Variant Effect Prediction (VEP)")
            plt.ylabel("Density")
            plt.legend(title="")
            plt.tight_layout()
            plt.show()
    else:
        print("No valid GMM models could be created. Check your data for sufficient non-NaN values.")


def get_example_data_for_decision_boundary(boundary_crossing_df, vep_df, top_n=3):
    variant_data_list = []
    for i, (_, row) in enumerate(boundary_crossing_df.sort_values('crossing_proportion', ascending=False).head(top_n).iterrows()):
        protein, mutant = row['protein'], row['mutant']
        variant_data = vep_df[(vep_df['protein'] == protein) & (vep_df['mutant'] == mutant)]
        variant_data = hs.add_haplotype_freqs(variant_data, verbose=False) 
        variant_data['Top Superpopulation'] = variant_data['top_superpop'].str.split(':').str[-1]
        variant_data['decision_boundary'] = row['decision_boundary']
        
        variant_data_list.append(variant_data)
    return variant_data_list

def plot_decision_boundaries(boundary_crossing_df,
                             vep_df, 
                             gmm_df, 
                             top_n=3):
    """
    Plot decision boundaries for each protein
    """
    print(f"Found {len(boundary_crossing_df)} variants across {boundary_crossing_df['protein'].nunique()} proteins that cross decision boundaries in different haplotypes")
        
   
    # Filter to only path variants
    # boundary_crossing_df = boundary_crossing_df.loc[boundary_crossing_df['clinsig'] == 'path']
    
    # Display the top variants with the most balanced crossing
    print("\nTop variants with most balanced boundary crossing:")
    print(boundary_crossing_df.sort_values('crossing_proportion', ascending=False).head(10))
    
    # Plot a few examples
    if len(boundary_crossing_df) > 0:
        fig, axes = plt.subplots(min(3, len(boundary_crossing_df)), 1, figsize=(10, 3*min(3, len(boundary_crossing_df))),
                                constrained_layout=True)
        if len(boundary_crossing_df) == 1:
            axes = [axes]  # Make axes iterable if there's only one subplot
        
        # For storing unique handles and labels
        all_handles, all_labels = [], []
        ref_handle_added = False
        variant_data_list = get_example_data_for_decision_boundary(boundary_crossing_df, vep_df, top_n=top_n)
            
        for i, variant_data in enumerate(variant_data_list):
            protein, mutant = variant_data['protein'].iloc[0], variant_data['mutant'].iloc[0]
            variant_data = hs.add_haplotype_freqs(variant_data, verbose=False) 
            variant_data['Top Superpopulation'] = variant_data['top_superpop'].str.split(':').str[-1]

            # Plot VEP scores for each haplotype
            ax = axes[i]
            scatter = sns.scatterplot(data=variant_data, 
                            x='VEP',
                            y="freq_1000GENOMES:phase_3:ALL",
                            hue="Top Superpopulation",
                            palette=utils.get_superpop_palette(),
                            ax=ax, alpha=0.85, 
                            size="top_superpop_freq",
                            sizes=(20, 200)  # Set a fixed range for sizes
                            )
            
            # Add decision boundary line
            ax.axvline(x=variant_data['decision_boundary'].iloc[0], color='red', linestyle='--')
            
            ax.set_title(f"{protein}; {mutant} ({variant_data['clinsig'].iloc[0]}); Haplotypes: {variant_data['haplotype'].nunique()}")
            ax.set_xlabel("VEP Score")
            
            # Add text annotation for decision boundary
            ax.text(variant_data['decision_boundary'].iloc[0] + 0.1, 0.95, f"Decision Boundary: {variant_data['decision_boundary'].iloc[0]:.2f}", 
                transform=ax.get_xaxis_transform(), va='top', color='red')
            
            # Get GMM stats for this protein
            if protein in gmm_df['protein'].to_list():
                gmm_df_protein = gmm_df.loc[gmm_df['protein'] == protein]
                # Add GMM model stats to the plot
                stats_text = (f"GMM Stats\n"
                            f"Accuracy: {gmm_df_protein['accuracy'].iloc[0]:.3f}\n"
                            f"Precision: {gmm_df_protein['precision'].iloc[0]:.3f}\n"
                            f"Recall: {gmm_df_protein['recall'].iloc[0]:.3f}\n"
                            f"AUC: {gmm_df_protein['auc'].iloc[0]:.3f}")
                ax.text(0.02, 1-0.02, stats_text, transform=ax.transAxes, 
                    fontsize=9, va='top', bbox=dict(boxstyle='round', 
                                                        facecolor='white', 
                                                        alpha=0.7))
            
            # Mark reference haplotypes with a black ring
            ref_haplotypes = variant_data[variant_data['is_ref'] == True]
            if not ref_haplotypes.empty:
                ref_scatter = ax.scatter(ref_haplotypes['VEP'], 
                        ref_haplotypes["freq_1000GENOMES:phase_3:ALL"],
                        s=ref_haplotypes["top_superpop_freq"]*100 + 50,  # Slightly larger than the original points
                        facecolors='none', 
                        edgecolors='black', 
                        linewidth=1,
                        label='Reference Haplotype')
                
                # Add the reference haplotype to the legend handles only once
                if not ref_handle_added:
                    all_handles.append(ref_scatter)
                    all_labels.append('Reference Haplotype')
                    ref_handle_added = True
            
            # Get handles and labels for the legend before removing it
            handles, labels = ax.get_legend_handles_labels()
            
            # Remove the size legend entries (they'll be added separately)
            size_handles = []
            size_labels = []
            non_size_handles = []
            non_size_labels = []
            
            for h, l in zip(handles, labels):
                if l not in all_labels and l != "top_superpop_freq":
                    if isinstance(l, float) or l.replace('.', '', 1).isdigit():
                        # This is a size entry
                        size_handles.append(h)
                        size_labels.append(l)
                    else:
                        # This is a regular entry
                        non_size_handles.append(h)
                        non_size_labels.append(l)
                        all_handles.append(h)
                        all_labels.append(l)
            
            # Remove legend from individual subplots
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        
        # Create a single legend for the entire figure
        if all_handles and all_labels:
            fig.legend(all_handles, all_labels, loc='center right', 
                    title='Top Superpopulation', bbox_to_anchor=(1.2, 0.5))
            
            # Add a separate size legend with sorted values
            if size_handles and size_labels:
                # Convert labels to floats and sort
                size_values = [(float(label), handle) for label, handle in zip(size_labels, size_handles)]
                size_values.sort()  # Sort by frequency value
                
                # Create new handles and labels in sorted order
                sorted_size_handles = [item[1] for item in size_values]
                sorted_size_labels = [f"{item[0]:.2f}" for item in size_values]
                
                # Add size legend below the main legend
                size_legend = fig.legend(sorted_size_handles, sorted_size_labels, 
                                        loc='center right', 
                                        title='Top Superpopulation\nFrequency', 
                                        bbox_to_anchor=(1.17, 0.2))
        
        add_arrows = False
        if add_arrows:
            # Add arrows indicating pathogenic and benign directions
            fig.subplots_adjust(bottom=0.15)  # Make room for the arrows
            arrow_ax = fig.add_axes([0.2, 0.02, 0.6, 0.05])  # [left, bottom, width, height]
            arrow_ax.set_xlim(0, 1)
            arrow_ax.set_ylim(0, 1)
            arrow_ax.axis('off')
            
            # Add left arrow (Pathogenic)
            y_pos = 0.25
            arrow_ax.annotate('Pathogenic', xy=(0.1, y_pos), xytext=(0.4, y_pos),
                            arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.5),
                            ha='right', va='center', fontsize=12, fontweight='bold')
            
            # Add right arrow (Benign)
            arrow_ax.annotate('Benign', xy=(0.8, y_pos), xytext=(0.6, y_pos),
                            arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.5),
                            ha='left', va='center', fontsize=12, fontweight='bold')
                    
        plt.tight_layout()
        plt.show()

def get_decision_boundaries(
    gmm_df, 
    vep_df, 
    gene_col="protein",
    mutant_col="mutant",    
    clinsig_col="clinsig", 
    plot=True
):
    """
    Find variants where the VEP crosses the decision boundary for some haplotypes but not others

    Parameters:
    -----------
    gmm_df : pd.DataFrame
        DataFrame containing GMM results
    vep_df : pd.DataFrame
        DataFrame containing VEP data
    plot : bool
        Whether to plot the decision boundaries
    Returns:
    --------
    pd.DataFrame
        DataFrame containing decision boundaries
    """

    if len(gmm_df) == 0:
        raise ValueError("gmm_df has zero rows.")
    # First, we need to get the decision boundary for each protein from the GMM models

    # Create a dictionary to store decision boundaries for each protein
    gene_decision_boundaries = {}
    variant_groups = vep_df.groupby([gene_col, mutant_col])

    # For each protein with a valid GMM model, calculate the decision boundary
    for _, model in gmm_df.iterrows():
        gene = model[gene_col]
        
        # Calculate the decision boundary where the two Gaussian components have equal probability
        # This is where: weight_0 * pdf_0(x) = weight_1 * pdf_1(x)
        # Solving this equation for x gives us the decision boundary
        
        # Extract model parameters
        mean_0, var_0, weight_0 = model['mean_0'], model['var_0'], model['weight_0']
        mean_1, var_1, weight_1 = model['mean_1'], model['var_1'], model['weight_1']
        
        # Skip if any parameter is invalid
        if np.isnan(mean_0) or np.isnan(mean_1) or np.isnan(var_0) or np.isnan(var_1) or var_0 <= 0 or var_1 <= 0:
            continue
        
        # Calculate coefficients for the quadratic equation: ax^2 + bx + c = 0
        a = 1/(2*var_0) - 1/(2*var_1)
        b = mean_1/var_1 - mean_0/var_0
        c = mean_0**2/(2*var_0) - mean_1**2/(2*var_1) + np.log(weight_1/weight_0) + np.log(np.sqrt(var_0)/np.sqrt(var_1))
        
        # If a is very close to zero, the boundary is linear
        if abs(a) < 1e-10:
            if b != 0:
                boundary = -c/b
                gene_decision_boundaries[gene] = boundary
        else:
            # Solve the quadratic equation
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                # Choose the boundary that's between the two means
                x1 = (-b + np.sqrt(discriminant)) / (2*a)
                x2 = (-b - np.sqrt(discriminant)) / (2*a)
                
                # Select the boundary that's between the two means
                if min(mean_0, mean_1) <= x1 <= max(mean_0, mean_1):
                    gene_decision_boundaries[gene] = x1
                elif min(mean_0, mean_1) <= x2 <= max(mean_0, mean_1):
                    gene_decision_boundaries[gene] = x2
                else:
                    # If neither solution is between the means, take the one closer to the midpoint
                    midpoint = (mean_0 + mean_1) / 2
                    gene_decision_boundaries[gene] = x1 if abs(x1 - midpoint) < abs(x2 - midpoint) else x2

    # Now find variants where some haplotypes cross the decision boundary while others don't
    boundary_crossing_variants = []

    for (protein, mutant), variant_data in tqdm(variant_groups, 
                                                total=len(variant_groups), 
                                                desc="Finding boundary-crossing variants"):
        # Skip if we don't have a decision boundary for this protein
        if protein not in gene_decision_boundaries:
            continue
        
        # Get the decision boundary for this protein
        boundary = gene_decision_boundaries[protein]
        
        # Get VEP scores for this variant across different haplotypes
        vep_scores = variant_data['VEP'].dropna().tolist()
        
        # Skip if we don't have at least two valid VEP scores
        if len(vep_scores) < 2:
            continue
        
        # Check if some scores are above the boundary and some are below
        scores_above = [score for score in vep_scores if score > boundary]
        scores_below = [score for score in vep_scores if score <= boundary]
        
        if scores_above and scores_below:
            # This variant crosses the decision boundary
            clinsig = variant_data[clinsig_col].iloc[0]
            
            boundary_crossing_variants.append({
                gene_col:protein,    
                mutant_col:mutant,
                clinsig_col: clinsig,
                'decision_boundary': boundary,
                'min_vep': min(vep_scores),
                'max_vep': max(vep_scores),
                'vep_range': max(vep_scores) - min(vep_scores),
                'haplotype_count': len(vep_scores),
                'scores_above_boundary': len(scores_above),
                'scores_below_boundary': len(scores_below)
            })

    # Convert to DataFrame
    boundary_crossing_df = pd.DataFrame(boundary_crossing_variants)
     # Sort by the proportion of scores that cross the boundary
    print("Adding crossing proportion")
    boundary_crossing_df['crossing_proportion'] = boundary_crossing_df.apply(
        lambda x: min(x['scores_above_boundary'], x['scores_below_boundary']) / x['haplotype_count'], axis=1
    )
    # Display summary of boundary-crossing variants
    if not boundary_crossing_df.empty and plot:
        print("Plotting decision boundaries")
        plot_decision_boundaries(boundary_crossing_df, vep_df, gmm_df)
    else:
        print("No variants found that cross decision boundaries in different haplotypes")
    return boundary_crossing_df


def get_overlapping_variants(vep_df, plot=True):
    """
    Find variants where the VEP crosses the decision boundary for some haplotypes but not others
    """
    # First, group by protein and mutant to get all haplotypes for each variant
    variant_groups = vep_df.groupby(['protein', 'mutant'])

    # Store results
    overlap_results = []

    for (protein, mutant), variant_data in tqdm(variant_groups,
                                                total=len(variant_groups)):
        # Skip if we only have one haplotype for this variant
        if len(variant_data) <= 1:
            continue
        
        # Get the clinical significance of this variant
        clinsig = variant_data['clinsig'].iloc[0]
        
        # Skip if not clearly benign or pathogenic
        if clinsig not in ['benign', 'path']:
            continue
        
        # Get VEP scores for this variant across different haplotypes
        vep_scores = variant_data['VEP'].dropna().tolist()
        
        # Skip if we don't have at least two valid VEP scores
        if len(vep_scores) < 2:
            continue
        
        # Calculate min and max VEP for this variant
        min_vep = min(vep_scores)
        max_vep = max(vep_scores)
        
        # Store the range information
        overlap_results.append({
            'protein': protein,
            'mutant': mutant,
            'clinsig': clinsig,
            'min_vep': min_vep,
            'max_vep': max_vep,
            'vep_range': max_vep - min_vep,
            'haplotype_count': len(vep_scores)
        })

    # Convert to DataFrame
    haplotype_effect_df = pd.DataFrame(overlap_results)

    # Now check for overlaps between benign and pathogenic variants
    if len(haplotype_effect_df) > 0:
        # Get ranges for benign and pathogenic variants
        benign_ranges = haplotype_effect_df[haplotype_effect_df['clinsig'] == 'benign']
        path_ranges = haplotype_effect_df[haplotype_effect_df['clinsig'] == 'path']
        
        # Find proteins that have both benign and pathogenic variants
        common_proteins = set(benign_ranges['protein']) & set(path_ranges['protein'])
        
        # Check for each protein if there's overlap in some cases but not others
        overlap_proteins = []
        
        for protein in common_proteins:
            protein_benign = benign_ranges[benign_ranges['protein'] == protein]
            protein_path = path_ranges[path_ranges['protein'] == protein]
            
            # Get overall min/max for benign and pathogenic
            benign_min = protein_benign['min_vep'].min()
            benign_max = protein_benign['max_vep'].max()
            path_min = protein_path['min_vep'].min()
            path_max = protein_path['max_vep'].max()
            
            # Check if there's overlap in the overall ranges
            overall_overlap = (benign_min <= path_max) and (path_min <= benign_max)
            
            # Check individual variants for non-overlapping cases
            has_non_overlapping = False
            
            for _, benign_var in protein_benign.iterrows():
                for _, path_var in protein_path.iterrows():
                    # Check if these specific variants don't overlap
                    if (benign_var['max_vep'] < path_var['min_vep']) or (path_var['max_vep'] < benign_var['min_vep']):
                        has_non_overlapping = True
                        break
                if has_non_overlapping:
                    break
            
            # If we have overall overlap but also some non-overlapping cases
            if overall_overlap and has_non_overlapping:
                overlap_proteins.append({
                    'protein': protein,
                    'benign_variants': len(protein_benign),
                    'path_variants': len(protein_path),
                    'benign_min': benign_min,
                    'benign_max': benign_max,
                    'path_min': path_min,
                    'path_max': path_max
                })
        
        # Convert to DataFrame
        haplotype_overlap_df = pd.DataFrame(overlap_proteins)
        
        # Display results
        if len(haplotype_overlap_df) > 0:
            print(f"\nFound {len(haplotype_overlap_df)} proteins where VEP of pathogenic variants overlaps with benign variants for some haplotypes but not others:")
            print(haplotype_overlap_df)
            
            # Visualize an example if available
            if len(haplotype_overlap_df) > 0 and plot:
                example_protein = haplotype_overlap_df['protein'].iloc[0]
                
                # Get data for this protein
                protein_data = vep_df[vep_df['protein'] == example_protein]
                
                # Plot distribution by haplotype
                plt.figure(figsize=(12, 6))
                sns.stripplot(data=protein_data, x='haplotype', y='VEP', hue='clinsig', 
                            palette=utils.get_clinsig_palette(), dodge=True)
                plt.title(f"VEP Distribution by Haplotype for {example_protein}")
                plt.xlabel("Haplotype")
                plt.ylabel("Variant Effect Prediction (VEP)")
                plt.xticks(rotation=90)
                plt.legend(title="Clinical Significance")
                plt.tight_layout()
                plt.show()
        else:
            print("\nNo proteins found where VEP of pathogenic variants overlaps with benign variants for some haplotypes but not others.")
    else:
        print("\nInsufficient data to analyze haplotype-specific VEP overlaps.")


def plot_vep_histograms_with_boundaries(
    vep_df,
    boundary_crossing_df=None,
    gene_col="protein",
    mutant_col="mutant",
    target_sites=None,
    bins="auto",
    facet_height=2.5,
    facet_aspect=1.75,
    col_wrap=2,
    facet_col=None,
    title="VEP Distributions with Decision Boundaries",
    x_label="VEP",
    y1_label="Haplotypes",
    y2_label="Density",
    x_offset = 0.01,
    y_offset_factor = 0.725,
    label_yspacing=None,
    sharex=True,
    sharey=False,
    flip_xaxis=False,
    log_yaxis=False,
    palette=["blue", "red"],
    show=True,
    show_superpops=False,
    superpop_height_ratio=0.2,
    superpop_binwidth=None,
    superpop_bins=None,
    superpop_default_bins=30,
    show_superpop_legend=True,
    superpop_legend_x=0.5,
    superpop_legend_y=-0.05,
    superpop_legend_row=1,
    superpop_legend_bottom=True,
    format_variant_p_notation=False,
    show_superpop_legend_title=False,
    superpop_legend_border=False,
    superpop_legend_border_padding=(0.04, 0.015, 0.015, 0.015),
    facet_row_spacing=0.4,
    show_twinx_labels=True,
    facet_title_x=0.5,
    n_xticks=None,
    n_yticks=None,
    show_vertical_lines=True,
    show_vertical_line_labels=True,
    shared_yaxis_label=False,
    shared_yaxis_label_x=-0.25
):
    """
    Plot histograms of VEP scores for selected sites, coloring bars by distance from decision boundary,
    and overlaying KDE and vertical lines for VEP_REF, VEP_mean, and decision boundary.

    Parameters
    ----------
    vep_df : pd.DataFrame
        DataFrame containing VEP scores and related columns.
    target_sites : pd.DataFrame
        A list of sites to plot.
    boundary_crossing_df : pd.DataFrame
        DataFrame with decision boundaries (must have "protein", "mutant", "decision_boundary").
    bins : int
        Number of histogram bins.
    n_facets : int
        Number of facets (sites) to plot.
    facet_height : float
        Height of each facet.
    facet_aspect : float
        Aspect ratio of each facet.
    col_wrap : int
        Number of columns in the facet grid.
    facet_row_spacing : float, default=0.4
        Vertical spacing (padding) between facet rows, as a fraction of the
        average subplot height. Higher values create more space between rows.
        This controls the `hspace` parameter in `plt.subplots_adjust()`.
    show : bool
        Whether to call plt.show() at the end.
    flip_xaxis : bool
        Whether to flip the x-axis.
    show_superpops : bool, default=False
        If True, add a thin row of stacked barplots (colored by superpopulation) 
        underneath each corresponding density/histogram plot.
    superpop_height_ratio : float, default=0.15
        Height of superpopulation subplots relative to main plot height.
        A value of 0.15 means superpop bars are 15% of the main plot height.
    superpop_binwidth : float or None, default=None
        Bin width for superpopulation histograms. If None, automatically calculates
        based on the main plot bins (using wider bins for cleaner stacked bars).
        If specified, uses this exact bin width. Ignored if superpop_bins is specified.
    superpop_bins : int or None, default=None
        Number of bins for superpopulation histograms. If specified, calculates
        binwidth from the maximum VEP range across all facets to ensure visual
        consistency. Takes precedence over superpop_binwidth.
        If None, uses superpop_binwidth or automatic calculation based on
        superpop_default_bins.
    superpop_default_bins : int, default=30
        Default number of bins to use for superpopulation stacked histograms
        when neither superpop_bins nor superpop_binwidth is specified.
        Binwidth will be calculated per-facet based on each facet's VEP range.
        Only used when show_superpops=True and both superpop_bins and
        superpop_binwidth are None.
    show_superpop_legend : bool, default=True
        Whether to show the superpopulation legend. Only used when show_superpops=True.
    superpop_legend_x : float or None, default=0.5
        X position for the superpopulation legend (bbox_to_anchor x-coordinate).
        Value of 0.5 centers the legend horizontally. Values > 1.0 place the legend
        outside the plot boundaries to the right. If None and superpop_legend_bottom=True,
        defaults to 0.5 (centered).
    superpop_legend_y : float or None, default=-0.05
        Y position for the superpopulation legend (bbox_to_anchor y-coordinate).
        Value between 0 and 1 places it within the figure, where 1.0 is the top edge.
        Negative values (e.g., -0.05) place it below the plot boundaries.
        If None, defaults to -0.05 (below the plot).
    superpop_legend_row : int or None, default=None
        Number of rows for the superpopulation legend layout.
        If None, automatically determines:
        - Single row if ≤4 superpopulations
        - Two rows if >4 superpopulations
    superpop_legend_bottom : bool, default=True
        If True, force horizontal single-row layout for the legend.
        Position can still be adjusted with superpop_legend_x and superpop_legend_y.
        If False, use superpop_legend_x, superpop_legend_y, and superpop_legend_row
        for full custom positioning and layout.
    format_variant_p_notation : bool, default=False
        If True, reformat the clinical variant in facet title from format like "C121F"
        to protein notation format like "p.F>121C" (p.alt>positionref).
        Only applies when facet_col is None (default facet label generation).
    show_superpop_legend_title : bool, default=True
        Whether to show the "Superpopulation" title above the legend.
        Only used when show_superpop_legend=True.
    superpop_legend_border : bool, default=False
        Whether to draw a border around the entire legend frame.
        Only used when show_superpop_legend=True.
    superpop_legend_border_padding : tuple, default=(0.04, 0.015, 0.015, 0.015)
        Padding within the legend border as (top, right, bottom, left) in figure coordinates.
        Only used when superpop_legend_border=True and show_superpop_legend=True.
        Default provides extra padding at top (0.04) to accommodate labels above patches
        in single-row layout, and equal padding (0.015) on other sides.
    show_twinx_labels : bool, default=True
        Whether to show the y-axis labels and tick labels on the twinx axis (KDE density axis).
        If False, the twinx axis will still be created for the KDE plot, but labels and ticks
        will be hidden.
    x_offset : float, default=0.01
        Horizontal offset for vertical line labels, as a fraction of the x-axis range.
        For example, 0.1 means the label will be offset by 10% of the x-axis range to the right
        (or left if flip_xaxis=True). This ensures consistent visual spacing across subplots
        with different x-axis scales.
    y_offset_factor : float, default=0.725
        Vertical starting position for the decision-boundary / VEP flag labels,
        expressed in axes fraction coordinates (0–1). Labels are placed starting
        at this fraction of the y-axis and staggered downward.
    label_yspacing : float or None, default=None
        Vertical spacing between successive flag labels (Boundary, VEP_ref, VEP_mean),
        expressed as a fraction of the y-axis in axes coordinates. If None, defaults
        to 0.06.
    facet_title_x : float, default=0.5
        X-position of facet subplot titles in axes coordinates (0-1). 
        Value of 0.5 centers the title horizontally. Values < 0.5 move the title left,
        values > 0.5 move it right.
    n_xticks : int or None, default=None
        Maximum number of x-axis tick labels to display. If None, uses matplotlib's default.
        Uses MaxNLocator to automatically choose nice tick positions.
    n_yticks : int or None, default=None
        Maximum number of y-axis tick labels to display. If None, uses automatic selection
        with MaxNLocator. Uses MaxNLocator to automatically choose nice tick positions.
    show_vertical_lines : bool, default=True
        Whether to show vertical lines for VEP_REF, VEP_mean, and decision_boundary.
    show_vertical_line_labels : bool, default=True
        Whether to show text labels next to the vertical lines.
    shared_yaxis_label : bool, default=False
        If True, only show y-axis label on the leftmost column(s) of subplots,
        centered vertically. If False, show y-axis label on all subplots.
    shared_yaxis_label_x : float, default=-0.2
        X-coordinate position for the shared y-axis label when shared_yaxis_label=True.
        This controls the horizontal padding/position of the label. Negative values
        move the label further left (more padding), positive values move it right.
        The y-coordinate is automatically centered at 0.5.
    Returns
    -------
    dict
        Dictionary with keys 'fig', 'axes', and 'data' containing the figure,
        axes (including superpop axes if show_superpops=True), and the plot DataFrame.
    """
 
    if target_sites is not None:
        plot_df = vep_df.loc[vep_df["site"].isin(target_sites)].copy()
    else:
        plot_df = vep_df.copy()
    # Sort according to site order in target_sites
    plot_df["site"] = pd.Categorical(plot_df["site"], categories=target_sites, ordered=True)
    plot_df = plot_df.sort_values("site")

    if facet_col is not None:
        plot_df["facet_label"] = plot_df[facet_col]
    else:
        # Format variant if requested
        if format_variant_p_notation: 
            
            formatted_variant = plot_df[mutant_col].apply(utils.standardize_variant)
        else:
            formatted_variant = plot_df[mutant_col]
        
        # plot_df["facet_label"] = (
        #     "Gene: " + plot_df["GENEINFO"].str.split(":").str[0]
        #     + ", Variant: " + formatted_variant
        #     + " (Haplotypes: " + plot_df["n_haplotypes"].astype(str) + ")")
        plot_df["facet_label"] = (
            plot_df["GENEINFO"].str.split(":").str[0]
            + " + " + formatted_variant 
            + " (" + plot_df["n_haplotypes"].astype(str) + " haplotypes)")


    if flip_xaxis:
        x_offset = -x_offset

    # Add superpopulation column if show_superpops is True
    if show_superpops:
        # Check if superpopulation column exists, if not try to extract from top_superpopulation
        if "superpopulation" not in plot_df.columns:
            if "top_superpopulation" in plot_df.columns:
                plot_df["superpopulation"] = plot_df["top_superpopulation"].str.split(":").str[-1]
            else:
                # Try to add haplotype frequencies which should include top_superpopulation
                print("Adding haplotype frequencies to extract superpopulation data...")
                # Prefer the common VEP dataframe protein identifier column when available.
                # Falling back to haplosaurus defaults otherwise.
                if "protein" in plot_df.columns:
                    plot_df = hs.add_haplotype_freqs(plot_df, protein_id_col="protein")
                else:
                    plot_df = hs.add_haplotype_freqs(plot_df)
                if "top_superpopulation" in plot_df.columns:
                    plot_df["superpopulation"] = plot_df["top_superpopulation"].str.split(":").str[-1]
                else:
                    raise ValueError("Could not extract superpopulation data. Ensure 'top_superpopulation' column exists or can be added via hs.add_haplotype_freqs()")

    if boundary_crossing_df is not None:
        # Merge with boundary_crossing_df, but also get decision boundaries for all proteins
        # First, create a protein-level decision boundary mapping
        protein_boundaries = boundary_crossing_df.groupby(gene_col)['decision_boundary'].first().to_dict()
        # Add decision boundary for each protein
        plot_df['decision_boundary'] = plot_df[gene_col].map(protein_boundaries)
    
        # Also merge the full boundary_crossing_df for additional info if available
        plot_df = plot_df.merge(boundary_crossing_df, on=[gene_col, mutant_col], how="left", suffixes=('', '_crossing'))

    cmap = mpl.colors.LinearSegmentedColormap.from_list("red_blue", palette)

    # Custom histogram function to color bars along a red-blue continuum based on distance from boundary
    def colored_histplot_with_kde(data, color, **kwargs):
        ax = plt.gca()
        # Get the decision boundary for this facet
        decision_boundary = data["decision_boundary"].iloc[0] if "decision_boundary" in data.columns else None
        # Compute histogram
        values = data["VEP"].dropna().values
        bins_ = kwargs.get("bins", 10)
        # Always use counts, not density
        counts, bin_edges = np.histogram(values, bins=bins_, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Color bars based on decision boundary if available, otherwise use VEP values
        if decision_boundary is not None and not pd.isnull(decision_boundary):
            # Use decision boundary for coloring
            distances = bin_centers - decision_boundary
            if len(distances) > 0:
                max_abs_dist = np.max(np.abs(distances))
                if max_abs_dist == 0:
                    normed = np.zeros_like(distances)
                else:
                    normed = distances / max_abs_dist 
                color_vals = (normed + 1) / 2
                bar_colors = [cmap(val) for val in color_vals]
            else:
                bar_colors = ["gray"] * len(bin_centers)
        else:
            # Fallback: color by VEP values (more negative = red, less negative = blue)
            if len(bin_centers) > 0:
                # For VEP values: vep_min is most negative (right), vep_max is least negative (left)
                vep_min, vep_max = values.min(), values.max()
                if vep_max > vep_min:
                    # Invert so more negative values (closer to vep_min) get higher color values (red)
                    normed_vep = (vep_max - bin_centers) / (vep_max - vep_min)
                else:
                    normed_vep = np.zeros_like(bin_centers)
                bar_colors = [cmap(val) for val in normed_vep]
            else:
                bar_colors = ["gray"] * len(bin_centers)

        # Plot bars
        width = bin_edges[1] - bin_edges[0]
        ax.bar(bin_centers, counts, width=width, color=bar_colors, alpha=0.9, edgecolor=None, align="center", zorder=2)
        # Always set ylabel here (will be removed from non-shared axes later if shared_yaxis_label=True)
        ax.set_ylabel(y1_label)
        ax.set_xlabel(x_label)

        max_count = counts.max() if len(counts) > 0 else 0
        if log_yaxis:
            # Built-in matplotlib log scaling; avoid including 0 in the visible range.
            ax.set_yscale("log")
            ymin = 0.8  # shows count=1 cleanly while avoiding 0 on a log axis
            ymax = max(1.0, float(max_count)) * 1.2
            ax.set_ylim(bottom=ymin, top=ymax)
            # Log ticks are not integers; use matplotlib's log locator/formatter.
            numticks = n_yticks if n_yticks is not None else 5
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=numticks))
            ax.yaxis.set_major_formatter(mticker.LogFormatter(base=10))
        else:
            # Set y-axis ticks to discrete integers only, ensuring 0 is NOT included in labels
            ax.set_ylim(bottom=0, top=max(1, max_count + 0.5))
            # Use MaxNLocator
            nbins_y = n_yticks if n_yticks is not None else 'auto'
            locator = mticker.MaxNLocator(integer=True, prune=None, nbins=nbins_y, steps=[1,2,5,10])
            # Get tick values
            ticks = locator.tick_values(0, max(1, max_count + 0.5))
            # Filter to only include ticks > 0 (exclude 0 from labels but keep y-axis starting at 0)
            ticks = ticks[ticks > 0]
            # Sort and remove duplicates
            ticks = np.unique(np.sort(ticks))
            ax.set_yticks(ticks)
        
        # Set x-axis ticks if n_xticks is specified
        if n_xticks is not None:
            x_locator = mticker.MaxNLocator(nbins=n_xticks)
            ax.xaxis.set_major_locator(x_locator)

        # Add KDE on top, but scale it to match the histogram counts
        if len(values) > 1:
            ax2 = ax.twinx()
            kde = sns.kdeplot(
                values,
                ax=ax2,
                color="gray",
                fill=True,
                linewidth=2,
                bw_adjust=1,
                edgecolor=None,
                common_norm=False,
            )
            if show_twinx_labels:
                ax2.set_ylabel(y2_label, color="gray")
                ax2.tick_params(axis='y', labelcolor='gray')
            else:
                ax2.set_ylabel("")
                ax2.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
            ax2.set_ylim(bottom=0)
            for line in ax2.get_lines():
                if line.get_color() == "gray":
                    line.set_alpha(0.2)
            for c in ax2.collections:
                if hasattr(c, "get_facecolor") and c.get_facecolor().shape[0] > 0:
                    c.set_alpha(0.2)

    # Add vertical lines at the VEP_REF (black), VEP_mean (goldenrod), and decision_boundary (red) values for each site
    def add_vep_ref_and_mean_lines(data, color, label_yspacing=None, **kwargs):
        """
        Add vertical lines for VEP_REF, VEP_mean, and decision_boundary, with labels.
        You can set label_yspacing (float, fraction of y-axis) to control label vertical increments (default: 0.06).
        Now with a white background rectangle behind each label (alpha=0.25).
        """
        ax = plt.gca()
        vep_mean = data["VEP_mean"].iloc[0] if "VEP_mean" in data.columns else None
        vep_ref = data["VEP_REF"].iloc[0] if "VEP_REF" in data.columns else None
        decision_boundary = data["decision_boundary"].iloc[0] if "decision_boundary" in data.columns else None

        # Place label y-positions using axes-fraction coordinates (independent of linear/log scale).
        # Start at y_offset_factor (0–1) and step downward.
        y_start_frac = y_offset_factor
        # label_yspacing is interpreted as a fraction of the y-axis (default: 0.06)
        dy_frac = label_yspacing if label_yspacing is not None else 0.06

        # Calculate relative x_offset based on x-axis range to ensure consistent visual spacing
        # across subplots with different x-axis scales
        # x_offset is interpreted as a fraction of the x-axis range (e.g., 0.1 = 10% of range)
        xlim = ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        relative_x_offset = x_offset * x_range if x_range > 0 else 0

        halign = "left"
        valign = "center"
        xlabels = []

        # Prepare label info in top-down order
        if decision_boundary is not None and pd.notnull(decision_boundary):
            xlabels.append((decision_boundary, r"Boundary", "red"))
        if vep_ref is not None and pd.notnull(vep_ref):
            xlabels.append((vep_ref, r"$VEP_{\text{ref}}$", "grey"))
        if vep_mean is not None and pd.notnull(vep_mean):
            xlabels.append((vep_mean, r"$VEP_{\text{mean}}$", "goldenrod"))
      

        # Draw lines and staggered labels from top downward
        for i, (xpos, label, color) in enumerate(xlabels):
            if show_vertical_lines:
                ax.axvline(xpos, color=color, linestyle="--", label=label)
            if show_vertical_line_labels:
                ax.text(
                    xpos + relative_x_offset,
                    y_start_frac - dy_frac * i,
                    label,
                    color=color,
                    rotation=0,
                    va=valign,
                    ha=halign,
                    fontsize="medium",
                    transform=ax.get_xaxis_transform(),
                    bbox=dict(facecolor="white", alpha=.95, edgecolor=color, boxstyle="round,pad=0.1")
                )

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    g = sns.FacetGrid(
        plot_df,
        col="facet_label",
        sharex=sharex,
        sharey=sharey,
        height=facet_height,
        aspect=facet_aspect,
        col_wrap=col_wrap
    )
    g.map_dataframe(colored_histplot_with_kde, x="VEP", bins=bins)
    g.map_dataframe(add_vep_ref_and_mean_lines, label_yspacing=label_yspacing)
    g.set_titles(col_template="{col_name}")

    # Handle shared y-axis label if requested
    if shared_yaxis_label:
        # Get all axes - FacetGrid.axes is typically a numpy array
        # Convert to list and filter None values
        all_axes_list = []
        if hasattr(g, 'axes'):
            # FacetGrid.axes can be 1D or 2D numpy array
            axes_array = np.asarray(g.axes)
            all_axes_list = [ax for ax in axes_array.flat if ax is not None]
        
        if len(all_axes_list) > 0:
            # Determine which axis should get the shared label
            # For a single shared label, use the middle axis (or first if only one)
            if col_wrap == 1:
                # Single column - use the middle axis for the shared label
                middle_idx = len(all_axes_list) // 2
                shared_axis = all_axes_list[middle_idx]
            else:
                # Multiple columns - use the middle row of the leftmost column
                # Leftmost column axes are at indices: 0, col_wrap, 2*col_wrap, etc.
                leftmost_indices = list(range(0, len(all_axes_list), col_wrap))
                if leftmost_indices:
                    middle_idx = len(leftmost_indices) // 2
                    shared_axis = all_axes_list[leftmost_indices[middle_idx]]
                else:
                    shared_axis = all_axes_list[0]
            
            # Remove ylabels from all axes
            for ax in all_axes_list:
                if ax is not None:
                    ax.set_ylabel("")
            
            # Set ylabel only on the shared axis, centered vertically
            if shared_axis is not None:
                shared_axis.set_ylabel(y1_label)
                # Center the label vertically (x position controlled by shared_yaxis_label_x, y centered at 0.5)
                shared_axis.yaxis.set_label_coords(shared_yaxis_label_x, 0.5)

    # Adjust facet title x-positions
    for ax in g.axes.flatten():
        if ax.title.get_text():  # Only adjust if title exists
            # Get current title position (y-coordinate) and set new x-position
            current_y = ax.title.get_position()[1]
            ax.title.set_position((facet_title_x, current_y))

    # Fix for x-axis inversion not working in some cases
    for ax in g.axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for twin_ax in ax.figure.axes:
            if twin_ax is not ax:
                if twin_ax.get_position().bounds == ax.get_position().bounds:
                    twin_ax.spines['top'].set_visible(False)
                    twin_ax.spines['right'].set_visible(False)

        if flip_xaxis:
            xlim = ax.get_xlim()
            if xlim[0] < xlim[1]:
                ax.set_xlim(xlim[1], xlim[0])

    if title is not None:
        g.fig.suptitle(title, fontsize="large")
        # Adjust layout to make room for suptitle
        plt.subplots_adjust(top=0.88)
    
    # Adjust bottom margin early if legend will be below the plot
    # This needs to happen BEFORE tight_layout so superpop axes are positioned correctly
    if show_superpops and show_superpop_legend:
        # Check if legend will be at bottom (either via superpop_legend_bottom or negative y)
        will_be_bottom = (superpop_legend_bottom or 
                         (superpop_legend_y is not None and superpop_legend_y < 0) or
                         (superpop_legend_y is None))  # Default is below
        if will_be_bottom:
            # Legend will be below - make room for it early
            current_bottom = g.fig.subplotpars.bottom
            legend_space = 0.1  # ~10% of figure height for legend
            new_bottom = max(0.05, current_bottom - legend_space)
            plt.subplots_adjust(bottom=new_bottom)
    
    # Do initial tight_layout to establish main axes positions
    plt.tight_layout()
    # Add padding between facet rows
    plt.subplots_adjust(hspace=facet_row_spacing)
    
    # Add superpopulation stacked barplots below each facet if requested
    superpop_axes = []
    # Get the figure and existing axes (needed in both branches)
    fig = g.figure
    main_axes = g.axes.flatten()
    
    if show_superpops:
        # Get superpopulation palette
        superpop_palette = utils.get_superpop_palette()
        
        # Create a mapping from facet_label to data
        facet_labels = plot_df["facet_label"].unique()
        
        # Pre-calculate a standardized binwidth for visual consistency across facets
        # This ensures bins appear the same width visually regardless of x-axis range
        standardized_binwidth = None
        if superpop_bins is not None:
            # Calculate binwidth based on the maximum range across ALL facets
            # This ensures visual consistency - all facets use the same absolute binwidth
            all_vep_ranges = []
            for facet_label in facet_labels:
                facet_data = plot_df[plot_df["facet_label"] == facet_label]
                facet_data_filtered = facet_data.loc[
                    (facet_data["superpopulation"] != "REF") & 
                    (facet_data["superpopulation"].notna())
                ]
                if len(facet_data_filtered) > 0:
                    vep_min = facet_data_filtered["VEP"].min()
                    vep_max = facet_data_filtered["VEP"].max()
                    if vep_max > vep_min:
                        all_vep_ranges.append(vep_max - vep_min)
            
            if all_vep_ranges:
                max_range = max(all_vep_ranges)
                standardized_binwidth = max_range / superpop_bins
            else:
                standardized_binwidth = 0.1  # Fallback
        elif superpop_binwidth is not None:
            # Use user-specified fixed binwidth (same for all facets)
            standardized_binwidth = superpop_binwidth
        
        # Store original positions and calculate new layout
        # We'll create superpop axes below each main axis using the specified height ratio
        
        # For each main axis, create a corresponding superpop axis below it
        for idx, main_ax in enumerate(main_axes):
            if idx >= len(facet_labels):
                break
                
            facet_label = facet_labels[idx]
            facet_data = plot_df[plot_df["facet_label"] == facet_label]
            
            # Get position of main axis (after tight_layout)
            pos = main_ax.get_position()
            
            # Create new axis below the main one
            # Calculate new position: same left, right, but lower and thinner
            new_bottom = pos.y0 - pos.height * superpop_height_ratio
            new_height = pos.height * superpop_height_ratio * 0.8  # Slightly smaller to avoid overlap (80% of ratio)
            
            superpop_ax = fig.add_axes([pos.x0, new_bottom, pos.width, new_height])
            superpop_axes.append(superpop_ax)
            
            # Filter out REF if present
            facet_data_filtered = facet_data.loc[
                (facet_data["superpopulation"] != "REF") & 
                (facet_data["superpopulation"].notna())
            ].copy()
            
            # Calculate binwidth for this specific facet
            if len(facet_data_filtered) > 0:
                # Get VEP range for this facet
                vep_min_facet = facet_data_filtered["VEP"].min()
                vep_max_facet = facet_data_filtered["VEP"].max()
                
                if standardized_binwidth is not None:
                    # Use the standardized binwidth calculated above (ensures visual consistency)
                    binwidth_superpop = standardized_binwidth
                else:
                    # Calculate binwidth per-facet based on this facet's range
                    # This ensures appropriate binning for each facet's data distribution
                    if vep_max_facet > vep_min_facet:
                        binwidth_superpop = (vep_max_facet - vep_min_facet) / superpop_default_bins
                    else:
                        binwidth_superpop = 0.1
                
                # Plot stacked histogram by superpopulation
                sns.histplot(
                    facet_data_filtered,
                    x="VEP",
                    hue="superpopulation",
                    palette=superpop_palette,
                    multiple="fill",
                    binwidth=binwidth_superpop,
                    legend=False,
                    ax=superpop_ax,
                    edgecolor='black',
                    linewidth=0.25  # Half the previous thickness
                )
                
                # Style the superpop axis
                # Get font size from main axis labels and tick labels to match them
                main_ylabel = main_ax.get_ylabel()
                main_xlabel = main_ax.get_xlabel()
                # Get font size from main axis y-label
                main_ylabel_obj = main_ax.yaxis.label
                if main_ylabel_obj:
                    main_label_fontsize = main_ylabel_obj.get_fontsize()
                else:
                    main_label_fontsize = 10  # Default fallback
                
                main_ytick_labels = main_ax.get_yticklabels()
                if len(main_ytick_labels) > 0:
                    main_tick_fontsize = main_ytick_labels[0].get_fontsize()
                else:
                    main_tick_fontsize = 10  # Default fallback
                
                superpop_ax.set_ylabel("Prop.", fontsize=main_label_fontsize)
                superpop_ax.set_xlabel("")
                superpop_ax.tick_params(labelsize=main_tick_fontsize)
                # Set y-axis to only show tick label at 0.5
                superpop_ax.set_yticks([0.5])
                superpop_ax.spines['top'].set_visible(False)
                superpop_ax.spines['right'].set_visible(False)
                
                # Set x-axis limits to match main axis
                superpop_ax.set_xlim(main_ax.get_xlim())
                
                # Remove x-axis tick labels except for the bottom row
                # We'll handle this after all axes are created
                if flip_xaxis:
                    xlim = superpop_ax.get_xlim()
                    if xlim[0] < xlim[1]:
                        superpop_ax.set_xlim(xlim[1], xlim[0])
            else:
                # No data to plot, just style the empty axis
                # Get font size from main axis labels and tick labels to match them
                main_ylabel_obj = main_ax.yaxis.label
                if main_ylabel_obj:
                    main_label_fontsize = main_ylabel_obj.get_fontsize()
                else:
                    main_label_fontsize = 10  # Default fallback
                
                main_ytick_labels = main_ax.get_yticklabels()
                if len(main_ytick_labels) > 0:
                    main_tick_fontsize = main_ytick_labels[0].get_fontsize()
                else:
                    main_tick_fontsize = 10  # Default fallback
                
                superpop_ax.set_ylabel("Prop.", fontsize=main_label_fontsize)
                superpop_ax.set_xlabel("")
                superpop_ax.tick_params(labelsize=main_tick_fontsize)
                # Set y-axis to only show tick label at 0.5
                superpop_ax.set_yticks([0.5])
                superpop_ax.spines['top'].set_visible(False)
                superpop_ax.spines['right'].set_visible(False)
                superpop_ax.set_xlim(main_ax.get_xlim())
        
        # Since each facet has different scales (sharex=False), show x-axis tick labels on ALL facets
        # But only show axis titles on the bottom row
        # Calculate which superpop axes are in the bottom row
        n_main = len(main_axes)
        n_rows = (n_main + col_wrap - 1) // col_wrap  # Ceiling division
        bottom_row_start = (n_rows - 1) * col_wrap
        
        # Get font size from main axis x-label to match it
        main_axes_xlabel_obj = main_axes[0].xaxis.label if len(main_axes) > 0 else None
        if main_axes_xlabel_obj:
            main_xlabel_fontsize = main_axes_xlabel_obj.get_fontsize()
        else:
            main_xlabel_fontsize = 10  # Default fallback
        
        for idx, superpop_ax in enumerate(superpop_axes):
            if idx >= bottom_row_start:
                # Bottom row - show axis title with same font size as main axes
                superpop_ax.set_xlabel(x_label, fontsize=main_xlabel_fontsize)
            else:
                # Not bottom row - no axis title, but tick labels will be visible
                superpop_ax.set_xlabel("")
        
        # Remove x-axis labels and tick labels from main axes (since superpop axes will have tick labels)
        for main_ax in main_axes:
            main_ax.set_xlabel("")
            main_ax.tick_params(labelbottom=False)  # Hide x-tick labels
            # Also hide x-tick labels on any twin axes (e.g., KDE plot axes)
            for twin_ax in main_ax.figure.axes:
                if twin_ax is not main_ax:
                    if twin_ax.get_position().bounds == main_ax.get_position().bounds:
                        twin_ax.tick_params(labelbottom=False)  # Hide x-tick labels on twin axes
        
        # Add a single legend for superpopulations in the upper right
        if show_superpop_legend:
            # Get all unique superpopulations from the data (excluding REF)
            all_superpops = plot_df.loc[
                (plot_df["superpopulation"] != "REF") & 
                (plot_df["superpopulation"].notna())
            ]["superpopulation"].unique()
            
            if len(all_superpops) > 0:
                # Sort superpops for consistent legend order
                all_superpops = sorted(all_superpops)
                
                # Create legend handles with border lines around each rectangle
                handles = [plt.Rectangle((0, 0), 1, 1, 
                                         facecolor=superpop_palette.get(sp, "gray"),
                                         edgecolor='black',
                                         linewidth=0.25)  # Half the previous thickness
                          for sp in all_superpops]
                labels = all_superpops
                
                # Determine legend position and layout based on superpop_legend_bottom
                if superpop_legend_bottom:
                    # Force horizontal single-row layout, but use x/y for positioning
                    # Use custom x/y positioning if provided, otherwise use defaults
                    legend_x = superpop_legend_x if superpop_legend_x is not None else 0.5  # Default: center horizontally
                    if superpop_legend_y is not None:
                        legend_y = superpop_legend_y
                    else:
                        legend_y = -0.05  # Default: below the plot
                    # Force single row for horizontal bottom layout
                    ncol = len(labels)  # All items in one row
                    is_single_row = True
                else:
                    # Use custom x/y positioning
                    legend_x = superpop_legend_x
                    if superpop_legend_y is not None:
                        legend_y = superpop_legend_y
                    else:
                        # Default: position below the plot (negative y is below figure)
                        legend_y = -0.05
                    # Use existing row/column logic
                    if superpop_legend_row is not None:
                        ncol = (len(labels) + superpop_legend_row - 1) // superpop_legend_row
                        is_single_row = (superpop_legend_row == 1)
                    else:
                        # Auto-determine: single row if ≤4 superpops, two rows if >4
                        if len(labels) <= 4:
                            ncol = len(labels)  # Single row
                            is_single_row = True
                        else:
                            ncol = (len(labels) + 1) // 2  # Two rows
                            is_single_row = False
                
                # Adjust legend parameters for single-row layout
                if is_single_row:
                    # Reduce horizontal spacing between items
                    columnspacing = 0.5  # Reduced from default ~1.0
                    # Use empty labels initially, we'll add text above
                    legend_labels = [''] * len(labels)
                else:
                    columnspacing = 1.0  # Default spacing
                    legend_labels = labels
                
                # Add legend to figure
                # Use 'upper center' loc to center the legend horizontally
                legend_kwargs = {
                    'handles': handles,
                    'labels': legend_labels,
                    'loc': 'upper center',
                    'bbox_to_anchor': (legend_x, legend_y),
                    'ncol': ncol,
                    'columnspacing': columnspacing,
                    'frameon': superpop_legend_border,
                    'fancybox': False,
                    'shadow': False
                    # fontsize not specified - uses matplotlib rcParams default
                }
                
                # Add title only if requested
                if show_superpop_legend_title:
                    legend_kwargs['title'] = 'Superpopulation'
                    # title_fontsize not specified - uses matplotlib rcParams default
                
                legend = fig.legend(**legend_kwargs)
                
                # For single-row layout, add labels above the patches
                label_texts = []
                if is_single_row:
                    # Force a layout calculation to get accurate patch positions
                    fig.canvas.draw_idle()
                    
                    # Get the actual legend bounding box in figure coordinates
                    legend_bbox = legend.get_window_extent(fig.canvas.get_renderer())
                    legend_bbox_fig = legend_bbox.transformed(fig.transFigure.inverted())
                    
                    # Get the individual patch positions
                    legend_patches = legend.get_patches()
                    if len(legend_patches) == len(labels):
                        # Calculate positions from actual patch locations
                        for patch, label in zip(legend_patches, labels):
                            # Get patch bounding box in display coordinates
                            patch_bbox = patch.get_window_extent(fig.canvas.get_renderer())
                            
                            # Convert to figure coordinates
                            patch_bbox_fig = patch_bbox.transformed(fig.transFigure.inverted())
                            
                            # Get center x position of the patch
                            patch_center_x = (patch_bbox_fig.x0 + patch_bbox_fig.x1) / 2
                            
                            # Position label above the patch (slightly above the top)
                            patch_top_y = patch_bbox_fig.y1
                            label_y = patch_top_y + 0.01  # Small offset above patch
                            
                            # Add text label centered above the patch
                            text_obj = fig.text(patch_center_x, label_y, label,
                                               ha='center', va='bottom',
                                               fontsize=8, transform=fig.transFigure)
                            label_texts.append(text_obj)
                    else:
                        # Fallback: use estimated positions if patch count doesn't match
                        n_items = len(labels)
                        # Use the legend bbox to estimate spacing
                        legend_width = legend_bbox_fig.width
                        patch_width_est = legend_width / (n_items + (n_items - 1) * columnspacing)
                        spacing_est = patch_width_est * columnspacing
                        
                        # Calculate starting position
                        start_x = legend_bbox_fig.x0 + patch_width_est / 2
                        label_y = legend_bbox_fig.y1 + 0.01
                        
                        # Add text labels above each patch
                        for i, label in enumerate(labels):
                            x_pos = start_x + i * (patch_width_est + spacing_est)
                            text_obj = fig.text(x_pos, label_y, label,
                                               ha='center', va='bottom',
                                               fontsize=8, transform=fig.transFigure)
                            label_texts.append(text_obj)
                else:
                    # Left-justify the legend text for multi-row layout
                    for t in legend.get_texts():
                        t.set_ha('left')
                
                # Configure border and background fill based on argument
                if superpop_legend_border:
                    # Show border with default styling
                    legend.get_frame().set_linewidth(1)
                    legend.get_frame().set_edgecolor('black')
                    legend.get_frame().set_facecolor('white')
                    
                    # If single-row with labels above, expand frame to include them
                    if is_single_row and label_texts:
                        # Redraw to get updated positions after labels are added
                        fig.canvas.draw_idle()
                        
                        # Get bounding box of legend patches
                        legend_bbox = legend.get_window_extent(fig.canvas.get_renderer())
                        legend_bbox_fig = legend_bbox.transformed(fig.transFigure.inverted())
                        
                        # Get bounding box of all label texts
                        from matplotlib.transforms import Bbox
                        label_bboxes = [t.get_window_extent(fig.canvas.get_renderer()) 
                                       for t in label_texts]
                        if label_bboxes:
                            label_bbox = Bbox.union([b for b in label_bboxes])
                            label_bbox_fig = label_bbox.transformed(fig.transFigure.inverted())
                            
                            # Calculate expanded bounds to include labels above
                            # Use padding tuple (top, right, bottom, left)
                            padding_top, padding_right, padding_bottom, padding_left = superpop_legend_border_padding
                            
                            # Calculate the union of legend and label bounding boxes
                            combined_left = min(legend_bbox_fig.x0, label_bbox_fig.x0)
                            combined_right = max(legend_bbox_fig.x1, label_bbox_fig.x1)
                            combined_bottom = legend_bbox_fig.y0  # Bottom is the legend patches
                            combined_top = label_bbox_fig.y1  # Top is the label text
                            
                            # Apply padding from tuple
                            new_left = combined_left - padding_left
                            new_right = combined_right + padding_right
                            new_bottom = combined_bottom - padding_bottom
                            new_top = combined_top + padding_top
                            
                            # Update frame bounds to include labels
                            # Manually create a border rectangle that encompasses everything
                            from matplotlib.patches import Rectangle
                            
                            legend_frame = legend.get_frame()
                            frame_width = new_right - new_left
                            frame_height = new_top - new_bottom
                            
                            # Hide the original legend frame
                            legend_frame.set_visible(False)
                            
                            # Create a custom border rectangle in figure coordinates
                            border_rect = Rectangle(
                                (new_left, new_bottom), frame_width, frame_height,
                                transform=fig.transFigure,
                                linewidth=1,
                                edgecolor='lightgray',
                                facecolor='white',
                                zorder=legend_frame.get_zorder() - 1  # Behind legend content
                            )
                            fig.patches.append(border_rect)
                else:
                    # No border or background fill
                    legend.get_frame().set_linewidth(0)
                    legend.get_frame().set_facecolor('none')
                    legend.get_frame().set_edgecolor('none')
                
                # Adjust spacing between title and legend items to avoid overlap (only if title is shown)
                if show_superpop_legend_title:
                    if hasattr(legend, '_legend_box'):
                        # Increase the separator between title and legend items
                        legend._legend_box.sep = 15  # Increase vertical spacing (default is typically 5-7)
                    
                    # Additionally, adjust the title's vertical position upward
                    title = legend.get_title()
                    if title:
                        # Get current title position and move it up
                        title.set_y(title.get_position()[1] + 0.02)  # Move title up by 0.02 in normalized coordinates
                
                # Adjust layout to make room for external legend if it's outside plot boundaries
                # Note: Bottom margin was already adjusted earlier if legend is below
                if legend_x > 1.0:
                    # Legend is on the right side - adjust right margin
                    current_right = fig.subplotpars.right
                    legend_space = 0.15
                    new_right = max(0.75, current_right - legend_space)
                    fig.subplots_adjust(right=new_right)
                # Bottom margin adjustment was already done earlier before superpop axes creation
    else:
        # No superpop plots - show x-axis tick labels on all main axes
        # But only show axis titles on the bottom row
        # Calculate which main axes are in the bottom row
        n_main = len(main_axes)
        n_rows = (n_main + col_wrap - 1) // col_wrap  # Ceiling division
        bottom_row_start = (n_rows - 1) * col_wrap
        
        for idx, main_ax in enumerate(main_axes):
            if idx >= bottom_row_start:
                # Bottom row - show axis title
                main_ax.set_xlabel(x_label, fontsize='medium')
            else:
                # Not bottom row - no axis title, but tick labels will be visible
                main_ax.set_xlabel("")
    
    if show:
        plt.show()

    # Return both main axes and superpop axes
    if show_superpops:
        all_axes = list(g.axes.flatten()) + superpop_axes
    else:
        all_axes = list(g.axes.flatten())
    return {'fig':g.figure, 'axes':all_axes, 'data':plot_df}
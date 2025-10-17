import numpy as np
import matplotlib.pyplot as plt
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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as mticker

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
    y1_label="Haplotype Count",
    y2_label="Density",
    x_offset = 0.1,
    y_offset_factor = 0.725,
    sharex=True,
    sharey=False,
    flip_xaxis=False,
     palette=["blue", "red"],
    show=True
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
    show : bool
        Whether to call plt.show() at the end.
    flip_xaxis : bool
        Whether to flip the x-axis.
    Returns
    -------
    g : sns.FacetGrid
        The FacetGrid object.
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
        plot_df["facet_label"] = (
        "Gene: " + plot_df["GENEINFO"].str.split(":").str[0]
        + " Clinical Variant: " + plot_df[mutant_col]
        + " (Haplotypes: " + plot_df["n_haplotypes"].astype(str) + ")")


    if flip_xaxis:
        x_offset = -x_offset


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
        ax.set_ylabel(y1_label)
        ax.set_xlabel(x_label)

        # Set y-axis ticks to discrete integers only
        max_count = counts.max() if len(counts) > 0 else 0
        ax.set_ylim(bottom=0, top=max(1, max_count + 0.5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, prune=None, nbins='auto', steps=[1,2,5,10]))

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
            ax2.set_ylabel(y2_label, color="gray")
            ax2.tick_params(axis='y', labelcolor='gray')
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

        y_offset = ax.get_ylim()[1] * y_offset_factor

        # Place y-offsets starting from the top and moving downward
        ymax = ax.get_ylim()[1]*.95
        # Allow custom spacing for y increment, otherwise default to 0.06
        dy_frac = label_yspacing if label_yspacing is not None else 0.15
        delta_y = ymax * dy_frac  # Stagger distance down from the top

        halign = "left"
        valign = "center"
        xlabels = []

        # Prepare label info in top-down order
        if decision_boundary is not None and pd.notnull(decision_boundary):
            xlabels.append((decision_boundary, r"Boundary", "red"))
        if vep_ref is not None and pd.notnull(vep_ref):
            xlabels.append((vep_ref, r"$VEP_{ref}$", "grey"))
        if vep_mean is not None and pd.notnull(vep_mean):
            xlabels.append((vep_mean, r"$VEP_{mean}$", "goldenrod"))
      

        # Draw lines and staggered labels from top downward
        for i, (xpos, label, color) in enumerate(xlabels):
            ax.axvline(xpos, color=color, linestyle="--", label=label)
            y_text = ymax - delta_y * i
            ax.text(
                xpos + x_offset, y_text, label, color=color, rotation=0,
                va=valign, ha=halign, fontsize="medium",
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
    g.map_dataframe(add_vep_ref_and_mean_lines)
    g.set_titles(col_template="{col_name}")

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
    
    plt.tight_layout()
    
    if show:
        plt.show()

    return {'fig':g.figure, 'axes':g.axes, 'data':plot_df}
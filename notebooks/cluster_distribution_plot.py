import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch

def create_cluster_distribution_plot(ecdf_umap_df, vep_ecdf, vep_ecdf_pivot):
    # Create the main figure
    fig = plt.figure(figsize=(20, 15))
    
    # Create the main UMAP scatter plot
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    scatter = sns.scatterplot(data=ecdf_umap_df,
                            x='UMAP_1',
                            y='UMAP_2',
                            hue='cluster',
                            size='n_haplotypes',
                            style='scoring_strategy',
                            palette='Set1',
                            ax=ax_main)
    
    # Calculate cluster centers
    cluster_centers = ecdf_umap_df.groupby('cluster')[['UMAP_1', 'UMAP_2']].mean()
    
    # Create small distribution plots for each cluster
    for cluster in sorted(ecdf_umap_df['cluster'].unique()):
        # Get data for this cluster
        ids = vep_ecdf_pivot.loc[vep_ecdf_pivot['cluster'] == cluster].reset_index()['id'].unique()
        cluster_data = vep_ecdf.loc[vep_ecdf['id'].isin(ids)]
        
        # Calculate mean ECDF and KDE
        mean_ecdf = cluster_data.groupby('grid_index')['ecdf'].mean()
        mean_kde = cluster_data.groupby('grid_index')['kde'].mean()
        
        # Create small subplot for this cluster
        # Position the subplot based on cluster number
        row = (cluster - 1) // 3
        col = (cluster - 1) % 3
        ax_dist = plt.subplot2grid((3, 3), (row, col), fig=fig)
        
        # Plot mean ECDF
        ax_dist.plot(mean_ecdf.index, mean_ecdf.values, color='blue', label='ECDF')
        
        # Create second y-axis for KDE
        ax2 = ax_dist.twinx()
        ax2.plot(mean_kde.index, mean_kde.values, color='red', label='KDE')
        
        # Customize the small plot
        ax_dist.set_title(f'Cluster {cluster}')
        ax_dist.set_ylabel('ECDF')
        ax2.set_ylabel('KDE')
        
        # Connect the cluster center to the distribution plot
        center = cluster_centers.loc[cluster]
        con = ConnectionPatch(
            xyA=(center['UMAP_1'], center['UMAP_2']),
            xyB=(0.5, 0.5),
            coordsA='data',
            coordsB='axes fraction',
            axesA=ax_main,
            axesB=ax_dist,
            color='gray',
            alpha=0.3
        )
        ax_main.add_artist(con)
    
    # Adjust layout
    plt.tight_layout()
    return fig

# Example usage:
# fig = create_cluster_distribution_plot(ecdf_umap_df, vep_ecdf, vep_ecdf_pivot)
# plt.show() 
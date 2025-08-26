"""
Module: test_normality

This module provides functions to test for normality of grouped data and to visualize the results
by clinical significance. It is intended for use with variant effect prediction (VEP) scores or
similar metrics, grouped by relevant biological or experimental columns.

Functions
---------
- test_normality(df, groupby_cols=None, value_col="VEP", min_n=20, show_progress=True)
    For each group in the DataFrame, tests for normality of the specified value column using
    scipy.stats.normaltest. Returns a DataFrame with the test statistic, p-value, and sample count
    for each group.

- plot_test_normality(normality_results, utils, figsize=(7, 4))
    Visualizes the proportion of groups with normal, non-normal, or untested distributions,
    grouped by clinical significance. Uses color to indicate clinical significance and hatching
    to indicate normality status.

Dependencies
------------
- pandas
- scipy.stats
- tqdm
- matplotlib
- numpy

Example
-------
>>> results = test_normality(df)
>>> plot_test_normality(results, utils)
"""

from scipy.stats import normaltest
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import src.utils as utils
import src.vep_analysis as va




def get_kde(x, 
            title=None,
            bw_method="ISJ", #'scott', #'silverman',
            grid_points=100,
            return_df=False,
            plot=False):
    
    # Calculate kernel density estimate
    # Use KDE with adaptive bandwidth for better handling of multimodal data
    # Scott's rule or Silverman's rule might not be optimal for multimodal distributions
  

    # For more flexibility with multimodal data, we could use a mixture of Gaussians
    # from sklearn.mixture import GaussianMixture
    # gmm = GaussianMixture(n_components=3, 
    #                       random_state=0).fit(dat['VEP'].values.reshape(-1, 1))
    # gmm_df = pd.DataFrame(gmm.means_, columns=['mean'])
    # gmm_df['std'] = np.sqrt(gmm.covariances_)
    # gmm_df['weight'] = gmm.weights_
    # gmm_df
    # Or use a variable bandwidth KDE which adapts better to multimodal distributions

   

    # Evaluate the KDE at these points
    if bw_method == "ISJ":
        from KDEpy import FFTKDE
        kde = FFTKDE(kernel='gaussian', bw=bw_method)
        density = kde.fit(np.array(x)).evaluate(grid_points)
    else:
        from scipy import stats
         # Create a range of x values to evaluate the KDE
        x_grid = np.linspace(x.min(), 
                             x.max(), 
                             num=grid_points)  
        kde_scipy = stats.gaussian_kde(x, bw_method=bw_method)  # Start with Silverman's rule
        density = kde_scipy(x_grid)

    # Create a DataFrame with the KDE results
    kde_df = pd.DataFrame({'grid_index': range(len(x_grid)),
                           'x': x_grid,
                           'y': density}).reset_index()

    # # Find peaks in the density to identify modes
    n_peaks = get_peaks(density)
    kde_df['n_peaks'] = n_peaks

    # # Plot the KDE
    if plot:
        plt.figure(figsize=(8, 6))
        x.hist(bins=grid_points, density=True, color="grey")
        sns.lineplot(data=kde_df, 
                     x='x', 
                     y='y')
        plt.title(f"{title}\n Peaks = {n_peaks}")
    
    # Return the KDE dataframe
    if return_df:
        return  kde_df
    else:
        return n_peaks 
    

def get_peaks(x,
              cwt=True,
              **kwargs):
    """
    Get the peaks of a list of values.
    """
    from scipy import signal
    if cwt:
        peaks = signal.find_peaks_cwt(x, widths=np.arange(1,10), **kwargs)
    else:
        peaks, _ = signal.find_peaks(x, **kwargs)
    n_peaks = len(peaks) if len(peaks) > 0 else 1
    return n_peaks

def ecdf(x):
    #     x = np.sort(data)
    #     n = len(x)
    #     y = np.arange(1, n + 1) / n
    #     return x, y
    from scipy import stats
    res = stats.ecdf(x)
    return res
    
def get_ecdf(x,
             return_df=False,
             plot=False, 
             grid_points=100,
             title=None,
             error=True):
    """
    Get the Empirical Cumulative Distribution Function of a list of values.
    """
    
    try:
        x_array = np.array(x) # Get the number of peaks in the KDE
        # Ensure x_array has no NaN values
        x_array = x_array[~np.isnan(x_array)]
        assert len(x_array) > 1, "Only one data point, cannot calculate density"

       
        # Create a grid that extends beyond the data points to avoid the "Every data point must be inside of the grid" error
        from KDEpy import FFTKDE
        kde_x, kde_y = FFTKDE(bw='ISJ').fit(x_array).evaluate(grid_points=grid_points)
        n_peaks = get_peaks(kde_y)

        # Get the ECDF as a dataframe
        ecdf_res = ecdf(x_array)
        x_cdf_sampled = np.linspace(x.min(), x.max(), grid_points)
        y_cdf_sampled = ecdf_res.cdf.evaluate(x_cdf_sampled)
        cdf_df = pd.DataFrame(
            {'x': x_cdf_sampled,
             'ecdf': y_cdf_sampled,
             'kde': kde_y
            })
        cdf_df['n_peaks'] = n_peaks
        cdf_df['n_samples'] = len(x_array)

         # Calculate summary statistics
        stats = {
            'min': x_array.min(),
            'max': x_array.max(),
            'range': x_array.max() - x_array.min(),
            'mean': x_array.mean(),
            'std': x_array.std(),
            'var': x_array.var(),
            'median': np.median(x_array)
        }
        # Add summary statistics to the dataframe
        for stat_name, stat_value in stats.items():
            cdf_df["x_"+stat_name] = stat_value

        # Plot the ECDF
        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))

            # Create primary y-axis for histogram
            sns.histplot(x, 
                        bins=grid_points, 
                        stat='density',
                        kde=True, 
                        ax=ax, 
                        color='skyblue',
                        # cumulative=True,
                        alpha=0.6, 
                        label='Histogram with KDE')

            # Create secondary y-axis for CDF
            ax2 = ax.twinx()
            sns.lineplot(data=cdf_df, x='x', y='ecdf',
                        ax=ax2, 
                        color='red', 
                        label='Empirical CDF')
            ax.set_title(title)
            # Add labels and title
            ax.set_xlabel('VEP Score')
            # ax.set_ylabel('Frequency')
            ax2.set_ylabel('Cumulative Probability')
            plt.title(f"Distribution and CDF (Number of modes: {n_peaks})")

            # Add legend - only include one instance of each label
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            # Create combined legend with unique labels
            ax.legend(lines1 + lines2, 
                    labels1 + labels2, 
                    loc='upper left')

            plt.tight_layout()
        # Return the CDF dataframe
        if return_df:
            return cdf_df
        else:
            return n_peaks
    except Exception as e:
        if error:
            raise e
        else:
            return None

def prepare_umap_data(vep_df, 
                      groupby_cols=['model_location', 'protein', 'scoring_strategy', 'mutant', 'clinsig'],
                      grid_points=100):
    """
    Prepare data for UMAP by combining ECDF values with summary statistics.
    
    Args:
        vep_df: DataFrame containing VEP scores
        groupby_cols: Columns to group by when calculating ECDFs
        grid_points: Number of points to use for ECDF grid
        
    Returns:
        DataFrame with combined ECDF values and summary statistics
    """
    # Group by specified columns and calculate ECDF for each group
    ecdf_dfs = []
    for _, group in vep_df.groupby(groupby_cols):
        ecdf_df = get_ecdf(group['VEP'], 
                          return_df=True,
                          grid_points=grid_points)
        if ecdf_df is not None:
            # Add group identifiers
            for col in groupby_cols:
                ecdf_df[col] = group[col].iloc[0]
            ecdf_dfs.append(ecdf_df)
    
    # Combine all ECDF dataframes
    combined_df = pd.concat(ecdf_dfs, ignore_index=True)
    
    # Create a pivot table with ECDF values and summary statistics
    pivot_df = combined_df.pivot_table(
        index=groupby_cols,
        columns='x',
        values='y'
    )
    
    # Add summary statistics as additional columns
    stats_cols = ['min', 'max', 'range', 'mean', 'std', 'median', 'q1', 'q3', 'iqr', 'n_peaks', 'n_samples']
    for col in stats_cols:
        pivot_df[col] = combined_df.groupby(groupby_cols)[col].first()
    
    return pivot_df


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch
from scipy.spatial import distance
import math

def create_cluster_distribution_plot(ecdf_umap_df, 
                                     vep_ecdf, 
                                     vep_ecdf_pivot, 
                                     num_subplots=16, 
                                     min_distance=0.8, 
                                     connector_alpha=0.7,
                                     label_font_size=12, 
                                     clusters=None,
                                     palette='gist_rainbow'):
    # Create the main figure
    fig = plt.figure(figsize=(20, 15))
    
    # Create the main UMAP scatter plot in the center
    ax_main = plt.subplot2grid((1, 1), (0, 0))
    scatter = sns.scatterplot(data=ecdf_umap_df,
                            x='UMAP_1',
                            y='UMAP_2',
                            hue='cluster',
                            sizes=(1, 30),
                            size='n_haplotypes',
                            style='scoring_strategy',
                            palette=palette,
                            ax=ax_main)
    
    # Calculate cluster centers
    cluster_centers = ecdf_umap_df.groupby('cluster')[['UMAP_1', 'UMAP_2']].mean()
    
    # Get clusters sorted by their centers
    if clusters is None:
        clusters = sorted(cluster_centers.index)
        clusters = clusters[:min(num_subplots, len(clusters))]
    
    # Get the limits of the main plot
    x_min, x_max = ax_main.get_xlim()
    y_min, y_max = ax_main.get_ylim()
    
    # Calculate positions around the border of the main plot
    # We'll place subplots outside the main plot area to avoid overlap
    positions = []
    
    # Calculate angles around the center
    angles = np.linspace(0, 2*np.pi, num_subplots, endpoint=False)
    
    # Calculate positions outside the edge of the main plot
    # Using min_distance parameter to control how far plots are from the center
    for angle in angles:
        # Use angle to determine position (x, y) outside the main plot
        x = min_distance * np.cos(angle)  # Normalized to [-min_distance, min_distance]
        y = min_distance * np.sin(angle)  # Normalized to [-min_distance, min_distance]
        positions.append((x, y))
    
    # Get the color map used in the scatter plot
    color_palette = sns.color_palette(palette, n_colors=len(ecdf_umap_df['cluster'].unique()))
    cluster_colors = {cluster: color_palette[i] for i, cluster in enumerate(sorted(ecdf_umap_df['cluster'].unique()))}
    
    # Add cluster labels on the UMAP plot
    for cluster, center in cluster_centers.iterrows():
        if cluster in clusters:
            # Create a semi-transparent white box with black border for the label
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
            # Add the cluster label at the center of each cluster
            ax_main.text(center['UMAP_1'], center['UMAP_2'], f"{cluster}", 
                        ha='center', va='center', fontsize=label_font_size, 
                        bbox=bbox_props, zorder=10)
    
    # Create small distribution plots for each cluster
    for i, cluster in enumerate(clusters):
        if i >= len(positions):
            break
            
        # Get data for this cluster
        ids = vep_ecdf_pivot.loc[vep_ecdf_pivot['cluster'] == cluster].reset_index()['id'].unique()
        cluster_data = vep_ecdf.loc[vep_ecdf['id'].isin(ids)]
        
        # Calculate mean ECDF and KDE
        mean_ecdf = cluster_data.groupby('grid_index')['ecdf'].mean()
        mean_kde = cluster_data.groupby('grid_index')['kde'].mean()
        
        # Calculate position for this subplot
        pos_x, pos_y = positions[i]
        
        # Convert normalized position to data coordinates
        data_x = x_min + (x_max - x_min) * (pos_x + 0.5)
        data_y = y_min + (y_max - y_min) * (pos_y + 0.5)
        
        # Create a small inset axes for this cluster
        # Position is relative to the main axes but placed outside the data area
        # Reduce the size of the inset plots to 0.15 of the main plot
        ax_dist = fig.add_axes([
            ax_main.get_position().x0 + (ax_main.get_position().width * (pos_x + 0.5) * 0.9),
            ax_main.get_position().y0 + (ax_main.get_position().height * (pos_y + 0.5) * 0.9),
            ax_main.get_position().width * 0.15,
            ax_main.get_position().height * 0.15
        ])
        
        # Plot mean ECDF
        ax_dist.plot(mean_ecdf.index, mean_ecdf.values, color='blue', label='ECDF')
        
        # Create second y-axis for KDE
        ax2 = ax_dist.twinx()
        ax2.plot(mean_kde.index, mean_kde.values, color='red', label='KDE')
        
        # Customize the small plot
        ax_dist.set_title(f'Cluster {cluster}', fontsize=8)
        ax_dist.tick_params(axis='both', which='major', labelsize=6)
        ax2.tick_params(axis='both', which='major', labelsize=6)
        
        # Remove axis labels to save space
        ax_dist.set_xlabel('')
        ax_dist.set_ylabel('')
        ax2.set_ylabel('')
        
        # Add a background to the subplot to make it stand out
        ax_dist.set_facecolor('white')
        ax_dist.patch.set_alpha(0.8)
        
        # Connect the cluster center to the distribution plot
        if cluster in cluster_centers.index:
            center = cluster_centers.loc[cluster]
            # Get the correct color for this specific cluster
            cluster_color = cluster_colors[cluster]
            con = ConnectionPatch(
                xyA=(center['UMAP_1'], center['UMAP_2']),
                xyB=(0.5, 0.5),
                coordsA='data',
                coordsB='axes fraction',
                arrowstyle='<|-',
                axesA=ax_main,
                axesB=ax_dist,
                color=cluster_color,
                alpha=connector_alpha,
                linewidth=2.0
            )
            ax_main.add_artist(con)
    
    # Add a legend to the main plot
    ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
    # Adjust layout
    plt.tight_layout()
    return fig


def test_normality(df, 
                   groupby_cols=None, 
                   value_col="VEP", 
                   min_n=20, 
                   show_progress=True):
    """
    Test for normality of a value column within groups of a DataFrame.

    For each group (e.g., each variant), this function tests for normality of the distribution
    of the specified value column using scipy.stats.normaltest. It also records the number of
    samples in each group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data to be grouped and tested.
    groupby_cols : list of str, optional
        Columns to group by. If None, defaults to
        ['model_location','protein','clinsig','mutant','scoring_strategy'].
    value_col : str, default "VEP"
        The column containing the values to test for normality.
    min_n : int, default 20
        Minimum number of samples required to perform the normality test. Groups with fewer
        samples will have NaN for statistic and p-value.
    show_progress : bool, default True
        Whether to show a progress bar during computation.

    Returns
    -------
    pd.DataFrame
        DataFrame with groupby columns and columns: 'statistic', 'pvalue', 'n_samples'.
    """
    if groupby_cols is None:
        groupby_cols = ['model_location','protein','clinsig','mutant','scoring_strategy']

    def normality_with_n(x):
        """
        Helper function to compute normality test and sample count for a group.

        Parameters
        ----------
        x : pd.Series
            Series of values for the group.

        Returns
        -------
        pd.Series
            Series with 'statistic', 'pvalue', and 'n_samples'.
        """
        x_nonan = x.dropna()
        n = len(x_nonan)
        if n > min_n:
            stat, pval = normaltest(x_nonan)
        else:
            stat, pval = float("nan"), float("nan")
        return pd.Series({"statistic": stat, "pvalue": pval, "n_samples": n})

    if show_progress:
        tqdm.pandas()
        grouped = df.groupby(groupby_cols, observed=True)[value_col].progress_apply(normality_with_n)
    else:
        grouped = df.groupby(groupby_cols, observed=True)[value_col].apply(normality_with_n)

    results = grouped.reset_index().rename(columns={f"level_{len(groupby_cols)}": "metric"}).pivot_table(index=groupby_cols, columns="metric", values=value_col).reset_index()
    return results

def plot_test_normality(normality_results, 
                        bar_width=0.8, 
                        figsize=(4, 4),
                        legend_bbox_to_anchor=(1.4, 0.6),
                        legend_loc='upper right',
                        title=None,
                        legend_left=False):
    """
    Plot the proportion of groups with normal, not normal, or not testable distributions,
    grouped by clinical significance.

    This function visualizes the results of test_normality by showing, for each clinical
    significance group, the proportion of groups that are normal, not normal, or not testable
    (due to insufficient sample size).

    Parameters
    ----------
    normality_results : pd.DataFrame
        DataFrame with at least columns 'clinsig' and 'pvalue', as returned by test_normality.
    utils : module
        Module with a get_clinsig_palette() function for color mapping.
    figsize : tuple, default (7, 4)
        Figure size for the plot.
    legend_bbox_to_anchor : tuple, default (1.4, 0.6)
        Bounding box to anchor the legend.
    legend_loc : str, default 'upper right'
        Location of the legend.
         title : str, default None
         Title of the plot.
     legend_left : bool, default False
         If True, place the legend on the left side of the plot. If False, place it on the right side.

    Returns
    -------
    dict
        Dictionary with keys 'fig', 'ax', and 'data' (the proportions DataFrame).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    import pandas as pd

    def normal_category(row):
        """
        Categorize each group as 'Normal', 'Not normal', or 'Not testable' based on p-value.

        Parameters
        ----------
        row : pd.Series
            Row from the normality_results DataFrame.

        Returns
        -------
        str or bool
            'Not testable' if p-value is NaN, True if p-value > 0.05, False otherwise.
        """
        if pd.isna(row["pvalue"]):
            return "Not testable"
        elif row["pvalue"] > 0.05:
            return True
        else:
            return False

    normality_results = normality_results.copy()
    normality_results["normal_category"] = normality_results.apply(normal_category, axis=1)

    # Group by 'clinsig' and 'normal_category', count number of groups in each category
    normality_by_clinsig = (
        normality_results.groupby(['clinsig', 'normal_category'])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure all three categories are present as columns
    for cat in [False, True, "Not testable"]:
        if cat not in normality_by_clinsig.columns:
            normality_by_clinsig[cat] = 0

    # Calculate proportions within each clinsig
    normality_by_clinsig_prop = normality_by_clinsig.div(normality_by_clinsig.sum(axis=1), axis=0)

    # Get clinsig palette and order
    palette = utils.get_clinsig_palette()
    clinsig_order = [c for c in palette if c in normality_by_clinsig_prop.index]

    # Plot with color by clinsig, hatching by normal/not normal/not testable
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(normality_by_clinsig_prop.index))
    hatch_map = {False: "//", True: "", "Not testable": "...."}
    label_map = {False: "Not normal", True: "Normal", "Not testable": "Not testable"}
    stack_order = ["Not testable", False, True]

    for i, clinsig in enumerate(normality_by_clinsig_prop.index):
        bottom = 0
        for normal in stack_order:
            value = normality_by_clinsig_prop.loc[clinsig, normal] if normal in normality_by_clinsig_prop.columns else 0
            color = palette.get(clinsig, "gray")
            hatch = hatch_map[normal]
            bar = ax.bar(
                x[i], value, bar_width,  
                bottom=bottom,
                color=color,
                edgecolor='black',
                hatch=hatch,
            )
            bottom += value

    # Legends
    clinsig_patches = [Patch(facecolor=palette[c], edgecolor='black', label=c) for c in clinsig_order]
    hatch_patches = [
        Patch(facecolor='white', edgecolor='black', hatch=hatch_map[True], label=label_map[True]),
        Patch(facecolor='white', edgecolor='black', hatch=hatch_map[False], label=label_map[False]),
        Patch(facecolor='white', edgecolor='black', hatch=hatch_map["Not testable"], label=label_map["Not testable"])
    ]

    # Only show hatching legend by default; add clinsig legend if desired
    # Adjust legend position based on figure size to avoid overlap
    if legend_left:
        # Place legend on the left side of the plot
        if figsize[0] <= 5:  # For smaller figure widths, position legend outside plot with adjusted positioning
            ax.legend(handles=hatch_patches, title='Status', loc='center right', bbox_to_anchor=(-0.25, 0.5))
        else:  # For larger figure widths, position legend outside plot
            ax.legend(handles=hatch_patches, title='Status', loc='center right', bbox_to_anchor=(-0.5, 0.5))
    else:
        # Place legend on the right side of the plot (default behavior)
        if figsize[0] <= 5:  # For smaller figure widths, position legend outside plot with adjusted positioning
            ax.legend(handles=hatch_patches, title='Status', loc='center left', bbox_to_anchor=(1.02, 0.5))
        else:  # For larger figure widths, position legend outside plot
            ax.legend(handles=hatch_patches, title='Status', loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)

    ax.set_xticks(x)
    ax.set_xticklabels(normality_by_clinsig_prop.index, rotation=30, ha='right')
    ax.set_ylabel('Proportion of Variants')

    if title is None:
        proportion_tested = 1 - normality_by_clinsig_prop["Not testable"].mean()
        title = f'Proportion of Variants\nwith Normal VEP Distributions\n({proportion_tested*100:.1f}% testable)'
    ax.set_title(title)
    
    # Adjust margins when legend is on the left to prevent overlap with y-axis title
    if legend_left:
        plt.subplots_adjust(left=0.25)
    
    plt.tight_layout()
    plt.show()
    return {'fig': fig, 'ax': ax, 'data': normality_by_clinsig_prop}


def plot_ref_percentile_schematic(
    ax=None, 
    show=False, 
    barplot_ylim=None, 
    schematic_heights=[0.35, 0.30, 0.35], 
    title_loc="left", 
    title_fontsize=10,
    show_xlabel=(True, True), 
    show_ylabel=(True, True),
    gradient_granularity=None,
    ):
    """
    Plot a schematic showing how REF can under- or over-estimate pathogenicity
    using a 3-row grid: top (REF far right), blank, bottom (REF far left).
    If ax is provided, draws the schematic into that axis (as a single column).
    If ax is None, creates a new figure and axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        If provided, draws the schematic into this axis (as a single column).
        If None, creates a new figure and axes.
    show : bool
        Whether to call plt.show() (only if ax is None).
    barplot_ylim : tuple or None
        If provided, (ymin, ymax) to align the schematic's top and bottom with the barplot.
    schematic_heights : list, optional
        Heights of the three schematic subplots as fractions of total height [top, blank, bottom] (default: [0.35, 0.30, 0.35]).
    title_loc : str, optional
        Location for the schematic titles (default: "left").
        Valid options include: "left", "center", "right", "upper left", "upper center", "upper right", etc.
    title_fontsize : int, optional
        Fontsize for the schematic titles (default: 10).
    show_xlabel : tuple of bool, optional
        Whether to show x-axis labels for the schematics (default: (True, True)).
        First boolean controls top schematic, second controls bottom schematic.
    show_ylabel : tuple of bool, optional
        Whether to show y-axis labels for the schematics (default: (True, True)).
        First boolean controls top schematic, second controls bottom schematic.
    gradient_granularity : int or None, optional
        If provided, controls the number of color steps in the gradient fill. If None, uses full resolution.

    Returns
    -------
    dict with keys:
        "fig": matplotlib.figure.Figure
        "axs": list of matplotlib.axes.Axes (top, blank, bottom)
        "data": pd.DataFrame (x, y for the normal curve)
    """
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import numpy as np
    import pandas as pd

    x = np.linspace(-3, 3, 500)
    y = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    data = pd.DataFrame({"x": x, "y": y})

    palette = utils.get_clinsig_palette()
    cmap = LinearSegmentedColormap.from_list("red_blue", [palette["path"], palette["benign"]])

    def gradient_fill(ax, x, y, cmap, alpha=0.5, zorder=1, granularity=None):
        x_norm = (x - x.min()) / (x.max() - x.min())
        if granularity is None:
            # Default: smooth gradient
            for i in range(len(x) - 1):
                ax.fill_between(
                    x[i:i+2], y[i:i+2], color=cmap(x_norm[i]), alpha=alpha, zorder=zorder
                )
        else:
            # Discretize the gradient into 'granularity' bins
            bins = np.linspace(x.min(), x.max(), granularity + 1)
            for i in range(granularity):
                mask = (x >= bins[i]) & (x <= bins[i+1])
                x_bin = x[mask]
                y_bin = y[mask]
                if len(x_bin) < 2:
                    continue
                x_norm_bin = (x_bin - x.min()) / (x.max() - x.min())
                color = cmap((bins[i] - x.min()) / (x.max() - x.min()))
                ax.fill_between(x_bin, y_bin, color=color, alpha=alpha, zorder=zorder)

    # The normal curve's max is about 0.4, min is 0
    # We'll use this to map the barplot's y-axis to the schematic's y-axis
    schematic_ymin = 0
    schematic_ymax = 0.4
    if barplot_ylim is not None:
        barplot_ymin, barplot_ymax = barplot_ylim
        # We'll map barplot_ymin to schematic_ymin and barplot_ymax to schematic_ymax
        # For the schematic, set ylim to (schematic_ymin, schematic_ymax)
        # But for the top and bottom axes, we want the top of the top schematic to align with barplot_ymax,
        # and the bottom of the bottom schematic to align with barplot_ymin.
        # So we set the ylims of both to (schematic_ymin, schematic_ymax)
        # and set the position of the axes to fill the vertical space from 0 to 1 in the parent axis.
        # This is handled below.

    if ax is not None:
        # Draw the schematic as a 3-row grid inside a single axis using manually positioned axes

        fig = ax.figure
        axs = []

        # We'll use 3 axes positioned manually to align with the barplot
        # The top schematic should align with the top of the barplot, bottom with bottom
        # Add some spacing between the schematics
        heights = schematic_heights  # top, blank, bottom (configurable heights)
        
        # Get the parent axis position
        parent_pos = ax.get_position()
        parent_x0, parent_y0, parent_width, parent_height = parent_pos.x0, parent_pos.y0, parent_pos.width, parent_pos.height
        
        # Calculate positions for each sub-axis with proper spacing
        y_positions = []
        y0 = parent_y0 + parent_height  # Start from top of parent
        for h in heights:
            y_positions.append((y0 - h * parent_height, h * parent_height))
            y0 -= h * parent_height
        
        for i, (y_pos, height) in enumerate(y_positions):
            if i == 1:  # blank axis
                blank_ax = fig.add_axes([parent_x0, y_pos, parent_width, height])
                blank_ax.axis('off')
                axs.append(blank_ax)
            else:
                sub_ax = fig.add_axes([parent_x0, y_pos, parent_width, height])
                axs.append(sub_ax)

        ax_top, ax_blank, ax_bottom = axs

        # Set ylims to align with barplot if provided
        if barplot_ylim is not None:
            barplot_ymin, barplot_ymax = barplot_ylim
            # Keep the schematic's natural y-axis range for proper curve display
            ax_top.set_ylim(schematic_ymin, schematic_ymax)
            ax_bottom.set_ylim(schematic_ymin, schematic_ymax)
        else:
            ax_top.set_ylim(schematic_ymin, schematic_ymax)
            ax_bottom.set_ylim(schematic_ymin, schematic_ymax)

        # Top subplot: REF far right
        ax_top.plot(x, y, color='black', lw=2)
        gradient_fill(ax_top, x, y, cmap, alpha=0.7, granularity=gradient_granularity)
        ref_x = 2.2
        ax_top.axvline(ref_x, color='grey', linestyle='--', lw=2, zorder=10)
        ax_top.text(ref_x-0.1, 0.25, "REF", color='grey', fontsize=10, fontweight=None, va='center', ha='right', rotation=90)
        if show_ylabel[0]:  # Show ylabel for top schematic if first boolean is True
            ax_top.set_ylabel("Density")
        ax_top.set_yticks([])
        # fontweight does not apply to LaTeX text; use \mathbf{} for bold in LaTeX
        ax_top.set_title(r"$\mathbf{VEP_{REF}\ underestimates}$" + "\n" + r"$\mathbf{pathogenicity}$", 
                        fontsize=title_fontsize, 
                        loc=title_loc,
                        fontweight='bold')
        if show_xlabel[0]:  # Show xlabel for top schematic if first boolean is True
            ax_top.set_xlabel(r"$VEP_{REF}$ percentile")
        ax_top.set_xlim(-3, 3)
        ax_top.set_xticks([-3, 0, 3])
        ax_top.set_xticklabels(['0', '50', '100'])
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['left'].set_visible(False)
        ax_top.spines['top'].set_visible(False)

        # Middle subplot: blank space
        ax_blank.axis('off')

        # Bottom subplot: REF far left
        ax_bottom.plot(x, y, color='black', lw=2)
        gradient_fill(ax_bottom, x, y, cmap, alpha=0.7, granularity=gradient_granularity)
        ref_x = -2.2
        ax_bottom.axvline(ref_x, color='grey', linestyle='--', lw=2, zorder=10)
        ax_bottom.text(ref_x-0.1, 0.25, "REF", color='grey', fontsize=10, fontweight=None, va='center', ha='right', rotation=90)
        if show_ylabel[1]:  # Show ylabel for bottom schematic if second boolean is True
            ax_bottom.set_ylabel("Density")
        ax_bottom.set_yticks([])
        ax_bottom.set_title(r"$\mathbf{VEP_{REF}\ overestimates}$" + "\n" + r"$\mathbf{pathogenicity}$", 
                            fontsize=title_fontsize, loc=title_loc,
                            fontweight='bold')
        if show_xlabel[1]:  # Show xlabel for bottom schematic if second boolean is True
            ax_bottom.set_xlabel(r"$VEP_{REF}$ percentile")
        ax_bottom.set_xlim(-3, 3)
        ax_bottom.set_xticks([-3, 0, 3])
        ax_bottom.set_xticklabels(['0', '50', '100'])
        ax_bottom.spines['right'].set_visible(False)
        ax_bottom.spines['left'].set_visible(False)
        ax_bottom.spines['top'].set_visible(False)

        # Hide axis frame, ticks, and labels for the parent axis
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

        return {"fig": fig, "axs": [ax_top, ax_blank, ax_bottom], "data": data}

    else:
        # Standalone schematic as before
        fig = plt.figure(figsize=(4, 10.5))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
        ax_top = fig.add_subplot(gs[0])
        ax_blank = fig.add_subplot(gs[1])
        ax_bottom = fig.add_subplot(gs[2], sharex=ax_top)
        axs = [ax_top, ax_blank, ax_bottom]

        ax_top.set_ylim(schematic_ymin, schematic_ymax)
        ax_bottom.set_ylim(schematic_ymin, schematic_ymax)

        # Top subplot: REF far right
        ax_top.plot(x, y, color='black', lw=2)
        gradient_fill(ax_top, x, y, cmap, alpha=0.7, granularity=gradient_granularity)
        ref_x = 2.2
        ax_top.axvline(ref_x, color='grey', linestyle='--', lw=2, zorder=10)
        ax_top.text(ref_x-0.1, 0.25, "REF", color='grey', fontsize=10, fontweight=None, va='center', ha='right', rotation=90)
        if show_ylabel[0]:  # Show ylabel for top schematic if first boolean is True
            ax_top.set_ylabel("Density")
        ax_top.set_yticks([])
        ax_top.set_title(r"$VEP_{REF}$ underestimates\npathogenicity", fontweight='bold', loc=title_loc)
        if show_xlabel[0]:  # Show xlabel for top schematic if first boolean is True
            ax_top.set_xlabel(r"$VEP_{REF}$ percentile")
        ax_top.set_xlim(-3, 3)
        ax_top.set_xticks([-3, 0, 3])
        ax_top.set_xticklabels(['0', '50', '100'])
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['top'].set_visible(False)

        # Middle subplot: blank space
        ax_blank.axis('off')

        # Bottom subplot: REF far left
        ax_bottom.plot(x, y, color='black', lw=2)
        gradient_fill(ax_bottom, x, y, cmap, alpha=0.7, granularity=gradient_granularity)
        ref_x = -2.2
        ax_bottom.axvline(ref_x, color='grey', linestyle='--', lw=2, zorder=10)
        ax_bottom.text(ref_x-0.1, 0.25, "REF", color='grey', fontsize=10, fontweight=None, va='center', ha='right', rotation=90)
        if show_ylabel[1]:  # Show ylabel for bottom schematic if second boolean is True
            ax_bottom.set_ylabel("Density")
        ax_bottom.set_yticks([])
        ax_bottom.set_title(r"$VEP_{REF}$ overestimates\npathogenicity", 
                            fontweight='bold', 
                            loc=title_loc)
        if show_xlabel[1]:  # Show xlabel for bottom schematic if second boolean is True
            ax_bottom.set_xlabel(r"$VEP_{REF}$ percentile")
        ax_bottom.set_xlim(-3, 3)
        ax_bottom.set_xticks([-3, 0, 3])
        ax_bottom.set_xticklabels(['0', '50', '100'])
        ax_bottom.spines['right'].set_visible(False)
        ax_bottom.spines['top'].set_visible(False)

        plt.tight_layout()
        if show:
            plt.show()
        return {"fig": fig, "axs": axs, "data": data}


def plot_ref_vep_percentile_stacked_bar(
    vep_df, 
    groupby_cols=['model_location','protein','clinsig','mutant','scoring_strategy'],
    y='VEP_percentile',
    n_bins=10, 
    figsize=(10, 4), 
    label_padding=0.15, 
    is_ref=True,
    title=r"$VEP_{REF}$ Percentiles Relative to Full $VEP$ Distribution",
    x_label=r"$VEP_{mean}$ Quantile",
    y_label="Proportion of Variants",
    show_arrows=False,
    show_vertical_arrows=False,
    show_schematic=True,
    legend_on_right=True,
    legend_height_factor=1.5,
    schematic_width_ratio=1.3,
    barplot_width_ratio=5,
    schematic_heights=[0.2, 0.3, 0.2],
    schematic_padding_left=0.12,
    schematic_padding_top=0.1,
    schematic_title_loc="left",
    show_schematic_xlabel=(False, True),
    show_schematic_ylabel=(False, False),
    use_quantile_labels=True
):
    """
    Plot a stacked bar plot showing the distribution of REF VEP percentiles
    relative to the full VEP distribution, binned by VEP_mean quantiles.
    Optionally, add a schematic illustration to the right of the plot.

    Parameters
    ----------
    vep_df : pd.DataFrame
        DataFrame containing at least 'VEP_mean' and 'VEP_percentile' columns.
    groupby_cols : list, optional
        Columns to group by for computing representativeness stats (default: ['model_location','protein','clinsig','mutant','scoring_strategy']).
    y : str, optional
        Column name for y-axis values (default: 'VEP_percentile').
    n_bins : int, optional
        Number of quantile bins for VEP_mean (default: 10).
    figsize : tuple, optional
        Figure size for the plot (default: (9, 4)).
    label_padding : float, optional
        Vertical padding between the two y-axis text labels (default: 0.15).
    is_ref : bool, optional
        Whether to compute REF statistics (default: True).
    title : str, optional
        Plot title (default: "REF VEP Percentiles Relative to Full VEP Distribution").
    x_label : str, optional
        X-axis label (default: "VEP Quantile").
    y_label : str, optional
        Y-axis label (default: "Proportion of Variants").
    show_arrows : bool, optional
        Whether to show arrows and labels indicating under/overestimation (default: False).
    show_vertical_arrows : bool, optional
        Whether to show vertical up/down arrows connecting the barplot to the schematic (default: True).
    legend_on_right : bool, optional
        Whether to position the legend on the right side of the plot (default: False).
        If True, the schematic will be moved further right to make room for the legend.
    legend_height_factor : float, optional
        Factor to control the height of the legend proportionally (default: 1.0).
        If < 1.0, the legend height is shrunk proportionally.
        If > 1.0, the legend height is grown proportionally.
    show_schematic : bool, optional
        Whether to show a schematic illustration to the right of the plot (default: True).
    schematic_width_ratio : float, optional
        Width ratio for the schematic subplot (default: 1.2).
    barplot_width_ratio : float, optional
        Width ratio for the barplot subplot (default: 4).
    schematic_heights : list, optional
        Heights of the three schematic subplots as fractions of total height [top, blank, bottom] (default: [0.35, 0.30, 0.35]).
    schematic_padding_left : float, optional
        Horizontal padding between the barplot and schematic plots (default: 0.05).
    schematic_padding_top : float, optional
        Vertical padding above the schematic column, moving it downward (default: 0.0).
    schematic_title_loc : str, optional
        Location for the schematic titles (default: "left").
        Valid options include: "left", "center", "right", "upper left", "upper center", "upper right", etc.
    show_schematic_xlabel : tuple of bool, optional
        Whether to show x-axis labels for the schematics (default: (True, True)).
        First boolean controls top schematic, second controls bottom schematic.
    show_schematic_ylabel : tuple of bool, optional
        Whether to show y-axis labels for the schematics (default: (True, True)).
        First boolean controls top schematic, second controls bottom schematic.
    use_quantile_labels : bool, optional
        Whether to use quantile labels ("Q1", "Q2", etc.) for x-axis tick labels (default: True).
    """
    from matplotlib import cm
    import matplotlib.pyplot as plt

    vep_df = vep_df.copy()

    data = va.compute_representativeness_stats(vep_df, groupby_cols=groupby_cols, y=y, is_ref=is_ref)

    # Bin VEP_mean into quantile bins (x-axis)
    vep_binned, bin_edges = pd.qcut(data['VEP_mean'], q=n_bins, labels=False, retbins=True, duplicates='drop')
    data = data.copy()
    data['VEP_binned'] = vep_binned

    # Create bin labels based on use_quantile_labels parameter
    if use_quantile_labels:
        # Use quantile labels like "Q1", "Q2", etc.
        bin_labels = [f"Q{i+1}" for i in range(n_bins)]
    else:
        # Create bin range labels as strings, e.g. "0.12–0.34"
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            left = bin_edges[i]
            right = bin_edges[i + 1]
            left_str = f"{left:.2g}" if abs(left) < 1e4 else f"{left:.2e}"
            right_str = f"{right:.2g}" if abs(right) < 1e4 else f"{right:.2e}"
            bin_labels.append(f"{left_str}\n→\n{right_str}")

    # Map integer bin codes to string labels
    data['VEP_binned_label'] = data['VEP_binned'].map(lambda x: bin_labels[int(x)] if pd.notnull(x) else np.nan)

    # Bin VEP_percentile into deciles (y-axis bins)
    percentile_bins = np.linspace(0, 100, 11)
    percentile_labels = [f"{int(percentile_bins[i])}-{int(percentile_bins[i+1])}%" for i in range(10)]
    data['VEP_percentile_decile'] = pd.cut(
        data['VEP_percentile'],
        bins=percentile_bins,
        labels=percentile_labels,
        include_lowest=True,
        right=True
    )

    # Prepare data for stacked bar plot (as proportions)
    stacked = data.groupby(['VEP_binned_label', 'VEP_percentile_decile']).size().unstack(fill_value=0)
    stacked_prop = stacked.div(stacked.sum(axis=1), axis=0)
    stacked_prop = stacked_prop.reindex(bin_labels)

    # Get the number of deciles for the color map
    n_cats = len(percentile_labels)
    cmap = cm.get_cmap('coolwarm', n_cats)
    colors = [cmap(i) for i in range(n_cats)]

    # Flip the order of the columns (deciles) and colors for the legend (bottom to top)
    reversed_labels = percentile_labels[::-1]
    reversed_colors = colors[::-1]
    stacked_prop = stacked_prop[reversed_labels]

    # If showing schematic, use gridspec to allocate space for the schematic
    if show_schematic:
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=figsize)
        
        if legend_on_right:
            # When legend is on right, create 3 columns: barplot, legend, schematic
            legend_width_ratio = 0.8  # Dedicated space for legend
            gs = gridspec.GridSpec(1, 3, width_ratios=[barplot_width_ratio, legend_width_ratio, schematic_width_ratio], wspace=schematic_padding_left)
            ax = fig.add_subplot(gs[0])
            legend_ax = fig.add_subplot(gs[1])  # Dedicated axis for legend
            schematic_ax = fig.add_subplot(gs[2])
            
            # Hide the legend axis (it's just for spacing)
            legend_ax.set_xticks([])
            legend_ax.set_yticks([])
            legend_ax.spines['top'].set_visible(False)
            legend_ax.spines['right'].set_visible(False)
            legend_ax.spines['bottom'].set_visible(False)
            legend_ax.spines['left'].set_visible(False)
        else:
            # Original layout: 2 columns - barplot and schematic
            gs = gridspec.GridSpec(1, 2, width_ratios=[barplot_width_ratio, schematic_width_ratio], wspace=schematic_padding_left)
            ax = fig.add_subplot(gs[0])
            schematic_ax = fig.add_subplot(gs[1])
            legend_ax = None
        
        # Apply top padding to schematic column if specified
        if schematic_padding_top > 0:
            # Adjust the schematic axis position to add top padding
            schematic_pos = schematic_ax.get_position()
            new_y0 = schematic_pos.y0 - schematic_padding_top
            new_height = schematic_pos.height
            schematic_ax.set_position([schematic_pos.x0, new_y0, schematic_pos.width, new_height])
    else:
        fig, ax = plt.subplots(figsize=figsize)
        schematic_ax = None
        legend_ax = None

    stacked_prop.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=reversed_colors,
        width=0.95,
        legend=False  # Disable automatic legend to prevent duplication
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Add grey dotted horizontal line at y=0.5 across the entire plot
    ax.axhline(y=0.5, color='grey', linestyle=':', linewidth=2, alpha=0.7, zorder=10, xmax=2)
    
    # Ensure y-axis is constrained to 0-1 range for proportions
    ax.set_ylim(0, 1)
    handles, legend_labels = ax.get_legend_handles_labels()
    
    if legend_on_right:
        # Legend in the dedicated legend axis (middle column)
        legend_ax.legend(
            handles[::-1],
            reversed_labels,
            title=r"$VEP_{REF}$" + "\n" + "percentile bin",
            loc='upper center',
            borderaxespad=0.0,
            bbox_to_anchor=(0.4, .97),  # Add a bit of padding to the top 
        )
    else:
        # Legend on the left side of the barplot (original behavior)
        # Calculate legend position and height based on height factor
        if legend_height_factor != 1.0:
            # Adjust y-position based on height factor
            # For height_factor > 1, move legend down; for < 1, move it up
            y_offset = (legend_height_factor - 1.0) * 0.5  # Scale factor for y-position adjustment
            y_pos = 1.0 + y_offset
        else:
            y_pos = 1.0
        
        # Create legend with height adjustment
        if legend_height_factor != 1.0:
            # Calculate custom height for legend
            # Use bbox_to_anchor with height adjustment
            legend = ax.legend(
                handles[::-1],
                reversed_labels,
                title=r"$VEP_{REF}$" + "\n" + "percentile bin",
                bbox_to_anchor=(-0.15, y_pos),
                loc='upper right',
                borderaxespad=0.0,
                # Adjust legend height by modifying the layout
                ncol=1,  # Ensure single column for height control
                frameon=True  # Keep frame for better height control
            )
            
            # Apply height scaling by adjusting the legend's internal spacing
            if hasattr(legend, '_legend_box'):
                legend._legend_box.sep = legend._legend_box.sep * legend_height_factor
        else:
            # Default legend without height adjustment
            legend = ax.legend(
                handles[::-1],
                reversed_labels,
                title=r"$VEP_{REF}$" + "\n" + "percentile bin",
                bbox_to_anchor=(-0.15, y_pos),
                loc='upper right',
                borderaxespad=0.0
            )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # Remove top and right margin lines (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    # --- Add up/down arrows to the right of the barplot that align with the schematic ---
    if show_schematic and show_vertical_arrows:
        # Get axis limits for y
        ymin, ymax = ax.get_ylim()
        ycenter = (ymin + ymax) / 2  # Center of the barplot
        
        if legend_on_right:
            # When legend is on right, arrows go between barplot and legend
            xlim = ax.get_xlim()
            x_arrow = xlim[1] + 0.05
        else:
            # Original behavior: arrows go between barplot and schematic
            xlim = ax.get_xlim()
            x_arrow = xlim[1] + 0.05
        
        # Length of arrows - extend almost to top and bottom with small gap in middle
        arrow_gap = (ymax - ymin) * 0.05  # Small gap in the middle
        top_arrow_length = (ymax - ymin) * 0.45  # Extend almost to top
        bottom_arrow_length = (ymax - ymin) * 0.45  # Extend almost to bottom
        
        # Arrow for "REF Underestimates Pathogenicity" (upward)
        ax.annotate(
            "",
            xy=(x_arrow, ymax - arrow_gap),
            xytext=(x_arrow, ycenter + arrow_gap),
            arrowprops=dict(arrowstyle="->", color="black", lw=2.5),
            annotation_clip=False
        )
        
        # Arrow for "REF Overestimates Pathogenicity" (downward)
        ax.annotate(
            "",
            xy=(x_arrow, ymin + arrow_gap),
            xytext=(x_arrow, ycenter - arrow_gap),
            arrowprops=dict(arrowstyle="->", color="black", lw=2.5),
            annotation_clip=False
        )
        
        # Add horizontal dotted lines connecting schematics to barplot
        if show_schematic and schematic_ax is not None:
            if legend_on_right:
                # When legend is on right, lines go from barplot to legend area
                # Get the right edge of the barplot
                barplot_right = ax.get_position().x1
                
                # Get the y-positions of the schematic subplots for reference
                schematic_pos = schematic_ax.get_position()
                schematic_y0, schematic_height = schematic_pos.y0, schematic_pos.height
                
                # Calculate y-positions for the connecting lines
                # Top schematic: connect from middle of top schematic to barplot
                top_y = schematic_y0 + schematic_height * (1 - schematic_heights[0]/2)
                # Bottom schematic: connect from middle of bottom schematic to barplot  
                bottom_y = schematic_y0 + schematic_height * schematic_heights[2]/2
                
                # Transform the y-positions from display coordinates to data coordinates
                top_title_y = schematic_y0 + schematic_height * (1 - schematic_heights[0] * 0.25)
                bottom_title_y = schematic_y0 + schematic_height * schematic_heights[2] * 0.75
                
                # Convert these display coordinates to data coordinates for the barplot
                top_data_y = ax.transData.inverted().transform((0, top_title_y))[1]
                bottom_data_y = ax.transData.inverted().transform((0, bottom_title_y))[1]
                
                # Draw lines from barplot right edge to arrows (in legend area)
                ax.plot([barplot_right, x_arrow], [top_data_y, top_data_y], color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
                ax.plot([barplot_right, x_arrow], [bottom_data_y, bottom_data_y], color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
                
                # Expand the xlim to make sure arrows and labels are visible
                ax.set_xlim(xlim[0], x_arrow + 0.6)
            else:
                # Original behavior for 2-column layout
                # Get the right edge of the barplot
                barplot_right = ax.get_position().x1
                
                # Get the left edge of the schematic
                schematic_left = schematic_ax.get_position().x0
                
                # Get the y-positions of the schematic subplots
                schematic_pos = schematic_ax.get_position()
                schematic_y0, schematic_height = schematic_pos.y0, schematic_pos.height
                
                # Calculate y-positions for the connecting lines
                # Top schematic: connect from middle of top schematic to barplot
                top_y = schematic_y0 + schematic_height * (1 - schematic_heights[0]/2)
                # Bottom schematic: connect from middle of bottom schematic to barplot  
                bottom_y = schematic_y0 + schematic_height * schematic_heights[2]/2
                
                # Draw horizontal dotted lines from barplot right edge to schematic left edge
                # Use data coordinates for x-axis and transform y-positions to data coordinates
                x_data_right = ax.get_xlim()[1]  # Right edge of barplot data
                
                # Transform the y-positions from display coordinates to data coordinates
                # We want the lines to align with the title positions in each schematic
                # Top schematic title is roughly at 75% of its height
                top_title_y = schematic_y0 + schematic_height * (1 - schematic_heights[0] * 0.25)
                # Bottom schematic title is roughly at 25% of its height  
                bottom_title_y = schematic_y0 + schematic_height * schematic_heights[2] * 0.75
                
                # Convert these display coordinates to data coordinates for the barplot
                top_data_y = ax.transData.inverted().transform((0, top_title_y))[1]
                bottom_data_y = ax.transData.inverted().transform((0, bottom_title_y))[1]
                
                # Draw lines from barplot right edge to arrows
                ax.plot([x_data_right, x_arrow], [top_data_y, top_data_y], color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
                ax.plot([x_data_right, x_arrow], [bottom_data_y, bottom_data_y], color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
                
                # Expand the xlim to make sure arrows and labels are visible
                ax.set_xlim(xlim[0], x_arrow + 0.6)

    # --- Optionally add arrows and labels along the y-axis, outside the right margin ---
    if show_arrows:
        # Get axis limits for y
        ymin, ymax = ax.get_ylim()
        ycenter = 0.5  # Origin for arrows

        # Set the x position for the arrows and labels just outside the right of the plot
        xlim = ax.get_xlim()
        x_arrow = xlim[1] + 0.1

        # Length of arrows (as a fraction of y-axis)
        arrow_length = (ymax - ymin) * 0.35

        # Arrow for "REF Underestimates Pathogenicity" (upward)
        fontsize = 8
        ax.annotate(
            "",
            xy=(x_arrow, ycenter + arrow_length),
            xytext=(x_arrow, ycenter),
            arrowprops=dict(arrowstyle="->", color="black", lw=2),
            annotation_clip=False
        )
        ax.text(
            x_arrow + 0.08,
            ycenter + arrow_length/2 + label_padding/2,
            r"$VEP_{REF}$ underestimates\npathogenicity",
            va='center', ha='left', rotation=90, fontsize=fontsize, fontweight='bold'
        )

        # Arrow for "REF Overestimates Pathogenicity" (downward)
        ax.annotate(
            "",
            xy=(x_arrow, ycenter - arrow_length),
            xytext=(x_arrow, ycenter),
            arrowprops=dict(arrowstyle="->", color="black", lw=2),
            annotation_clip=False
        )
        ax.text(
            x_arrow + 0.08,
            ycenter - arrow_length/2 - label_padding/2,
            r"$VEP_{REF}$ overestimates\npathogenicity",
            va='center', ha='left', rotation=90, fontsize=fontsize, fontweight='bold'
        )

        # Optionally, expand the xlim to make sure arrows and labels are visible
        ax.set_xlim(xlim[0], x_arrow + 0.75)

    # --- Optionally add schematic to the right of the plot ---
    if show_schematic and schematic_ax is not None:
        # Draw the schematic into the provided axis, aligning top/bottom with barplot
        barplot_ylim = ax.get_ylim()
        # Remove all content from schematic_ax, then fill it with 3 axes that fill the vertical space
        # We'll use inset_axes with bbox_to_anchor covering the full vertical range 
        schematic_ax.set_xticks([])
        schematic_ax.set_yticks([])
        schematic_ax.set_frame_on(False)
        schematic_ax.set_title("")
        schematic_ax.set_xlabel("")
        schematic_ax.set_ylabel("")
        # Remove all children from schematic_ax
        for child in schematic_ax.get_children():
            try:
                child.remove()
            except Exception:
                pass
        # Now, fill schematic_ax with the schematic, using the full vertical space
        plot_ref_percentile_schematic(ax=schematic_ax, show=False, barplot_ylim=barplot_ylim, schematic_heights=schematic_heights, title_loc=schematic_title_loc, show_xlabel=show_schematic_xlabel, show_ylabel=show_schematic_ylabel)

    plt.show()
    # Return both axes if schematic is shown
    if show_schematic and schematic_ax is not None:
        return {"fig": fig, "axes": (ax, schematic_ax), "data": data}
    else:
        return {"fig": fig, "axes": ax, "data": data}


def plot_ref_vep_std_stacked_bar(vep_df, 
                                groupby_cols = ['model_location','protein','clinsig','mutant','scoring_strategy'],
                                y='VEP_percentile',
                                n_bins=10, 
                                figsize=(9, 4), 
                                label_padding=0.15, 
                                is_ref=True,
                                title="Standard Deviations Separating REF VEP from full VEP Distribution Mean",
                                x_label="VEP Quantile",
                                y_label="Proportion of Variants"):
    """
    Plot a stacked bar plot of VEP percentiles, binned by quantiles and standard deviation categories.
    Adds arrows and labels to indicate under/overestimation of pathogenicity.
    Returns the matplotlib figure and axis, and the processed data.
    """
    from matplotlib import cm
    import matplotlib.pyplot as plt

    vep_df = vep_df.copy()
    data = va.compute_representativeness_stats(vep_df, groupby_cols=groupby_cols, y=y, is_ref=is_ref)

    # Bin VEP into quantile bins and get bin edges for labeling
    vep_binned, bin_edges = pd.qcut(data['VEP_mean'], q=n_bins, labels=False, retbins=True, duplicates='drop')
    data = data.copy()
    data['VEP_binned'] = vep_binned

    # Create bin range labels as strings, e.g. "0.12–0.34"
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        left_str = f"{left:.2g}" if abs(left) < 1e4 else f"{left:.2e}"
        right_str = f"{right:.2g}" if abs(right) < 1e4 else f"{right:.2e}"
        bin_labels.append(f"{left_str}\n↓\n{right_str}")

    # Map integer bin codes to string labels
    data['VEP_binned_label'] = data['VEP_binned'].map(lambda x: bin_labels[int(x)] if pd.notnull(x) else np.nan)

    # For each row, compute (VEP - VEP_mean) / VEP_std using the row's own mean and std
    def pct_group(row):
        if pd.isnull(row['VEP']) or pd.isnull(row['VEP_mean']) or pd.isnull(row['VEP_std']) or row['VEP_std'] == 0:
            return np.nan
        return np.round((row['VEP'] - row['VEP_mean']) / row['VEP_std'], 1)

    data['VEP_pct_group'] = data.apply(pct_group, axis=1)

    # For plotting, bin VEP_pct_group into categories with a central bin of +/-0.5 SD
    bins = [-np.inf, -2, -1, -0.5, 0.5, 1, 2, np.inf]
    labels = ['<-2', '-1 → -2', '-0.5 → -1', '-0.5 ↔ 0.5', '0.5 → 1', '1 → 2', '>2']
    data['VEP_pct_group_cat'] = pd.cut(data['VEP_pct_group'], bins=bins, labels=labels)

    # Prepare data for stacked bar plot (as proportions), using the string bin labels for x-axis
    stacked = data.groupby(['VEP_binned_label', 'VEP_pct_group_cat']).size().unstack(fill_value=0)
    stacked_prop = stacked.div(stacked.sum(axis=1), axis=0)  # Proportion (0-1)

    # Ensure the x-axis bins are in the correct order
    stacked_prop = stacked_prop.reindex(bin_labels)

    # Get the number of categories for the color map
    n_cats = len(labels)
    cmap = cm.get_cmap('coolwarm', n_cats)
    colors = [cmap(i) for i in range(n_cats)]

    # Flip the order of the columns (categories) and colors for the legend (bottom to top)
    reversed_labels = labels[::-1]
    reversed_colors = colors[::-1]
    stacked_prop = stacked_prop[reversed_labels]

    fig, ax = plt.subplots(figsize=figsize)
    stacked_prop.plot(
        kind='bar', 
        stacked=True, 
        ax=ax,
        color=reversed_colors,
        width=0.95
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1], 
        reversed_labels, 
        title='Standard\nDeviations', 
        bbox_to_anchor=(-0.15, 1),
        loc='upper right',
        borderaxespad=0.0
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()

    # --- Add arrows and labels along the y-axis, outside the right margin ---
    ymin, ymax = ax.get_ylim()
    ycenter = 0.5
    xlim = ax.get_xlim()
    x_arrow = xlim[1] + 0.1
    arrow_length = (ymax - ymin) * 0.35
    fontsize = 8

    ax.annotate(
        "",
        xy=(x_arrow, ycenter + arrow_length),
        xytext=(x_arrow, ycenter),
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
        annotation_clip=False
    )
    ax.text(
        x_arrow + 0.08,
        ycenter + arrow_length/2 + label_padding/2,
        r"$\mathbf{VEP_{REF}}$"+"\n"+r"underestimates"+"\n"+r"pathogenicity",
        va='center', ha='left', rotation=90, fontsize=fontsize, fontweight='bold'
    )

    ax.annotate(
        "",
        xy=(x_arrow, ycenter - arrow_length),
        xytext=(x_arrow, ycenter),
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
        annotation_clip=False
    )
    ax.text(
        x_arrow + 0.08,
        ycenter - arrow_length/2 - label_padding/2,
        r"$\mathbf{VEP_{REF}}$"+"\n"+r"overestimates"+"\n"+r"pathogenicity",
        va='center', ha='left', rotation=90, fontsize=fontsize, fontweight='bold'
    )

    ax.set_xlim(xlim[0], x_arrow + 0.75)
    plt.show()
    return {"fig": fig, "ax": ax, "data": data}

def plot_ref_vep_diff_stacked_bar(vep_df, 
                                groupby_cols = ['model_location','protein','clinsig','mutant','scoring_strategy'],
                                y='VEP_mean_diff',
                                n_bins=5, 
                                figsize=(9, 4), 
                                label_padding=0.15, 
                                is_ref=True,
                                title="VEP Differences (REF - Full) by VEP Quantile",
                                x_label="VEP Quantile",
                                y_label="Proportion of Variants",
                                n_diff_bins=5):
    """
    Plot a stacked bar plot of VEP percentiles, binned by quantiles and VEP difference categories.
    Uses VEP_diff column instead of standard deviations.
    Adds arrows and labels to indicate under/overestimation of pathogenicity.
    Returns the matplotlib figure and axis, and the processed data.

    Parameters
    ----------
    vep_df : pd.DataFrame
        DataFrame containing VEP results.
    groupby_cols : list, optional
        Columns to group by for computing representativeness stats (default: ['model_location','protein','clinsig','mutant','scoring_strategy']).
    y : str, optional
        Column name for y-axis values (default: 'VEP_percentile').
    n_bins : int, optional
        Number of quantile bins for VEP_mean (default: 5).
    figsize : tuple, optional
        Figure size (default: (9, 4)).
    label_padding : float, optional
        Vertical padding between arrow labels (default: 0.15).
    is_ref : bool, optional
        Whether to compute REF statistics (default: True).
    title : str, optional
        Plot title (default: "VEP Differences (REF - Full) by VEP Quantile").
    x_label : str, optional
        X-axis label (default: "VEP Quantile").
    y_label : str, optional
        Y-axis label (default: "Proportion of Variants").
    n_diff_bins : int, optional
        Number of bins for VEP_diff categorization (default: 5).
    """
    from matplotlib import cm
    import matplotlib.pyplot as plt

    vep_df = vep_df.copy()
    data = va.compute_representativeness_stats(vep_df, groupby_cols=groupby_cols, y=y, is_ref=is_ref)
    
    # Check what columns are available in the data
    print(f"Available columns in data: {data.columns.tolist()}")
    print(f"Looking for column: {y}")
    
    # Ensure the y column exists
    if y not in data.columns:
        available_cols = [col for col in data.columns if 'diff' in col.lower() or 'std' in col.lower()]
        if available_cols:
            print(f"Column '{y}' not found. Available similar columns: {available_cols}")
            y = available_cols[0]  # Use the first available column
            print(f"Using column: {y}")
        else:
            raise ValueError(f"Column '{y}' not found in data. Available columns: {data.columns.tolist()}")

    # Bin VEP into quantile bins and get bin edges for labeling
    vep_binned, bin_edges = pd.qcut(data["VEP_mean"], q=n_bins, labels=False, retbins=True, duplicates='drop')
    data = data.copy()
    data['VEP_binned'] = vep_binned

    # Create bin range labels as strings, e.g. "0.12–0.34"
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        left_str = f"{left:.2g}" if abs(left) < 1e4 else f"{left:.2e}"
        right_str = f"{right:.2g}" if abs(right) < 1e4 else f"{right:.2e}"
        bin_labels.append(f"{left_str}\n→\n{right_str}")

    # Map integer bin codes to string labels
    data['VEP_binned_label'] = data['VEP_binned'].map(lambda x: bin_labels[int(x)] if pd.notnull(x) else np.nan)

    # Use VEP_diff column directly for categorization
    # Bin VEP_diff into categories based on n_diff_bins parameter
    if n_diff_bins == 5:
        # Default 5-bin categorization
        bins = [-np.inf, -0.5, -0.2, 0.2, 0.5, np.inf]
        labels = ['<-0.5', '-0.5 → -0.2', '-0.2 ↔ 0.2', '0.2 → 0.5', '>0.5']
    elif n_diff_bins == 3:
        # 3-bin categorization
        bins = [-np.inf, -0.2, 0.2, np.inf]
        labels = ['<-0.2', '-0.2 ↔ 0.2', '>0.2']
    elif n_diff_bins == 7:
        # 7-bin categorization
        bins = [-np.inf, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, np.inf]
        labels = ['<-1.0', '-1.0 → -0.5', '-0.5 → -0.2', '-0.2 ↔ 0.2', '0.2 → 0.5', '0.5 → 1.0', '>1.0']
    else:
        # Dynamic binning based on n_diff_bins
        # Create evenly spaced bins around 0
        max_diff = data[y].abs().max()
        if max_diff > 0:
            bin_edges = np.linspace(-max_diff, max_diff, n_diff_bins + 1)
            bins = [-np.inf] + list(bin_edges[1:-1]) + [np.inf]
            labels = []
            for i in range(len(bins) - 1):
                if i == 0:
                    labels.append(f'<{bins[1]:.2f}')
                elif i == len(bins) - 2:
                    labels.append(f'>{bins[-2]:.2f}')
                else:
                    labels.append(f'{bins[i]:.2f} → {bins[i+1]:.2f}')
        else:
            # Fallback to default 5-bin if no variation
            bins = [-np.inf, -0.5, -0.2, 0.2, 0.5, np.inf]
            labels = ['<-0.5', '-0.5 → -0.2', '-0.2 ↔ 0.2', '0.2 → 0.5', '>0.5']
    
    data[y+"_cat"] = pd.cut(data[y], bins=bins, labels=labels)
    
    # Check if we have the expected columns for grouping
    print(f"Columns after categorization: {data.columns.tolist()}")
    print(f"VEP_binned_label unique values: {data['VEP_binned_label'].unique()}")
    print(f"Category column '{y}_cat' unique values: {data[y+'_cat'].unique()}")

    # Prepare data for stacked bar plot (as proportions), using the string bin labels for x-axis
    cat_column = y+"_cat"
    stacked = data.groupby(['VEP_binned_label', cat_column]).size().unstack(fill_value=0)
    stacked_prop = stacked.div(stacked.sum(axis=1), axis=0)  # Proportion (0-1)

    # Ensure the x-axis bins are in the correct order
    stacked_prop = stacked_prop.reindex(bin_labels)

    # Get the number of categories for the color map
    n_cats = len(labels)
    cmap = cm.get_cmap('coolwarm', n_cats)
    colors = [cmap(i) for i in range(n_cats)]

    # Flip the order of the columns (categories) and colors for the legend (bottom to top)
    reversed_labels = labels[::-1]
    reversed_colors = colors[::-1]
    stacked_prop = stacked_prop[reversed_labels]

    fig, ax = plt.subplots(figsize=figsize)
    stacked_prop.plot(
        kind='bar', 
        stacked=True, 
        ax=ax,
        color=reversed_colors,
        width=0.95
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1], 
        reversed_labels, 
        title=f'{y.replace("_", " ").title()}\nCategories', 
        bbox_to_anchor=(-0.15, 1),
        loc='upper right',
        borderaxespad=0.0
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()

    # --- Add arrows and labels along the y-axis, outside the right margin ---
    ymin, ymax = ax.get_ylim()
    ycenter = 0.5
    xlim = ax.get_xlim()
    x_arrow = xlim[1] + 0.1
    arrow_length = (ymax - ymin) * 0.35
    fontsize = 8

    ax.annotate(
        "",
        xy=(x_arrow, ycenter + arrow_length),
        xytext=(x_arrow, ycenter),
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
        annotation_clip=False
    )
    ax.text(
        x_arrow + 0.08,
        ycenter + arrow_length/2 + label_padding/2,
        r"$VEP_{REF}$ underestimates\npathogenicity",
        va='center', ha='left', rotation=90, fontsize=fontsize, fontweight='bold'
    )

    ax.annotate(
        "",
        xy=(x_arrow, ycenter - arrow_length),
        xytext=(x_arrow, ycenter),
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
        annotation_clip=False
    )
    ax.text(
        x_arrow + 0.08,
        ycenter - arrow_length/2 - label_padding/2,
        r"$VEP_{REF}$ overestimates\npathogenicity",
        va='center', ha='left', rotation=90, fontsize=fontsize, fontweight='bold'
    )

    ax.set_xlim(xlim[0], x_arrow + 0.75)
    plt.show()
    return {"fig": fig, "ax": ax, "data": data}


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
                        ylim=(0,100),
                        height=3,
                        aspect=.9,
                        title_y=1,
                        x_rotation=45,
                        x_ha='right',
                        sharex=True,
                        sharey=True,
                        save_path=None,
                        invert_xaxis=True,
                        cut=0,  # Parameter to control violin plot distribution inference (0 = no inference beyond data) 
                        **kwargs):

    def _format_label(label):
        # Replace "path" with "pathogenic" and "_" with " "
        label = label.replace("path", "pathogenic")
        label = label.replace("_", " ")
        return label

    if suptitle is None and is_ref:
        suptitle = 'Reference Representativeness'
    elif suptitle is None and not is_ref:
        suptitle = _format_label(y)

    pct_df = va.compute_representativeness_stats(vep_df, groupby_cols=groupby_cols, y=y, is_ref=is_ref)

    # Sort by scoring strategy
    pct_df = utils.sort_by_reverse_string(pct_df, 
                                                column='scoring_strategy', 
                                                extra_sort_cols=['model_location','clinsig'],
                                                ascending=[False, True, True])
    
    pct_stats = pct_df.loc[pct_df['is_ref']==True]['VEP_percentile'].describe()

    g = sns.FacetGrid(data=pct_df, 
                      col=col, 
                      row=row, 
                      height=height, 
                      aspect=aspect, 
                      margin_titles=True, 
                      ylim=ylim,
                      sharex=sharex, 
                      sharey=sharey)
    
    # Handle hue=None gracefully
    map_df_kwargs = dict(x=x, y=y)
    if hue is not None:
        map_df_kwargs['hue'] = hue
    if palette is not None and hue is not None:
        map_df_kwargs['palette'] = palette

    # Add cut parameter to violin plot to control distribution inference
    if func == sns.violinplot:
        map_df_kwargs['cut'] = cut
        g.map_dataframe(func, **map_df_kwargs, **kwargs)
    else:
        g.map_dataframe(func, **map_df_kwargs, **kwargs)
        
    # Add a dotted horizontal line at 0.5
    for ax in g.axes.flat:
        ax.axhline(y=50, linestyle='--', color='gray', alpha=0.7)

    # Add lines above and below the 50th percentile showing the STD
    for ax in g.axes.flat:
        ax.axhline(y=pct_stats['50%'] - pct_stats['std'], linestyle=':', color='gray', alpha=0.7)
        ax.text(1.01, pct_stats['50%'] - pct_stats['std'], "-1 SD", transform=ax.get_yaxis_transform(), 
                ha='left', va='center', color='gray', alpha=0.7)
        ax.axhline(y=pct_stats['50%'] + pct_stats['std'], linestyle=':', color='gray', alpha=0.7)
        ax.text(1.01, pct_stats['50%'] + pct_stats['std'], "+1 SD", transform=ax.get_yaxis_transform(), 
                ha='left', va='center', color='gray', alpha=0.7)

    # Count the number of unique haplotypes per group and update x-axis labels
    if 'haplotype' in pct_df.columns:
        for ax in g.axes.flat:
            if not ax.get_xlabel():
                continue
            
            # Get the current tick labels
            tick_labels = [item.get_text() for item in ax.get_xticklabels()]
            
            # Count unique haplotypes for each group
            counts = {}
            for label in tick_labels:
                # Unformat label for lookup: reverse _format_label
                lookup_label = label.split('\n')[0]  # Remove (n=...) if present
                # Try to reverse the formatting for lookup
                # Replace "pathogenic" with "path" and " " with "_"
                lookup_label_raw = lookup_label.replace("pathogenic", "path").replace(" ", "_")
                if lookup_label_raw in pct_df[x].values:
                    counts[label] = pct_df[pct_df[x] == lookup_label_raw]['haplotype'].nunique()
                else:
                    # Try original label as fallback
                    if lookup_label in pct_df[x].values:
                        counts[label] = pct_df[pct_df[x] == lookup_label]['haplotype'].nunique()
                    else:
                        counts[label] = 0
            
            # Update labels with counts of unique haplotypes and apply formatting
            new_labels = [f"{_format_label(label)}\n(n={counts.get(label, 0)})" for label in tick_labels]
            ax.set_xticklabels(new_labels, rotation=x_rotation, ha=x_ha)

    # Remove subplot titles and add margin titles
    g.figure.suptitle(suptitle, y=title_y)  # Remove overall title if any

    # Adjust y-axis text label content
    for ax in g.axes.flat:
        ax.set_ylabel("Reference Sequence VEP Percentile")
    
    # Adjust x-axis labels
    for ax in g.axes.flat:
        ax.set_xlabel(f"{_format_label(x)} (unique haplotypes)")

    va.rm_subplot_prefixes(g)

    if invert_xaxis:
        g.axes.flat[0].invert_xaxis() 

    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path)
        
    plt.show()

    return {'fig':g, 'axes':ax, 'data':pct_df}

def fix_clinsig_labels(df, 
                       palette=None, 
                       clinsig_col="clinsig", 
                       mutant_col="mutant"):
    
    # Normalize 'clinsig' column
    df[clinsig_col] = df[clinsig_col].str.replace("path$", "pathogenic", regex=True).str.replace("_", " ")

    # Get palette and normalize its keys
    if palette is None:
        palette = utils.get_clinsig_palette()
    palette = {k.replace("path$", "pathogenic").replace("_", " "): v for k, v in palette.items()}

    # Recompute mutant counts after normalization
    if mutant_col is not None and mutant_col in df.columns:
        mutant_counts = df.groupby(clinsig_col)[mutant_col].nunique()
    else:
        mutant_counts = None
        
    clinsig_label_map = {
        clinsig: f"{clinsig} ({mutant_counts.get(clinsig, 0)} variants)" if mutant_counts is not None else clinsig
        for clinsig in df[clinsig_col].unique()
    }
    df[clinsig_col+"_label"] = df[clinsig_col].map(clinsig_label_map)
    palette = {clinsig_label_map.get(k, k): v for k, v in palette.items()}

    return df, palette

def plot_vep_histogram_with_arrows(
    vep_df,
    model_name=None, 
    min_haplotype_seq_len_pct=None,
    figsize=(9, 4),
    add_arrows=True,
    palette=utils.get_clinsig_palette(),
    arrow_y=-0.25,
    arrow_length=0.25,
    arrow_head_width=0.012,
    arrow_head_length=0.065,
    arrow_linewidth=0,
    legend_title="Clinical Signifance",
    external_legend_annotation=False,
    arrow_text_fontsize=11,
    x_label=r"$VEP_{mean}$",
    y_label="Probability",
    title="Variant Effect Prediction (VEP) Distributions",
):
    """
    Plot a histogram of VEP scores by clinical significance, with custom arrows and annotation.

    Parameters
    ----------
    vep_df : pd.DataFrame
        DataFrame containing VEP results, must have columns: 'haplotype', 'protein', 'haplotype_sequence_len_pct', 'model_location', 'scoring_strategy', 'mutant', 'clinsig', 'VEP'.
    model_name : str
        Name of the model (for annotation).
    utils : module
        Module with get_clinsig_palette().
    min_haplotype_seq_len_pct : float
        Minimum percent of haplotype sequence length to include.
    figsize : tuple
        Figure size.
    arrow_y : float
        Y position of the arrows (axes fraction).
    arrow_length : float
        Length of the arrows (axes fraction).
    arrow_head_width : float
        Width of the arrow head (axes fraction).
    arrow_head_length : float
        Length of the arrow head (axes fraction).
    arrow_linewidth : float
        Line width of the arrows.
    legend_title : str
        Title for the legend.
    external_legend_annotation : bool, optional
        If True, places the legend and annotation text outside the plot to the right (default: False).
    arrow_text_fontsize : int, optional
        Font size for the text labels on the bottom arrows (default: 11).
    x_label : str, optional
        Label for the x-axis (default: "VEP Score").
    y_label : str, optional
        Label for the y-axis (default: "Probability").
    title : str, optional
        Title for the plot (default: None, no title).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import matplotlib.patches as mpatches

    vep_df = vep_df.copy()

    n_haplotypes = vep_df["haplotype"].nunique()
    n_proteins = vep_df["protein"].nunique()

    # Filter and aggregate
    if min_haplotype_seq_len_pct is not None:
        if 'haplotype_sequence_len_pct' not in vep_df.columns:
            raise ValueError("min_haplotype_seq_len_pct is set, but 'haplotype_sequence_len_pct' column is not present in the dataframe")
        vep_df = vep_df.loc[vep_df['haplotype_sequence_len_pct'] > min_haplotype_seq_len_pct]

    hist_df = vep_df \
        .groupby(['model_location', 'scoring_strategy', "protein", "mutant", "clinsig"]) \
        .agg({"VEP": "mean", "haplotype": "count"}).reset_index()

    # Normalize 'clinsig' column
    hist_df, palette_labels = fix_clinsig_labels(hist_df, clinsig_col="clinsig", mutant_col="mutant")

    # Ensure 'clinsig' is a categorical with the palette order
    clinsig_order = list(palette_labels.keys())
    hist_df['clinsig'] = pd.Categorical(hist_df['clinsig'], categories=clinsig_order, ordered=True)
    hist_df = hist_df.sort_values('clinsig')

    fig, ax = plt.subplots(figsize=figsize)
    hist = sns.histplot(
        hist_df,
        x='VEP',
        hue='clinsig_label',
        multiple='layer',
        stat="probability",
        palette=palette_labels,
        ax=ax
    )

    # Set axis labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    # Change legend title and position
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title(legend_title)
        if external_legend_annotation:
            # Move legend outside the plot to the right
            legend.set_bbox_to_anchor((1.02, 1.0))
            legend.set_loc('upper left')

    # Concise annotation string with italic prefixes
    lines = [
        r'$\it{proteins:}$ ' + str(n_proteins),
        r'$\it{haplotypes:}$ ' + str(n_haplotypes),
    ]
    if model_name is not None:
        lines.append(r'$\it{model:}$ ' + str(model_name))
    textstr = '\n'.join(lines)
    
    if external_legend_annotation:
        # Place annotation text outside the plot to the right, below the legend
        # Position it lower to avoid overlap with the legend
        ax.text(
            1.02, 0.5, textstr, fontsize=12, va='top', ha='left',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )
    else:
        # Place annotation text inside the plot (original behavior)
        ax.text(
            0.02, 0.98, textstr, fontsize=12, va='top', ha='left',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                )
    
    if external_legend_annotation:
        # Adjust layout to make room for external legend and annotation
        plt.subplots_adjust(right=0.75)
    else:
        plt.tight_layout()
    
    if add_arrows:  
        # _draw_vep_direction_arrows_chunky(
        #     ax=ax,
        #     fig=fig,
        #     arrow_y=arrow_y,
        #     arrow_length=arrow_length,
        #     arrow_head_width=arrow_head_width,
        #     arrow_head_length=arrow_head_length,
        #     arrow_linewidth=arrow_linewidth,
        #     arrow_text_fontsize=arrow_text_fontsize,
        #     external_legend_annotation=external_legend_annotation
        # )
        _draw_vep_direction_arrows(ax, palette, reverse=True) 

    plt.show()
    return {'fig':fig, 'axes':ax, 'data':hist_df}



def plot_vep_variance(vep_df,
                      groupby_cols = ['model_location','protein','clinsig','mutant','scoring_strategy'],
                      x='clinsig',
                      y='VEP_variance',
                      hue='clinsig',
                      row='model_location',
                      col='scoring_strategy',
                      normalize=True,
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
    if normalize:
        vep_df["VEP"] = vep_df["VEP"]/vep_df["VEP"].max() 
    vep_variance = vep_df.groupby(groupby_cols)["VEP"].var().reset_index().rename(columns={"VEP":'VEP_variance'})

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

    va.rm_subplot_prefixes(g)
    va.add_legend(g, palette=palette) 

    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()

    if return_df:
        return vep_variance
    

def _draw_vep_direction_arrows(ax, palette, reverse=False):
    """
    Draws arrows and labels underneath the x-axis to indicate 'Pathogenic' and 'Benign' directions.
    If reverse=True, swaps the directions and sides of 'Pathogenic' and 'Benign'.
    """
    # Get axis limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Move arrows and text further down
    arrow_y = ymin - 0.12 * (ymax - ymin)
    label_y = arrow_y - 0.04 * (ymax - ymin)

    # Compute center and spacing for arrows
    center_x = (xmin + xmax) / 2
    arrow_inner_offset = 0.10 * (xmax - xmin)  # distance from center to inner end of each arrow
    arrow_length = 0.20 * (xmax - xmin)        # length of each arrow

    # Use palette keys, fallback to "path" for "pathogenic"
    path_color = palette.get("pathogenic", palette.get("path", "#d62728"))
    benign_color = palette.get("benign", "#1f77b4")

    if not reverse:
        # Red right arrow for "Pathogenic"
        right_arrow_start = center_x + arrow_inner_offset
        right_arrow_end = right_arrow_start + arrow_length
        ax.annotate(
            '',
            xy=(right_arrow_end, arrow_y),
            xytext=(right_arrow_start, arrow_y),
            arrowprops=dict(facecolor=path_color, edgecolor=path_color, arrowstyle='->', lw=2),
            annotation_clip=False
        )
        ax.text(
            (right_arrow_start + right_arrow_end) / 2, label_y, "Pathogenic", 
            color=path_color, ha='center', va='top', fontsize=12, fontweight='bold'
        )

        # Blue left arrow for "Benign"
        left_arrow_start = center_x - arrow_inner_offset
        left_arrow_end = left_arrow_start - arrow_length
        ax.annotate(
            '',
            xy=(left_arrow_end, arrow_y),    # arrow tip (left)
            xytext=(left_arrow_start, arrow_y),  # arrow tail (right, closer to center)
            arrowprops=dict(facecolor=benign_color, edgecolor=benign_color, arrowstyle='-|>', lw=2),
            annotation_clip=False
        )
        ax.text(
            (left_arrow_start + left_arrow_end) / 2, label_y, "Benign", 
            color=benign_color, ha='center', va='top', fontsize=12, fontweight='bold'
        )
    else:
        # Red left arrow for "Pathogenic" (now on the left)
        left_arrow_start = center_x - arrow_inner_offset
        left_arrow_end = left_arrow_start - arrow_length
        ax.annotate(
            '',
            xy=(left_arrow_end, arrow_y),
            xytext=(left_arrow_start, arrow_y),
            arrowprops=dict(facecolor=path_color, edgecolor=path_color, arrowstyle='-|>', lw=2),
            annotation_clip=False
        )
        ax.text(
            (left_arrow_start + left_arrow_end) / 2, label_y, "Pathogenic", 
            color=path_color, ha='center', va='top', fontsize=12, fontweight='bold'
        )

        # Blue right arrow for "Benign" (now on the right)
        right_arrow_start = center_x + arrow_inner_offset
        right_arrow_end = right_arrow_start + arrow_length
        ax.annotate(
            '',
            xy=(right_arrow_end, arrow_y),    # arrow tip (right)
            xytext=(right_arrow_start, arrow_y),  # arrow tail (left, closer to center)
            arrowprops=dict(facecolor=benign_color, edgecolor=benign_color, arrowstyle='->', lw=2),
            annotation_clip=False
        )
        ax.text(
            (right_arrow_start + right_arrow_end) / 2, label_y, "Benign", 
            color=benign_color, ha='center', va='top', fontsize=12, fontweight='bold'
        )

def _draw_vep_direction_arrows_chunky(ax, fig, 
                                      subplots_adjust_kwargs={"bottom":.2},
                                      arrow_y=-0.25, 
                                      arrow_length=0.25, 
                                      arrow_head_width=0.012, 
                                      arrow_head_length=0.065, 
                                      arrow_linewidth=0, 
                                      arrow_text_fontsize=11, 
                                      palette=utils.get_clinsig_palette(),
                                      external_legend_annotation=False):
        """
        Add pathogenic and benign arrows underneath the plot.
        """
        import matplotlib.patches as mpatches

        fig.subplots_adjust(**subplots_adjust_kwargs)  # Make room for arrows

        # Helper to convert data coordinate x to axes fraction
        def data_to_axes(x, ax):
            x0, x1 = ax.get_xlim()
            return (x - x0) / (x1 - x0)

        zero_axes = data_to_axes(0, ax) 

        # Adjust arrow parameters when external legend is enabled to prevent overlap
        if external_legend_annotation:
            # Keep arrows visible but adjust positioning for smaller plot area
            adjusted_arrow_length = arrow_length * 0.8  # Moderate reduction
            adjusted_head_width = arrow_head_width * 0.85  # Keep heads visible
            adjusted_head_length = arrow_head_length * 0.85  # Keep heads visible
            arrow_spacing = 0.008  # Reduced spacing to prevent overlap
        else:
            # Use original arrow parameters
            adjusted_arrow_length = arrow_length
            adjusted_head_width = arrow_head_width
            adjusted_head_length = arrow_head_length
            arrow_spacing = 0.02

        # Pathogenic (left) arrow
        left_arrow_start = zero_axes - adjusted_head_width * 2
        left_arrow_end = left_arrow_start - adjusted_arrow_length
        left_arrow = mpatches.FancyArrowPatch(
            (left_arrow_start, arrow_y), (left_arrow_end, arrow_y),
            mutation_scale=25,
            arrowstyle=f'-|>,head_length={int(adjusted_head_length*100)},head_width={int(adjusted_head_width*100)}',
            color=palette.get("path", "#d62728"),
            linewidth=arrow_linewidth,
            transform=ax.transAxes,
            zorder=10,
            clip_on=False
        )
        ax.add_patch(left_arrow)
        left_label_x = (left_arrow_start + left_arrow_end) / 2
        ax.text(
            left_label_x - .04, arrow_y, "pathogenic",
            color="white", fontsize=arrow_text_fontsize, fontweight='bold', ha='left', va='center',
            transform=ax.transAxes, zorder=11
        )

        # Benign (right) arrow
        right_arrow_start = zero_axes + arrow_spacing
        right_arrow_end = right_arrow_start + adjusted_arrow_length
        right_arrow = mpatches.FancyArrowPatch(
            (right_arrow_start, arrow_y), (right_arrow_end, arrow_y),
            mutation_scale=25,
            arrowstyle=f'-|>,head_length={int(adjusted_head_length*100)},head_width={int(adjusted_head_width*100)}',
            color=palette.get("benign", "#2ca02c"),
            linewidth=arrow_linewidth,
            transform=ax.transAxes,
            zorder=10,
            clip_on=False
        )
        ax.add_patch(right_arrow)
        right_label_x = (right_arrow_start + right_arrow_end) / 2
        ax.text(
            right_label_x, arrow_y, "benign",
            color="white", fontsize=arrow_text_fontsize, fontweight='bold', ha='right', va='center',
            transform=ax.transAxes, zorder=11
        )
    
def plot_vep_kde_with_arrows(vep_df, 
                                    vep_col="VEP",
                                    clinsig_col="clinsig",
                                    site_col="site", 
                                    title="VEP Distributions by Clinical Significance",
                                    x_label=r"$VEP_{mean}$",
                                    y_label="Density",  
                                    palette = utils.get_clinsig_palette(),
                                    figsize=(8, 5),
                                    save_path=None,
                                    dpi=300,
                                    plot_kwargs={},
                                    save_kwargs={},
                                    ):
    """
    Plot VEP distributions stratified by clinical significance categories,
    with annotated arrows for 'Pathogenic' and 'Benign' directions.

    Parameters
    ----------
    vep_df : pd.DataFrame
        DataFrame with columns ["clinsig", "site", "VEP"] (and possibly "clinsig_label").
    save_path : str
        Path to save the output figure.
    dpi : int
        DPI of the output figure.
    plot_kwargs : dict
        Keyword arguments for the plot.
    save_kwargs : dict
        Keyword arguments for the save function.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=figsize)
    vep_df = vep_df.dropna(subset=[vep_col,clinsig_col]).groupby([clinsig_col,site_col], observed=True)[vep_col].mean().reset_index(name=vep_col)

    vep_df, palette_labeled = fix_clinsig_labels(vep_df, clinsig_col=clinsig_col, mutant_col=site_col)

    ax = sns.kdeplot(
        data=vep_df,
        x=vep_col,
        hue=clinsig_col+"_label",
        multiple="fill",
        fill=True,
        cut=0, 
        palette=palette_labeled,
        **plot_kwargs
    )
    plt.axvline(x=0.2, color='white', linestyle='--', linewidth=2, alpha=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title) 

    _draw_vep_direction_arrows(ax, palette)

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", **save_kwargs)

    plt.show()

    return {'fig':ax, 'axes':ax, 'data':vep_df}

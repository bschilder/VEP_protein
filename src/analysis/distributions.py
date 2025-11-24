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

import os
from scipy.stats import normaltest
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import BayesianGaussianMixture

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

def get_peaks_bgm(x,
                  max_components=50,
                  random_state=42,
                  bgm_kwargs={}):
    """
    Get the peaks of a list of values using the Bayesian Gaussian Mixture Model.

    Parameters
    ----------
    x : array-like
        The input data as an array-like sequence (e.g., numpy array, pandas Series, or list).
        This should be one-dimensional (shape (n_samples,)) as it will be reshaped for modeling.
    max_components : int, optional
        Maximum number of mixture components to consider in the Bayesian Gaussian Mixture.
        If set higher than the number of data points, it is reduced to len(x).
        Default is 50.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    bgm_kwargs : dict, optional
        Additional keyword arguments to pass to sklearn.mixture.BayesianGaussianMixture.

    Returns
    -------
    bgm : sklearn.mixture.BayesianGaussianMixture
        The fitted BayesianGaussianMixture object.
    component_labels : ndarray of shape (n_samples,)
        Cluster/component assignments for each data point in x.
    n_peaks : int
        Number of unique mixture components assigned, i.e., the number of peaks found in x.

    Notes
    -----
    This function fits a Bayesian Gaussian Mixture Model (BGM) to the univariate data ``x``.
    Peaks are defined as unique mixture components assigned to the data.
    The number of peaks is the count of unique predicted labels.

    #### Parameters  ####
    # Docs: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html
    # Tweak parameters to reduce peak overestimation:
    # - weight_concentration_prior_type="dirichlet_process" encourages fewer, more distinct components
    # - weight_concentration_prior=0.8 (increase from default) for more sparsity (fewer active components)
    # - covariance_type="full" for flexibility, but could use "diag" if desired
    # - init_params="kmeans" can help with initialization

    Example
    -------
    >>> x = np.array([1,2,3,10,11,12])
    >>> bgm, component_labels, n_peaks = get_peaks_bgm(x)
    >>> print(n_peaks)
    2

    """
    from sklearn.mixture import BayesianGaussianMixture
    # Make sure max_components is not greater than the number of data points
    if max_components > len(x):
        max_components = len(x)

    bgm = BayesianGaussianMixture(
        n_components=max_components,
        random_state=random_state,
        **bgm_kwargs
    ).fit(np.array(x).reshape(-1, 1))
    
    component_labels = bgm.predict(np.array(x).reshape(-1, 1))
    n_peaks = len(np.unique(component_labels))
    
    return bgm, component_labels, n_peaks

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
             max_components=50,
             random_state=42,
             bgm_kwargs={},
             title=None,
             error=True):
    """
    Get the Empirical Cumulative Distribution Function of a list of values.
    """
    from KDEpy import FFTKDE
    
    try:
        x_array = np.array(x) # Get the number of peaks in the KDE
        # Ensure x_array has no NaN values
        x_array = x_array[~np.isnan(x_array)]
        assert len(x_array) > 1, "Only one data point, cannot calculate density"

       
       #### KDE + scipy.signal method ####
        # Create a grid that extends beyond the data points to avoid the "Every data point must be inside of the grid" error
        kde_x, kde_y = FFTKDE(bw='ISJ').fit(x_array).evaluate(grid_points=grid_points)
        # n_peaks = get_peaks(kde_y) 

        # Make sure max_components is not greater than the number of data points
        bgm, component_labels, n_peaks = get_peaks_bgm(x_array, 
                                                       max_components=max_components, 
                                                       random_state=random_state, 
                                                       bgm_kwargs=bgm_kwargs)

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
        

def estimate_modality(
    vep_df,
    groupby_cols=['model_location', 'protein', 'scoring_strategy', 'mutant', 'site', 'clinsig'],
    min_haplotypes=None,
    vep_col='VEP',
    save_path='results/vep_ecdf.parquet',
    max_components=50,
    bgm_kwargs={},
    error=False,
    force=False
):
    """
    Estimate the modality (number of modes) of VEP score distributions for grouped data.

    This function computes the empirical cumulative distribution function (ECDF) and related
    statistics for VEP scores, grouped by the specified columns. It also estimates the number
    of modes (peaks) in each group's distribution. Results can be cached to disk for faster
    subsequent loading.

    Parameters
    ----------
    vep_df : pd.DataFrame
        DataFrame containing VEP scores and associated metadata.
    vep_col : str, optional
        Column name containing VEP scores. Default is 'VEP'.
    groupby_cols : list of str, optional
        Columns to group by when estimating modality. Default is
        ['model_location', 'protein', 'scoring_strategy', 'mutant', 'site', 'clinsig'].
    save_path : str or None, optional
        Path to save or load the cached ECDF and modality results. If None, results are not cached.
        Default is 'results/vep_ecdf.parquet'.
    error : bool, optional
        If True, raise an error if get_ecdf() fails. If False, return None. Default is False.
    force : bool, optional
        If True, force recalculation even if a cached file exists. If False and the file exists,
        load the cached results. Default is False.
    bgm_kwargs : dict, optional
        Keyword arguments for the Bayesian Gaussian Mixture Model. Default is {}.
        See https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html
        for available parameters.
         
    Returns
    -------
    vep_ecdf : pd.DataFrame
        DataFrame containing ECDF values, modality estimates, and group identifiers for each group.

    Notes
    -----
    - The function uses `get_ecdf` to compute ECDF and modality for each group.
    - The number of unique haplotypes per group is also calculated and included.
    - If `save_path` is provided, results are cached as a parquet file for future use.
    """
    
    if save_path is not None and os.path.exists(save_path) and not force:
        print("Loading cached ECDF data")
        vep_ecdf = pd.read_parquet(save_path)
    else:
        vep_df = vep_df.copy()
        # Calculate the number of unique haplotypes per group
        vep_df['n_haplotypes'] = vep_df.groupby(groupby_cols)['haplotype'].transform('nunique')

        if min_haplotypes is not None:
            vep_df = vep_df[vep_df['n_haplotypes'] >= min_haplotypes]

        tqdm.pandas(desc="Estimating modality")
        vep_ecdf = (
            vep_df
            .dropna(subset=[vep_col])
            .sort_values([vep_col])
            .groupby(groupby_cols + ['n_haplotypes'])[vep_col]
            .progress_apply(
                get_ecdf,
                return_df=True,
                max_components=max_components,
                bgm_kwargs=bgm_kwargs,
                error=error
            )
            .reset_index()
        )
        # Rename the automatically generated index column for grid points
        vep_ecdf.rename(columns={f'level_{len(groupby_cols)+1}': 'grid_index'}, inplace=True)
        # Create a unique identifier for each group
        vep_ecdf['id'] = vep_ecdf[groupby_cols].astype(str).agg('_'.join, axis=1)

        if save_path is not None:
            print("Caching results ==>", save_path)
            vep_ecdf.to_parquet(save_path)

    # Report
    print(vep_ecdf.shape)
    return vep_ecdf

def plot_estimate_modality(vep_ecdf,
                           x="n_peaks", 
                           site_col="site",
                           hue_col="clinsig",
                           figsize=(8, 6),
                           min_haplotypes=20, 
                           title="VEP Modality",
                           x_label="Mixture Components (Modes)",
                           y_label="Clinical Variants",
                           palette=utils.get_clinsig_palette(),
                           flip_axes=False,
                           legend_title="Clinical Significance",
                           show_percentages=False,
                           percentage_fmt=".1f",
                           legend_loc=None,
                           legend_bbox_to_anchor=None,
                           legend_ncol=1,
                           legend_frameon=False,
                           show_legend=True,
                           **kwargs):
    """
    Plot the estimate modality of VEP scores as a stacked barplot.

    Parameters
    ----------
    flip_axes : bool, optional
        If True, flip the x and y axes to create a horizontal barplot (default: False).
    legend_title : str or None, optional
        Title for the legend. If None, uses hue_col (default: None).
    show_percentages : bool, optional
        If True, show percentage labels above each bar (or to the right if flip_axes=True)
        indicating the percentage of variants with that number of peaks (default: False).
    percentage_fmt : str, optional
        Format string for the percentage labels. Default is ".1f".
    legend_loc : str or int or tuple, optional
        Location of the legend. Can be a string (e.g., 'upper right', 'center', 'best'),
        an int (0-10), or a tuple of (x, y) coordinates. If None, uses matplotlib default.
    legend_bbox_to_anchor : tuple, optional
        Bounding box for the legend. Tuple of (x, y) or (x, y, width, height).
        Used with legend_loc for fine positioning. If None, not used.
    legend_ncol : int, optional
        Number of columns in the legend (default: 1).
    legend_frameon : bool, optional
        Whether to show a frame around the legend (default: False).
    show_legend : bool, optional
        Whether to show the legend (default: True). If False, no legend is displayed.
    """
    import matplotlib.ticker as mticker

    fig, ax = plt.subplots(figsize=figsize)
    
    plot_df = vep_ecdf.loc[vep_ecdf['n_haplotypes'] >= min_haplotypes].copy()
    # Get one row per "site"
    plot_df = plot_df.drop_duplicates(subset=[site_col])
    
    # Clean up clinsig labels: remove underscores and spell out "path" -> "pathogenic"
    if hue_col in plot_df.columns:
        plot_df[hue_col] = plot_df[hue_col].str.replace("path$", "pathogenic", regex=True).str.replace("_", " ")
        # Update palette keys to match cleaned labels
        if palette is not None:
            palette_cleaned = {}
            for k, v in palette.items():
                k_cleaned = k.replace("path", "pathogenic").replace("_", " ")
                palette_cleaned[k_cleaned] = v
            palette = palette_cleaned
    
    numerator = plot_df.loc[(plot_df['n_haplotypes'] >= min_haplotypes) & (plot_df['n_peaks'] > 1), site_col].nunique()
    denominator = plot_df.loc[plot_df['n_haplotypes'] >= min_haplotypes, site_col].nunique()
    percent = numerator / denominator * 100
    print(f"Remaining sites @ n_haplotypes >= {min_haplotypes}: {numerator} / {denominator} = {percent:.10f}%")
    
    # Prepare data for stacked barplot
    stacked_data = (
        plot_df
        .groupby([x, hue_col])[site_col]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )
    stacked_data['total'] = stacked_data.sum(axis=1)

    # Plot stacked barplot with outlined bars
    bottom = None
    for idx, clinsig in enumerate([x for x in stacked_data.columns if x != 'total']):
        values = stacked_data[clinsig]
        if flip_axes:
            ax.barh(
                stacked_data.index,
                values,
                left=bottom,
                label=clinsig,
                color=palette[clinsig] if palette and clinsig in palette else None,
                edgecolor='grey',  # Outline bars
                linewidth=0.8,      # Set outline thickness
                **kwargs
            )
        else:
            ax.bar(
                stacked_data.index,
                values,
                bottom=bottom,
                label=clinsig,
                color=palette[clinsig] if palette and clinsig in palette else None,
                edgecolor='grey',  # Outline bars
                linewidth=0.8,      # Set outline thickness
                **kwargs
            )
        if bottom is None:
            bottom = values
        else:
            bottom = bottom + values

    # Add percentage labels if requested
    if show_percentages:
        total_variants = stacked_data['total'].sum()
        percentages_raw = (stacked_data['total'] / total_variants * 100)
        percentages = percentages_raw.round(1)
        
        # Helper function to convert number to superscript format
        def to_superscript(value):
            """Convert a number to superscript format (e.g., 1.23×10⁻²)"""
            # Format in scientific notation first
            sci_str = f'{value:.2e}'
            # Parse mantissa and exponent
            if 'e' in sci_str:
                mantissa, exp_str = sci_str.split('e')
                exp = int(exp_str)
                
                # Superscript Unicode characters
                superscript_map = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
                exp_superscript = str(exp).translate(superscript_map)
                
                return f'{mantissa}×10{exp_superscript}'
            else:
                return sci_str
        
        # Calculate offset based on data range for better positioning
        max_value = stacked_data['total'].max()
        offset = max_value * 0.02  # 2% of maximum bar value
        
        for peak_num in stacked_data.index:
            pct_rounded = percentages[peak_num]
            pct_raw = percentages_raw[peak_num]
            
            # If rounded percentage is 0.0 but raw value is non-zero, use superscript format
            if abs(pct_rounded) < 0.05 and abs(pct_raw) > 1e-10:  # Rounded to 0.0 but raw value is non-zero
                label_text = f'{to_superscript(pct_raw)}%'
            else:
                label_text = f'{pct_rounded:{percentage_fmt}}%'
            
            if flip_axes:
                # For horizontal bars, place label to the right
                bar_width = stacked_data.loc[peak_num, 'total']
                ax.text(
                    bar_width + offset,
                    peak_num,
                    label_text,
                    ha='left',
                    va='center',
                    fontsize='medium'
                )
            else:
                # For vertical bars, place label above
                bar_height = stacked_data.loc[peak_num, 'total']
                ax.text(
                    peak_num,
                    bar_height + offset,
                    label_text,
                    ha='center',
                    va='bottom',
                    fontsize='medium'
                )

    # Set axis formatter - use x-axis formatter when flipped, y-axis when not
    # Format counts in units of 1000 (e.g., 5000 -> 5k)
    def format_thousands(x, pos):
        """Format number in thousands with 'k' suffix"""
        if x >= 1000:
            return f'{x/1000:.0f}k'
        else:
            return f'{x:.0f}'

    # Set ticks to show all peak numbers
    peak_numbers = stacked_data.index.tolist()

    # --- Ensure clinical variants axis (i.e., the counts axis) uses *evenly spaced*, nicely formatted ticks ---
    import numpy as np

    def get_nice_ticks(vmin, vmax, min_ticks=3, max_ticks=5):
        """
        Return evenly spaced "nice" ticks between vmin and vmax (inclusive), for count axes.
        The result will always include vmin and vmax and be suitable for thousands-formatting.
        """
        if vmin == vmax:
            # Just one value; expand and return min_ticks ticks
            spread = vmin if vmin != 0 else 1
            vmin = 0
            vmax = spread
        # Use MaxNLocator to pick good ticks in integer steps, then round to nearest 100/1000 etc if possible.
        locator = mticker.MaxNLocator(nbins=max_ticks, steps=[1, 2, 2.5, 5, 10], integer=True, prune=None)
        ticks = locator.tick_values(vmin, vmax)
        # Filter ticks to be inside the desired range (inclusive) and round nicely
        ticks = [tick for tick in ticks if vmin <= tick <= vmax]
        if len(ticks) < min_ticks:
            # If too few, fall back to linspace and round to nearest hundred
            ticks = np.linspace(vmin, vmax, min_ticks)
            # Avoid negative/very small values
            ticks = [np.round(tick, -2) if vmax > 1500 else int(round(tick)) for tick in ticks]
        return ticks

    if flip_axes:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_thousands))
        # Set y-axis ticks to show all peak numbers
        ax.set_yticks(peak_numbers)
        ax.invert_yaxis()  # Invert y-axis for horizontal barplot
        # Ensure count axis (x-axis) uses nicely spaced ticks
        count_min = stacked_data['total'].min()
        count_max = stacked_data['total'].max()
        xticks = get_nice_ticks(count_min, count_max)
        ax.set_xticks(xticks)
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_thousands))
        # Set x-axis ticks to show all peak numbers
        ax.set_xticks(peak_numbers)
        # Ensure count axis (y-axis) uses nicely spaced ticks
        count_min = stacked_data['total'].min()
        count_max = stacked_data['total'].max()
        yticks = get_nice_ticks(count_min, count_max)
        ax.set_yticks(yticks)
    
    if title:
        plt.title(title)
    
    # Swap labels when axes are flipped
    if flip_axes:
        if y_label:
            plt.xlabel(y_label)
        if x_label:
            plt.ylabel(x_label)
    else:
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)

    # Handle spines and axis positioning based on flip_axes
    if flip_axes:
        # For horizontal barplot, move x-axis to top
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # Move x-axis tick labels and label to top
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
    else:
        # For vertical barplot, standard positioning
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Use legend_title parameter, defaulting to hue_col if None
    legend_title_to_use = legend_title if legend_title is not None else hue_col

    # Only create legend if show_legend is True
    if show_legend:
        # Build legend kwargs
        legend_kwargs = {'title': legend_title_to_use}
        if legend_loc is not None:
            legend_kwargs['loc'] = legend_loc
        if legend_bbox_to_anchor is not None:
            legend_kwargs['bbox_to_anchor'] = legend_bbox_to_anchor
        if legend_ncol is not None:
            legend_kwargs['ncol'] = legend_ncol

        leg = ax.legend(**legend_kwargs)
        # Set frame visibility
        if leg is not None:
            leg.set_frame_on(legend_frameon)

    plt.show()
    return {'fig': fig, 'axes': ax, "data": stacked_data}


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


def cluster_and_annotate_umap(
    ecdf_umap_df, 
    vep_ecdf_pivot, 
    groupby_cols=['model_location', 'protein', 'scoring_strategy', 'mutant', 'n_haplotypes', 'n_peaks'],
    n_clusters=30, 
    n_samples=None, 
    show_dendrogram=True,
    method="dbscan",
    random_state=42,
    cluster_kwargs={},
    cluster_on_umap=False
):
    """
    Cluster the provided VEP ECDF pivot table or UMAP embedding, assign cluster labels,
    and annotate the UMAP embedding DataFrame with these cluster assignments.

    Args:
        vep_ecdf_pivot (pd.DataFrame): Pivot table containing ECDF and summary statistics for VEP data.
        ecdf_umap_df (pd.DataFrame): DataFrame containing UMAP embedding coordinates and associated metadata.
        n_clusters (int): Number of clusters to form (if applicable).
        n_samples (int): Maximum number of samples to use for clustering (for efficiency).
        show_dendrogram (bool): Whether to display a dendrogram of the hierarchical clustering (if applicable).
        method (str): Clustering method to use. Options: "kmeans", "dbscan". Default is "dbscan".
        cluster_on_umap (bool): If True, cluster on UMAP coordinates instead of ECDF features.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - ecdf_umap_df: The input UMAP DataFrame with an added 'cluster' column.
            - vep_ecdf_pivot: The input pivot table with an added 'cluster' column (if clustering on ECDF features).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if cluster_on_umap:
        # Cluster directly on UMAP coordinates
        umap_cols = [col for col in ecdf_umap_df.columns if col.startswith("UMAP_")]
        data_for_clustering = ecdf_umap_df[umap_cols].copy()
        # Optionally subsample for efficiency
        if n_samples is None or n_samples >= len(data_for_clustering):
            data_sample = data_for_clustering.reset_index(drop=True)
            n_samples = len(data_for_clustering)
        else:
            n_samples = min(n_samples, len(data_for_clustering))
            sampled_indices = np.random.choice(
                len(data_for_clustering), 
                size=n_samples, 
                replace=False
            )
            data_sample = data_for_clustering.iloc[sampled_indices].reset_index(drop=True)
        
        if 'cluster' in ecdf_umap_df.columns:
            ecdf_umap_df = ecdf_umap_df.drop(columns=['cluster'])

        if method.lower() == "kmeans":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            if len(data_for_clustering) > n_samples:
                kmeans.fit(data_sample)
                clusters = kmeans.predict(data_for_clustering)
            else:
                clusters = kmeans.fit_predict(data_for_clustering)
            ecdf_umap_df['cluster'] = clusters + 1  # 1-based cluster labels
            if show_dendrogram:
                print("Dendrogram not available for kmeans clustering.")
            print(f"Number of clusters (kmeans, UMAP): {n_clusters}")

        elif method.lower() == "dbscan":
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(**cluster_kwargs)
            if len(data_for_clustering) > n_samples:
                dbscan.fit(data_sample)
                mask = dbscan.labels_ != -1
                if np.any(mask):
                    from sklearn.neighbors import KNeighborsClassifier
                    knn = KNeighborsClassifier()
                    knn.fit(data_sample[mask], dbscan.labels_[mask])
                    clusters = knn.predict(data_for_clustering)
                else:
                    dbscan_full = DBSCAN(**cluster_kwargs)
                    clusters = dbscan_full.fit_predict(data_for_clustering)
            else:
                clusters = dbscan.fit_predict(data_for_clustering)
            ecdf_umap_df['cluster'] = np.where(clusters == -1, 0, clusters + 1)
            n_found = len(set(ecdf_umap_df['cluster'])) - (1 if 0 in ecdf_umap_df['cluster'].values else 0)
            print(f"Number of clusters (dbscan, UMAP): {n_found} (noise labeled as 0)")
            if show_dendrogram:
                print("Dendrogram not available for DBSCAN clustering.")
        else:
            raise ValueError("Invalid method. Choose 'kmeans' or 'dbscan'.")

        print("Cluster distribution (UMAP):")
        print(pd.Series(ecdf_umap_df['cluster']).value_counts().sort_index())

        # Optionally, propagate cluster labels to vep_ecdf_pivot if possible
        # (not always possible if indices do not match)
        return ecdf_umap_df, vep_ecdf_pivot

    else:
        # Prepare data for clustering: only columns with numeric names
        numeric_cols = [col for col in vep_ecdf_pivot.columns if isinstance(col, (int, float))]
        data_for_clustering = vep_ecdf_pivot[numeric_cols].copy()

        # Optionally subsample for efficiency
        if n_samples is None or n_samples >= len(data_for_clustering):
            data_sample = data_for_clustering.reset_index(drop=True)
            n_samples = len(data_for_clustering)
        else:
            n_samples = min(n_samples, len(data_for_clustering))
            sampled_indices = np.random.choice(
                len(data_for_clustering), 
                size=n_samples, 
                replace=False
            )
            data_sample = data_for_clustering.iloc[sampled_indices].reset_index(drop=True)

        # Clustering
        if method.lower() == "kmeans":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            if len(data_for_clustering) > n_samples:
                kmeans.fit(data_sample)
                clusters = kmeans.predict(data_for_clustering)
            else:
                clusters = kmeans.fit_predict(data_for_clustering)
            vep_ecdf_pivot['cluster'] = clusters + 1  # 1-based cluster labels for consistency
            if show_dendrogram:
                print("Dendrogram not available for kmeans clustering.")
            print(f"Number of clusters (kmeans): {n_clusters}")

        elif method.lower() == "dbscan":
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(**cluster_kwargs)
            if len(data_for_clustering) > n_samples:
                dbscan.fit(data_sample)
                mask = dbscan.labels_ != -1
                if np.any(mask):
                    from sklearn.neighbors import KNeighborsClassifier
                    knn = KNeighborsClassifier()
                    knn.fit(data_sample[mask], dbscan.labels_[mask])
                    clusters = knn.predict(data_for_clustering)
                else:
                    dbscan_full = DBSCAN(**cluster_kwargs)
                    clusters = dbscan_full.fit_predict(data_for_clustering)
            else:
                clusters = dbscan.fit_predict(data_for_clustering)
            vep_ecdf_pivot['cluster'] = np.where(clusters == -1, 0, clusters + 1)
            n_found = len(set(vep_ecdf_pivot['cluster'])) - (1 if 0 in vep_ecdf_pivot['cluster'].values else 0)
            print(f"Number of clusters (dbscan): {n_found} (noise labeled as 0)")
            if show_dendrogram:
                print("Dendrogram not available for DBSCAN clustering.")
        else:
            raise ValueError("Invalid method. Choose 'kmeans' or 'dbscan'.")

        print("Cluster distribution:")
        print(pd.Series(vep_ecdf_pivot['cluster']).value_counts().sort_index())

        # Map cluster assignments to the UMAP DataFrame using groupby columns as keys
        ecdf_umap_df['cluster'] = ecdf_umap_df.set_index(groupby_cols).index.map(
            vep_ecdf_pivot.reset_index().set_index(groupby_cols)['cluster']
        )

        # Compute and display summary statistics for each cluster (if columns exist)
        cluster_stats = None
        if 'n_peaks' in vep_ecdf_pivot.columns and 'n_haplotypes' in vep_ecdf_pivot.columns:
            cluster_stats = vep_ecdf_pivot.reset_index().groupby('cluster').agg({
                'n_peaks': ['mean', 'median', 'min', 'max', 'count'],
                'n_haplotypes': ['mean', 'median', 'min', 'max']
            })
            print("Cluster statistics:")
            print(cluster_stats)

        return ecdf_umap_df, vep_ecdf_pivot

def run_rescale_ecdf(vep_ecdf_pivot):
    """
    Rescale the ECDF values in the input DataFrame to the range for each row,
    based on the 'x_min' and 'x_max' index levels.
    This adds back information about the scale of the ECDF values.

    Parameters
    ----------
    vep_ecdf_pivot : pd.DataFrame
        A DataFrame with a MultiIndex that includes 'x_min' and 'x_max' as levels,
        and numeric columns representing ECDF values at different grid points.

    Returns
    -------
    vep_ecdf_pivot_rescaled : pd.DataFrame
        A DataFrame of the same shape as the input, with numeric columns rescaled
        to the range of the ECDF values for each row according to its 'x_min' and 'x_max' index values.
    """
    x_min = vep_ecdf_pivot.index.get_level_values('x_min')
    x_max = vep_ecdf_pivot.index.get_level_values('x_max')
    num_cols = vep_ecdf_pivot.select_dtypes(include='number').columns
    vep_ecdf_pivot_rescaled = vep_ecdf_pivot.copy()
    vep_ecdf_pivot_rescaled[num_cols] = (
        vep_ecdf_pivot[num_cols].subtract(x_min.values, axis=0)
        .div((x_max - x_min).values, axis=0)
    )
    return vep_ecdf_pivot_rescaled


def ecdf_to_umap(vep_ecdf, 
                 rescale_ecdf=False,
                 columns="grid_index",
                 values="ecdf",
                 groupby_cols=['model_location', 'protein', 'scoring_strategy', 'mutant', 'clinsig']
                 ):
    
    # Create a matrix with both ECDF values and VEP statistics
    vep_ecdf_pivot = vep_ecdf.pivot_table(
        index=['id',]+groupby_cols+['n_haplotypes', 'n_peaks',
            'x_min', 'x_max','x_range', 'x_std', 'x_var'],
        columns=columns,
        values=values
    )

    if rescale_ecdf:
        vep_ecdf_pivot = run_rescale_ecdf(vep_ecdf_pivot)

    # Run UMAP
    model, embedding, nan_indices = utils.run_umap(vep_ecdf_pivot, 
                                                filter=False) 
    # Create a dataframe with the UMAP embedding
    ecdf_umap_df = pd.DataFrame(embedding, 
                                index=vep_ecdf_pivot.index, 
                                columns=[f'UMAP_{i+1}' for i in range(embedding.shape[1])]
                                ).reset_index()
    ecdf_umap_df['x_range_log'] = np.log2(ecdf_umap_df['x_range'])  
    return ecdf_umap_df, vep_ecdf_pivot

def plot_cluster_umap(ecdf_umap_df,
                      x="UMAP_1",
                      y="UMAP_2",
                      hue_var="n_peaks",
                      size_var="n_haplotypes",
                      style_var="scoring_strategy",
                      palette="gnuplot2", 
                      alpha=1,
                      sizes=(.0001, 15),
                      figsize=None,
                       **kwargs): 

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    sns.scatterplot(data=ecdf_umap_df,
                    x=x,
                    y=y,
                    size=size_var,
                    sizes=sizes,
                    alpha=1,
                    hue=hue_var,
                    palette=palette,
                    style=style_var,
                    **kwargs
                    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return {'fig':fig, 'axes':ax, 'data':ecdf_umap_df}




def plot_cluster_umap_with_distribution_plots(
    ecdf_umap_df, 
    vep_ecdf,  
    vep_ecdf_pivot,
    x='UMAP_1',
    y='UMAP_2',
    hue='cluster',
    sizes=(1, 30),
    size='n_haplotypes',
    style='scoring_strategy',
    max_subplots=None, 
    min_distance=0.8, 
    connector_alpha=0.7,
    label_font_size=12, 
    figsize=(20, 15),
    clusters=None,
    palette='gist_rainbow',
    line_color=None,
    subplot_kwargs={"ecdf": {"color": "blue", "label": "ECDF"}, 
                    "kde": {"color": "red", "label": "VEP"}},
    subplot_x="grid_index",
):
    """
    Plot UMAP clusters with distribution plots arranged to minimize connector line crossings.
    """
    from matplotlib.patches import ConnectionPatch
    from scipy.optimize import linear_sum_assignment

    if max_subplots is None:
        max_subplots = len(ecdf_umap_df['cluster'].unique())

    # Create the main figure
    fig = plt.figure(figsize=figsize)
    ax_main = plt.subplot2grid((1, 1), (0, 0))
    scatter = sns.scatterplot(
        data=ecdf_umap_df,
        x=x,
        y=y,
        hue=hue,
        sizes=sizes,
        size=size,
        style=style,
        palette=palette,
        ax=ax_main
    )
    # Remove all margin lines and x-axis tick labels from the main UMAP plot
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['left'].set_visible(False)
    ax_main.spines['bottom'].set_visible(False)
    ax_main.set_xticklabels([])
    ax_main.set_yticklabels([])

    # Calculate cluster centers
    cluster_centers = ecdf_umap_df.groupby('cluster')[[x, y]].mean()

    # Get clusters sorted by their centers
    if clusters is None:
        clusters = sorted(cluster_centers.index)
        clusters = clusters[:min(max_subplots, len(clusters))]

    # Get the limits of the main plot
    x_min, x_max = ax_main.get_xlim()
    y_min, y_max = ax_main.get_ylim()

    # Calculate positions around the border of the main plot (in normalized coordinates)
    n_clusters = len(clusters)
    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    # Use min_distance to control how far plots are from the center (in normalized [-0.5, 0.5])
    positions = []
    for angle in angles:
        pos_x = min_distance * np.cos(angle)
        pos_y = min_distance * np.sin(angle)
        positions.append((pos_x, pos_y))

    # Get the color map used in the scatter plot
    if isinstance(palette, str):
        color_palette = sns.color_palette(palette, n_colors=len(ecdf_umap_df['cluster'].unique()))
        cluster_colors = {cluster: color_palette[i] for i, cluster in enumerate(sorted(ecdf_umap_df['cluster'].unique()))}
    elif isinstance(palette, dict):
        cluster_colors = palette
    else:
        raise ValueError(f"Invalid palette: {palette}")

    # Add cluster labels on the UMAP plot
    for cluster, center in cluster_centers.iterrows():
        if cluster in clusters:
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
            ax_main.text(center[x], center[y], f"{cluster}", 
                         ha='center', va='center', fontsize=label_font_size, 
                         bbox=bbox_props, zorder=10)

    # --- Arrange subplots to minimize connector line crossings ---
    # 1. Get cluster center coordinates (in data coordinates)
    cluster_coords = np.array([cluster_centers.loc[cluster, [x, y]].values for cluster in clusters])
    # 2. Get candidate subplot positions (in normalized [-0.5, 0.5] coordinates)
    subplot_coords = np.array(positions)
    # 3. Convert subplot positions to data coordinates for distance calculation
    subplot_data_coords = np.zeros_like(subplot_coords)
    for i, (pos_x, pos_y) in enumerate(subplot_coords):
        subplot_data_coords[i, 0] = x_min + (x_max - x_min) * (pos_x + 0.5)
        subplot_data_coords[i, 1] = y_min + (y_max - y_min) * (pos_y + 0.5)
    # 4. Compute cost matrix (distance from each cluster center to each subplot position)
    cost_matrix = np.linalg.norm(cluster_coords[:, None, :] - subplot_data_coords[None, :, :], axis=2)
    # 5. Find optimal assignment using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Now, for cluster i, assign subplot at positions[col_ind[i]]

    # --- Create small distribution plots for each cluster ---
    subplot_keys = list(subplot_kwargs.keys())
    for i, cluster in enumerate(clusters):
        subplot_idx = col_ind[i]
        pos_x, pos_y = positions[subplot_idx]

        # Get data for this cluster
        ids = vep_ecdf_pivot.loc[vep_ecdf_pivot['cluster'] == cluster].reset_index()['id'].unique()
        cluster_data = vep_ecdf.loc[vep_ecdf['id'].isin(ids)]

        # Calculate mean ECDF and KDE
        mean_1 = cluster_data.groupby(subplot_x)[subplot_keys[0]].mean()
        mean_2 = cluster_data.groupby(subplot_x)[subplot_keys[1]].mean()

        # Convert normalized position to data coordinates (for possible use)
        data_x = x_min + (x_max - x_min) * (pos_x + 0.5)
        data_y = y_min + (y_max - y_min) * (pos_y + 0.5)

        # Create a small inset axes for this cluster
        ax_dist = fig.add_axes([
            ax_main.get_position().x0 + (ax_main.get_position().width * (pos_x + 0.5) * 0.9),
            ax_main.get_position().y0 + (ax_main.get_position().height * (pos_y + 0.5) * 0.9),
            ax_main.get_position().width * 0.15,
            ax_main.get_position().height * 0.15
        ])

        # Plot mean ECDF
        ax_dist.plot(mean_1.index, mean_1.values, 
                     color=subplot_kwargs[subplot_keys[0]]['color'], 
                     label=subplot_keys[0])

        # Create second y-axis for KDE
        ax2 = ax_dist.twinx()
        ax2.plot(mean_2.index, mean_2.values, 
                 color=subplot_kwargs[subplot_keys[1]]['color'], 
                 label=subplot_keys[1])

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
            line_color = line_color if line_color is not None else cluster_colors[cluster]
            con = ConnectionPatch(
                xyA=(center[x], center[y]),
                xyB=(0.5, 0.5),
                coordsA='data',
                coordsB='axes fraction',
                arrowstyle='<|-',
                axesA=ax_main,
                axesB=ax_dist,
                color=line_color,
                alpha=connector_alpha,
                linewidth=2.0
            )
            ax_main.add_artist(con)

    # Add a legend to the main plot
    ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    # Adjust layout
    plt.tight_layout()
    return {'fig': fig, 'axes': ax_main, 'data': {'ecdf_umap_df': ecdf_umap_df, 'vep_ecdf_pivot': vep_ecdf_pivot}}

def test_normality(df, 
                   groupby_cols=None, 
                   value_col="VEP", 
                   min_n=20, 
                   show_progress=True,
                   save_path=None,
                   force=False):
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
    save_path : str or None, optional
        Path to save or load the cached normality results. If None, results are not cached.
    force : bool, default False
        If True, force recalculation even if a cached file exists. If False and the file exists,
        load the cached results.

    Returns
    -------
    pd.DataFrame
        DataFrame with groupby columns and columns: 'statistic', 'pvalue', 'n_samples'.
    """
    if save_path is not None and os.path.exists(save_path) and not force:
        print("Loading cached normality results")
        return pd.read_parquet(save_path)

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

    if save_path is not None:
        print("Saving normality results")
        results.to_parquet(save_path)
    return results

def plot_test_normality(normality_results, 
                        bar_width=0.8, 
                        figsize=(4, 4),
                        legend_bbox_to_anchor=(1.4, 0.6),
                        legend_loc='upper right',
                        title=None,
                        x_label="Clinical Significance",
                        y_label="Proportion of Variants",
                        legend_left=False,
                        legend_columnspacing=0.5,
                        legend_position='bottom',
                        legend_title='Status',
                        xtick_rotation=0,
                        show_xticklabels=True):
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
     legend_columnspacing : float, default 0.5
         Horizontal spacing between legend columns (in units of font size).
     legend_position : str, default 'bottom'
         Position of the legend: 'bottom', 'top' (horizontal), or 'right' (vertical).
     legend_title : str or None, default 'Status'
         Title for the legend. Set to None to remove the legend title.
     xtick_rotation : float, default 0
         Rotation angle in degrees for x-axis tick labels. When rotation is non-zero,
         labels will not be broken into multiple lines.
     show_xticklabels : bool, default True
         If False, omit x-axis tick labels entirely.

    Returns
    -------
    dict
        Dictionary with keys 'fig', 'ax', and 'data' (the proportions DataFrame).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch, Rectangle
    import pandas as pd
    import re

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

    ax.set_xticks(x)
    if show_xticklabels:
        # Format x-axis labels: remove underscores, replace "path" with "pathogenic"
        # If rotation is specified, don't break into multiple lines
        formatted_labels = []
        for label in normality_by_clinsig_prop.index:
            # Remove underscores
            formatted_label = label.replace('_', ' ')
            # Replace "path" with "pathogenic" (handling word boundaries)
            formatted_label = re.sub(r'\bpath\b', 'pathogenic', formatted_label)
            # Break into multiple lines at spaces only if rotation is 0
            if xtick_rotation == 0:
                formatted_label = formatted_label.replace(' ', '\n')
            formatted_labels.append(formatted_label)
        ax.set_xticklabels(formatted_labels, rotation=xtick_rotation, ha='right' if xtick_rotation != 0 else 'center')
    else:
        ax.set_xticklabels([])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if title is None:
        proportion_tested = 1 - normality_by_clinsig_prop["Not testable"].mean()
        title = f'Proportion of Variants\nwith Normal VEP Distributions\n({proportion_tested*100:.1f}% testable)'
    
    # Place legend based on legend_position
    if legend_position == 'top':
        # Horizontal legend at the top
        # Set title higher to avoid overlap with legend
        # Use y parameter to position title above the axes area
        ax.set_title(title, y=1.08, pad=10)
        # Position legend just below title in axes coordinates
        ax.legend(handles=hatch_patches, title=legend_title, loc='lower center', 
                 bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False, 
                 columnspacing=legend_columnspacing)
        # Increase top margin to accommodate both title and legend
        plt.subplots_adjust(top=0.80)
    elif legend_position == 'right':
        # Vertical legend on the right with labels above rectangles
        ax.set_title(title)
        
        # Create custom legend with labels above rectangles
        # Get the legend position
        legend_x = 1.08  # Positioned further to the right with more space from plot
        legend_y_center = 0.5  # Center vertically
        
        # Calculate spacing between legend entries
        n_entries = len(hatch_patches)
        total_height = 0.6  # Total height for all entries
        entry_height = total_height / n_entries
        start_y = legend_y_center + total_height / 2 - entry_height / 2
        
        # Transform to axes coordinates
        transform = ax.transAxes
        
        # Reverse label_map to look up hatch from label
        reverse_label_map = {v: k for k, v in label_map.items()}
        
        # Draw legend entries with labels above
        for i, patch in enumerate(hatch_patches):
            y_pos = start_y - i * entry_height
            label = patch.get_label()
            
            # Get hatch pattern from label
            hatch_key = reverse_label_map.get(label, True)
            hatch_pattern = hatch_map.get(hatch_key, "")
            
            # Draw the patch (rectangle)
            patch_width = 0.15
            patch_height = 0.08
            rect = Rectangle(
                (legend_x, y_pos - patch_height / 2),
                patch_width, patch_height,
                facecolor=patch.get_facecolor(),
                edgecolor=patch.get_edgecolor(),
                hatch=hatch_pattern,
                transform=transform,
                clip_on=False
            )
            ax.add_patch(rect)
            
            # Add label above the rectangle (replace spaces with newlines)
            label_y = y_pos + patch_height / 2 + 0.02
            label_multiline = label.replace(' ', '\n')
            ax.text(
                legend_x + patch_width / 2, label_y,
                label_multiline,
                transform=transform,
                ha='center', va='bottom',
                fontsize='small',
                clip_on=False
            )
        
        # Add legend title if provided
        if legend_title is not None:
            title_y = start_y + entry_height / 2 + 0.05
            ax.text(
                legend_x + patch_width / 2, title_y,
                legend_title,
                transform=transform,
                ha='center', va='bottom',
                fontsize='small',
                fontweight='bold',
                clip_on=False
            )
        
        # Adjust right margin to accommodate legend (more space on left of legend)
        plt.subplots_adjust(right=0.70)
    else:  # 'bottom'
        # Horizontal legend at the bottom
        ax.set_title(title)
        ax.legend(handles=hatch_patches, title=legend_title, loc='upper center', 
                 bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, 
                 columnspacing=legend_columnspacing)
        plt.subplots_adjust(bottom=0.2)
    # Remove top and right borders (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # Report statistics
    normality_mean_prop = normality_by_clinsig_prop.mean(axis=0)
    print(f"Total proteins: {normality_results['protein'].nunique()}")
    print(f"Total sites: {normality_results['site'].nunique()}")
    print(f"Sites with normal distribution (Testable): {normality_results['site'].nunique() * (1-normality_mean_prop['Not testable']):0} ({1-normality_mean_prop['Not testable']:.2%})")
    print(f"Sites with normal distribution (True): {normality_results['site'].nunique() * normality_mean_prop[True]:0} ({normality_mean_prop[True]:.2%})")
    print(f"Sites with non-normal distribution: {normality_results['site'].nunique() * normality_mean_prop[False]:0} ({normality_mean_prop[False]:.2%})")

    return {'fig': fig, 'ax': ax, 'data': normality_by_clinsig_prop}


def plot_ref_percentile_schematic(
    ax=None, 
    show=False, 
    barplot_ylim=None, 
    schematic_heights=[0.35, 0.30, 0.35], 
    title_loc="left", 
    title_fontsize='small',
    title_fontweight=None,
    show_xlabel=(True, True), 
    show_ylabel=(True, True),
    gradient_granularity=None,
    ref_x_positions=(-2.2, 2.2), 
    ref_label=r"$VEP_{ref}$",
    ref_label_fontsize='small',
    ref_label_color='grey',
    ref_label_rotation=0,
    ref_label_side=('left','right'),
    ref_label_offset=None,
    ref_label_y_position=(0.3, 0.3),
    ref_line_kwargs=dict(color='grey', linestyle='--', lw=1),
    ):
    """
    Plot a schematic showing how REF can under- or over-estimate pathogenicity
    using a 3-row grid: top (REF ref_x_positions[0]), blank, bottom (REF ref_x_positions[1]).
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
    title_fontweight : str, optional
        Fontweight for the schematic titles (default: 'bold').
    show_xlabel : tuple of bool, optional
        Whether to show x-axis labels for the schematics (default: (True, True)).
        First boolean controls top schematic, second controls bottom schematic.
    show_ylabel : tuple of bool, optional
        Whether to show y-axis labels for the schematics (default: (True, True)).
        First boolean controls top schematic, second controls bottom schematic.
    gradient_granularity : int or None, optional
        If provided, controls the number of color steps in the gradient fill. If None, uses full resolution.
    ref_x_positions : tuple or list, optional
        X-axis position(s) for the REF line/text in the (top, bottom) subplots (default: (2.2, -2.2)).
    ref_label_side : str, tuple, or None, optional
        Which side of the vertical line to anchor the label on. Can be 'left' or 'right'.
        If tuple, specifies (top, bottom) sides separately. If None, automatically determines
        based on ref_x_positions (default: None).
    ref_label_offset : float, tuple, or None, optional
        Offset distance from the vertical line for the label. Positive values move the label
        away from the line. If tuple, specifies (top, bottom) offsets separately. If None,
        uses default offset of 0.1 (default: None).
    ref_label_y_position : float, tuple, or None, optional
        Y-axis position for the ref label in data coordinates. If tuple, specifies (top, bottom)
        y-positions separately. If None, uses default y-position of 0.25 (default: None).

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
    cmap = LinearSegmentedColormap.from_list("red_blue", [palette["benign"], palette["path"]])

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

    # Helper: get REF label text position alignment depending on ref_x location (left/right)
    def get_ref_text_align(ref_x, side=None):
        # If side is explicitly provided, use it
        if side is not None:
            return 'left' if side == 'right' else 'right'
        # Otherwise, automatically determine: If REF marker at left, align text right; if at right, align left
        return 'left' if ref_x > 0 else 'right'
    
    # Helper: calculate label position and alignment
    def get_label_position_and_alignment(ref_x, side=None, offset=None):
        """
        Calculate the x position and horizontal alignment for the ref label.
        
        Returns:
            x_pos: x coordinate for the label
            ha: horizontal alignment ('left' or 'right')
        """
        # Determine side
        if side is None:
            # Auto-determine based on ref_x position
            determined_side = 'right' if ref_x > 0 else 'left'
        else:
            determined_side = side
        
        # Determine offset
        if offset is None:
            offset = 0.1
        
        # Calculate position: if side is 'right', label goes to the right (positive offset)
        # if side is 'left', label goes to the left (negative offset)
        if determined_side == 'right':
            x_pos = ref_x + offset
            ha = 'left'
        else:  # determined_side == 'left'
            x_pos = ref_x - offset
            ha = 'right'
        
        return x_pos, ha

    # The normal curve's max is about 0.4, min is 0
    schematic_ymin = 0
    schematic_ymax = 0.42  # Add tiny bit of extra room at top to prevent cutoff
    if barplot_ylim is not None:
        barplot_ymin, barplot_ymax = barplot_ylim

    # Normalize ref_label_side and ref_label_offset to tuples (top, bottom)
    if ref_label_side is None:
        ref_label_side_top = None
        ref_label_side_bottom = None
    elif isinstance(ref_label_side, (tuple, list)) and len(ref_label_side) == 2:
        ref_label_side_top, ref_label_side_bottom = ref_label_side
    else:
        ref_label_side_top = ref_label_side_bottom = ref_label_side
    
    if ref_label_offset is None:
        ref_label_offset_top = None
        ref_label_offset_bottom = None
    elif isinstance(ref_label_offset, (tuple, list)) and len(ref_label_offset) == 2:
        ref_label_offset_top, ref_label_offset_bottom = ref_label_offset
    else:
        ref_label_offset_top = ref_label_offset_bottom = ref_label_offset
    
    if ref_label_y_position is None:
        ref_label_y_position_top = 0.25
        ref_label_y_position_bottom = 0.25
    elif isinstance(ref_label_y_position, (tuple, list)) and len(ref_label_y_position) == 2:
        ref_label_y_position_top, ref_label_y_position_bottom = ref_label_y_position
    else:
        ref_label_y_position_top = ref_label_y_position_bottom = ref_label_y_position

    if ax is not None:
        fig = ax.figure
        axs = []

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
                blank_ax.set_facecolor('none')
                axs.append(blank_ax)
            else:
                sub_ax = fig.add_axes([parent_x0, y_pos, parent_width, height])
                sub_ax.set_facecolor('none')
                axs.append(sub_ax)

        ax_top, ax_blank, ax_bottom = axs

        # Set ylims to align with barplot if provided
        ax_top.set_ylim(schematic_ymin, schematic_ymax)
        ax_bottom.set_ylim(schematic_ymin, schematic_ymax)

        # Top subplot: REF at custom x position
        ax_top.plot(x, y, color='black', lw=2)
        gradient_fill(ax_top, x, y, cmap, alpha=0.7, granularity=gradient_granularity)
        ref_x_top = ref_x_positions[0]
        ax_top.axvline(ref_x_top, **ref_line_kwargs, zorder=10)
        label_x_top, label_ha_top = get_label_position_and_alignment(
            ref_x_top, side=ref_label_side_top, offset=ref_label_offset_top
        )
        ax_top.text(label_x_top, ref_label_y_position_top, ref_label, 
                    color=ref_label_color, fontsize=ref_label_fontsize, fontweight=None, va='center',
                    ha=label_ha_top, rotation=ref_label_rotation)
        if show_ylabel[0]:
            ax_top.set_ylabel("Density")
        ax_top.set_yticks([])
        ax_top.set_title(
            r"$\mathbf{VEP_{ref}\ underestimates}$" + "\n" + r"$\mathbf{pathogenicity}$", 
            fontsize=title_fontsize, 
            loc=title_loc,
            fontweight=title_fontweight
        )
        if show_xlabel[0]:
            ax_top.set_xlabel(r"$VEP_{ref}$ percentile")
        ax_top.set_xlim(-3, 3)
        ax_top.set_xticks([-3, 0, 3])
        ax_top.set_xticklabels([])
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['left'].set_visible(False)
        ax_top.spines['top'].set_visible(False)

        ax_blank.axis('off')

        # Bottom subplot: REF at custom x position
        ax_bottom.plot(x, y, color='black', lw=2)
        gradient_fill(ax_bottom, x, y, cmap, alpha=0.7, granularity=gradient_granularity)
        ref_x_bottom = ref_x_positions[1]
        ax_bottom.axvline(ref_x_bottom, **ref_line_kwargs, zorder=10)
        label_x_bottom, label_ha_bottom = get_label_position_and_alignment(
            ref_x_bottom, side=ref_label_side_bottom, offset=ref_label_offset_bottom
        )
        ax_bottom.text(label_x_bottom, ref_label_y_position_bottom, ref_label,
                       color=ref_label_color, fontsize=ref_label_fontsize, fontweight=None, va='center',
                       ha=label_ha_bottom, rotation=ref_label_rotation)
        if show_ylabel[1]:
            ax_bottom.set_ylabel("Density")
        ax_bottom.set_yticks([])
        ax_bottom.set_title(
            r"$\mathbf{VEP_{ref}\ overestimates}$" + "\n" + r"$\mathbf{pathogenicity}$", 
            fontsize=title_fontsize, loc=title_loc,
            fontweight=title_fontweight
        )
        if show_xlabel[1]:
            ax_bottom.set_xlabel(r"$VEP_{ref}$ percentile")
        ax_bottom.set_xlim(-3, 3)
        ax_bottom.set_xticks([-3, 0, 3])
        # If near right, labels left-to-right, else right-to-left
        if ref_x_bottom < 0:
            ax_bottom.set_xticklabels(['0', '50', '100'])
        else:
            ax_bottom.set_xticklabels(['100', '50', '0'])
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
        fig.patch.set_facecolor('none')
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
        ax_top = fig.add_subplot(gs[0])
        ax_top.set_facecolor('none')
        ax_blank = fig.add_subplot(gs[1])
        ax_blank.set_facecolor('none')
        ax_bottom = fig.add_subplot(gs[2], sharex=ax_top)
        ax_bottom.set_facecolor('none')
        axs = [ax_top, ax_blank, ax_bottom]

        ax_top.set_ylim(schematic_ymin, schematic_ymax)
        ax_bottom.set_ylim(schematic_ymin, schematic_ymax)

        # Top subplot: REF at custom x position
        ax_top.plot(x, y, color='black', lw=2)
        gradient_fill(ax_top, x, y, cmap, alpha=0.7, granularity=gradient_granularity)
        ref_x_top = ref_x_positions[0]
        ax_top.axvline(ref_x_top, **ref_line_kwargs, zorder=10)
        label_x_top, label_ha_top = get_label_position_and_alignment(
            ref_x_top, side=ref_label_side_top, offset=ref_label_offset_top
        )
        ax_top.text(label_x_top, ref_label_y_position_top, ref_label,
                    color=ref_label_color, fontsize=ref_label_fontsize, fontweight=None, va='center',
                    ha=label_ha_top, rotation=ref_label_rotation)
        if show_ylabel[0]:
            ax_top.set_ylabel("Density")
        ax_top.set_yticks([])
        ax_top.set_title(r"$VEP_{ref}$ underestimates\npathogenicity", fontweight=title_fontweight, loc=title_loc)
        if show_xlabel[0]:
            ax_top.set_xlabel(r"$VEP_{ref}$ percentile")
        ax_top.set_xlim(-3, 3)
        ax_top.set_xticks([-3, 0, 3])
        ax_top.set_xticklabels([])
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['top'].set_visible(False)

        ax_blank.axis('off')

        # Bottom subplot: REF at custom x position
        ax_bottom.plot(x, y, color='black', lw=2)
        gradient_fill(ax_bottom, x, y, cmap, alpha=0.7, granularity=gradient_granularity)
        ref_x_bottom = ref_x_positions[1]
        ax_bottom.axvline(ref_x_bottom, **ref_line_kwargs, zorder=10)
        label_x_bottom, label_ha_bottom = get_label_position_and_alignment(
            ref_x_bottom, side=ref_label_side_bottom, offset=ref_label_offset_bottom
        )
        ax_bottom.text(label_x_bottom, ref_label_y_position_bottom, ref_label,
                       color=ref_label_color, fontsize=ref_label_fontsize, fontweight=None, va='center',
                       ha=label_ha_bottom, rotation=ref_label_rotation)
        if show_ylabel[1]:
            ax_bottom.set_ylabel("Density")
        ax_bottom.set_yticks([])
        ax_bottom.set_title(r"$VEP_{ref}$ overestimates\npathogenicity", 
                            fontweight=title_fontweight, 
                            loc=title_loc)
        if show_xlabel[1]:
            ax_bottom.set_xlabel(r"$VEP_{ref}$ percentile")
        ax_bottom.set_xlim(-3, 3)
        ax_bottom.set_xticks([-3, 0, 3])
        if ref_x_bottom < 0:
            ax_bottom.set_xticklabels(['0', '50', '100'])
        else:
            ax_bottom.set_xticklabels(['100', '50', '0'])
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
    n_ref_pct_bins=5,
    figsize=(10, 4), 
    label_padding=0.15, 
    is_ref=True,
    title=r"$VEP_{ref}$ Percentiles Relative to Full $VEP$ Distribution",
    x_label=r"$VEP_{mean}$ Quantile",
    y_label="Proportion of Variants",
    show_arrows=False,
    show_vertical_arrows=False,
    show_schematic=True,
    legend_on_right=True,
    legend_loc='upper right',
    legend_height_factor=1.5,
    schematic_width_ratio=1.3,
    barplot_width_ratio=5,
    schematic_heights=[0.2, 0.3, 0.2],
    schematic_padding_left=0.12,
    schematic_padding_top=0.1,
    schematic_title_loc="left",
    show_schematic_xlabel=(False, True),
    show_schematic_ylabel=(False, False),
    use_quantile_labels=True,
    flip_xaxis=False,  # <--- New argument to flip the x-axis direction
    legend_frame=True,  # <--- Whether to draw a frame around the legend
    edgecolor='black',  # <--- Edge color for bars
    linewidth=0.5,  # <--- Line width for bar edges
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
    flip_xaxis : bool, optional
        If True, reverse the direction of the x-axis (default: False).
    legend_frame : bool, optional
        Whether to draw a frame/border around the percentile bin legend (default: True).
    edgecolor : str or color, optional
        Edge color for the bars (default: 'black').
    linewidth : float, optional
        Line width for bar edges (default: 0.5).
    """
    from matplotlib import cm
    import matplotlib.pyplot as plt

    vep_df = vep_df.copy()

    if "is_ref" not in vep_df.columns:
        vep_df["is_ref"] = vep_df["sample"] == "REF"

    groupby_cols = [x for x in groupby_cols if x in vep_df.columns]

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
    def _bin_label_mapper(x):
        if pd.notnull(x):
            idx = int(x)
            return bin_labels[idx]
        else:
            return np.nan
    data['VEP_binned_label'] = data['VEP_binned'].map(_bin_label_mapper)

    # Optionally flip the bin labels and bin order for x-axis
    if flip_xaxis:
        # Create a mapping from original labels to flipped labels
        original_labels = bin_labels.copy()
        bin_labels = bin_labels[::-1]
        # Map the data labels to the flipped labels
        label_mapping = {original_labels[i]: bin_labels[i] for i in range(n_bins)}
        data['VEP_binned_label'] = data['VEP_binned_label'].map(label_mapping)

    # Bin VEP_percentile into deciles (y-axis bins)
    percentile_bins = np.linspace(0, 100, n_ref_pct_bins+1)
    percentile_labels = [f"$P_{{{int(percentile_bins[i])}-{int(percentile_bins[i+1])}}}$" for i in range(n_ref_pct_bins)]
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
            ax.set_facecolor('none')  # Make background transparent
            legend_ax = fig.add_subplot(gs[1])  # Dedicated axis for legend
            legend_ax.set_facecolor('none')  # Make background transparent
            schematic_ax = fig.add_subplot(gs[2])
            schematic_ax.set_facecolor('none')  # Make background transparent
            
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
            ax.set_facecolor('none')  # Make background transparent
            schematic_ax = fig.add_subplot(gs[1])
            schematic_ax.set_facecolor('none')  # Make background transparent
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
        ax.set_facecolor('none')  # Make background transparent
        schematic_ax = None
        legend_ax = None

    stacked_prop.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=reversed_colors,
        width=0.95,
        legend=False,  # Disable automatic legend to prevent duplication
        edgecolor=edgecolor,
        linewidth=linewidth
    )
    
    # Ensure edge lines are applied to all bar patches (for stacked bars)
    for container in ax.containers:
        for patch in container.patches:
            patch.set_edgecolor(edgecolor)
            patch.set_linewidth(linewidth)
    
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
        legend_obj = legend_ax.legend(
            handles[::-1],
            reversed_labels,
            title=r"$VEP_{ref}$" + " percentile bin",
            loc='center left',  # Center the legend content vertically
            borderaxespad=0.0,
            frameon=legend_frame,
        )
        
        # After creating the legend, adjust its position to account for the title
        # This ensures the legend content (not including title) is centered
        if legend_obj is not None:
            # Get the legend's bounding box
            legend_bbox = legend_obj.get_window_extent()
            # Transform to display coordinates
            legend_bbox_display = legend_bbox.transformed(legend_ax.transAxes.inverted())
            
            # Calculate the height of the legend content (excluding title)
            legend_height = legend_bbox_display.height
            title_height = legend_height * 0.15  # Approximate title height as fraction of total legend height
            
            # Adjust y position to center the content (excluding title)
            # Move up by half the title height to center the content
            # For the right legend, we need to adjust the bbox_to_anchor
            legend_obj.set_bbox_to_anchor((0.5, 0.5 + (title_height / 2)))
    else:
        # Legend on the left side of the barplot (original behavior)
        # Calculate legend position and height based on height factor
        if legend_height_factor != 1.0:
            # Adjust y-position based on height factor
            # For height_factor > 1, move legend down; for < 1, move it up
            y_offset = (legend_height_factor - 1.0) * 0.5  # Scale factor for y-position adjustment
            y_pos = 0.5 + y_offset  # Center at 0.5 instead of 1.0
        else:
            y_pos = 0.5  # Center vertically
        
        # Create legend with height adjustment
        if legend_height_factor != 1.0:
            # Calculate custom height for legend
            # Use bbox_to_anchor with height adjustment
            legend = ax.legend(
                handles[::-1],
                reversed_labels,
                title=r"$VEP_{ref}$" + "\n" + "percentile bin",
                bbox_to_anchor=(-0.15, y_pos),
                loc=legend_loc,
                borderaxespad=0.0,
                # Adjust legend height by modifying the layout
                ncol=1,  # Ensure single column for height control
                frameon=legend_frame
            )
            
            # Apply height scaling by adjusting the legend's internal spacing
            if hasattr(legend, '_legend_box'):
                legend._legend_box.sep = legend._legend_box.sep * legend_height_factor
        else:
            # Default legend without height adjustment
            legend = ax.legend(
                handles[::-1],
                reversed_labels,
                title=r"$VEP_{ref}$" + "\n" + "percentile bin",
                bbox_to_anchor=(-0.15, y_pos),
                loc=legend_loc,
                borderaxespad=0.0,
                frameon=legend_frame
            )
        
        # After creating the legend, adjust its position to account for the title
        # This ensures the legend content (not including title) is centered
        if legend is not None:
            # Get the legend's bounding box
            legend_bbox = legend.get_window_extent()
            # Transform to display coordinates
            legend_bbox_display = legend_bbox.transformed(ax.transAxes.inverted())
            
            # Calculate the height of the legend content (excluding title)
            legend_height = legend_bbox_display.height
            title_height = legend_height * 0.15  # Approximate title height as fraction of total legend height
            
            # Adjust y position to center the content (excluding title)
            # Move up by half the title height to center the content
            adjusted_y_pos = y_pos + (title_height / 2)
            
            # Update the legend position
            legend.set_bbox_to_anchor((-0.15, adjusted_y_pos))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # Remove top and right margin lines (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Flip the x-axis if requested
    if flip_xaxis:
        ax.invert_xaxis()

    plt.tight_layout()

    # --- Add up/down arrows to the right of the barplot that align with the schematic ---
    if show_schematic and show_vertical_arrows:
        # Get axis limits for y
        ymin, ymax = ax.get_ylim()
        ycenter = (ymin + ymax) / 2  # Center of the barplot
        
        if legend_on_right:
            # When legend is on right, arrows go between barplot and legend
            xlim = ax.get_xlim()
            x_arrow = xlim[1] + 0.05 if not flip_xaxis else xlim[0] - 0.05
        else:
            # Original behavior: arrows go between barplot and schematic
            xlim = ax.get_xlim()
            x_arrow = xlim[1] + 0.05 if not flip_xaxis else xlim[0] - 0.05
        
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
                
                # Draw lines from barplot right edge all the way to the legend
                # Extend beyond the arrows to reach the legend area
                legend_x = x_arrow + 0.3 if not flip_xaxis else x_arrow - 0.3  # Extend further right/left to reach the legend
                ax.plot([barplot_right, legend_x], [top_data_y, top_data_y], color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
                ax.plot([barplot_right, legend_x], [bottom_data_y, bottom_data_y], color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
                
                # Expand the xlim to make sure arrows, labels, and extended lines are visible
                if not flip_xaxis:
                    ax.set_xlim(xlim[0], legend_x + 0.1)
                else:
                    ax.set_xlim(legend_x - 0.1, xlim[1])
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
                x_data_left = ax.get_xlim()[0]   # Left edge of barplot data
                
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
                if not flip_xaxis:
                    ax.plot([x_data_right, x_arrow], [top_data_y, top_data_y], color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
                    ax.plot([x_data_right, x_arrow], [bottom_data_y, bottom_data_y], color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
                    ax.set_xlim(xlim[0], x_arrow + 0.6)
                else:
                    ax.plot([x_data_left, x_arrow], [top_data_y, top_data_y], color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
                    ax.plot([x_data_left, x_arrow], [bottom_data_y, bottom_data_y], color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
                    ax.set_xlim(x_arrow - 0.6, xlim[1])

    # --- Optionally add arrows and labels along the y-axis, outside the right margin ---
    if show_arrows:
        # Get axis limits for y
        ymin, ymax = ax.get_ylim()
        ycenter = 0.5  # Origin for arrows

        # Set the x position for the arrows and labels just outside the right of the plot
        xlim = ax.get_xlim()
        x_arrow = xlim[1] + 0.1 if not flip_xaxis else xlim[0] - 0.1

        # Length of arrows (as a fraction of y-axis)
        arrow_length = (ymax - ymin) * 0.35

        # Arrow for "REF Underestimates Pathogenicity" (upward)
        fontsize = 'medium'
        fontweight = None
        ax.annotate(
            "",
            xy=(x_arrow, ycenter + arrow_length),
            xytext=(x_arrow, ycenter),
            arrowprops=dict(arrowstyle="->", color="black", lw=2),
            annotation_clip=False
        )
        ax.text(
            x_arrow + (0.08 if not flip_xaxis else -0.08),
            ycenter + arrow_length/2 + label_padding/2,
            r"$VEP_{ref}$ underestimates\npathogenicity",
            va='center', ha='left' if not flip_xaxis else 'right', rotation=90, fontsize=fontsize, fontweight=fontweight
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
            x_arrow + (0.08 if not flip_xaxis else -0.08),
            ycenter - arrow_length/2 - label_padding/2,
            r"$VEP_{ref}$ overestimates\npathogenicity",
            va='center', ha='left' if not flip_xaxis else 'right', rotation=90, fontsize=fontsize, fontweight=fontweight
        )

        # Optionally, expand the xlim to make sure arrows and labels are visible
        if not flip_xaxis:
            ax.set_xlim(xlim[0], x_arrow + 0.75)
        else:
            ax.set_xlim(x_arrow - 0.75, xlim[1])

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


def plot_ref_vep_std_stacked_bar(
    vep_df, 
    groupby_cols = ['model_location','protein','clinsig','mutant','scoring_strategy'],
    y='VEP_percentile',
    n_bins=10, 
    figsize=(9, 4), 
    label_padding=0.15, 
    is_ref=True,
    title=r"Standard Deviations Separating $VEP_{ref}$ from $VEP_{mean}$",
    x_label=r"$VEP_{mean}$ Quantile",
    y_label="Proportion of Variants",
    xaxis_quartile_labels=True,
    add_arrows=True,
    flip_xaxis=False
):
    """
    Plot a stacked bar plot of VEP percentiles, binned by quantiles and standard deviation categories.
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
        Number of quantile bins for VEP_mean (default: 10).
    figsize : tuple, optional
        Figure size (default: (9, 4)).
    label_padding : float, optional
        Vertical padding between arrow labels (default: 0.15).
    is_ref : bool, optional
        Whether to compute REF statistics (default: True).
    title : str, optional
        Plot title.
    x_label : str, optional
        X-axis label.
    y_label : str, optional
        Y-axis label.
    xaxis_quartile_labels : bool, optional
        If True, label x-axis ticks as Q1, Q2, ... instead of bin ranges.
    flip_xaxis : bool, optional
        If True, reverse the x-axis direction (default: False).
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

    # Optionally relabel x-axis ticks as Q1, Q2, ...
    if xaxis_quartile_labels:
        # Only relabel visible ticks (i.e., those with data)
        n_visible = len(stacked_prop)
        quartile_labels = [f"Q{i+1}" for i in range(n_visible)]
        ax.set_xticklabels(quartile_labels, rotation=0)
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()

    # --- Add arrows and labels along the y-axis, outside the right margin ---
    if add_arrows:
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
            r"$\mathbf{VEP_{ref}}$"+"\n"+r"underestimates"+"\n"+r"pathogenicity",
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
            r"$\mathbf{VEP_{ref}}$"+"\n"+r"overestimates"+"\n"+r"pathogenicity",
            va='center', ha='left', rotation=90, fontsize=fontsize, fontweight='bold'
        )
        if flip_xaxis:
            ax.set_xlim(x_arrow + 0.75, xlim[0])
        else:
            ax.set_xlim(xlim[0], x_arrow + 0.75)
    if flip_xaxis:
        ax.invert_xaxis()
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
        r"$VEP_{ref}$ underestimates\npathogenicity",
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
        r"$VEP_{ref}$ overestimates\npathogenicity",
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

def fix_clinsig_labels(
    df, 
    palette=None, 
    clinsig_col="clinsig", 
    mutant_col="mutant",
    suffix="variants",
    format_k=False,
    k_precision=1
):
    """
    Normalize clinsig labels and optionally format mutant counts as #.#k.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    palette : dict or None
        Color palette for clinsig labels.
    clinsig_col : str
        Column name for clinical significance.
    mutant_col : str
        Column name for mutant/haplotype/variant.
    suffix : str
        Suffix for label (e.g., "variants").
    format_k : bool, optional
        If True, format counts as #.#k (e.g., 1.2k).
    k_precision : int, optional
        Number of decimal places for k-format (default: 1).
    """
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

    def _format_count(n):
        if not format_k:
            return str(n)
        if n >= 1000:
            fmt = f"{{:.{k_precision}f}}"
            return fmt.format(n / 1000).rstrip('0').rstrip('.') + "k"
        return str(n)

    clinsig_label_map = {
        clinsig: (
            f"{clinsig} ({_format_count(mutant_counts.get(clinsig, 0))}{'' if suffix is None else f' {suffix}'})"
            if mutant_counts is not None else clinsig
        )
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
    legend_loc="best",
    x_label=r"$VEP_{mean}$",
    y_label=None,
    stat="frequency",
    title="VEP Distributions",
    suffix="variants",
    format_k=False,
    k_precision=1
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

    if y_label is None:
        y_label = stat.title()

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
    hist_df, palette_labels = fix_clinsig_labels(hist_df, 
                                                 clinsig_col="clinsig", 
                                                 mutant_col="mutant", 
                                                 suffix=suffix, 
                                                 format_k=format_k, 
                                                 k_precision=k_precision)

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
        legend.set_loc(legend_loc)
        if external_legend_annotation:
            # Move legend outside the plot to the right
            legend.set_bbox_to_anchor((1.02, 1.0))
            legend.set_loc("upper left")

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

    # Remove the top and right spines (margin lines) for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Remove border around legend if legend exists
    if legend is not None:
        legend.get_frame().set_linewidth(0)
        legend.get_frame().set_edgecolor('none')
    
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
    

def _draw_vep_direction_arrows(ax, palette, reverse=False, lw=2, fontsize="medium", fontweight='bold', 
y_arrow_offset=0.12, y_label_offset=0.04,):
    """
    Draws arrows and labels underneath the x-axis to indicate 'Pathogenic' and 'Benign' directions.
    If reverse=True, swaps the directions and sides of 'Pathogenic' and 'Benign'.
    y_offset: float
        Offset of the arrows and text from the bottom of the plot, as a fraction of the plot height.
    """
    # Get axis limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Move arrows and text further down
    arrow_y = ymin - y_arrow_offset * (ymax - ymin)
    label_y = arrow_y - y_label_offset * (ymax - ymin)

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
            arrowprops=dict(facecolor=path_color, edgecolor=path_color, arrowstyle='->', lw=lw),
            annotation_clip=False
        )
        ax.text(
            (right_arrow_start + right_arrow_end) / 2, label_y, "Pathogenic", 
            color=path_color, ha='center', va='top', fontsize=fontsize, fontweight=fontweight
        )

        # Blue left arrow for "Benign"
        left_arrow_start = center_x - arrow_inner_offset
        left_arrow_end = left_arrow_start - arrow_length
        ax.annotate(
            '',
            xy=(left_arrow_end, arrow_y),    # arrow tip (left)
            xytext=(left_arrow_start, arrow_y),  # arrow tail (right, closer to center)
            arrowprops=dict(facecolor=benign_color, edgecolor=benign_color, arrowstyle='-|>', lw=lw),
            annotation_clip=False
        )
        ax.text(
            (left_arrow_start + left_arrow_end) / 2, label_y, "Benign", 
            color=benign_color, ha='center', va='top', fontsize=fontsize, fontweight=fontweight
        )
    else:
        # Red left arrow for "Pathogenic" (now on the left)
        left_arrow_start = center_x - arrow_inner_offset
        left_arrow_end = left_arrow_start - arrow_length
        ax.annotate(
            '',
            xy=(left_arrow_end, arrow_y),
            xytext=(left_arrow_start, arrow_y),
            arrowprops=dict(facecolor=path_color, edgecolor=path_color, arrowstyle='-|>', lw=lw),
            annotation_clip=False
        )
        ax.text(
            (left_arrow_start + left_arrow_end) / 2, label_y, "Pathogenic", 
            color=path_color, ha='center', va='top', fontsize=fontsize, fontweight=fontweight
        )

        # Blue right arrow for "Benign" (now on the right)
        right_arrow_start = center_x + arrow_inner_offset
        right_arrow_end = right_arrow_start + arrow_length
        ax.annotate(
            '',
            xy=(right_arrow_end, arrow_y),    # arrow tip (right)
            xytext=(right_arrow_start, arrow_y),  # arrow tail (left, closer to center)
            arrowprops=dict(facecolor=benign_color, edgecolor=benign_color, arrowstyle='->', lw=lw),
            annotation_clip=False
        )
        ax.text(
            (right_arrow_start + right_arrow_end) / 2, label_y, "Benign", 
            color=benign_color, ha='center', va='top', fontsize=fontsize, fontweight=fontweight
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

def plot_vep_kde_with_arrows(
    vep_df, 
    vep_col="VEP",
    clinsig_col="clinsig",
    site_col="site", 
    title="VEP Distributions",
    x_label=r"$VEP_{mean}$",
    y_label="Proportion",  
    legend_title="Clinical Significance",
    palette=utils.get_clinsig_palette(),
    hline_x=None,
    figsize=(8, 5),
    save_path=None, 
    kde_kwargs={"cut":0},
    save_kwargs=utils.FIG_SAVE_KWARGS,
    suffix=None,
    format_k=False,
    k_precision=1,
    legend_loc="best",
    legend_kwargs={},
    flip_xaxis=False,
    add_histogram=False,
    hist_kwargs={"bins":100,"edgecolor":"grey"},
    title_y=0.9,
    height_ratios=(.2,1),
    hist_log_y=False,
    hist_y_label="Count",
    hist_y_format="k",
    title_kwargs={},    
    arrow_kwargs={},
    extend_kde_to_outliers=False,
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
    kde_kwargs : dict
        Keyword arguments for the plot.
    save_kwargs : dict
        Keyword arguments for the save function.
    hline_x : float
        X position of the horizontal line.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    title : str
        Title for the plot.
    legend_title : str
        Title for the legend.
    palette : dict
        Palette for the plot.
    figsize : tuple
        Figure size.
    suffix : str
        Suffix for the legend.
    format_k : bool
        If True, format the legend as #.#k.
    k_precision : int
        Number of decimal places for k-format.
    legend_loc : str or int, optional
        Location of the legend (default: "best").
    legend_kwargs : dict or None, optional
        Additional keyword arguments for ax.legend().
    flip_xaxis : bool, optional
        If True, flip the direction of the x-axis (default: False).
    add_histogram : bool, optional
        If True, add a histogram on top of the KDE plot showing counts of vep_df[vep_col] (default: False).
    hist_kwargs : dict, optional
        Keyword arguments for the histogram plot (default: {}).
    height_ratios : list, optional
        Height ratios for [histogram, KDE] subplots when add_histogram=True (default: [1, 3]).
    title_y : float, optional
        Y position of the title when add_histogram=True, as a fraction of figure height (default: 0.98).
        Only used when add_histogram=True.
    hist_log_y : bool, optional
        If True, use log scale for the histogram y-axis (default: False).
        Only used when add_histogram=True.
    hist_y_label : str, optional
        Label for the histogram y-axis (default: "Count").
        Only used when add_histogram=True.
    hist_y_format : str or None, optional
        Format for histogram y-axis units: None for raw units, "k" for thousands, "m" for millions (default: "k").
        Only used when add_histogram=True.
    arrow_kwargs : dict, optional
        Keyword arguments for the _draw_vep_direction_arrows function (default: {}).
    extend_kde_to_outliers : bool, optional
        If True, extend the KDE plot to show outliers by adding boundary points at xlim extremes.
        This ensures the KDE covers the full range visible in the histogram (default: False).
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns

    # Create aggregated plot_df first to check its range
    plot_df = vep_df.copy().dropna(subset=[vep_col, clinsig_col]).groupby([clinsig_col, site_col], observed=True)[vep_col].mean().reset_index(name=vep_col)
    
    # Get the actual data range to set x-axis limits based on observed min/max
    raw_min = vep_df[vep_col].dropna().min()
    raw_max = vep_df[vep_col].dropna().max()
    agg_min = plot_df[vep_col].dropna().min()
    agg_max = plot_df[vep_col].dropna().max()
    
    # Set xlim based on extend_kde_to_outliers setting
    if extend_kde_to_outliers:
        # Use the wider range to ensure both histogram and KDE can show all data including outliers
        xlim_min = min(raw_min, agg_min)
        xlim_max = max(raw_max, agg_max)
    else:
        # Use only aggregated data range to match KDE's natural range
        # This constrains the histogram to the same range as the KDE
        xlim_min = agg_min
        xlim_max = agg_max
    
    if add_histogram:
        # Don't use sharex=True if we need to flip, as it interferes with invert_xaxis()
        fig, (ax_hist, ax) = plt.subplots(2, 1, figsize=figsize, sharex=False, 
                                          gridspec_kw={'height_ratios': height_ratios, 'hspace': 0})
        # Set xlim immediately after creating axes
        ax_hist.set_xlim(xlim_min, xlim_max)
    else:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Set xlim on KDE axis immediately
    ax.set_xlim(xlim_min, xlim_max)


    plot_df = utils.sort_by_clinsig(plot_df, clinsig_col=clinsig_col)

    plot_df, palette_labeled = fix_clinsig_labels(
        plot_df, 
        clinsig_col=clinsig_col, 
        mutant_col=site_col, 
        suffix=suffix, 
        format_k=format_k, 
        k_precision=k_precision
    )

    if legend_title is not None:
        plot_df[legend_title] = plot_df[clinsig_col + "_label"]
    else:
        legend_title = clinsig_col + "_label"

    # Extend plot_df with boundary points to force KDE evaluation at xlim extremes
    # This ensures the KDE extends to show outliers visible in the histogram
    if extend_kde_to_outliers and (raw_min < agg_min or raw_max > agg_max):
        # Create extended dataframe with boundary points for each group
        extended_dfs = []
        for group_name in plot_df[legend_title].unique():
            group_df = plot_df[plot_df[legend_title] == group_name].copy()
            # Add boundary points at xlim_min and xlim_max if they're outside the group's range
            group_min = group_df[vep_col].min()
            group_max = group_df[vep_col].max()
            
            # Create boundary rows by duplicating the closest existing points
            boundary_rows_list = []
            if xlim_min < group_min:
                # Add a point at xlim_min (duplicate the closest existing point)
                closest_idx = group_df[vep_col].idxmin()
                boundary_row = group_df.loc[[closest_idx]].copy()
                boundary_row[vep_col] = xlim_min
                boundary_rows_list.append(boundary_row)
            if xlim_max > group_max:
                # Add a point at xlim_max (duplicate the closest existing point)
                closest_idx = group_df[vep_col].idxmax()
                boundary_row = group_df.loc[[closest_idx]].copy()
                boundary_row[vep_col] = xlim_max
                boundary_rows_list.append(boundary_row)
            
            if boundary_rows_list:
                boundary_df = pd.concat(boundary_rows_list, ignore_index=True)
                extended_dfs.append(pd.concat([group_df, boundary_df], ignore_index=True))
            else:
                extended_dfs.append(group_df)
        
        plot_df_extended = pd.concat(extended_dfs, ignore_index=True)
    else:
        plot_df_extended = plot_df
    
    sns.kdeplot(
        data=plot_df_extended,
        x=vep_col,
        hue=legend_title,
        multiple="fill",
        fill=True, 
        palette=palette_labeled,
        ax=ax,
        **kde_kwargs
    )
    if hline_x is not None:
        ax.axvline(x=hline_x, color='black', linestyle='--', linewidth=2, alpha=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Set title - use fig.suptitle when histogram is enabled to position it above
    if add_histogram:
        fig.suptitle(title, y=title_y, verticalalignment='bottom', **title_kwargs)
        ax.set_title('')  # Remove title from KDE plot
    else:
        ax.set_title(title, **title_kwargs)

    # Add histogram on top if requested
    if add_histogram:
        # Filter histogram data to match xlim range to avoid abrupt cutoffs
        hist_data = vep_df[vep_col].dropna()
        hist_data_filtered = hist_data[(hist_data >= xlim_min) & (hist_data <= xlim_max)]
        
        # Plot histogram on top axes with filtered data
        # Use lighter grey as default, but allow override via hist_kwargs
        hist_color = hist_kwargs.pop('color', 'lightgrey')
        # Ensure density=False for log scale to work correctly with counts
        if 'density' not in hist_kwargs:
            hist_kwargs['density'] = False
        hist_data_filtered.hist(ax=ax_hist, color=hist_color, **hist_kwargs)
        
        # Ensure xlim is set after histogram plotting (in case it was changed)
        ax_hist.set_xlim(xlim_min, xlim_max)
        
        # Set log scale for y-axis BEFORE formatting (if requested)
        if hist_log_y:
            # This is a logarithmic scale for the y-axis of the histogram (ax_hist)
            # Set log scale first, before any formatting
            ax_hist.set_yscale('log')
        
        # Format y-axis based on hist_y_format parameter
        # Note: When using log scale, the formatter receives log-space values
        if hist_y_format is not None and not hist_log_y:
            # Only apply custom formatter for non-log scales
            if hist_y_format == "k":
                def format_units(x, pos):
                    """Format number in thousands with 'k' suffix"""
                    if x >= 1000:
                        return f'{x/1000:.0f}k'
                    else:
                        return f'{x:.0f}'
            elif hist_y_format == "m":
                def format_units(x, pos):
                    """Format number in millions with 'm' suffix"""
                    if x >= 1000000:
                        return f'{x/1000000:.1f}m'
                    elif x >= 1000:
                        return f'{x/1000:.0f}k'
                    else:
                        return f'{x:.0f}'
            else:
                raise ValueError(f"hist_y_format must be None, 'k', or 'm', got '{hist_y_format}'")
            ax_hist.yaxis.set_major_formatter(mticker.FuncFormatter(format_units))
        
        # Set y-tick labels: one in the middle and one at the top
        # Get current y-axis limits after any scaling
        ymin, ymax = ax_hist.get_ylim()
        if hist_log_y:
            # For log scale, use geometric mean for middle and don't round aggressively
            # Let matplotlib handle log ticks naturally, just set reasonable bounds
            # Don't force specific tick values as it can cause display issues
            pass  # Let matplotlib auto-generate log ticks
        else:
            # For linear scale, use arithmetic mean for middle
            y_middle = (ymin + ymax) / 2
            # Round to nearest 100k (100,000)
            y_middle_rounded = np.round(y_middle / 100000) * 100000
            ymax_rounded = np.round(ymax / 100000) * 100000
            ax_hist.set_yticks([y_middle_rounded, ymax_rounded])
        
        # Remove grid lines
        ax_hist.grid(False)
        
        ax_hist.set_ylabel(hist_y_label)
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.set_xlabel('')  # Remove x-label from top plot
        ax_hist.set_xticklabels([])  # Remove x-axis tick labels

    # Ensure xlim is set on KDE axis after plotting (in case seaborn changed it)
    ax.set_xlim(xlim_min, xlim_max)

    _draw_vep_direction_arrows(ax, palette,reverse=flip_xaxis, **arrow_kwargs)

    # Flip the x-axis if requested
    if flip_xaxis:
        ax.invert_xaxis()
        if add_histogram:
            ax_hist.invert_xaxis()
    
    # Re-enforce xlim after all operations to ensure it's based on actual data range
    ax.set_xlim(xlim_min, xlim_max)
    if add_histogram:
        ax_hist.set_xlim(xlim_min, xlim_max)
    
    # Set legend location and title (do this LAST, after all other operations)
    legend = ax.get_legend()
    if legend is not None:
        # Get handles and labels - this should work correctly with seaborn
        handles, labels = ax.get_legend_handles_labels()
        
        # Only proceed if we have valid handles and labels
        if len(handles) > 0 and len(labels) > 0:
            # Remove the old legend
            legend.remove()
            
            # Re-add legend with user-specified location, preserving all labels
            new_legend = ax.legend(
                handles, labels, 
                title=legend_title, 
                loc=legend_loc, 
                **legend_kwargs
            )

    # Adjust layout to accommodate title when histogram is enabled
    if add_histogram:
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for suptitle
        
        # Align y-axis labels vertically by using the same labelpad
        # Get the labelpad from the KDE plot and apply it to the histogram
        kde_labelpad = ax.yaxis.labelpad
        ax_hist.yaxis.labelpad = kde_labelpad
        
        # Force a draw and check if further alignment is needed
        fig.canvas.draw_idle()
        try:
            # Get the x-positions of both y-axis labels in figure coordinates
            kde_label_bbox = ax.yaxis.label.get_window_extent(fig.canvas.get_renderer())
            kde_label_bbox_fig = kde_label_bbox.transformed(fig.transFigure.inverted())
            kde_x = kde_label_bbox_fig.x0
            
            hist_label_bbox = ax_hist.yaxis.label.get_window_extent(fig.canvas.get_renderer())
            hist_label_bbox_fig = hist_label_bbox.transformed(fig.transFigure.inverted())
            hist_x = hist_label_bbox_fig.x0
            
            # If still misaligned, adjust histogram labelpad to match
            if abs(hist_x - kde_x) > 0.001:
                x_diff_fig = kde_x - hist_x
                # Convert figure coordinate difference to points
                fig_width_inches = fig.get_figwidth()
                x_diff_points = x_diff_fig * fig_width_inches * 72
                current_pad = ax_hist.yaxis.labelpad
                ax_hist.yaxis.labelpad = current_pad + x_diff_points
        except:
            pass  # If alignment fails, at least they have the same labelpad
    
    # Re-apply legend location after tight_layout (tight_layout can reposition it)
    legend = ax.get_legend()
    if legend is not None and legend_loc != "best":
        # Get handles and labels
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0 and len(labels) > 0:
            legend.remove()
            ax.legend(
                handles, labels, 
                title=legend_title, 
                loc=legend_loc, 
                **legend_kwargs
            )

    if save_path is not None:
        fig.savefig(save_path, **save_kwargs)

    plt.show()

    result = {'fig': fig, 'axes': ax, 'data': plot_df}
    if add_histogram:
        result['hist_axes'] = ax_hist
    return result


def test_vep_clinsig_separation(
    vep_agg, 
    clinsig_col="clinsig", 
    vep_col="VEP", 
    model_col="model_location", 
    clinsig_groups=None
):
    """
    Test the separation between all specified clinsig groups in each model using Mann-Whitney U test.

    Parameters
    ----------
    vep_agg : pd.DataFrame
        DataFrame containing at least columns for model, clinsig, and VEP values.
    clinsig_col : str
        Name of the column containing clinical significance labels.
    vep_col : str
        Name of the column containing VEP values.
    model_col : str
        Name of the column containing model identifiers.
    clinsig_groups : list or None
        List of clinsig group labels to test. If None, defaults to ["benign", "likely_benign", "path", "likely_path"].

    Returns
    -------
    vep_diff_df : pd.DataFrame
        DataFrame with Mann-Whitney U test results for all pairwise group comparisons per model.
    summary : pd.DataFrame
        Aggregated mean pvalue and statistic per model.
    """
    from scipy.stats import mannwhitneyu
    import itertools

    if clinsig_groups is None:
        clinsig_groups = vep_agg[clinsig_col].unique()

    if clinsig_col not in vep_agg.columns:
        raise ValueError(f"vep_agg must contain a '{clinsig_col}' column with clinical significance labels.")

    results = []
    for model, group in vep_agg.groupby(model_col):
        # For each group, extract VEP values
        group_veps = {}
        for label in clinsig_groups:
            mask = group[clinsig_col].str.fullmatch(label, case=False, na=False)
            vep_vals = group.loc[mask, vep_col].dropna()
            group_veps[label] = vep_vals

        # For each pairwise combination, perform Mann-Whitney U test if both groups have data
        for g1, g2 in itertools.combinations(clinsig_groups, 2):
            vep1 = group_veps[g1]
            vep2 = group_veps[g2]
            if len(vep1) > 0 and len(vep2) > 0:
                stat, pval = mannwhitneyu(vep1, vep2, alternative="two-sided")
                results.append({
                    model_col: model,
                    "group1": g1,
                    "group2": g2,
                    "n_group1": len(vep1),
                    "n_group2": len(vep2),
                    "statistic": stat,
                    "pvalue": pval,
                    "group1_median": vep1.median(),
                    "group2_median": vep2.median()
                })
            else:
                results.append({
                    model_col: model,
                    "group1": g1,
                    "group2": g2,
                    "n_group1": len(vep1),
                    "n_group2": len(vep2),
                    "statistic": None,
                    "pvalue": None,
                    "group1_median": vep1.median() if len(vep1) > 0 else None,
                    "group2_median": vep2.median() if len(vep2) > 0 else None
                })

    vep_diff_df = pd.DataFrame(results).sort_values(by=[model_col, 'group1', 'group2'])
    summary = vep_diff_df.groupby(model_col).agg({"pvalue": "mean", "statistic": "mean"}).sort_values(by="statistic", ascending=False)
    return vep_diff_df, summary

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, median_abs_deviation

def variant_embedding(
    df: pd.DataFrame, 
    score_col: str = "VEP", 
    quantiles: list = None
) -> dict:
    """
    Generate a numerical summary ('embedding') of variant effect scores within a group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing variant effect score data for a single site.
    score_col : str, default="VEP"
        Column name holding the effect scores to embed.
    quantiles : list of float, optional
        List of quantile fractions (e.g., 0.01) to compute. If None, defaults to [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99].

    Returns
    -------
    emb : dict
        Dictionary including quantiles, moments (mean, std, skew, kurtosis, min, max, mad), and missingness counts.
    """
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    scores = df[score_col].values

    # Exclude NaNs
    scores_nonan = scores[~np.isnan(scores)]
    n_total = len(scores)
    n_nonan = len(scores_nonan)
    n_nan = n_total - n_nonan

    # Handle fully missing case
    if n_nonan == 0:
        # All values are NaN: provide np.nan in all keys, explicit missingness
        emb = {f"q{int(q*100)}": np.nan for q in quantiles}
        emb.update({
            "mean": np.nan,
            "std": np.nan,
            "mad": np.nan,
            "skew": np.nan,
            "kurt": np.nan,
            "min": np.nan,
            "max": np.nan,
            "n_nonan": 0,
            "n_nan": n_total,
            "prop_missing": 1.0,
        })
        return emb

    # If only 1 or 2 values: quantiles/mean/mad/var ok, but skew/kurtosis undefined
    if n_nonan < 3:
        moments = {
            "mean": np.nanmean(scores_nonan),
            "std": np.nanstd(scores_nonan),
            "mad": median_abs_deviation(scores_nonan, nan_policy="omit"),
            "skew": np.nan,
            "kurt": np.nan,
            "min": np.nanmin(scores_nonan),
            "max": np.nanmax(scores_nonan),
        }
    else:
        moments = {
            "mean": np.nanmean(scores_nonan),
            "std": np.nanstd(scores_nonan),
            "mad": median_abs_deviation(scores_nonan, nan_policy="omit"),
            "skew": skew(scores_nonan, nan_policy="omit"),
            "kurt": kurtosis(scores_nonan, nan_policy="omit"),
            "min": np.nanmin(scores_nonan),
            "max": np.nanmax(scores_nonan),
        }

    # Quantiles
    emb = {f"q{int(q*100)}": np.nanquantile(scores_nonan, q) for q in quantiles}
    emb.update(moments)
    emb["n_nonan"] = n_nonan
    emb["n_nan"] = n_nan
    emb["prop_missing"] = n_nan / n_total if n_total > 0 else np.nan

    return emb

def build_variant_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build summary/embedding rows for each variant site in the provided DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'site' column (e.g. unique variant key) and a score column.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of embeddings per site.
    """
    from tqdm import tqdm

    rows = []
    for var, sub in tqdm(df.groupby("site"), desc="Building variant embeddings"):
        emb = variant_embedding(sub)
        emb["site"] = var
        rows.append(emb)
    return pd.DataFrame(rows)

def variant_embedding_umap(variant_embed: pd.DataFrame,
umap_kwargs={"n_components": 3,
        "random_state": 42}) -> pd.DataFrame:
    """
    Build UMAP embeddings for each variant site in the provided DataFrame.

    Parameters
    ----------
    variant_embed : pd.DataFrame
        Input DataFrame with a 'site' column (e.g. unique variant key) and a score column.
    
    Returns
    -------
    pd.DataFrame    

    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import umap

    feature_cols = [c for c in variant_embed.columns if c.startswith("q") or c in ["mad","std","skew","kurt"]]

    # Standardize
    X = StandardScaler().fit_transform(variant_embed[feature_cols].dropna())

    # PCA to 30–50 dims
    # pca = PCA(n_components=10)
    # X_pca = pca.fit_transform(X)
    # variant_embed["PC1"] = X_pca[:, 0]
    # variant_embed["PC2"] = X_pca[:, 1]
 
    um = umap.UMAP( 
        **umap_kwargs
    )
    X_umap = um.fit_transform(X)

    # Create a UMAP DataFrame with only the variants that were not dropped for UMAP (i.e., same order as X_umap)
    umap_df = variant_embed.loc[variant_embed[feature_cols].dropna().index].copy()
    umap_df["UMAP1"] = X_umap[:, 0]
    umap_df["UMAP2"] = X_umap[:, 1]
    umap_df["UMAP3"] = X_umap[:, 2]
    # for col in ["clinsig", "GENE", "protein", "mutant"]:
    #     if col not in umap_df.columns:
    #         umap_df[col] = umap_df["site"].map(dict(zip(vep_df["site"], vep_df[col])))

    return umap_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def normalize_rows(X: np.ndarray, 
                   keep_nan: bool = False) -> np.ndarray:
    """Normalize rows of a matrix to sum to 1.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input matrix to normalize
    keep_nan : bool, default=False
        If True, keep NaN values in the output. If False, replace NaN values with 0.
    retain_row_weights : bool, default=False
        If True, multiply the normalized values by the original row sums to retain the original weights.
        
    Returns
    -------
    numpy.ndarray
        Matrix with normalized rows
    """
    row_sums = np.nansum(X.copy(), axis=1, keepdims=True)
    if keep_nan:
        X = np.divide(X, row_sums)
    else:
        X = np.divide(X, row_sums, 
                      where=(row_sums!=0) & (row_sums!=np.nan))
        
    return X


def pad_matrices_to_max_shape(matrices,
                              pad_value=np.nan):
    """
    Pad all matrices in the list to the maximum shape among them using np.nan.
    """
    max_shape = np.array([m.shape for m in matrices]).max(axis=0)
    padded_matrices = []
    for m in matrices:
        pad_height = max_shape[0] - m.shape[0]
        pad_width = max_shape[1] - m.shape[1]
        if pad_height > 0 or pad_width > 0:
            pad_widths = ((0, pad_height), (0, pad_width))
            m_padded = np.pad(m, pad_widths, 
                                mode='constant', 
                                constant_values=pad_value)
            padded_matrices.append(m_padded)
        else:
            padded_matrices.append(m)
    return padded_matrices

def average_matrices(matrices, 
                     weights=None, 
                     scale_multipliers=None,
                     normalize_scale=True, 
                     normalize_rows=False, 
                     
                     use_max=False):
    """
    Average a list of matrices with different weights.
    
    Parameters
    ----------
    matrices : list of numpy.ndarray
        List of matrices to average
    weights : list of float, optional
        Weights for each matrix. If None, equal weights are used.
    normalize_scale : bool or list of bool, default=True
        Whether to normalize each matrix to [0,1] scale before averaging.
        If a list, each boolean corresponds to the matrix at the same index.
    normalize_rows : bool or list of bool, default=False
        Whether to normalize each row of each matrix to sum to 1 before averaging.
        If a list, each boolean corresponds to the matrix at the same index.
        
    Returns
    -------
    numpy.ndarray
        Weighted average of the input matrices
    """

    if isinstance(matrices, dict):
        matrices = list(matrices.values()) 

    # Subset to matrices that are the same size as the REF
    # If matrices are not all the same size, pad the smaller matrices to match the size of the largest
    matrices = pad_matrices_to_max_shape(matrices)

    if weights is None:
        weights = [1] * len(matrices)

    # Convert single boolean to list if needed
    if isinstance(normalize_scale, bool):
        normalize_scale = [normalize_scale] * len(matrices)
    
    # Apply scale normalization only to matrices where normalize_scale is True
    for i, (matrix, should_normalize) in enumerate(zip(matrices, normalize_scale)):
        if should_normalize:
            matrices[i] = (matrix - np.nanmin(matrix)) / (np.nanmax(matrix) - np.nanmin(matrix))
    
    # Convert single boolean to list if needed
    if isinstance(normalize_rows, bool):
        normalize_rows = [normalize_rows] * len(matrices)
    
    # Apply row normalization only to matrices where normalize_rows is True
    for i, (matrix, should_normalize) in enumerate(zip(matrices, normalize_rows)):
        if should_normalize:
            row_sums = np.nansum(matrix, axis=1, keepdims=True)
            matrices[i] = np.divide(matrix, row_sums, 
                                  where=(row_sums!=0) & (row_sums!=np.nan))

    if scale_multipliers is not None:
        for i, (matrix, multiplier) in enumerate(zip(matrices, scale_multipliers)):
            matrices[i] = matrix * multiplier

    if use_max:
        return np.maximum.reduce([weights[i] * matrices[i] for i in range(len(matrices))])
    else:
        return sum(weights[i] * matrices[i] for i in range(len(matrices))) / sum(weights)

def bin_matrix(X, bin_size=10, agg_func=np.nanmax):
    """
    Bin a matrix by aggregating values within bins of specified size.
    
    Args:
        X (np.ndarray): Input matrix to be binned
        bin_size (int, optional): Size of each bin. Defaults to 10.
        agg_func (callable, optional): Aggregation function to apply to each bin. Defaults to np.nanmax.
        
    Returns:
        np.ndarray: Binned matrix with dimensions reduced by bin_size
        
    Example:
        >>> X = np.random.rand(10, 10)
        >>> binned = bin_matrix(X, bin_size=2)
        >>> print(binned.shape)  # (5, 5)
    """
    if bin_size == 1 or bin_size is None:
        return X
    if isinstance(X, pd.DataFrame):
        X = X.values
        
    # Calculate number of bins that fit in the matrix
    n_bins = X.shape[0] // bin_size
    
    # Reshape into 4D array of bins, then aggregate along bin dimensions
    return agg_func(
        X[:n_bins*bin_size, :n_bins*bin_size].reshape(n_bins, bin_size, n_bins, bin_size), 
        axis=(1,3)
    )

def expand_matrix(X, target_size=None,
                  verbose=False):
    """
    Expand a binned matrix back to original dimensions by repeating values.
    
    Args:
        X (np.ndarray): Input binned matrix
        target_size (int): Target size for the expanded matrix (default: 1863)
        verbose (bool): Whether to print verbose output (default: True).
        
    Returns:
        np.ndarray: Expanded matrix with dimensions (target_size, target_size)
    """
    if target_size is None:
        target_size = X.shape[0]
        if verbose:
            print(f"No target size specified, using input size {target_size}")
    if isinstance(target_size, tuple):
        target_size = target_size[0]

    if X.shape[0] == target_size and X.shape[1] == target_size:
        if verbose:
            print(f"Matrix already has target size {target_size}x{target_size}")
        return X
    
    # Calculate expansion factor based on input and target sizes
    input_size = X.shape[0]
    expansion_factor = target_size // input_size

    # Expand the binned matrix back to original dimensions by repeating values
    expanded_matrix = np.repeat(np.repeat(X, expansion_factor, axis=0), expansion_factor, axis=1)

    # Ensure final dimensions are target_size x target_size by padding or truncating if necessary
    current_size = expanded_matrix.shape[0]

    if current_size < target_size:
        # Pad with zeros if too small
        if verbose:
            print(f"Expanding x-axis by {target_size - current_size}")
            print(f"Expanding y-axis by {target_size - current_size}")
        expanded_matrix = np.pad(expanded_matrix, 
                               ((0, target_size - current_size), 
                                (0, target_size - current_size)), 
                               mode='constant')
    elif current_size > target_size:
        # Truncate if too large
        expanded_matrix = expanded_matrix[:target_size, :target_size]

    return expanded_matrix

def label_bins(bin_size, n_bins, max_labels=7, xtick_rotation=0, ytick_rotation=None):
    """
    Create and set bin labels for a contact map plot based on residue positions.
    
    Args:
        bin_size (int): Size of each bin in residues
        n_bins (int): Number of bins in the contact map
        max_labels (int, optional): Maximum number of labels to show. Defaults to 10.
        xtick_rotation (int, optional): Rotation of the x-axis labels. Defaults to 90.
        ytick_rotation (int, optional): Rotation of the y-axis labels. Defaults to 0.
    Returns:
        None: Modifies the current matplotlib plot's axis labels
    """
    # Calculate optimal spacing between labels to show max_labels
    spacing = max(1, n_bins // max_labels)
    
    # Generate labels with optimal spacing
    bin_labels = [f"{i*bin_size + 1}" if i % spacing == 0 else "" for i in range(n_bins)]
    
    plt.xticks(range(n_bins), bin_labels, rotation=xtick_rotation)
    plt.yticks(range(n_bins), bin_labels, rotation=ytick_rotation)

def nonzero_mean(arr, axis=None):
    """
    Compute the mean of nonzero elements in an array, optionally along a given axis.

    Args:
        arr (array-like): Input array.
        axis (int or None): Axis along which to compute the mean. If None, compute over the flattened array.

    Returns:
        float or np.ndarray: Mean of nonzero elements (np.nan if all are zero).
    """
    arr = np.array(arr)
    mask = arr != 0
    # If axis is None, just flatten
    if axis is None:
        if np.any(mask):
            return np.nanmean(arr[mask])
        else:
            return np.nan
    else:
        # Compute mean only over nonzero elements along the given axis
        # To avoid broadcasting issues, use masked arrays
        arr_masked = np.ma.masked_where(~mask, arr)
        mean = arr_masked.mean(axis=axis)
        # Convert masked means to np.nan where all values were masked
        return mean.filled(np.nan)

def find_nearest_neighbor_path(distance_matrix, start_idx=0):
    """Find path through all points using nearest neighbor algorithm"""
    n = len(distance_matrix)
    unvisited = set(range(n))
    path = [start_idx]
    unvisited.remove(start_idx)
    
    current = start_idx
    while unvisited:
        # Find nearest unvisited neighbor
        min_dist = float('inf')
        nearest = None
        
        for neighbor in unvisited:
            dist = distance_matrix[current, neighbor]
            if dist < min_dist:
                min_dist = dist
                nearest = neighbor
        
        path.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return path
 

def animate_matrices_variation(matrices, 
                                 n_frames=None,
                                 bin_size=10,
                                 pow=1,
                                 cmap="gnuplot2",
                                 figsize=(8, 6),
                                 dpi=100,
                                 duration=500,
                                 output_path=None):
    """
    Create an animated GIF showing matrices from different structures.
    
    Parameters
    ----------
    matrices : dict
        Dictionary mapping names to matrix arrays
    n_frames : int, optional
        Number of frames to include (None for all)
    bin_size : int, default=10
        Size of bins for matrix binning
    pow : int, default=4
        Power to raise matrices to
    cmap : str, default="gnuplot2"
        Colormap for visualization
    figsize : tuple, default=(8, 6)
        Figure size
    dpi : int, default=100
        DPI for saved images
    duration : int, default=500
        Duration per frame in milliseconds
    output_path : str, default=None
        Path to save the GIF
        
    Returns
    -------
    list
        List of PIL Image objects (frames)
    """
    import matplotlib.animation as animation
    from PIL import Image
    import io
    
    frames = []
    matrices_subset = list(matrices.items())[:n_frames]
    
    for i, (name, matrix) in enumerate(tqdm(matrices_subset)):
        # Bin the matrix   
        matrix_binned = bin_matrix(X=matrix**pow, 
                                           bin_size=bin_size)
         
        # Create a single plot for each frame
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix_binned, 
                        cmap=cmap, 
                        interpolation="nearest")
         
        
        # Set haplotype ID left justified
        ax.set_title(f"{name}", fontsize=12, loc='left', pad=10)
        
        # Add frame counter right justified
        ax.text(0.98, 0.98, f"({i+1} / {len(matrices_subset)})", 
                transform=ax.transAxes, fontsize=12, ha='right', va='top', color='white')
        
        ax.axis('off')
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img)
        plt.close()

    # Save as GIF
    if frames:
        frames[0].save(output_path, 
                       save_all=True, 
                       append_images=frames[1:], 
                       duration=duration,
                       loop=0)
        from IPython.display import display, HTML
        abs_path = os.path.abspath(output_path)
        display(HTML(f'<a href="file://{abs_path}" target="_blank">GIF saved as \'{abs_path}\'</a>'))
    
    return frames 

def animate_matrices_interpolation(X1, 
                                    X2, 
                                    num_frames=30, 
                                    duration=200, # 200ms per frame
                                    loop=0,
                                    pow=1,
                                    figsize=(8, 6),
                                    dpi=100,
                                    format="png",
                                    cmap="gnuplot2",
                                    cbar_label="Matrix Value",
                                    save_path=None,
                                    ):
    """
    Create a smooth animation transitioning between two matrices using the trained autoencoder.
    
    Args:   
        X1: First matrix (numpy array)
        X2: Second matrix (numpy array) 
        num_frames: Number of frames in the animation
        pow: Power to raise the contact map to
        output_path: Path to save the GIF
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import io
    
    # Ensure maps have the same shape
    if X1.shape != X2.shape:
        # Resize X2 to match X1's shape
        from scipy.ndimage import zoom
        zoom_factors = (X1.shape[0] / X2.shape[0], X1.shape[1] / X2.shape[1])
        X2 = zoom(X2, zoom_factors, order=1) 

        X1 = np.power(X1, pow)
        X2 = np.power(X2, pow)
        # Fallback: simple linear interpolation without autoencoder
        frames = []
        for i in range(num_frames):
            alpha = i / (num_frames - 1)
            interpolated_matrix = alpha * X2 + (1 - alpha) * X1
            
            # Create frame
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(interpolated_matrix, cmap=cmap, interpolation='nearest')
            ax.set_title(f'Linear Interpolation Frame {i+1}/{num_frames} (α={alpha:.2f})')
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(cbar_label)
            
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
            buf.seek(0)
            frame = Image.open(buf)
            frames.append(frame)
            plt.close()
        
        # Save as GIF
        if frames:
            frames[0].save(
                save_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=loop
            )
            print(f"Linear interpolation animation saved to {save_path}")
        
        return frames
    
def animate_matrices_morphing(
    matrices, 
    pow=4,
    n_frames_per_transition=15,
    bin_size=1,
    random_order=True,
    save_path=None,
    cbar_label="Matrix Value",
    cmap="gnuplot2",
    interval=50,
    fps=10,
    dpi=100,
):
    """
    Create and save an animation morphing between matrices.

    Parameters:
        matrices (dict): Dictionary of matrices.
        pow (int): Power to raise matrices before binning.
        n_frames_per_transition (int): Frames per transition.
        bin_size (int): Bin size for bin_matrix.
        random_order (bool): If True, use random order; else, nearest neighbor path.
        save_path (str): Output GIF filename.
        interval (int): Interval between frames in milliseconds.
        fps (int): Frames per second for GIF.
        dpi (int): DPI for saved images.    

    Example:
        >>> # Suppose you have a dictionary of matrices (numpy arrays) keyed by sample names:
        >>> matrices = {
        ...     "sample1": np.random.rand(50, 50),
        ...     "sample2": np.random.rand(50, 50),
        ...     "sample3": np.random.rand(50, 50),
        ... }
        >>> animate_matrices_morphing(
        ...     contact_maps,
        ...     pow=2,
        ...     n_frames_per_transition=10,
        ...     bin_size=1,
        ...     random_order=True,
        ...     save_path='matrices_demo.gif',
        ...     fps=5
        ... )
        # This will display the animation and save it as 'contact_map_demo.gif'
    """ 
    import matplotlib.animation as animation  
    from scipy.spatial.distance import pdist, squareform 

    # Bin and power contact maps
    matrices_binned = {k: bin_matrix(X=v**pow, bin_size=bin_size) for k, v in matrices.items()}
    sample_ids = list(matrices_binned.keys())
    first_sample = sample_ids[0]
    first_matrix = matrices_binned[first_sample]**pow

    # Prepare arrays for distance calculation
    print("Computing similarity matrix between matrices...")
    matrix_arrays = []
    valid_samples = []
    for sample in sample_ids:
        matrix = matrices_binned[sample]
        if matrix is not None:
            matrix_arrays.append(matrix.flatten())
            valid_samples.append(sample)
    matrix_arrays = np.array(matrix_arrays)

    # Compute pairwise distances
    distances = pdist(matrix_arrays, metric='euclidean')
    distance_matrix = squareform(distances)

    # Find the optimal path
    if random_order:
        np.random.seed(42)
        optimal_path = np.random.permutation(len(valid_samples))
    else:
        optimal_path = find_nearest_neighbor_path(distance_matrix)

    # Set up the animation figure
    fig, ax = plt.subplots()
    ax.set_title("Matrix Morphing Animation", fontsize=16)

    # Normalize all matrices to same range for consistent visualization
    all_matrices = [matrices_binned[valid_samples[i]] for i in optimal_path]
    vmin = min(np.min(matrix) for matrix in all_matrices)
    vmax = max(np.max(matrix) for matrix in all_matrices)

    # Create initial image
    img = ax.imshow(first_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(img, ax=ax, label=cbar_label)

    # Add text annotation
    text_annotation = ax.text(
        0.02, 0.98,
        f'Sample: {os.path.basename(valid_samples[optimal_path[0]]).split(".")[0]}',
        transform=ax.transAxes, fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    def animate(frame, n_frames_per_transition=n_frames_per_transition):
        """Animate function for smooth morphing between contact maps"""
        n_maps = len(optimal_path)
        total_frames = (n_maps - 1) * n_frames_per_transition

        if frame >= total_frames:
            frame = total_frames - 1

        map_idx = frame // n_frames_per_transition
        transition_progress = (frame % n_frames_per_transition) / n_frames_per_transition

        # Get the two maps to interpolate between
        map1_idx = optimal_path[map_idx]
        map2_idx = optimal_path[map_idx + 1]

        map1 = matrices_binned[valid_samples[map1_idx]]
        map2 = matrices_binned[valid_samples[map2_idx]]

        # Linear interpolation between the two maps
        interpolated_matrix = (1 - transition_progress) * map1 + transition_progress * map2

        # Update the image
        img.set_array(interpolated_matrix)

        # Update the text annotation
        def get_sample_name(sample):
            # Try to extract a meaningful sample name
            base = os.path.basename(sample)
            if "_unrelaxed" in base:
                base = base.split("_unrelaxed")[0]
            parts = base.split("_")
            if len(parts) > 1:
                return parts[1]
            return base.split(".")[0]

        sample1_name = get_sample_name(valid_samples[map1_idx])
        sample2_name = get_sample_name(valid_samples[map2_idx])
        text_annotation.set_text(
            f'Transition: {sample1_name} → {sample2_name}\nProgress: {map_idx}/{n_maps} ({transition_progress:.1%})'
        )
        text_annotation.set_position((0.98, 0.98))  # Position at top-right corner
        text_annotation.set_horizontalalignment('right')  # Anchor text to the right

        return [img, text_annotation]

    # Create animation
    n_maps = len(optimal_path)
    total_frames = (n_maps - 1) * n_frames_per_transition

    print(f"Creating animation with {total_frames} frames...")
    print(f"Transitioning through {n_maps} matrices")

    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames,
        interval=interval, blit=True, repeat=True
    )

    plt.tight_layout()
    plt.show()

    # Save the animation as GIF
    print("Saving animation as GIF...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)
    print(f"Animation saved as '{save_path}'")

    # Display some statistics about the path
    print(f"\nAnimation Statistics:")
    print(f"Number of contact maps: {n_maps}")
    print(f"Total animation frames: {total_frames}")
    print(f"Frames per transition: {n_frames_per_transition}")
    print(f"Animation duration: {total_frames * 0.1:.1f} seconds")
    print(f"GIF saved with {fps} FPS")

    # Show the path taken
    print(f"\nPath taken through matrices:")
    for i, idx in enumerate(optimal_path):
        sample_name = os.path.basename(valid_samples[idx]).split(".")[0]
        print(f"{i+1:2d}. {sample_name}")
    
    return {
        "anim": anim,
        "path": optimal_path,
        "valid_samples": valid_samples,
        "matrices_binned": matrices_binned,
        "matrix_arrays": matrix_arrays,
        "distances": distances,
        "distance_matrix": distance_matrix,
    }



def find_dense_submatrix(
    Xwt, 
    window_height=10, 
    window_width=10, 
    frac_min=0.2, 
    frac_max=0.8, 
    plot=True, 
    verbose=True, 
    include_any_1_col=False,
    include_any_1_row=False,
    **clustermap_kwargs
):
    """
    Cluster the matrix, then slide a window to find a submatrix with a fraction of 1s between frac_min and frac_max.
    Optionally plot the heatmap of the found submatrix.

    Parameters
    ----------
    Xwt : pd.DataFrame
        Binary matrix to search.
    window_height : int
        Height of the sliding window.
    window_width : int
        Width of the sliding window.
    frac_min : float
        Minimum fraction of 1s in the submatrix.
    frac_max : float
        Maximum fraction of 1s in the submatrix.
    plot : bool
        Whether to plot the heatmap of the found submatrix.
    verbose : bool
        Whether to print information about the found submatrix.
    include_any_1_col : bool
        If True, after finding the submatrix, include any columns from the original matrix where the value is 1 for at least one of the selected rows.
    include_any_1_row : bool
        If True, after finding the submatrix, include any rows from the original matrix where the value is 1 for at least one of the selected columns.
    **clustermap_kwargs : dict
        Additional arguments to pass to sns.clustermap.

    Returns
    -------
    submatrix : pd.DataFrame or None
        The found submatrix, or None if not found.
    (row_start, row_end), (col_start, col_end) : tuple of ints or None
        The indices of the found submatrix, or None if not found.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create a 20x20 random binary matrix with a dense 1s block in the center
    >>> np.random.seed(0)
    >>> X = np.random.binomial(1, 0.1, size=(20, 20))
    >>> X[5:10, 7:12] = 1  # Insert a dense block of 1s
    >>> df = pd.DataFrame(X, index=[f"row{i}" for i in range(20)], columns=[f"col{j}" for j in range(20)])
    >>> submatrix, row_idx, col_idx = find_dense_submatrix(df, window_height=5, window_width=5, frac_min=0.7, frac_max=1.0, plot=False)
    Found 5x5 submatrix at rows 5-10, cols 7-12 with 100.0% 1s
    >>> print(submatrix)
           col7  col8  col9  col10  col11
    row5      1     1     1      1      1
    row6      1     1     1      1      1
    row7      1     1     1      1      1
    row8      1     1     1      1      1
    row9      1     1     1      1      1
    """
    import seaborn as sns
    # Perform clustering and get the reordered matrix
    if plot:
        cg = sns.clustermap(Xwt, figsize=(10,10), cmap="viridis", **clustermap_kwargs)
    else:
        # Suppress plotting by using a dummy matplotlib backend and closing the figure
        import matplotlib
        import matplotlib.pyplot as plt
        backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        cg = sns.clustermap(Xwt, figsize=(10,10), cmap="viridis", **clustermap_kwargs)
        plt.close('all')
        matplotlib.use(backend)
    Xwt_clustered = Xwt.iloc[cg.dendrogram_row.reordered_ind, cg.dendrogram_col.reordered_ind]

    for i in range(Xwt_clustered.shape[0] - window_height + 1):
        for j in range(Xwt_clustered.shape[1] - window_width + 1):
            sub = Xwt_clustered.iloc[i:i+window_height, j:j+window_width]
            frac_ones = sub.values.sum() / sub.size
            if frac_min <= frac_ones <= frac_max:
                if verbose:
                    print(f"Found {window_height}x{window_width} submatrix at rows {i}-{i+window_height}, cols {j}-{j+window_width} with {frac_ones*100:.1f}% 1s")
                # Handle both include_any_1_col and include_any_1_row
                if include_any_1_col or include_any_1_row:
                    selected_rows = sub.index
                    selected_cols = sub.columns
                    # Start with the submatrix
                    rows_to_use = selected_rows
                    cols_to_use = selected_cols
                    if include_any_1_col:
                        # Find all columns in the original matrix where at least one of these rows has a 1
                        cols_with_1 = Xwt.loc[selected_rows].any(axis=0)
                        cols_to_use = cols_with_1[cols_with_1].index
                    if include_any_1_row:
                        # Find all rows in the original matrix where at least one of these columns has a 1
                        rows_with_1 = Xwt.loc[:, cols_to_use].any(axis=1)
                        rows_to_use = rows_with_1[rows_with_1].index
                    sub_expanded = Xwt.loc[rows_to_use, cols_to_use]
                    if plot:
                        sns.heatmap(sub_expanded, cmap="viridis", cbar=True)
                    return sub_expanded, (i, i+window_height), (j, j+window_width)
                else:
                    if plot:
                        sns.heatmap(sub, cmap="viridis", cbar=True)
                    return sub, (i, i+window_height), (j, j+window_width)
    if verbose:
        print(f"No {window_height}x{window_width} submatrix found with {int(frac_min*100)}-{int(frac_max*100)}% 1s.")
    return None, None, None


def calc_percent_nas(X):
    """
    Print the percentage of NaN (missing) values in a DataFrame or array.

    Parameters
    ----------
    X : pandas.DataFrame or array-like
        The matrix to check for NaN values.

    Returns
    -------
    None
        Prints the percentage of NaN values in X.
    """
    percent_nas = X.isna().sum().sum() / X.size * 100
    print(f"Percent of cells in X that are NaN: {percent_nas:.2f}%") 


def fill_na(X, 
            fillna_method=None,
            verbose=True):
    """
    Fill missing (NaN) values in a DataFrame using various strategies.

    Parameters
    ----------
    X : pandas.DataFrame
        The input matrix with possible NaN values.
    fillna_method : str, int, float, or None, optional
        Method to use for filling NaNs:
            - "colmean": fill NaNs in each column with the column mean
            - "colmedian": fill NaNs in each column with the column median
            - "rowmean": fill NaNs in each row with the row mean
            - "rowmedian": fill NaNs in each row with the row median
            - int or float: fill NaNs with this constant value
            - None or False: do not fill NaNs
    verbose : bool, default=True
        If True, print information about the filling process.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with NaNs filled according to the specified method.

    Raises
    ------
    ValueError
        If an invalid fillna_method is provided.
    """
    if verbose:
        calc_percent_nas(X)
        print(f"Filling NaNs with {fillna_method}") 

    # Compute fill value for missing data
    if fillna_method == "colmean":
        # Fill NaNs in each column with the column mean
        col_means = X.mean(axis=0, skipna=True)
        X = X.fillna(col_means)
    elif fillna_method == "colmedian":
        # Fill NaNs in each column with the column median
        col_medians = X.median(axis=0, skipna=True)
        X = X.fillna(col_medians)
    elif fillna_method == "rowmean":
        # Fill NaNs in each row with the row mean
        row_means = X.mean(axis=1, skipna=True)
        X = X.T.fillna(row_means).T
    elif fillna_method == "rowmedian":
        # Fill NaNs in each row with the row median
        row_medians = X.median(axis=1, skipna=True)
        X = X.T.fillna(row_medians).T
    elif isinstance(fillna_method, (int, float)):
        X = X.fillna(fillna_method)
    elif fillna_method is None or fillna_method is False:
        # Do not fill NaNs, just return as is
        pass
    else:
        raise ValueError(f"Invalid fillna_method: {fillna_method}")
    
    if verbose:
        calc_percent_nas(X)
        print(f"Final matrix shape: {X.shape}")
    
    return X

def minmax_normalize(X, procedure=["rows", "cols"], verbose=True):
    """
    Min-max normalize a matrix by columns and/or rows in a specified order.
    Args:
        X: Matrix to normalize (pd.DataFrame or np.ndarray)
        procedure: List of procedures to apply. Can be "rows" or "cols".
    Returns:
        Normalized matrix
    """

    if not isinstance(X, pd.DataFrame) and isinstance(X, np.ndarray):
        X = pd.DataFrame(X)    

    def normalize_rows(X):
        X = X.sub(X.min(axis=1), axis=0)
        X = X.div(X.max(axis=1), axis=0)
        return X
    
    def normalize_cols(X):
        X = X.sub(X.min(axis=0), axis=1)
        X = X.div(X.max(axis=0), axis=1)
        return X
    
    for proc in procedure:
        if proc == "rows":
            if verbose:
                print("Normalizing rows")
            X = normalize_rows(X)
        elif proc == "cols":
            if verbose:
                print("Normalizing columns")
            X = normalize_cols(X)
        else:
            raise ValueError(f"Invalid procedure: {proc}")
    return X


def minmax_normalize_numpy(X):
    """
    Min-max normalize a matrix by columns and/or rows in a specified order.
    Args:
        X: Matrix to normalize (pd.DataFrame or np.ndarray)
    Returns:
        Normalized matrix
    """
    X_min = np.nanmin(X, axis=1, keepdims=True)
    X_max = np.nanmax(X, axis=1, keepdims=True)
    X = (X - X_min) / (X_max - X_min + 1e-8)
    return X

def fill_coordinates(df, 
                     full_length,
                     x_id_col='variant',
                     y_id_col='mutant',
                     x_pos_col='variant_position',
                     y_pos_col='mutant_position',
                     value_col='VEP',
                     aggfunc='mean', 
                     dropna=False,
                     **kwargs):
    """
    Fill a coordinate matrix with values from a DataFrame, creating a complete grid of positions.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the data in long format (one row per x-y coordinate)
        x_id_col (str): Column name for x-axis identifiers
        y_id_col (str): Column name for y-axis identifiers
        x_pos_col (str): Column name for x-axis positions
        y_pos_col (str): Column name for y-axis positions
        value_col (str): Column name for the values to fill in the matrix
        full_length (int): Length of the complete position range
        aggfunc (str): Aggregation function to use for duplicate values
        dropna (bool): Whether to drop NaN values
        **kwargs: Additional arguments passed to pd.pivot_table
        
    Returns:
        pd.DataFrame: Pivoted matrix with filled coordinates
    """
    # Select and deduplicate relevant columns
    dat = df[[x_id_col, y_id_col, x_pos_col, y_pos_col, value_col]].drop_duplicates().copy()
    
    # Convert position columns to integers, handling NaN values
    dat[x_pos_col] = dat[x_pos_col].astype('Int64')
    dat[y_pos_col] = dat[y_pos_col].astype('Int64')

    # Drop rows with NaN values in x_pos_col or y_pos_col
    dat.dropna(subset=[x_pos_col, y_pos_col], inplace=True)

    # Create complete range of positions
    all_positions = pd.DataFrame({
        y_pos_col: range(1, full_length + 1),
        x_pos_col: range(1, full_length + 1)
    })

    # Merge with original data to include all positions
    dat = pd.merge(
        dat,
        all_positions,
        on=[y_pos_col, x_pos_col],
        how='outer'
    )

    # Create pivoted matrix
    X = dat.pivot_table(
        index=x_pos_col, 
        columns=y_pos_col, 
        values=value_col, 
        aggfunc=aggfunc, 
        dropna=dropna,
        **kwargs
    )
    return X
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

def label_bins(bin_size, n_bins, max_labels=10):
    """
    Create and set bin labels for a contact map plot based on residue positions.
    
    Args:
        bin_size (int): Size of each bin in residues
        n_bins (int): Number of bins in the contact map
        max_labels (int, optional): Maximum number of labels to show. Defaults to 10.
        
    Returns:
        None: Modifies the current matplotlib plot's axis labels
    """
    # Calculate optimal spacing between labels to show max_labels
    spacing = max(1, n_bins // max_labels)
    
    # Generate labels with optimal spacing
    bin_labels = [f"{i*bin_size + 1}" if i % spacing == 0 else "" for i in range(n_bins)]
    
    plt.xticks(range(n_bins), bin_labels, rotation=90)
    plt.yticks(range(n_bins), bin_labels)

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

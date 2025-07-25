# Source: https://github.com/sokrypton/ColabFold
# Source: https://github.com/YoshitakaMo/localcolabfold
# NOTES:
# - First must install localcolabfold via instructions here: https://github.com/YoshitakaMo/localcolabfold
# - By default, MSAs for each sequence in a fasta file will be generated via the public colabfold MSA server using MMseqs2
# - If you have issues with CuDNN, try: 
#   -  pip install "tensorflow[and-cuda]"
#   -  Or follow the instructions here: https://github.com/YoshitakaMo/localcolabfold/issues/210#issuecomment-2456052121
# - ColabFold currently require tensorflow >=2.16.2: https://github.com/sokrypton/ColabFold/blob/d1b8ec153b65a6b90e75a8d12f058ee292473962/pyproject.toml#L27C1-L28C1
# - High pLDDT and low PAE: Indicates high confidence in both the local and relative positions of residues/domains.
# - In the outputs .pdb files, rank indicates which model performs best for that sequence. 
#   Model indicates which model was used, since AlphaFold2 uses an ensemble of models.
#   Unrelaxed structures are those that have not undergone a refinement process ("_unrelaxed_rank_"), 
#   while relaxed structures are those that have been refined to improve their accuracy ("_relaxed_rank_"). 
#   - e.g. ENSP00000497069_492V_I_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb
#   - The "relaxed_rank" files will only be generated when the "--amber" flag is used.
# - Embeddings will not saved by default, but can be saved by using the following flags:
#   - "--save-single-representations": Saves per-residue embeddings that capture the local structural context 
#     and evolutionary information for each amino acid position
#   - "--save-pair-representations": Saves pairwise embeddings that encode the relationships and interactions 
#     between pairs of residues, useful for understanding residue-residue contacts and structural constraints
# - If on Elzar HPC, load CUDA and GCC first with: module load CUDA/12.3.0 GCC
# Known bugs:
# - `/lib64/libstdc++.so.6: version GLIBCXX_3.4.20' not found`: 
#   - https://github.com/YoshitakaMo/localcolabfold/issues/179#issuecomment-1742059553
#   - https://github.com/YoshitakaMo/localcolabfold#for-linux

##### Example usage: #####
# module load CUDA/12.3.0 GCC
# export PATH="/home/schilder/projects/localcolabfold/colabfold-conda/bin:$PATH"
# export CUDA_VISIBLE_DEVICES=1,3
# python >> import src.haplosaurus as hs;hs.haplotypes_to_fasta(tx_ids="ENST00000357654")
# mkdir ENST00000357654 & cd ENST00000357654 
# scp $HOME/projects/data/1KG/fasta/split/ENST00000357654.fasta.gz .
# gunzip ENST00000357654.fasta.gz
#### Then run using different MSAs for each sequence (constructed using MMseqs2)
# colabfold_batch --save-single-representations --save-pair-representations ENST00000357654.fasta af2
#### OR using custom MSA (constructed using the same MSA template for all sequences)
# colabfold_batch --save-single-representations --save-pair-representations af2_sameMSA/ af2_sameMSA/

from Bio.PDB import PDBParser, Selection
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import alphafold_db as af
from Bio.PDB.Structure import Structure

from typing import Optional
import pooch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os

def import_pdb(pdb_path: str, protein_name: str = "BRCA1") -> Structure:
    """
    Import a protein structure from a PDB file (local or remote).
    
    Args:
        pdb_path: Path to the PDB file (local file path or URL)
        protein_name: Name to assign to the structure
        
    Returns:
        Bio.PDB.Structure: Parsed protein structure
        
    Raises:
        Exception: If download or file reading fails
        
    Example:
        >>> structure = import_pdb("https://alphafold.ebi.ac.uk/files/AF-Q8CGX5-F1-model_v4.pdb")
        >>> structure = import_pdb("/path/to/local/file.pdb")
        >>> print(f"Number of models: {len(structure)}")
    """
    import requests
    from io import StringIO
    import os
    
    parser = PDBParser()
    
    # Check if pdb_path is a URL (starts with http/https)
    if pdb_path.startswith(('http://', 'https://')):
        # Handle remote file
        response = requests.get(pdb_path)
        if response.status_code == 200:
            pdb_data = StringIO(response.text)
            structure = parser.get_structure(protein_name, pdb_data)
            return structure
        else:
            raise Exception(f"Failed to download PDB file from {pdb_path}")
    else:
        # Handle local file
        if os.path.exists(pdb_path):
            structure = parser.get_structure(protein_name, pdb_path)
            return structure
        else:
            raise Exception(f"Local PDB file not found: {pdb_path}")

def import_mmcif(protein_id,
                 protein_name: Optional[str] = None,
                 cache = pooch.os_cache("alphafold_db")) -> Structure:
    """
    Import a protein structure from AlphaFold database in mmCIF format.
    See example here: https://alphafold.ebi.ac.uk/entry/P38398
    
    Args:
        protein_name: Name to assign to the structure
        protein_id: UniProt ID of the protein (e.g. "P38398" for BRCA1)
        
    Returns:
        Bio.PDB.Structure: Parsed protein structure
        
    Example:
        >>> structure = import_mmcif(protein_name="BRCA1", protein_id="P38398")
        >>> print(f"Number of models: {len(structure)}")
    """ 
    if protein_name is None:
        protein_name = protein_id
        
    # Download mmCIF file from AlphaFold database 
    for pred in af.get_predictions(protein_id):
        mmCIF = af.download_cif_for(pred, directory=cache)
    
    # Parse mmCIF file into structure object
    parser = MMCIFParser()
    structure = parser.get_structure(protein_name, mmCIF)
    
    return structure

def get_contact_map(structure, 
                    max_distance: float = 8.0,  # Maximum distance to consider
                    continuous: bool = True,  # Flag to control whether contact map is continuous or binary
                    return_distance_map: bool = True,  # Flag to control whether to return distance map or contact map
                    chain_id: str = "A",
                    structure_id: int = 0,
                    ) -> np.ndarray:
    """Calculate contact map or distance map from a protein structure.
    
    Args:
        structure: Bio.PDB.Structure object containing the protein structure
        max_distance: Maximum distance (Å) to consider for contact scoring
        continuous: If True, returns continuous contact scores using inverse distance.
                   If False, returns binary contact map
        return_distance_map: If True, returns distance map normalized to [0,1].
                           If False, returns contact map
        chain_id: Chain identifier to analyze
        structure_id: Model number to analyze (default: 0 for first model)
        
    Returns:
        np.ndarray: Contact map or distance map matrix of shape (n_residues, n_residues)
        
    Example:
        >>> structure = import_mmcif(protein_id="P38398")
        >>> contact_map = get_contact_map(structure, max_distance=8.0, continuous=True)
        >>> print(f"Contact map shape: {contact_map.shape}")
    """
    # Get specified model and chain
    model = structure[structure_id]
    chain = model[chain_id]

    # Extract alpha carbon coordinates efficiently
    ca_coords = []
    for residue in chain.get_residues():
        for atom in residue:
            if atom.get_id() == "CA":
                ca_coords.append(atom.get_coord())
                break
    
    # Convert to numpy array for vectorized operations
    ca_coords = np.array(ca_coords)
    num_residues = len(ca_coords)
    
    # Use broadcasting for ultra-fast pairwise distance calculation
    # Reshape for broadcasting: (n, 1, 3) - (1, n, 3) = (n, n, 3)
    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff**2, axis=2))

    if return_distance_map:
        # Normalize distance matrix to [0,1] range
        max_dist = distance_matrix.max()
        contact_map = max_dist - distance_matrix
    else:
        # Convert distances to contact scores
        if continuous:
            # Continuous scoring using inverse distance with vectorized operations
            mask = distance_matrix < max_distance
            contact_map = np.zeros_like(distance_matrix)
            contact_map[mask] = 1.0 / (1.0 + distance_matrix[mask])
        else:
            # Binary scoring with vectorized operations
            contact_map = (distance_matrix < max_distance).astype(float)
            
    return contact_map

def get_plddt(structure) -> pd.DataFrame:
    """Extract pLDDT scores from a protein structure and categorize them by confidence level.
    
    Args:
        structure: A BioPython structure object containing the protein model
        
    Returns:
        pd.DataFrame: DataFrame containing pLDDT scores and confidence categories with columns:
            - residue: Residue number
            - pLDDT: pLDDT score (0-100)
            - confidence_category: Category name (very_high, high, low, very_low)
            - confidence_label: Human-readable category description
            - confidence_color: Hex color code for visualization
            
    Example:
        >>> structure = import_mmcif(protein_id="P38398")
        >>> plddt_df = get_plddt(structure)
        >>> print(f"Mean pLDDT score: {plddt_df['pLDDT'].mean():.2f}")
    """
    # Create a dictionary mapping pLDDT scores to confidence categories
    plddt_dict = {
        'very_high': {'label':'Very high (pLDDT > 90)', 'color':'#0053D6', 'min':90, 'max':100},
        'high': {'label':'High (90 > pLDDT > 70)', 'color':'#65CBF3', 'min':70, 'max':90},
        'low': {'label':'Low (70 > pLDDT > 50)', 'color':'#FFDB13', 'min':50, 'max':70},
        'very_low': {'label':'Very low (pLDDT < 50)', 'color':'#FF7D45', 'min':0, 'max':50}
    }

    # Extract pLDDT scores and create DataFrame
    plddt_data = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # Get pLDDT score from B-factor column
                plddt = residue['CA'].get_bfactor()
                # Determine confidence category
                category = next((k for k, v in plddt_dict.items() 
                            if v['min'] <= plddt < v['max']), 'very_low')
                
                plddt_data.append({
                    'residue': residue.get_id()[1],
                    'pLDDT': plddt,
                    'confidence_category': category,
                    'confidence_label': plddt_dict[category]['label'],
                    'confidence_color': plddt_dict[category]['color']
                })

    # Create DataFrame
    plddt_df = pd.DataFrame(plddt_data)

    # Print basic statistics
    print(f"Number of residues: {len(plddt_df)}")
    return plddt_df

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

def expand_matrix(X, target_size):
    """
    Expand a binned matrix back to original dimensions by repeating values.
    
    Args:
        X (np.ndarray): Input binned matrix
        target_size (int): Target size for the expanded matrix (default: 1863)
        
    Returns:
        np.ndarray: Expanded matrix with dimensions (target_size, target_size)
    """
    if target_size is None:
        raise ValueError("target_size must be specified")

    if X.shape[0] == target_size and X.shape[1] == target_size:
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

def plot_contact_map(contact_map,  
                     bin_size=2, 
                     reverse_sign=False,
                     normalize_rows=False,
                     normalize_scale=False,
                     dist_axis = None,
                     dist_metric = "cosine",
                     log_before_bin=False, 
                     log_after_bin=False,
                     log_func=lambda x: np.log10(x+1e-6),
                     max_labels=10, 
                     pow=4,
                     cmap="gnuplot2", 
                     agg_func=np.nanmax,
                     title=None,
                     cbar_label="Contact score",
                     x_label="Residue Position",
                     y_label="Residue Position"):
    
    import scipy.spatial.distance
 
    if reverse_sign:
        contact_map = -contact_map
    
    # Normalize each row to sum to 1, skipping NAs 
    if normalize_rows:
        row_sums = np.nansum(contact_map, axis=1, keepdims=True)
        contact_map = np.divide(contact_map, row_sums, where=(row_sums!=0) & (row_sums!=np.nan))


    # Normalize the whole matrix to a scale from 0 to 1
    if normalize_scale:
        contact_map = (contact_map - np.nanmin(contact_map)) / (np.nanmax(contact_map) - np.nanmin(contact_map))

    # Log before binning
    if log_before_bin:
        contact_map = log_func(contact_map) 

    if dist_axis == "x":
        # Calculate cosine similarity along rows
        vep_matrix = scipy.spatial.distance.cdist(vep_matrix, vep_matrix, metric=dist_metric)
        vep_matrix = np.nanmax(vep_matrix) - vep_matrix
    elif dist_axis == "y":
        # Calculate cosine similarity along columns
        vep_matrix = scipy.spatial.distance.cdist(vep_matrix.T, vep_matrix.T, metric=dist_metric)
        vep_matrix = np.nanmax(vep_matrix) - vep_matrix
 
    ### Bin the matrix   
    contact_map_binned = bin_matrix(X=contact_map, 
                                    bin_size=bin_size, 
                                    agg_func=agg_func)
    n_bins = contact_map_binned.shape[0]
    
    ### Log after binning
    if log_after_bin:
        contact_map_binned = log_func(contact_map_binned)

    ### Power transform the matrix
    if pow is not None:
        contact_map_binned = contact_map_binned**pow

    # Create the plot 
    plt.imshow(contact_map_binned, 
                cmap=cmap, 
                interpolation="nearest")
    plt.colorbar().ax.set_ylabel(cbar_label, 
                                 rotation=270, 
                                 va="bottom") 

    label_bins(bin_size, n_bins, max_labels=max_labels)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

    return contact_map, contact_map_binned

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

def animate_contact_maps_variation(contact_maps, 
                                 n_frames=None,
                                 bin_size=10,
                                 pow=4,
                                 cmap="gnuplot2",
                                 figsize=(8, 6),
                                 dpi=100,
                                 duration=500,
                                 output_path='results/plots/contact_maps_animation.gif'):
    """
    Create an animated GIF showing contact maps from different structures.
    
    Parameters
    ----------
    contact_maps : dict
        Dictionary mapping names to contact map arrays
    n_frames : int, optional
        Number of frames to include (None for all)
    bin_size : int, default=10
        Size of bins for matrix binning
    pow : int, default=4
        Power to raise contact maps to
    cmap : str, default="gnuplot2"
        Colormap for visualization
    figsize : tuple, default=(8, 6)
        Figure size
    dpi : int, default=100
        DPI for saved images
    duration : int, default=500
        Duration per frame in milliseconds
    output_path : str, default='results/plots/contact_maps_animation.gif'
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
    contact_maps_subset = list(contact_maps.items())[:n_frames]
    
    for i, (name, contact_map) in enumerate(tqdm(contact_maps_subset)):
        # Bin the matrix   
        contact_map_binned = bin_matrix(X=contact_map**pow, 
                                           bin_size=bin_size)
         
        # Create a single plot for each frame
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(contact_map_binned, 
                        cmap=cmap, 
                        interpolation="nearest")
        
        protein_id = os.path.basename(name).split("_unrelaxed")[0].split("_")[0]
        haplotype_id = os.path.basename(name).split("_unrelaxed")[0].replace(protein_id+"_","")
        
        # Set haplotype ID left justified
        ax.set_title(f"{haplotype_id}", fontsize=12, loc='left', pad=10)
        
        # Add frame counter right justified
        ax.text(0.98, 0.98, f"({i+1} / {len(contact_maps_subset)})", 
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




def animate_contact_map_interpolation(map1, 
                                    map2, 
                                    num_frames=30, 
                                    duration=200, # 200ms per frame
                                    loop=0,
                                    pow=1,
                                    figsize=(8, 6),
                                    dpi=100,
                                    format="png",
                                    cmap="gnuplot2",
                                    save_path="results/plots/contact_map_interpolation.gif",
                                    ):
    """
    Create a smooth animation transitioning between two contact maps using the trained autoencoder.
    
    Args:
        map1: First contact map (numpy array)
        map2: Second contact map (numpy array) 
        model: Trained autoencoder model
        num_frames: Number of frames in the animation
        pow: Power to raise the contact map to
        output_path: Path to save the GIF
    """
    
    # Ensure maps have the same shape
    if map1.shape != map2.shape:
        # Resize map2 to match map1's shape
        from scipy.ndimage import zoom
        zoom_factors = (map1.shape[0] / map2.shape[0], map1.shape[1] / map2.shape[1])
        map2 = zoom(map2, zoom_factors, order=1) 

        map1 = np.power(map1, pow)
        map2 = np.power(map2, pow)
        # Fallback: simple linear interpolation without autoencoder
        frames = []
        for i in range(num_frames):
            alpha = i / (num_frames - 1)
            interpolated_map = alpha * map2 + (1 - alpha) * map1
            
            # Create frame
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(interpolated_map, cmap=cmap, interpolation='nearest')
            ax.set_title(f'Linear Interpolation Frame {i+1}/{num_frames} (α={alpha:.2f})')
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Contact Probability')
            
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

def standardize_id(id, 
                    pattern=[" ", ">", ",", ":", "*", "{", "}", "(", ")"], 
                    replacement="_"):
        new_id = id
        for char in pattern:
            new_id = new_id.replace(char, replacement)
        return new_id

def create_haplotype_msas(template_msa, 
                          haplotypes_fasta,  
                          output_dir=None,
                          output_suffix="a3m",
                          description_prefix="",
                          return_files=True,
                          verbose=False,
                          force=False):
    """
    Create MSA files for each haplotype sequence by replacing the first sequence
    in a template MSA with each haplotype sequence.
    This ensures that AlphaFold2 can use the same MSA for all haplotypes.

    Args:
        template_msa (str): Path to template MSA file (fasta format)
        haplotypes_fasta (str): Path to fasta file containing haplotype sequences
        output_dir (str): Directory to save output MSA files
        output_suffix (str): File extension for output files (default: "a3m")
        verbose (bool): Whether to print progress messages (default: True)
        force (bool): Whether to overwrite existing files (default: False)
    Returns:
        list: List of paths to created MSA files

    Example:
        create_haplotype_msas(template_msa="~/projects/data/ProteinGym/subs/refseq-BRCA1-NM_007294.3.a2m", 
                              haplotypes_fasta="~/projects/data/1KG/fasta/split/ENST00000357654.fasta",  
                              output_dir="~/projects/data/colabfold/ENST00000357654/af2_sameMSA",
                              description_prefix="1000_Genomes_on_GRCh38-Haplosaurus-")
    """
    from Bio import SeqIO  

    if output_dir is None:
        import tempfile
        output_dir = tempfile.mkdtemp()
        if verbose:
            print(f"Setting output_dir to temporary directory: {output_dir}")

    # Expand the paths
    template_msa = os.path.expanduser(template_msa)
    haplotypes_fasta = os.path.expanduser(haplotypes_fasta)
    output_dir = os.path.expanduser(output_dir)

    # Import the haplotype sequences
    haplotype_seqs = list(SeqIO.parse(haplotypes_fasta, "fasta"))
 
    # Import template MSA as a list of SeqRecord objects
    msa_seqs = list(SeqIO.parse(template_msa, "fasta"))
 
    # Ensure output directory exists
    os.makedirs(os.path.expanduser(output_dir), exist_ok=True)
    
    haplotype_msas = []
    for hap_seq in tqdm(haplotype_seqs, 
                        desc="Processing haplotype sequences"): 
        standardized_id = standardize_id(hap_seq.id) 
        output_file = os.path.expanduser(f"{output_dir}/{standardized_id}.{output_suffix}")
        if os.path.exists(output_file) and not force:
            if verbose:
                print(f"Skipping {output_file} because it already exists")
            if return_files:
                haplotype_msas.append(output_file)
            else:
                haplotype_msas.append(msa_seqs)
            continue
        # Replace the first sequence with the haplotype sequence
        
        msa_seqs[0].id = standardized_id
        msa_seqs[0].name = hap_seq.name
        msa_seqs[0].description = f"{description_prefix}{hap_seq.id}"
        # AlphaFold2 doesn't supoort gaps. Can instead replace with X (unknown sequence):
        # https://github.com/google-deepmind/alphafold/issues/150#issuecomment-905341661
        msa_seqs[0].seq = hap_seq.seq.replace("-", "X") 
        
        # Write the alignment to file 
        SeqIO.write(msa_seqs, output_file, "fasta")
        
        if verbose:
            print(f"Alignment written to {output_file}")
        if return_files:
            haplotype_msas.append(output_file)
        else:
            haplotype_msas.append(msa_seqs)
    
    return haplotype_msas

def create_interactive_umap_contact_plot(af2_meta, contact_maps, 
                                        x_col='umap_1', y_col='umap_2',
                                        color_col=None, size_col=None,
                                        hover_cols=None, 
                                        contact_map_params=None,
                                        figsize=(1200, 600),
                                        title="Interactive UMAP with Contact Maps"):
    """
    Create an interactive 2D scatterplot of AlphaFold2 embeddings with dynamic contact map visualization.
    
    Args:
        af2_meta: DataFrame containing UMAP coordinates and metadata
        contact_maps: Dictionary mapping sequence identifiers to contact map arrays
        x_col: Column name for x-axis (default: 'umap_1')
        y_col: Column name for y-axis (default: 'umap_2')
        color_col: Column name for point colors (optional)
        size_col: Column name for point sizes (optional)
        hover_cols: List of columns to show in hover tooltip (optional)
        contact_map_params: Dictionary of parameters for contact map visualization
        figsize: Tuple of (width, height) in pixels
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot with scatter plot and contact map subplot
        
    Example:
        >>> fig = create_interactive_umap_contact_plot(
        ...     af2_meta=af2_meta,
        ...     contact_maps=contact_maps,
        ...     color_col='sample_id',
        ...     hover_cols=['sample_id', 'edit_distance', 'model_rank']
        ... )
        >>> fig.show()
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    # Default contact map parameters
    if contact_map_params is None:
        contact_map_params = {
            'bin_size': 2,
            'pow': 4,
            'cmap': 'viridis',
            'normalize_scale': True
        }
    # Map matplotlib colormap names to Plotly colorscales
    colormap_mapping = {
        'gnuplot2': 'viridis',
        'gnuplot': 'viridis',
        'plasma': 'plasma',
        'inferno': 'inferno',
        'magma': 'magma',
        'viridis': 'viridis',
        'cividis': 'cividis',
        'hot': 'hot',
        'cool': 'plasma',
        'spring': 'plasma',
        'summer': 'viridis',
        'autumn': 'viridis',
        'winter': 'viridis',
        'gray': 'gray',
        'bone': 'gray',
        'pink': 'pinkyl',
        'copper': 'oranges',
        'jet': 'jet',
        'hsv': 'hsv',
        'rainbow': 'rainbow',
        'ocean': 'haline',
        'gist_earth': 'earth',
        'terrain': 'earth',
        'gist_stern': 'viridis',
        'brg': 'rdbu',
        'CMRmap': 'viridis',
        'cubehelix': 'viridis',
        'flag': 'viridis',
        'prism': 'viridis',
        'nipy_spectral': 'spectral',
        'gist_ncar': 'spectral'
    }
    if contact_map_params['cmap'] in colormap_mapping:
        contact_map_params['cmap'] = colormap_mapping[contact_map_params['cmap']]
    
    # Create subplot layout: scatter plot on left, contact map on right
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=('UMAP Embeddings', 'Contact Map'),
        specs=[[{"type": "scatter"}, {"type": "heatmap"}]]
    )
    
    # Prepare scatter plot data
    scatter_data = af2_meta.copy()
    
    # Create hover text
    if hover_cols is None:
        hover_cols = ['sample_id'] if 'sample_id' in scatter_data.columns else []
    
    hover_text = []
    for idx, row in scatter_data.iterrows():
        text_parts = []
        for col in hover_cols:
            if col in row and pd.notna(row[col]):
                text_parts.append(f"{col}: {row[col]}")
        hover_text.append("<br>".join(text_parts))
    
    # Create scatter plot
    scatter_kwargs = {
        'x': scatter_data[x_col],
        'y': scatter_data[y_col],
        'mode': 'markers',
        'text': hover_text,
        'hovertemplate': '<b>%{text}</b><extra></extra>',
        'name': 'UMAP Points'
    }
    
    # Add color and size if specified
    marker_dict = {}
    
    if color_col and color_col in scatter_data.columns:
        # Create color mapping for categorical data
        unique_colors = scatter_data[color_col].unique()
        color_map = {val: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)] 
                    for i, val in enumerate(unique_colors)}
        colors = [color_map[val] for val in scatter_data[color_col]]
        marker_dict['color'] = colors
    
    if size_col and size_col in scatter_data.columns:
        # Normalize size to reasonable range
        sizes = scatter_data[size_col]
        if sizes.dtype in ['int64', 'float64']:
            min_size, max_size = 5, 20
            normalized_sizes = min_size + (sizes - sizes.min()) / (sizes.max() - sizes.min()) * (max_size - min_size)
            marker_dict['size'] = normalized_sizes
    
    if marker_dict:
        scatter_kwargs['marker'] = marker_dict
    
    # Add scatter plot to figure
    fig.add_trace(
        go.Scatter(**scatter_kwargs),
        row=1, col=1
    )
    
    # Create a placeholder contact map (will be updated on hover)
    if contact_maps:
        # Get the first contact map as placeholder
        first_key = list(contact_maps.keys())[0]
        first_contact_map = contact_maps[first_key]
        
        # Process contact map
        contact_map_processed = process_contact_map_for_display(
            first_contact_map, **contact_map_params
        )
        
        # Add placeholder heatmap
        fig.add_trace(
            go.Heatmap(
                z=contact_map_processed,
                colorscale=contact_map_params['cmap'],
                showscale=True,
                name='Contact Map',
                hovertemplate='Residue i: %{y}<br>Residue j: %{x}<br>Contact: %{z:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        width=figsize[0],
        height=figsize[1],
        showlegend=False,
        hovermode='closest'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text=x_col, row=1, col=1)
    fig.update_yaxes(title_text=y_col, row=1, col=1)
    fig.update_xaxes(title_text="Residue j", row=1, col=2)
    fig.update_yaxes(title_text="Residue i", row=1, col=2)
    
    # Add JavaScript for dynamic contact map updates
    # This will be handled by the hover events
    fig.update_traces(
        hoverinfo='skip',
        selector=dict(type='scatter')
    )
    
    return fig

def process_contact_map_for_display(contact_map, bin_size=2, pow=4, 
                                   normalize_scale=True, cmap='gnuplot2'):
    """
    Process a contact map for display in the interactive plot.
    
    Args:
        contact_map: Contact map array
        bin_size: Size of bins for matrix binning
        pow: Power to raise contact map to
        normalize_scale: Whether to normalize to [0,1] scale
        cmap: Colormap name
        
    Returns:
        np.ndarray: Processed contact map ready for display
    """
    # Apply power transformation
    if pow is not None:
        contact_map = contact_map**pow
    
    # Normalize scale if requested
    if normalize_scale:
        contact_map = (contact_map - np.nanmin(contact_map)) / (np.nanmax(contact_map) - np.nanmin(contact_map))
    
    # Bin the matrix
    if bin_size > 1:
        contact_map = bin_matrix(contact_map, bin_size=bin_size, agg_func=np.nanmax)
    
    return contact_map

def create_dash_interactive_umap_app(af2_meta, contact_maps, 
                                   x_col='umap_1', y_col='umap_2',
                                   color_col=None, size_col=None,
                                   hover_cols=None, 
                                   contact_map_params=None,
                                   title="Interactive UMAP with Contact Maps"):
    """
    Create a Dash web application for interactive UMAP visualization with dynamic contact maps.
    
    Args:
        af2_meta: DataFrame containing UMAP coordinates and metadata
        contact_maps: Dictionary mapping sequence identifiers to contact map arrays
        x_col: Column name for x-axis (default: 'umap_1')
        y_col: Column name for y-axis (default: 'umap_2')
        color_col: Column name for point colors (optional)
        size_col: Column name for point sizes (optional)
        hover_cols: List of columns to show in hover tooltip (optional)
        contact_map_params: Dictionary of parameters for contact map visualization
        title: Plot title
        
    Returns:
        dash.Dash: Dash application object
        
    Example:
        >>> app = create_dash_interactive_umap_app(
        ...     af2_meta=af2_meta,
        ...     contact_maps=contact_maps,
        ...     color_col='sample_id',
        ...     hover_cols=['sample_id', 'edit_distance']
        ... )
        >>> app.run_server(debug=True, port=8050)
    """
    try:
        import dash
        from dash import dcc, html, Input, Output, callback
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
    except ImportError:
        print("Dash is required for this functionality. Install with: pip install dash")
        return None
    
    # Default contact map parameters
    if contact_map_params is None:
        contact_map_params = {
            'bin_size': 2,
            'pow': 4,
            'cmap': 'viridis',
            'normalize_scale': True
        }
    colormap_mapping = {
        'gnuplot2': 'viridis',
        'gnuplot': 'viridis',
        'plasma': 'plasma',
        'inferno': 'inferno',
        'magma': 'magma',
        'viridis': 'viridis',
        'cividis': 'cividis',
        'hot': 'hot',
        'cool': 'plasma',
        'spring': 'plasma',
        'summer': 'viridis',
        'autumn': 'viridis',
        'winter': 'viridis',
        'gray': 'gray',
        'bone': 'gray',
        'pink': 'pinkyl',
        'copper': 'oranges',
        'jet': 'jet',
        'hsv': 'hsv',
        'rainbow': 'rainbow',
        'ocean': 'haline',
        'gist_earth': 'earth',
        'terrain': 'earth',
        'gist_stern': 'viridis',
        'brg': 'rdbu',
        'CMRmap': 'viridis',
        'cubehelix': 'viridis',
        'flag': 'viridis',
        'prism': 'viridis',
        'nipy_spectral': 'spectral',
        'gist_ncar': 'spectral'
    }
    if contact_map_params['cmap'] in colormap_mapping:
        contact_map_params['cmap'] = colormap_mapping[contact_map_params['cmap']]
    
    # Create Dash app
    app = dash.Dash(__name__)
    
    # Prepare data
    scatter_data = af2_meta.copy()
    
    # Create hover text
    if hover_cols is None:
        hover_cols = ['sample_id'] if 'sample_id' in scatter_data.columns else []
    
    hover_text = []
    for idx, row in scatter_data.iterrows():
        text_parts = []
        for col in hover_cols:
            if col in row and pd.notna(row[col]):
                text_parts.append(f"{col}: {row[col]}")
        hover_text.append("<br>".join(text_parts))
    
    # Create initial figure
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=('UMAP Embeddings', 'Contact Map'),
        specs=[[{"type": "scatter"}, {"type": "heatmap"}]]
    )
    
    # Add scatter plot
    scatter_kwargs = {
        'x': scatter_data[x_col],
        'y': scatter_data[y_col],
        'mode': 'markers',
        'text': hover_text,
        'hovertemplate': '<b>%{text}</b><extra></extra>',
        'name': 'UMAP Points'
    }
    
    # Add color and size if specified
    marker_dict = {}
    
    if color_col and color_col in scatter_data.columns:
        # Create color mapping for categorical data
        unique_colors = scatter_data[color_col].unique()
        color_map = {val: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)] 
                    for i, val in enumerate(unique_colors)}
        colors = [color_map[val] for val in scatter_data[color_col]]
        marker_dict['color'] = colors
    
    if size_col and size_col in scatter_data.columns:
        sizes = scatter_data[size_col]
        if sizes.dtype in ['int64', 'float64']:
            min_size, max_size = 5, 20
            normalized_sizes = min_size + (sizes - sizes.min()) / (sizes.max() - sizes.min()) * (max_size - min_size)
            marker_dict['size'] = normalized_sizes
    
    if marker_dict:
        scatter_kwargs['marker'] = marker_dict
    
    fig.add_trace(
        go.Scatter(**scatter_kwargs),
        row=1, col=1
    )
    
    # Add placeholder contact map
    if contact_maps:
        first_key = list(contact_maps.keys())[0]
        first_contact_map = contact_maps[first_key]
        contact_map_processed = process_contact_map_for_display(
            first_contact_map, **contact_map_params
        )
        
        fig.add_trace(
            go.Heatmap(
                z=contact_map_processed,
                colorscale=contact_map_params['cmap'],
                showscale=True,
                name='Contact Map',
                hovertemplate='Residue i: %{y}<br>Residue j: %{x}<br>Contact: %{z:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        width=1200,
        height=600,
        showlegend=False,
        hovermode='closest'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text=x_col, row=1, col=1)
    fig.update_yaxes(title_text=y_col, row=1, col=1)
    fig.update_xaxes(title_text="Residue j", row=1, col=2)
    fig.update_yaxes(title_text="Residue i", row=1, col=2)
    
    # App layout
    app.layout = html.Div([
        html.H1(title),
        dcc.Graph(
            id='umap-contact-plot',
            figure=fig,
            config={'displayModeBar': True}
        ),
        html.Div(id='hover-info', style={'marginTop': 20})
    ])
    
    # Callback to update contact map on hover
    @app.callback(
        Output('umap-contact-plot', 'figure'),
        Input('umap-contact-plot', 'hoverData')
    )
    def update_contact_map(hover_data):
        if hover_data is None or 'points' not in hover_data:
            return fig
        
        # Get the hovered point index
        point_index = hover_data['points'][0]['pointIndex']
        
        # Get the corresponding sample ID or use index
        if 'sample_id' in scatter_data.columns:
            sample_id = scatter_data.iloc[point_index]['sample_id']
        else:
            sample_id = scatter_data.index[point_index]
        
        # Find the corresponding contact map
        contact_map = None
        for key in contact_maps.keys():
            if sample_id in key or str(sample_id) in key:
                contact_map = contact_maps[key]
                break
        
        if contact_map is None:
            # Use first contact map as fallback
            contact_map = list(contact_maps.values())[0]
        
        # Process contact map
        contact_map_processed = process_contact_map_for_display(
            contact_map, **contact_map_params
        )
        
        # Update the figure
        fig.data[1].z = contact_map_processed
        fig.data[1].name = f'Contact Map: {sample_id}'
        
        return fig
    
    return app

def create_interactive_umap_with_contact_maps(af2_meta, contact_maps, 
                                            output_path=None,
                                            use_dash=False,
                                            use_bokeh=False,
                                            **kwargs):
    """
    Create and optionally save an interactive HTML plot with UMAP embeddings and contact maps.
    
    Args:
        af2_meta: DataFrame containing UMAP coordinates and metadata
        contact_maps: Dictionary mapping sequence identifiers to contact map arrays
        output_path: Path to save the interactive HTML file (optional)
        use_dash: Whether to use Dash for full interactivity (requires Dash installation)
        use_bokeh: Whether to use Bokeh for Jupyter notebook interactivity (requires Bokeh installation)
        **kwargs: Additional arguments passed to the respective plotting function
        
    Returns:
        plotly.graph_objects.Figure, dash.Dash, or bokeh.layouts.Column: Interactive plot
        
    Note:
        - For static HTML plots (use_dash=False, use_bokeh=False): Contact map shows first contact map only
        - For dynamic updates in Jupyter: Use use_bokeh=True (requires Bokeh installation)
        - For dynamic updates in web browser: Use use_dash=True (requires Dash installation)
        
    Example:
        >>> fig = create_interactive_umap_with_contact_maps(
        ...     af2_meta=af2_meta,
        ...     contact_maps=contact_maps,
        ...     output_path='interactive_umap_contact_maps.html',
        ...     color_col='sample_id',
        ...     hover_cols=['sample_id', 'edit_distance']
        ... )
    """
    if use_bokeh:
        # Create Bokeh app for Jupyter notebook interactivity
        layout = create_bokeh_interactive_umap(af2_meta, contact_maps, **kwargs)
        if layout:
            print("Bokeh interactive plot created for Jupyter notebook.")
            print("Use show(layout) to display it in the notebook.")
            return layout
        else:
            print("Falling back to static Plotly figure")
    
    elif use_dash:
        # Create Dash app for full interactivity
        app = create_dash_interactive_umap_app(af2_meta, contact_maps, **kwargs)
        if app:
            print("Dash app created. Run with: app.run_server(debug=True, port=8050)")
            return app
        else:
            print("Falling back to static Plotly figure")
    
    # Create the interactive plot
    fig = create_interactive_umap_contact_plot(af2_meta, contact_maps, **kwargs)
    
    # Add warning about static nature
    print("Note: This is a static Plotly figure. The contact map will not update on hover.")
    print("For dynamic contact map updates:")
    print("  - In Jupyter notebooks: use use_bokeh=True (requires Bokeh installation)")
    print("  - In web browsers: use use_dash=True (requires Dash installation)")
    
    # Save to HTML if output path is provided
    if output_path:
        fig.write_html(output_path)
        print(f"Interactive plot saved to: {output_path}")
    
    return fig

def create_bokeh_interactive_umap(af2_meta, contact_maps, 
                                 x_col='umap_1', y_col='umap_2',
                                 color_col=None, size_col=None,
                                 hover_cols=None, 
                                 contact_map_params=None,
                                 title="Interactive UMAP with Contact Maps",
                                 width=1200, height=600,
                                 use_server_callbacks=False):
    """
    Create an interactive UMAP visualization with dynamic contact maps using Bokeh.
    This works well in Jupyter notebooks and provides hover-based contact map updates.
    
    Args:
        af2_meta: DataFrame containing UMAP coordinates and metadata
        contact_maps: Dictionary mapping sequence identifiers to contact map arrays
        x_col: Column name for x-axis (default: 'umap_1')
        y_col: Column name for y-axis (default: 'umap_2')
        color_col: Column name for point colors (optional)
        size_col: Column name for point sizes (optional)
        hover_cols: List of columns to show in hover tooltip (optional)
        contact_map_params: Dictionary of parameters for contact map visualization
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels
        
    Returns:
        bokeh.layouts.Column: Bokeh layout with interactive plots
        
    Example:
        >>> layout = create_bokeh_interactive_umap(
        ...     af2_meta=af2_meta,
        ...     contact_maps=contact_maps,
        ...     color_col='sample_id',
        ...     hover_cols=['sample_id', 'edit_distance']
        ... )
        >>> show(layout)  # In Jupyter notebook
        
    Note:
        This version provides hover-based dynamic contact map updates in Jupyter notebooks.
        The contact map uses a continuous Viridis color palette and updates when you hover over points.
    """
    try:
        from bokeh.plotting import figure, show
        from bokeh.layouts import column, row
        from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper
        from bokeh.transform import factor_cmap, linear_cmap
        from bokeh.palettes import Category10, Set3, Viridis256
        from bokeh.io import output_notebook
        import bokeh
    except ImportError:
        print("Bokeh is required for this functionality. Install with: pip install bokeh")
        return None
    
    # Default contact map parameters
    if contact_map_params is None:
        contact_map_params = {
            'bin_size': 2,
            'pow': 4,
            'cmap': 'viridis',
            'normalize_scale': True
        }
    
    # Map matplotlib colormap names to Bokeh palettes
    colormap_mapping = {
        'gnuplot2': 'Viridis256',
        'gnuplot': 'Viridis256',
        'plasma': 'Plasma256',
        'inferno': 'Inferno256',
        'magma': 'Magma256',
        'viridis': 'Viridis256',
        'cividis': 'Cividis256',
        'hot': 'Hot256',
        'cool': 'Plasma256',
        'spring': 'Plasma256',
        'summer': 'Viridis256',
        'autumn': 'Viridis256',
        'winter': 'Viridis256',
        'gray': 'Greys256',
        'bone': 'Greys256',
        'pink': 'Pink256',
        'copper': 'Oranges256',
        'jet': 'Turbo256',
        'hsv': 'Hue256',
        'rainbow': 'Rainbow256'
    }
    
    # Convert matplotlib colormap to Bokeh palette if needed
    if contact_map_params['cmap'] in colormap_mapping:
        palette_name = colormap_mapping[contact_map_params['cmap']]
        try:
            palette = getattr(bokeh.palettes, palette_name)
        except AttributeError:
            palette = Viridis256
    else:
        palette = Viridis256
    
    # Prepare data
    scatter_data = af2_meta.copy()
    
    # Create hover tooltips
    if hover_cols is None:
        hover_cols = ['sample_id'] if 'sample_id' in scatter_data.columns else []
    
    tooltips = []
    for col in hover_cols:
        if col in scatter_data.columns:
            tooltips.append((col, f'@{col}'))
    
    # Create ColumnDataSource for scatter plot
    source_data = {
        'x': scatter_data[x_col],
        'y': scatter_data[y_col],
        'index': list(range(len(scatter_data)))
    }
    
    # Add hover columns to source
    for col in hover_cols:
        if col in scatter_data.columns:
            source_data[col] = scatter_data[col]
    
    # Add color and size if specified
    if color_col and color_col in scatter_data.columns:
        source_data['color_col'] = scatter_data[color_col]
    
    if size_col and size_col in scatter_data.columns:
        sizes = scatter_data[size_col]
        if sizes.dtype in ['int64', 'float64']:
            min_size, max_size = 5, 20
            normalized_sizes = min_size + (sizes - sizes.min()) / (sizes.max() - sizes.min()) * (max_size - min_size)
            source_data['size'] = normalized_sizes
    
    source = ColumnDataSource(source_data)
    
    # Create scatter plot
    p1 = figure(
        width=width//2, 
        height=height,
        title="UMAP Embeddings",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        tooltips=tooltips
    )
    
    # Add scatter points with color and size
    if color_col and color_col in scatter_data.columns:
        unique_colors = scatter_data[color_col].unique()
        # Get the appropriate palette based on number of unique colors
        if len(unique_colors) <= 10:
            palette = Category10[len(unique_colors)]
        else:
            # For more than 10 colors, use a different palette
            try:
                from bokeh.palettes import Set3
                palette = Set3[min(len(unique_colors), 12)]
            except (ImportError, KeyError):
                # Fallback to Category10 with cycling
                palette = Category10[10] * (len(unique_colors) // 10 + 1)
                palette = palette[:len(unique_colors)]
        color_mapper = factor_cmap('color_col', palette=palette, factors=unique_colors)
        size_col_name = 'size' if size_col and size_col in scatter_data.columns else 8
        p1.scatter('x', 'y', source=source, size=size_col_name, color=color_mapper, alpha=0.7)
    else:
        size_col_name = 'size' if size_col and size_col in scatter_data.columns else 8
        p1.scatter('x', 'y', source=source, size=size_col_name, alpha=0.7)
    
    p1.xaxis.axis_label = x_col
    p1.yaxis.axis_label = y_col
    
    # Create contact map plot
    p2 = figure(
        width=width//2, 
        height=height,
        title="Contact Map",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_range=(0, 100),  # Will be updated dynamically
        y_range=(0, 100)   # Will be updated dynamically
    )
    
    # Process first contact map for initial display
    if contact_maps:
        first_key = list(contact_maps.keys())[0]
        first_contact_map = contact_maps[first_key]
        contact_map_processed = process_contact_map_for_display(
            first_contact_map, **contact_map_params
        )
        
        # Create color mapper for contact map - use continuous palette
        contact_color_mapper = LinearColorMapper(
            palette=Viridis256,  # Use continuous Viridis palette
            low=contact_map_processed.min(),
            high=contact_map_processed.max()
        )
        
        # Create image data for contact map
        img_data = contact_map_processed
        img_glyph = p2.image(
            image=[img_data],
            x=0, y=0, dw=img_data.shape[1], dh=img_data.shape[0],
            color_mapper=contact_color_mapper
        )
        
        # Add colorbar
        color_bar = ColorBar(color_mapper=contact_color_mapper, location=(0, 0))
        p2.add_layout(color_bar, 'right')
    
    # Add hover tool for dynamic updates
    from bokeh.models import HoverTool, CustomJS
    
    # Prepare contact map data for JavaScript
    contact_map_data = {}
    for i, (key, contact_map) in enumerate(contact_maps.items()):
        processed_map = process_contact_map_for_display(contact_map, **contact_map_params)
        contact_map_data[i] = processed_map.tolist()
    
    # Create JavaScript code for dynamic updates on hover
    js_code = """
    var contact_maps = %s;
    var index = cb_obj.index;
    if (index !== undefined && index in contact_maps) {
        var new_data = contact_maps[index];
        var img_source = p2.select_one('image');
        if (img_source) {
            img_source.data_source.data['image'] = [new_data];
            img_source.data_source.change.emit();
            
            // Update color mapper
            var min_val = Math.min(...new_data.flat());
            var max_val = Math.max(...new_data.flat());
            var color_mapper = p2.select_one('LinearColorMapper');
            if (color_mapper) {
                color_mapper.low = min_val;
                color_mapper.high = max_val;
                color_mapper.change.emit();
            }
            
            // Update title
            p2.title.text = "Contact Map (Point " + index + ")";
        }
    }
    """ % contact_map_data
    
    # Add hover callback
    hover_callback = CustomJS(args=dict(p2=p2), code=js_code)
    
    # Create hover tool with callback
    hover_tool = HoverTool(
        tooltips=tooltips,
        callback=hover_callback
    )
    p1.add_tools(hover_tool)
    
    # Add instructions text
    from bokeh.models import Div
    instructions = Div(
        text="<b>Instructions:</b> Hover over points in the scatter plot to see contact maps update dynamically! "
             "The contact map will show the corresponding contact map for each point.",
        width=width,
        height=50,
        styles={'font-size': '12px', 'color': '#666666'}
    )
    
    # Create layout with instructions
    layout = column(instructions, row(p1, p2))
    
    return layout
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
## If using Elzar HPC:
# module load CUDA/12.3.0 GCC 
#
# export PATH="/home/schilder/projects/localcolabfold/colabfold-conda/bin:$PATH"
# export CUDA_VISIBLE_DEVICES=1,3
#
# python: 
# >> import src.haplosaurus as hs;
# >> import os
# >> haplotypes = hs.get_haplotypes(tx_ids=["ENST00000357654"], cache_only=True)
# >> hs.haplotypes_to_fasta(haplotypes=haplotypes,  
# >>                       save_dir=os.path.expanduser("~/projects/data/colabfold/ENST00000357654"), 
# >>                       split_subdir="",
# >>                       force=True,
# >>                       compress=False) # Must be uncompressed for colabfold_batch
#  
#### Then run using different MSAs for each sequence (constructed using MMseqs2)
# colabfold_batch --save-single-representations --save-pair-representations ENST00000357654.fasta af2
#
#### OR using custom MSA (constructed using the same MSA template for all sequences using the create_haplotype_msas() function)
# colabfold_batch --save-single-representations --save-pair-representations af2_sameMSA/ af2_sameMSA/

from Bio.PDB import PDBParser, Selection
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import alphafold_db as afdb
from Bio.PDB.Structure import Structure

from typing import Optional
import pooch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io 
from PIL import Image
from tqdm import tqdm
import os
import glob

AFDB_CACHE = pooch.os_cache("alphafold_db")

def search_files(base_dir,
                 tx_id,
                 subdir = "af2_sameMSA",
                 suffix = ".pdb",
                 model_number = 1):
    
    pdb_files = glob.glob(os.path.join(base_dir,
                                    tx_id,
                                    "**",
                                    #    "*_unrelaxed_rank_001_*.pdb"
                                    f"{subdir}/*_unrelaxed_*_model_{model_number}_*{suffix}"
                                    ),
                        recursive=True)
    print(len(pdb_files),"PDB files found") 
    return pdb_files

def import_pdb(pdb_path: str, 
               protein_name: str = "BRCA1") -> Structure:
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
                 cache = os.path.join(AFDB_CACHE, "cif")) -> Structure:
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
    # cif_path = f"{cache}/AF-{protein_id}-F1-model_v4.cif .cif"
    mmCIFs = [afdb.download_cif_for(pred, directory=cache) for pred in afdb.get_predictions(protein_id)][-1]
    
    # Parse mmCIF file into structure object using list comprehension (for demonstration)
    parser = MMCIFParser()
    structures = [parser.get_structure(protein_name, mmCIFs)]
    
    return structures

def get_contact_map(structure, 
                    max_distance: float = 8.0,  # Maximum distance to consider
                    continuous: bool = True,  # Flag to control whether contact map is continuous or binary
                    return_distance_map: bool = True,  # Flag to control whether to return distance map or contact map
                    return_raw_distance_map: bool = False,  # Flag to control whether to return raw distance map
                    chain_id: str = "A",
                    structure_id: int = 0,
                    verbose: bool = False,
                    ) -> np.ndarray:
    """Calculate contact map or distance map from a protein structure.
    
    Args:
        structure: Bio.PDB.Structure object containing the protein structure
        max_distance: Maximum distance in units of Angstroms (Å) to consider for contact scoring
        continuous: If True, returns continuous contact scores using inverse distance.
                   If False, returns binary contact map
        return_distance_map: If True, returns distance map normalized to [0,1].
                           If False, returns a binarized contact map.
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

    if return_raw_distance_map:
        return distance_matrix

    if return_distance_map:
        # Normalize distance matrix to [0,1] range
        max_dist = distance_matrix.max()
        contact_map = max_dist - distance_matrix
    else:
        # Convert distances to contact scores
        if continuous:
            # Continuous scoring using inverse distance with vectorized operations
            mask = distance_matrix < max_distance

            # Report the percentage of contacts
            if verbose:
                num_contacts = np.count_nonzero(mask) - distance_matrix.shape[0]  # exclude diagonal
                total_possible_contacts = distance_matrix.shape[0] * (distance_matrix.shape[0] - 1)
                percent_contacts = 100.0 * num_contacts / total_possible_contacts if total_possible_contacts > 0 else 0.0
                print(f"Percentage of contacts (distance < {max_distance} Å): {percent_contacts:.2f}%")
            
            contact_map = np.zeros_like(distance_matrix)
            # Transform distance to contact score
            contact_map[mask] = 1.0 / (1.0 + distance_matrix[mask])
        else:
            # Binary scoring with vectorized operations
            contact_map = (distance_matrix < max_distance).astype(float)
            
    return contact_map

def get_plddt(structure,
              verbose: bool = False) -> pd.DataFrame:
    """Extract pLDDT scores from a protein structure and categorize them by confidence level.
    
    Args:
        structure: A BioPython structure object containing the protein model
        verbose: If True, print the number of residues and the mean pLDDT score
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

    if verbose:
        # Print basic statistics
        print(f"Number of residues: {len(plddt_df)}")
        print(f"Mean pLDDT score: {plddt_df['pLDDT'].mean():.2f}")

    return plddt_df


def get_plddt_all(protein_ids,  
                  verbose: bool = False,
                  cache_only: bool = False,
                  cache = AFDB_CACHE, 
                  force=False):
    """
    Extract pLDDT scores and confidence annotations for multiple proteins.

    Args:
        protein_ids (list or iterable): List of UniProt IDs or protein identifiers to process.
        verbose (bool, optional): If True, print summary statistics. Defaults to False.
        cache_only (bool, optional): If True, only use cached CIF files. Defaults to False.
        cache (str, optional): Path to the cache directory. Defaults to AFDB_CACHE.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing pLDDT scores and confidence categories for all proteins.
            Columns include:
                - uniprot_id: Protein identifier
                - residue: Residue number
                - pLDDT: pLDDT score (0-100)
                - confidence_category: Category name (very_high, high, low, very_low)
                - confidence_label: Human-readable category description
                - confidence_color: Hex color code for visualization

    Example:
        >>> protein_ids = ["P38398", "Q9Y2T1"]
        >>> plddt_df = get_plddt_all(protein_ids)
        >>> print(plddt_df.head())
    """
    import os
    import glob
    from IPython.display import clear_output

    cache_cif = os.path.join(cache, "cif")
    os.makedirs(cache_cif, exist_ok=True)

    cache_plddt = os.path.join(cache, "plddt")
    os.makedirs(cache_plddt, exist_ok=True)


    if cache_only:
        # Infer the CIF paths from the protein IDs
        target_cif_paths = [f"{cache_cif}{os.path.sep}AF-{protein_id}-F1-model_v4.cif" for protein_id in protein_ids]
        # Get the cached CIF paths
        cached_cif_paths = glob.glob(os.path.join(cache_cif, "*.cif"))
        # Get the protein IDs from the cached CIF paths
        target_protein_ids = [cif_path.split(os.path.sep)[-1].split(".")[0].split("-")[1] for cif_path in target_cif_paths if cif_path in cached_cif_paths]
    else:
        target_protein_ids = protein_ids
  
    plddt_df = []
    for protein_id in tqdm(target_protein_ids): 
        save_path = os.path.join(cache_plddt, f"{protein_id}.parquet")
        # Read in cached dataframe if it exists and force is False
        if os.path.exists(save_path) and not force:
            plddt_df_protein = pd.read_parquet(save_path)
            plddt_df.append(plddt_df_protein)
            continue
        
        try:
            plddt_df_protein = []
            for structure in import_mmcif(protein_id=protein_id,  
                                          cache=cache_cif):
                plddt = get_plddt(structure)
                plddt.insert(0, "uniprot_id", protein_id)
                plddt_df_protein.append(plddt)
            # Concatenate protein-specific dataframes
            plddt_df_protein = pd.concat(plddt_df_protein)
            # Save protein-specific dataframe to parquet
            plddt_df_protein.to_parquet(save_path)
            # Append protein-specific dataframe to list
            plddt_df.append(plddt_df_protein)
        except Exception as e:
            print(f"{protein_id}: {e}")
            continue
        clear_output()

    # Concatenate all protein dataframes
    plddt_df = pd.concat(plddt_df)
    if verbose:
        print(f"plddt_df.shape: {plddt_df.shape}") 
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

def expand_matrix(X, target_size=None):
    """
    Expand a binned matrix back to original dimensions by repeating values.
    
    Args:
        X (np.ndarray): Input binned matrix
        target_size (int): Target size for the expanded matrix (default: 1863)
        
    Returns:
        np.ndarray: Expanded matrix with dimensions (target_size, target_size)
    """
    if target_size is None:
        target_size = X.shape[0]
        print(f"No target size specified, using input size {target_size}")

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
    
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os

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

def animate_contact_map_morphing(
    contact_maps, 
    pow=4,
    n_frames_per_transition=15,
    bin_size=1,
    random_order=True,
    gif_filename='results/plots/contact_map_interpolation.gif',
    interval=50,
    fps=10,
    dpi=100,
):
    """
    Create and save an animation morphing between contact maps.

    Parameters:
        contact_maps (dict): Dictionary of contact maps.
        cf: Object with bin_matrix and get_ref_key methods.
        pow (int): Power to raise contact maps before binning.
        n_frames_per_transition (int): Frames per transition.
        bin_size (int): Bin size for bin_matrix.
        random_order (bool): If True, use random order; else, nearest neighbor path.
        gif_filename (str): Output GIF filename.
        interval (int): Interval between frames in milliseconds.
        fps (int): Frames per second for GIF.
        dpi (int): DPI for saved images.    

    Example:
        >>> # Suppose you have a dictionary of contact maps (numpy arrays) keyed by sample names:
        >>> contact_maps = {
        ...     "sample1": np.random.rand(50, 50),
        ...     "sample2": np.random.rand(50, 50),
        ...     "sample3": np.random.rand(50, 50),
        ... }
        >>> # And a cf object with a bin_matrix method:
        >>> class DummyCF:
        ...     def bin_matrix(self, X, bin_size):
        ...         # Simple binning: just return X for demonstration
        ...         return X
        ...     def get_ref_key(self):
        ...         return "sample1"
        >>> cf = DummyCF()
        >>> animate_contact_map_morphing(
        ...     contact_maps,
        ...     cf,
        ...     pow=2,
        ...     n_frames_per_transition=10,
        ...     bin_size=1,
        ...     random_order=True,
        ...     gif_filename='contact_map_demo.gif',
        ...     fps=5
        ... )
        # This will display the animation and save it as 'contact_map_demo.gif'
    """ 
    import matplotlib.animation as animation  
    from scipy.spatial.distance import pdist, squareform 

    # Bin and power contact maps
    contact_maps_binned = {k: bin_matrix(X=v**pow, bin_size=bin_size) for k, v in contact_maps.items()}
    sample_ids = list(contact_maps_binned.keys())
    first_sample = sample_ids[0]
    first_contact_map = contact_maps_binned[first_sample]**pow

    # Prepare arrays for distance calculation
    print("Computing similarity matrix between contact maps...")
    contact_map_arrays = []
    valid_samples = []
    for sample in sample_ids:
        cm = contact_maps_binned[sample]
        if cm is not None:
            contact_map_arrays.append(cm.flatten())
            valid_samples.append(sample)
    contact_map_arrays = np.array(contact_map_arrays)

    # Compute pairwise distances
    distances = pdist(contact_map_arrays, metric='euclidean')
    distance_matrix = squareform(distances)

    # Find the optimal path
    if random_order:
        np.random.seed(42)
        optimal_path = np.random.permutation(len(valid_samples))
    else:
        optimal_path = find_nearest_neighbor_path(distance_matrix)

    # Set up the animation figure
    fig, ax = plt.subplots()
    ax.set_title("Contact Map Morphing Animation", fontsize=16)

    # Normalize all contact maps to same range for consistent visualization
    all_maps = [contact_maps_binned[valid_samples[i]] for i in optimal_path]
    vmin = min(np.min(cm) for cm in all_maps)
    vmax = max(np.max(cm) for cm in all_maps)

    # Create initial image
    img = ax.imshow(first_contact_map, cmap='gnuplot2', vmin=vmin, vmax=vmax)
    plt.colorbar(img, ax=ax, label='Contact Probability')

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

        map1 = contact_maps_binned[valid_samples[map1_idx]]
        map2 = contact_maps_binned[valid_samples[map2_idx]]

        # Linear interpolation between the two maps
        interpolated_map = (1 - transition_progress) * map1 + transition_progress * map2

        # Update the image
        img.set_array(interpolated_map)

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
    print(f"Transitioning through {n_maps} contact maps")

    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames,
        interval=interval, blit=True, repeat=True
    )

    plt.tight_layout()
    plt.show()

    # Save the animation as GIF
    print("Saving animation as GIF...")
    os.makedirs(os.path.dirname(gif_filename), exist_ok=True)
    anim.save(gif_filename, writer='pillow', fps=fps, dpi=dpi)
    print(f"Animation saved as '{gif_filename}'")

    # Display some statistics about the path
    print(f"\nAnimation Statistics:")
    print(f"Number of contact maps: {n_maps}")
    print(f"Total animation frames: {total_frames}")
    print(f"Frames per transition: {n_frames_per_transition}")
    print(f"Animation duration: {total_frames * 0.1:.1f} seconds")
    print(f"GIF saved with {fps} FPS")

    # Show the path taken
    print(f"\nPath taken through contact maps:")
    for i, idx in enumerate(optimal_path):
        sample_name = os.path.basename(valid_samples[idx]).split(".")[0]
        print(f"{i+1:2d}. {sample_name}")
    
    return {
        "anim": anim,
        "path": optimal_path,
        "valid_samples": valid_samples,
        "contact_maps_binned": contact_maps_binned,
        "contact_map_arrays": contact_map_arrays,
        "distances": distances,
        "distance_matrix": distance_matrix,
    }


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


def get_ref_key(contact_maps, pattern="REF"):
    """
    Find and return the key from contact_maps whose filename contains the given pattern
    in the second underscore-separated field.

    Args:
        contact_maps (dict): Dictionary where keys are file paths or names.
        pattern (str): Pattern to search for in the second field of the filename (default: "REF").

    Returns:
        str: The first key matching the pattern.

    Raises:
        IndexError: If no key matches the pattern.
        ValueError: If the filename does not have at least two underscore-separated fields.

    Example:
        >>> contact_maps = {
        ...     '/path/to/sample_REF_001.npy': ...,
        ...     '/path/to/sample_ALT_002.npy': ...,
        ... }
        >>> get_ref_key(contact_maps)
        '/path/to/sample_REF_001.npy'
    """
    return [x for x in contact_maps.keys() if pattern in os.path.basename(x).split("_")[1]][0]


def plot_contact_map_entropy(
    contact_maps,
    bin_size=None,
    cmap="viridis",
    figsize=(8, 6),
    dpi=100,
    agg_func=np.nanmean,
    title="Contact Map Entropy",
):
    """
    Plot the per-pixel entropy across a set of contact maps.

    This function computes the entropy at each (i, j) position across all provided contact maps,
    optionally bins the result, and displays it as a heatmap.

    Args:
        contact_maps (dict): Dictionary mapping keys (e.g., filenames) to 2D numpy arrays representing contact maps.
        bin_size (int, optional): If provided, bin the entropy map into square bins of this size using `agg_func`.
        cmap (str): Colormap to use for the heatmap.
        figsize (tuple): Figure size for matplotlib.
        dpi (int): Dots per inch for the figure.
        agg_func (callable): Aggregation function to use when binning (default: np.nanmax).
        title (str): Title for the plot.

    Returns:
        np.ndarray: The (possibly binned and expanded) entropy map as a numpy array.

    Raises:
        ValueError: If no contact maps match the reference number of residues.

    Example:
        >>> entropy_map = plot_contact_map_entropy(contact_maps, bin_size=10)
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    # Determine the reference contact map (assume first in dict is ref)
    ref_key = get_ref_key(contact_maps)
    ref_map = contact_maps[ref_key]
    ref_n_res = ref_map.shape[0]

    # Only include contact maps with the same number of residues as the reference
    filtered_contact_maps = {k: v for k, v in contact_maps.items() if v.shape[0] == ref_n_res}

    if len(filtered_contact_maps) == 0:
        raise ValueError("No contact maps match the reference number of residues.")

    # Stack all contact maps into a 3D tensor (n_maps, n_res, n_res)
    contact_map_tensor = torch.tensor(np.stack(list(filtered_contact_maps.values())), dtype=torch.float32)

    # Compute per-pixel entropy across all contact maps
    # For each (i,j), treat the values across maps as a distribution

    # Add a small epsilon to avoid log(0)
    eps = 1e-8
    probs = contact_map_tensor + eps
    probs = probs / probs.sum(dim=0, keepdim=True)

    # Entropy: -sum(p * log(p)) over the first axis (maps)
    # entropy_map = -torch.sum(probs * torch.log(probs), dim=0)
    # NOTE: The following is not true entropy, but is used for visualization
    entropy_map = -torch.sum(probs * probs, dim=0)

    # Convert to numpy for further plotting/analysis
    entropy_map_np = entropy_map.numpy()

    print("Entropy map shape:", entropy_map_np.shape)

    if bin_size is not None:
        entropy_map_np = bin_matrix(entropy_map_np, bin_size=bin_size, agg_func=agg_func)
        entropy_map_np = expand_matrix(entropy_map_np, target_size=ref_map.shape[0])

    plt.figure(figsize=figsize, dpi=dpi)
    im = plt.imshow(entropy_map_np, cmap=cmap, interpolation='nearest')
    plt.title(title)
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    plt.colorbar(im, label="Entropy")
    plt.tight_layout()
    plt.show()

    return entropy_map_np


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


def plot_contact_map_diff(
    contact_maps,
    bin_size=20,
    cmap="seismic_r",
    figsize=(8, 6),
    dpi=100,
    agg_func=nonzero_mean,
    title="Gain/Loss of Contact Relative to REF",
):
    """
    Plot the difference in contact probability between reference and non-reference contact maps.

    This function compares each non-reference contact map to the reference, binarizes contacts,
    and computes, for each (i, j), the fraction of non-reference maps that differ from the reference.
    The result is visualized as a heatmap, with negative values indicating loss of contact,
    positive values indicating gain, and zero indicating no change.

    Args:
        contact_maps (dict): Dictionary mapping keys (e.g., filenames) to 2D numpy arrays representing contact maps.
        bin_size (int): Bin size for aggregating the difference map (default: 20).
        cmap (str): Colormap to use for the heatmap.
        figsize (tuple): Figure size for matplotlib.
        dpi (int): Dots per inch for the figure.
        agg_func (callable): Aggregation function to use when binning (default: nonzero_mean).
        title (str): Title for the plot.

    Returns:
        np.ndarray: The difference map (not binned/expanded).

    Raises:
        ValueError: If no contact maps match the reference number of residues,
                    or if there are no non-reference contact maps to compare.

    Example:
        >>> diff_map = plot_contact_map_diff(contact_maps, bin_size=10)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Determine the reference contact map (assume first in dict is ref)
    ref_key = get_ref_key(contact_maps)
    ref_map = contact_maps[ref_key]
    ref_n_res = ref_map.shape[0]

    # Only include contact maps with the same number of residues as the reference
    filtered_contact_maps = {k: v for k, v in contact_maps.items() if v.shape[0] == ref_n_res}

    if len(filtered_contact_maps) == 0:
        raise ValueError("No contact maps match the reference number of residues.")

    # Identify non-REF keys (all except the first one)
    non_ref_keys = [k for k in filtered_contact_maps.keys() if k != ref_key]

    if len(non_ref_keys) == 0:
        raise ValueError("No non-REF contact maps to compare.")

    # Binarize the contact maps: treat values >=0.5 as 1, <0.5 as 0
    def binarize_map(contact_map, threshold=0.5):
        return (contact_map >= threshold).astype(int)

    ref_bin = binarize_map(ref_map)

    # Stack all non-REF binarized contact maps into a 3D array (n_nonref, n_res, n_res)
    nonref_bin_maps = np.stack([binarize_map(filtered_contact_maps[k]) for k in non_ref_keys])

    # Compute, for each (i,j), the proportion of non-REF maps that differ from REF
    # But also distinguish between loss and gain of contact:
    #   - If REF has contact (1) and most non-REF lose it (0), value will be negative
    #   - If REF has no contact (0) and most non-REF gain it (1), value will be positive
    #   - If no change, value is 0

    # For each (i,j), count how many non-REFs have a contact (1)
    nonref_contact_sum = nonref_bin_maps.sum(axis=0)  # shape: (n_res, n_res)
    n_nonref = nonref_bin_maps.shape[0]

    # For each (i,j), compute the difference in contact probability relative to REF
    # If REF has contact (1): (fraction of non-REFs with contact) - 1  (so negative if most lose)
    # If REF has no contact (0): (fraction of non-REFs with contact) - 0 (so positive if most gain)
    fraction_nonref_contact = nonref_contact_sum / n_nonref
    diff_vs_ref = fraction_nonref_contact - ref_bin  # shape: (n_res, n_res)
    # Range: -1 (all lost contact), 0 (no change), +1 (all gained contact)

    print("Difference-vs-REF map shape:", diff_vs_ref.shape)
    diff_vs_ref_binned = bin_matrix(diff_vs_ref, bin_size=bin_size, agg_func=agg_func)
    diff_vs_ref_binned = expand_matrix(diff_vs_ref_binned, target_size=ref_map.shape[0])

    plt.figure(figsize=figsize, dpi=dpi)
    im = plt.imshow(
        diff_vs_ref_binned,
        cmap=cmap,
        interpolation='nearest',
        vmin=-1, vmax=1
    )
    plt.title(title)
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    cbar = plt.colorbar(im, label="Contact change vs REF")
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(['Lost in all', '-0.5', 'No change', '+0.5', 'Gained in all'])
    plt.tight_layout()
    plt.show()

    return diff_vs_ref


def plot_contact_map_diff_barplot(
    diff_map, 
    title="Contact Points Gained vs Lost", 
    figsize=(4, 5), 
    cmap="seismic_r",
    percent_precision=2
):
    """
    Plot a barplot of the number of contact points gained and lost,
    showing both the count and the percent in parentheses.

    Args:
        diff_map: 2D numpy array of contact map differences.
        title: Title for the plot.
        figsize: Figure size for the plot.
        cmap: Colormap for the bars.
        percent_precision: Number of decimal places to show for percent values.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Only consider the upper triangle (excluding the diagonal) to avoid double counting
    # np.triu with k=1 already excludes the diagonal
    triu_mask = np.triu(np.ones(diff_map.shape, dtype=bool), k=1)
    total_contacts = np.sum(triu_mask)

    # Contacts gained: elements > 0 in upper triangle (excluding diagonal)
    contacts_gained = np.sum((diff_map > 0) & triu_mask)
    # Contacts lost: elements < 0 in upper triangle (excluding diagonal)
    contacts_lost = np.sum((diff_map < 0) & triu_mask)

    # Calculate percentages (excluding diagonal)
    percent_gained = 100.0 * contacts_gained / total_contacts if total_contacts > 0 else 0.0
    percent_lost = 100.0 * contacts_lost / total_contacts if total_contacts > 0 else 0.0

    # Store results in a DataFrame
    df = pd.DataFrame({
        'Type': ['Gained', 'Lost'],
        'Count': [contacts_gained, contacts_lost],
        'Percent': [percent_gained, percent_lost]
    })

    # Plot as barplot
    import matplotlib as mpl
    cmap_obj = mpl.colormaps.get_cmap(cmap)
    # For "Gained" (positive), use the high end; for "Lost" (negative), use the low end
    gained_color = cmap_obj(1.0)
    lost_color = cmap_obj(0.0)
    colors = [gained_color, lost_color]

    labels = df['Type']
    values = df['Count']

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Number of contact points')
    ax.set_title(title)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percent = df['Percent'].iloc[i]
        percent_fmt = f"{{:.{percent_precision}f}}"
        ax.annotate(f'{int(height)} ({percent_fmt.format(percent)}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

    # Expand the y-axis to avoid cutting off the text
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + max(values)*0.08 + 10)

    plt.show()

    return df


def revert_haplotype_naming(names, sep="_"):
    """
    Revert haplotype naming from "871P_L_1645R_T" -> "871P>L,1645R>T".
    Accepts a single string or an iterable of strings.
    Returns a single string or a list of strings, matching the input type.
    """
    def revert_one(name):
        parts = name.split(sep)
        reverted = []
        i = 0
        while i < len(parts):
            if i+1 < len(parts):
                orig = f"{parts[i]}>{parts[i+1]}"
                reverted.append(orig)
                i += 2
            else:
                reverted.append(parts[i])
                i += 1
        return ",".join(reverted)
    
    if isinstance(names, str):
        return revert_one(names)
    else:
        return [revert_one(name) for name in names]
    

def plot_most_different_contact_maps(
    contact_maps,
    max_subplots=6,
    bin_size=10,
    pow=4,
    diff_mode="global",
    conv_window=100,
    cmap="gnuplot2",
    show_diag=False,
    square_size=4,
    highlight_diff_regions=False,
    highlight_min_size=100,
    highlight_n_regions=2, 
    highlight_palette="Set3",
    highlight_color_fill=False,
    highlight_alpha=1,
    highlight_linewidth=2,
    highlight_box_offset=None,
    device=None,  # New argument to specify torch device, e.g., "cuda:2"
    split_ref_haplotype=False,  # NEW: if True, split each subplot along diagonal (bottom=REF, top=haplotype)
    split_divider_params={"linestyle": "solid", 
                          "color": "white", 
                          "width": 1, 
                          "alpha": 1.0},
    hspace=0.05, 
    wspace=0.001,
    axis_title_fontsize=10,  # NEW: controls x and y axis title text size
    dpi=100,
):
    """
    Plots a grid of the most different contact maps compared to the reference.

    Args:
        contact_maps (dict): Dictionary mapping keys to contact map numpy arrays.
        max_subplots (int): Maximum number of subplots to show (including reference).
        bin_size (int): Bin size for binning the contact map.
        pow (int): Power to raise the contact map values before binning.
        cmap (str): Colormap to use for imshow.
        show_diag (bool): Whether to show the diagonal in the contact maps.
        square_size (int): Size (in inches) of each subplot side.
        highlight_diff_regions (bool): If True, highlight most different regions (non-REF only).
        highlight_min_size (int): Minimum size (in residues) of highlighted region (rectangle side).
        highlight_n_regions (int): Number of most different regions to highlight per map.
        highlight_color (str): Color of the highlight rectangle.
        highlight_palette (str): Palette to use for the highlight rectangle.
        highlight_alpha (float): Alpha of the highlight rectangle.
        highlight_linewidth (float): Line width of the highlight rectangle.
        device (str or torch.device, optional): Device for torch operations, e.g., "cuda:2".
        split_ref_haplotype (bool): If True, each subplot is split along the diagonal: bottom=REF, top=haplotype.
        hspace (float): Vertical space between subplots.
        wspace (float): Horizontal space between subplots.
        axis_title_fontsize (int): Font size for x and y axis titles.
    """

    def get_most_different_contact_maps(
        contact_maps, 
        max_subplots=6,
        diff_mode="global",  # "global" or "local"
        conv_window=100      # int, only used if diff_mode=="local"
    ):
        """
        Selects the most different contact maps compared to the reference.

        Args:
            contact_maps (dict): Dictionary mapping keys to contact map numpy arrays.
            max_subplots (int): Maximum number of subplots to show (including reference).
            diff_mode (str): "global" (sum of all differences) or "local" (max difference in a window).
            conv_window (int or None): Window size for local difference (square side length).
        Returns:
            dict: Most different contact maps (excluding reference).
        """
        import numpy as np
        from scipy.ndimage import uniform_filter

        ref_key = get_ref_key(contact_maps)
        ref_map = contact_maps[ref_key]
        diff_scores = {}

        for k, v in contact_maps.items():
            # Ensure same shape and skip ref itself
            if k == ref_key or v.shape != ref_map.shape:
                continue
            # Use nan_to_num to ignore NaNs in the difference
            v_clean = np.nan_to_num(v)
            ref_clean = np.nan_to_num(ref_map)
            diff_abs = np.abs(v_clean - ref_clean)

            if diff_mode == "global":
                diff = np.nansum(diff_abs)
            elif diff_mode == "local":
                if conv_window is None or conv_window < 1:
                    raise ValueError("conv_window must be set to a positive integer for local diff_mode")
                # Use uniform_filter to compute local sums (moving window)
                # The maximum sum in any window is the local difference score
                # Use mode='constant', cval=0 to ignore out-of-bounds
                local_sum = uniform_filter(diff_abs, size=conv_window, mode='constant', cval=0) * (conv_window**2)
                diff = np.nanmax(local_sum)
            else:
                raise ValueError(f"Unknown diff_mode: {diff_mode}")

            diff_scores[k] = diff

        # Get the most different contact maps
        most_diff_keys = sorted(diff_scores, key=diff_scores.get, reverse=True)[:max_subplots]
        most_diff_contact_maps = {k: contact_maps[k] for k in most_diff_keys}
        return most_diff_contact_maps

    import matplotlib.patches as patches
    import matplotlib as mpl
    import torch    
    import torch.nn.functional as F

    nrows, ncols = 2, max_subplots // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * square_size, nrows * square_size),
                              dpi=dpi)
    axes = axes.flatten()

    if not split_ref_haplotype:
        max_subplots -= 1

    # Get the most different contact maps
    most_diff_contact_maps = get_most_different_contact_maps(contact_maps=contact_maps, 
                                                             max_subplots=max_subplots, 
                                                             diff_mode=diff_mode, 
                                                             conv_window=conv_window)
    

    # Add the reference as the first item, then the most different contact maps
    ref_key = get_ref_key(contact_maps)
    ref_map = contact_maps[ref_key]
    if split_ref_haplotype:
        items_to_plot = list(most_diff_contact_maps.items())
        
    else:
        items_to_plot = [(ref_key, ref_map)] + list(most_diff_contact_maps.items())
    items_to_plot = items_to_plot[:max_subplots] 
    


    highlight_boxes = []  # List of (color, [(x, y, w), ...])

    for i, (name, contact_map) in enumerate(tqdm(items_to_plot)):
        if not show_diag:
            np.fill_diagonal(contact_map, np.nan)
        # Prepare binned maps for both haplotype and REF
        contact_map_binned = bin_matrix(X=contact_map**pow, bin_size=bin_size)
        ref_map_binned = bin_matrix(X=ref_map**pow, bin_size=bin_size)
        binned_shape = contact_map_binned.shape[0]

        # Compose the split image if requested
        if split_ref_haplotype:
            # Create a new array for the split image
            split_img = np.zeros_like(contact_map_binned)
            # Fill upper triangle (excluding diagonal) with haplotype, lower triangle (including diagonal) with REF
            triu_idx = np.triu_indices(binned_shape, k=1)
            tril_idx = np.tril_indices(binned_shape, k=0)
            split_img[triu_idx] = contact_map_binned[triu_idx]
            split_img[tril_idx] = ref_map_binned[tril_idx]
            im = axes[i].imshow(split_img, cmap=cmap, interpolation="nearest", aspect='equal')

            # --- Split divider appearance controls ---
            # These can be set as function arguments or module-level variables as needed:
            # split_divider: None, "solid", "dashed", "dotted" (default: "dashed")
            # split_divider_color: e.g. "black", "#FF0000" (default: "black")
            # split_divider_width: float (default: 1.5)
            # split_divider_alpha: float (default: 1.0)
       
            if split_divider_params is not None:
                split_divider = split_divider_params.get("linestyle", "solid")
                split_divider_color = split_divider_params.get("color", "white")
                split_divider_width = split_divider_params.get("width", 1.5)
                split_divider_alpha = split_divider_params.get("alpha", 1.0)

                # Draw a diagonal line to indicate the split
                # Map the diagonal to image coordinates
                # imshow by default puts (0,0) at top-left, so diagonal is from (0,0) to (N-1,N-1)
                # The axes limits are -0.5 to N-0.5 for imshow
                N = binned_shape
                divider_styles = {
                    "solid": (0, ()),
                    "dashed": (0, (5, 5)),
                    "dotted": (0, (1, 3)),
                }
                linestyle = divider_styles.get(str(split_divider).lower(), (0, (5, 5)))
                axes[i].plot(
                    [-0.5, N-0.5],
                    [-0.5, N-0.5],
                    color=split_divider_color,
                    linewidth=split_divider_width,
                    alpha=split_divider_alpha,
                    linestyle=linestyle
                )
            # Add "REF" as y-axis label for the leftmost subplots
            # axes is a flat array, so leftmost subplots are those where i % ncols == 0
            if i % ncols == 0:
                # Place the "REF" label outside the plot, vertically centered
                axes[i].annotate(
                    "REF",
                    xy=(0, 0.5),
                    xycoords='axes fraction',
                    fontsize=axis_title_fontsize,
                    ha='right',
                    va='center',
                    rotation=90,
                    # fontweight='bold'
                )
        else:
            im = axes[i].imshow(contact_map_binned, cmap=cmap, interpolation="nearest", aspect='equal')

        protein_id = os.path.basename(name).split("_unrelaxed")[0].split("_")[0]
        haplotype_id = os.path.basename(name).split("_unrelaxed")[0].replace(protein_id + "_", "")
        axes[i].set_title(f"{revert_haplotype_naming(haplotype_id)}", fontsize=axis_title_fontsize, pad=2)
        axes[i].axis('off')
        axes[i].set_aspect('equal')

        # Highlight most different regions for non-REF subplots if requested
        if 'highlight_box_offset' not in locals() and 'highlight_box_offset' not in globals():
            highlight_box_offset = highlight_linewidth / 2.0  # Default: half the linewidth

        if (
            highlight_diff_regions
            and name != ref_key
            and contact_map.shape == ref_map.shape
        ):
            if split_ref_haplotype:
                color = "white"
            else:
                color_idx = i - 1  # i=0 is REF, so subtract 1
                better_cmap = mpl.cm.get_cmap(highlight_palette, max(1, len(items_to_plot)-1))
                color = mpl.colors.to_hex(better_cmap(color_idx % better_cmap.N))

            
            diff_map_np = np.abs(np.nan_to_num(contact_map) - np.nan_to_num(ref_map))
            L = diff_map_np.shape[0]
            min_size = highlight_min_size 
            coords = []
            lw = highlight_linewidth
            offset = highlight_box_offset
            if L <= min_size:
                rect = patches.Rectangle(
                    (-offset, -offset),
                    L + 2*offset,
                    L + 2*offset,
                    linewidth=lw,
                    edgecolor=color,
                    facecolor=color,
                    alpha=highlight_alpha,
                    fill=highlight_color_fill,
                )
                axes[i].add_patch(rect)
                coords.append((0, 0, L))
            else:
                if device is None:
                    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    torch_device = torch.device(device) if isinstance(device, str) else device
                diff_map = torch.from_numpy(diff_map_np).float().to(torch_device)
                window = min_size
                kernel = torch.ones((1, 1, window, window), dtype=torch.float32, device=torch_device)
                diff_map_4d = diff_map.unsqueeze(0).unsqueeze(0)  # shape: (1,1,L,L)
                
                sums = F.conv2d(diff_map_4d, kernel, stride=1).squeeze().cpu().numpy()  # shape: (L-window+1, L-window+1)

                sums_flat = sums.ravel()
                idx_sorted = np.argpartition(-sums_flat, highlight_n_regions*2)[:highlight_n_regions*4]
                idx_sorted = idx_sorted[np.argsort(-sums_flat[idx_sorted])]  # sort top indices

                mask = np.zeros_like(sums, dtype=bool)
                found = 0
                for idx in idx_sorted:
                    if found >= highlight_n_regions:
                        break
                    x = idx // sums.shape[1]
                    y = idx % sums.shape[1]
                    x_center = x + window // 2
                    y_center = y + window // 2
                    x0 = x_center - window // 2
                    y0 = y_center - window // 2
                    x0 = max(0, min(x0, L - window))
                    y0 = max(0, min(y0, L - window))
                    mask_x0 = x0
                    mask_x1 = x0 + window
                    mask_y0 = y0
                    mask_y1 = y0 + window
                    if not mask[mask_x0:mask_x1, mask_y0:mask_y1].any():
                        coords.append((x0, y0, window))
                        mask[mask_x0:mask_x1, mask_y0:mask_y1] = True
                        found += 1

                bin_scale = bin_size
                for (x, y, w) in coords:
                    x_binned = x / bin_scale
                    y_binned = y / bin_scale
                    w_binned = w / bin_scale
                    rect = patches.Rectangle(
                        (y_binned - offset, x_binned - offset),  # imshow: (col, row), offset outwards
                        w_binned + 2*offset,
                        w_binned + 2*offset,
                        linewidth=lw,
                        edgecolor=color,
                        facecolor=color,
                        alpha=highlight_alpha,
                        fill=highlight_color_fill,
                    )
                    axes[i].add_patch(rect)
                highlight_boxes.append((color, [(x / bin_scale, y / bin_scale, w / bin_scale) for (x, y, w) in coords]))

    # Optionally include all highlight boxes in the REF plot afterwards, color-coded
    if highlight_diff_regions and len(highlight_boxes) > 0 and not split_ref_haplotype:
        # The REF plot is always axes[0]
        lw = highlight_linewidth
        offset = highlight_box_offset
        for color, coords in highlight_boxes:
            for (x, y, w) in coords:
                rect = patches.Rectangle(
                    (y - offset, x - offset),  # imshow: (col, row), offset outwards
                    w + 2*offset,
                    w + 2*offset,
                    linewidth=lw,
                    edgecolor=color,
                    facecolor=color,
                    alpha=highlight_alpha,
                    fill=highlight_color_fill,
                )
                axes[0].add_patch(rect)

    # --- Add a grey border around the entire first subplot (REF) ---
    # Only if not using split_ref_haplotype, since there is no dedicated REF plot in split mode
    if not split_ref_haplotype:
        ref_binned_shape = bin_matrix(X=ref_map**pow, bin_size=bin_size).shape[0]
        border_color = "#888888"
        border_linewidth = 14  # You can adjust this for visibility
        border_shift_up = -8  # You can adjust this value as needed
        border_shift_left = -2  # You can adjust this value as needed
        border_rect = patches.Rectangle(
            (border_shift_left, border_shift_up),
            ref_binned_shape,
            ref_binned_shape,
            linewidth=border_linewidth,
            edgecolor=border_color,
            facecolor='none',
            zorder=-10,
            clip_on=False  # Allow the border to extend outside the axes
        )
        axes[0].add_patch(border_rect)
        axes[0].title.set_color('white')

    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    plt.show()

def get_ref_size(contact_maps):
    ref_key = get_ref_key(contact_maps)
    ref_map = contact_maps[ref_key]
    return ref_map.shape[0]
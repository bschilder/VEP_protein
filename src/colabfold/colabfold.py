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
# module load EBModules CUDA/12.6.0 GCC 
#
# export PATH="/home/schilder/projects/localcolabfold/colabfold-conda/bin:$PATH"
# export CUDA_VISIBLE_DEVICES=1,3
#
# python: 
# >> import pandas as pd; import os
# >> if 'NOTEBOOK_INITIALIZED' not in globals():
# >>     os.chdir(os.path.dirname(os.path.abspath('.')))
# >>     NOTEBOOK_INITIALIZED = True
# >> import src.haplosaurus as hs;
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

from re import M
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
import math
from scipy import stats
import sys

import src.utils as utils
import src.analysis.matrices as mc

AFDB_CACHE = pooch.os_cache("alphafold_db")

def search_files(
    base_dir,
    tx_id,
    subdir="af2_sameMSA",
    suffix=".pdb*",
    model_number=1,
    ref_only=False
):
    """
    Search for PDB files in a specified directory structure.

    This function searches recursively for PDB files matching a specific pattern
    within a given base directory and transcript ID. Optionally, it can filter
    to return only the reference (REF) PDB file.

    Args:
        base_dir (str): The base directory to search within.
        tx_id (str): The transcript ID subdirectory to search under.
        subdir (str, optional): Subdirectory name containing the PDB files. Default is "af2_sameMSA".
        suffix (str, optional): File suffix or pattern to match. Default is ".pdb*".
        model_number (int, optional): Model number to include in the filename pattern. Default is 1.
        ref_only (bool, optional): If True, return only the reference (REF) file. Default is False.

    Returns:
        list: List of matching PDB file paths. If ref_only is True, returns a list with a single REF file,
              or an empty list if not found.

    Raises:
        ValueError: If ref_only is True and no REF file is found for the given transcript ID.

    Example:
        >>> files = search_files("/data/colabfold", "ENST00000357654")
        >>> print(files)
        ['/data/colabfold/ENST00000357654/af2_sameMSA/sample_unrelaxed_rank_001_model_1.pdb', ...]
        >>> ref_file = search_files("/data/colabfold", "ENST00000357654", ref_only=True)
        >>> print(ref_file)
        ['/data/colabfold/ENST00000357654/af2_sameMSA/sample_REF_unrelaxed_model_1.pdb']
    """
    pdb_files = glob.glob(
        os.path.join(
            base_dir,
            tx_id,
            "**",
            f"{subdir}/*_unrelaxed_*_model_{model_number}_*{suffix}"
        ),
        recursive=True
    )
    if ref_only:
        from .colabfold import get_ref_key  # Ensure get_ref_key is imported if not already
        ref_file = get_ref_key(pdb_files, pattern="REF", error=False)
        pdb_files = [ref_file] if ref_file is not None else []
        if len(pdb_files) == 0:
            raise ValueError(f"No REF file found for {tx_id}")
    print(len(pdb_files), "PDB files found")
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

def import_contact_maps(
    pdb_files,
    return_distance_map=True,
    return_raw_distance_map=False,
    continuous=True,
    as_dict=True,
    verbose=True,
    **kwargs
):
    """
    Import contact maps from a list of PDB files.

    This function processes each PDB file in the provided list, extracts the structure,
    and computes the contact map using the specified parameters. Empty files are skipped.
    Handles both plain and gzipped (.gz) PDB files.

    Args:
        pdb_files (list of str): List of paths to PDB files.
        return_distance_map (bool, optional): If True, return the normalized distance map instead of a contact map. Default is True.
        return_raw_distance_map (bool, optional): If True, return the raw distance map instead of a contact map. Default is False.
        continuous (bool, optional): If True, use continuous scoring for contact map; otherwise, use binary scoring. Default is True.
        as_dict (bool, optional): If True, return a dictionary mapping each PDB file path to its corresponding contact map (numpy.ndarray). Default is True.
        verbose (bool, optional): If True, print progress and information. Default is True.
        **kwargs: Additional keyword arguments passed to get_contact_map.

    Returns:
        dict: A dictionary mapping each PDB file path to its corresponding contact map (numpy.ndarray). 
        If as_dict is False and there is only one PDB file, returns a single contact map.

    Example:
        >>> pdb_files = ["protein1.pdb", "protein2.pdb"]
        >>> contact_maps = import_contact_maps(pdb_files, continuous=False)
        >>> print(contact_maps["protein1.pdb"].shape)
    """
    import gzip
    import io

    contact_maps = {}
    for pdb_file in tqdm(pdb_files):
        # Check for empty file (works for both .gz and plain)
        try:
            if pdb_file.endswith('.gz'):
                with gzip.open(pdb_file, 'rt') as f:
                    first_char = f.read(1)
                    if not first_char:
                        print(f"Skipping {pdb_file} because it is empty")
                        continue
            else:
                if os.path.getsize(pdb_file) == 0:
                    print(f"Skipping {pdb_file} because it is empty")
                    continue
        except Exception as e:
            print(f"Error reading {pdb_file}: {e}")
            continue

        # For gzipped files, decompress to string and pass as file-like object
        if pdb_file.endswith('.gz'):
            with gzip.open(pdb_file, 'rt') as f:
                pdb_string = f.read()
            pdb_io = io.StringIO(pdb_string)
            # import_pdb expects a path or URL, not a file-like object, so we need to
            # call the parser directly here to avoid AttributeError
            from Bio.PDB import PDBParser
            parser = PDBParser()
            structure = parser.get_structure("structure", pdb_io)
        else:
            structure = import_pdb(pdb_file)
        contact_maps[pdb_file] = get_contact_map(
            structure,
            return_distance_map=return_distance_map,
            return_raw_distance_map=return_raw_distance_map,
            continuous=continuous,
            verbose=verbose,
            **kwargs
        )
    if not as_dict:
        if len(contact_maps) == 1:
            return list(contact_maps.values())[0]
        else:
            raise ValueError("Cannot return a single contact map if as_dict is False and there are multiple PDB files")
    return contact_maps

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


def plot_contact_map(contact_map,  
                     bin_size=2, 
                     figsize=None,
                     reverse_sign=False,
                     normalize_rows=False,
                     normalize_scale=False,
                     dist_axis = None,
                     dist_metric = "cosine",
                     log_before_bin=False, 
                     log_after_bin=False,
                     log_func=lambda x: np.log10(x+1e-6),
                     max_labels=5, 
                     pow=4,
                     cmap="gnuplot2", 
                     agg_func=np.nanmax,
                     title=None,
                     cbar_label=r"3D Distance (Å)",
                     x_label="Residue Position",
                     y_label="Residue Position"):
    
    import scipy.spatial.distance
    import matplotlib as mpl
 
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
    contact_map_binned = mc.bin_matrix(X=contact_map, 
                                    bin_size=bin_size, 
                                    agg_func=agg_func)
    n_bins = contact_map_binned.shape[0]
    
    ### Log after binning
    if log_after_bin:
        contact_map_binned = log_func(contact_map_binned)

    # Do NOT apply pow to the data, but to the color scale
    # Create the plot 
    fig, ax = plt.subplots(figsize=figsize)

    # Set up normalization for color scale
    if pow is not None and pow != 1:
        norm = mpl.colors.PowerNorm(gamma=pow, vmin=np.nanmin(contact_map_binned), vmax=np.nanmax(contact_map_binned))
    else:
        norm = None

    im = plt.imshow(contact_map_binned, 
                    cmap=cmap, 
                    interpolation="nearest",
                    norm=norm)
    # plt.colorbar(im).ax.set_ylabel(cbar_label, 
    #                                rotation=270, 
    #                                va="bottom") 

    mc.label_bins(bin_size, n_bins, max_labels=max_labels)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

    return {"fig": fig, 'axes': ax, 'data':{'contact_map': contact_map, 'contact_map_binned': contact_map_binned}} 


def get_haplotype_ids(names, revert_naming=True, as_dict=False):
    """
    Extracts the haplotype ID from a given filename.

    The function assumes the filename is of the form:
    "<protein_id>_<haplotype_id>_unrelaxed..." or "<protein_id>_<haplotype_id>..."

    Parameters
    ----------
    name : str
        The filename or path from which to extract the haplotype ID.
    revert_naming : bool, default=True
        If True, revert the haplotype naming to the original format.

    Returns
    -------
    str
        The haplotype ID extracted from the filename.
    """
    names = utils.as_list(names)

    def get_haplotype_id(name, revert_naming=True):
        protein_id = os.path.basename(name).split("_unrelaxed")[0].split("_")[0]
        haplotype_id = os.path.basename(name).split("_unrelaxed")[0].replace(protein_id + "_", "")
        if revert_naming:
            haplotype_id = revert_haplotype_naming(haplotype_id)
        return haplotype_id
    
    haplotype_ids = [get_haplotype_id(name, revert_naming=revert_naming) for name in names]
    if as_dict is True:
        return {name: haplotype_id for name, haplotype_id in zip(names, haplotype_ids)}
    elif as_dict == -1:
        return {haplotype_id: name for name, haplotype_id in zip(names, haplotype_ids)}
    else:
        return haplotype_ids


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
        contact_map = mc.bin_matrix(contact_map, bin_size=bin_size, agg_func=np.nanmax)
    
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


def get_ref_key(contact_maps, pattern="REF", error=True):
    """
    Find and return the key from contact_maps whose filename contains the given pattern
    anywhere in its basename path.

    Args:
        contact_maps (dict or list): Dictionary where keys are file paths/names, or list of such strings.
        pattern (str): Pattern to search for (default: "REF").
        error (bool): Whether to raise ValueError or return None on error.

    Returns:
        str: The first key whose basename contains the pattern.

    Raises:
        ValueError: If no key matches the pattern and error=True.
        ValueError: If more than one file matches and error=True.

    Example:
        >>> contact_maps = {
        ...     '/path/to/ENSP00000350283_REF_unrelaxed_rank_002_alphafold2_ptm_model_1_seed_000.pdb': ...,
        ...     '/path/to/ENSP00000350283_ALT_unrelaxed_rank_002_alphafold2_ptm_model_1_seed_000.pdb': ...,
        ... }
        >>> get_ref_key(contact_maps)
        '/path/to/ENSP00000350283_REF_unrelaxed_rank_002_alphafold2_ptm_model_1_seed_000.pdb'
    """
    import os
    if isinstance(contact_maps, dict):
        names = list(contact_maps.keys())
    elif isinstance(contact_maps, list):
        names = contact_maps
    else:
        raise ValueError(f"contact_maps must be a dict or list, not {type(contact_maps)}")

    matches = []
    for x in names:
        base = os.path.basename(x)
        if pattern in base:
            matches.append(x)
    if not matches:
        if error:
            raise ValueError(
                f"No key found in contact_maps with pattern '{pattern}' in basename."
            )
        else:
            return None
    if len(matches) > 1:
        import warnings
        warnings.warn(
            f"Multiple keys found in contact_maps with pattern '{pattern}' in basename: {matches}. Using the last match."
        )
        return matches[-1]
    return matches[-1]


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
        entropy_map_np = mc.bin_matrix(entropy_map_np, bin_size=bin_size, agg_func=agg_func)
        entropy_map_np = mc.expand_matrix(entropy_map_np, target_size=ref_map.shape[0])

    plt.figure(figsize=figsize, dpi=dpi)
    im = plt.imshow(entropy_map_np, cmap=cmap, interpolation='nearest')
    plt.title(title)
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    plt.colorbar(im, label="Entropy")
    plt.tight_layout()
    plt.show()

    return entropy_map_np

def plot_contact_map_diff(
    contact_maps,
    ref_key=None,
    bin_size=20,
    cmap="seismic_r",
    figsize=None,
    dpi=100,
    agg_func=mc.nonzero_mean,
    title="Gained/Lost Contacts Relative to REF",
    xlabel="Residue Position",
    ylabel="Residue Position",
    legend_title=r"Proportion of Haplotypes",
    show_plot=True,
    verbose=True,
    weights=None,
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
        show_plot (bool): Whether to show the plot (default: True).
        verbose (bool): Whether to print verbose output (default: True).
        ref_key (str): Key of the reference contact map (default: None).
        weights (dict or list or np.ndarray, optional): Weights for each non-ref contact map. If dict, keys must match non-ref keys.
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
    if ref_key is None:
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

    # Handle weights
    n_nonref = nonref_bin_maps.shape[0]
    if weights is not None:
        # Accept dict, list, or np.ndarray
        if isinstance(weights, dict):
            # Map non_ref_keys to weights
            weights_arr = np.array([weights[k] for k in non_ref_keys], dtype=float)
        else:
            weights_arr = np.array(weights, dtype=float)
            if weights_arr.shape[0] != n_nonref:
                raise ValueError(f"weights must have length {n_nonref} (number of non-ref contact maps)")
        # Normalize weights to sum to 1 (for proportions)
        weights_arr = weights_arr / np.sum(weights_arr)
        # Compute weighted sum for each (i,j)
        # shape: (n_nonref, n_res, n_res) * (n_nonref, 1, 1) -> (n_res, n_res)
        weighted_sum = np.tensordot(weights_arr, nonref_bin_maps, axes=([0], [0]))
        fraction_nonref_contact = weighted_sum  # already weighted average
    else:
        # For each (i,j), count how many non-REFs have a contact (1)
        nonref_contact_sum = nonref_bin_maps.sum(axis=0)  # shape: (n_res, n_res)
        fraction_nonref_contact = nonref_contact_sum / n_nonref

    # For each (i,j), compute the difference in contact probability relative to REF
    # If REF has contact (1): (fraction of non-REFs with contact) - 1  (so negative if most lose)
    # If REF has no contact (0): (fraction of non-REFs with contact) - 0 (so positive if most gain)
    diff_vs_ref = fraction_nonref_contact - ref_bin  # shape: (n_res, n_res)
    # Range: -1 (all lost contact), 0 (no change), +1 (all gained contact)

    if verbose:
        print("Difference-vs-REF map shape:", diff_vs_ref.shape)

    diff_vs_ref_binned = mc.bin_matrix(diff_vs_ref, bin_size=bin_size, agg_func=agg_func)
    diff_vs_ref_binned = mc.expand_matrix(diff_vs_ref_binned, target_size=ref_map.shape[0],
                                       verbose=verbose)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(
        diff_vs_ref_binned,
        cmap=cmap,
        interpolation='nearest',
        vmin=-1, vmax=1
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Use make_axes_locatable to create a colorbar axis that matches the heatmap height exactly
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, label=legend_title)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(['Lost in all', '-0.5', 'No change', '+0.5', 'Gained in all'])
    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {'fig':fig, 'axes':ax, 'data':diff_vs_ref}

def plot_contact_map_diff_barplot(
    diff_map, 
    title="Contact Points Gained vs. Lost", 
    ylabel="Contact Counts",
    figsize=(4, 5), 
    cmap="seismic_r",
    percent_precision=2,
    ax=None,
    annotation_fontsize=None
):
    """
    Plot a barplot of the number of contact points gained and lost,
    showing both the count and the percent in parentheses.

    Args:
        diff_map: 2D numpy array of contact map differences.
        title: Title for the plot.
        ylabel: Y-axis label.
        figsize: Figure size for the plot.
        cmap: Colormap for the bars.
        percent_precision: Number of decimal places to show for percent values.
        ax: Optional matplotlib Axes to plot on. If None, a new figure and axes are created.
        annotation_fontsize: Optional font size for bar annotations. If None, uses matplotlib rcParams default.

    Example:
        >>> diff_map = plot_contact_map_diff(contact_maps, bin_size=10)
        >>> plot_contact_map_diff_barplot(diff_map)
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
    # For "Gained" (positive), use the high end (0.85); for "Lost" (negative), use the low end (0.30)
    colors = [cmap_obj(0.85),  cmap_obj(0.30)]

    labels = df['Type']
    values = df['Count']

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percent = df['Percent'].iloc[i]
        percent_fmt = f"{{:.{percent_precision}f}}"
        annotate_kwargs = {
            'xy': (bar.get_x() + bar.get_width() / 2, height),
            'xytext': (0, 3),  # 3 points vertical offset
            'textcoords': "offset points",
            'ha': 'center', 
            'va': 'bottom'
        }
        if annotation_fontsize is not None:
            annotate_kwargs['fontsize'] = annotation_fontsize
        ax.annotate(f'{int(height)}\n({percent_fmt.format(percent)}%)',
                    **annotate_kwargs)

    # Expand the y-axis to avoid cutting off the text
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + max(values)*0.08 + 10)

    if created_fig:
        plt.show()

    return fig, df

def plot_contact_map_diff_barplot_grouped(
    bar_df,
    group_by="superpopulation",
    figsize=(10, 5),
    translate_superpop_names=True,
    ylabel="Contact Counts",
    xlabel=None,
    title=None,
    text_label_info =["count", "percent"],
    legend_title="Contact Change\nRelative to Ref",
    cmap="seismic_r", 
):
    """
    Plot a grouped barplot showing the number and percentage of contact map differences
    (e.g., gained/lost contacts) for each group (e.g., superpopulation).

    Args:
        bar_df (pd.DataFrame): DataFrame containing columns for group, 'Type' (e.g., 'Gained', 'Lost'),
            'Count' (number of contacts), and 'Percent' (percentage of contacts).
        group_by (str): Column name in bar_df to group bars by (default: "superpopulation").
        figsize (tuple): Size of the matplotlib figure (default: (10, 5)).
        cmap (str): Name of the matplotlib colormap to use for bar colors (default: "seismic_r").
        legend_title (str): Title for the legend (default: "Contact Change\nRelative to Ref").
        ylabel (str): Y-axis label (default: "Contact Counts").

    Returns:
        None. Displays the plot.

    Notes:
        - The function sorts the bars by 'Count' in descending order.
        - Bar colors are chosen for colorblind accessibility.
        - Each bar is annotated with its count and percentage.
    """

    bar_df = bar_df.copy()
    
    # Translate superpopulation names to more readable names in the subplot titles
    if translate_superpop_names:
        import src.onekg as og 
        bar_df[group_by] = bar_df[group_by].apply(lambda x: og.SUPERPOP_NAMES_DICT[x].replace(" ","\n")+"\n"*(3-og.SUPERPOP_NAMES_DICT[x].count(" "))+"("+x+")")

    # Validate data
    required_columns = [group_by, 'Type', 'Count', 'Percent']
    missing_columns = [col for col in required_columns if col not in bar_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Ensure Count and Percent are numeric
    bar_df['Count'] = pd.to_numeric(bar_df['Count'], errors='coerce')
    bar_df['Percent'] = pd.to_numeric(bar_df['Percent'], errors='coerce')
    
    # Check for any NaN values
    if bar_df[['Count', 'Percent']].isna().any().any():
        print("Warning: NaN values found in Count or Percent columns")
    
    # Sort by the sum of 'Count' for each group, descending
    group_sums = bar_df.groupby(group_by)["Count"].sum().sort_values(ascending=False)
    # Reorder the dataframe so that group_by is a categorical with the desired order
    bar_df[group_by] = pd.Categorical(bar_df[group_by], categories=group_sums.index, ordered=True)
    bar_df = bar_df.sort_values(by=group_by)


    # Use the first and last colors from seismic_r as the palette
    import matplotlib as mpl
    cmap_obj = mpl.colormaps.get_cmap(cmap)
    # For colorblind accessibility, use slightly different shades of blue and red
    # Use 0.85 and 0.30 for more distinguishable, less saturated colors
    palette = [cmap_obj(0.85), cmap_obj(0.30)]

    # Make the figure wider
    fig, ax = plt.subplots(figsize=figsize) 
    sns.barplot(
        data=bar_df,
        x=group_by,
        y="Count",
        hue="Type",
        ax=ax,
        palette=palette,
        dodge=True, 
    )

    # Add black borders around all bars
    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(1)

    # Remove top and right plot borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Get the legend handles and labels to understand the order
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    
    # Create a mapping from the data to ensure correct annotation placement
    # Get the unique groups and their order as they appear on the x-axis
    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    
    # Annotate each bar with its count and percentage
    # The patches are ordered by hue first, then by group
    # So for 2 hues and N groups, the order is: hue1_group1, hue1_group2, ..., hue1_groupN, hue2_group1, hue2_group2, ..., hue2_groupN
    
    n_groups = len(x_tick_labels)
    n_hues = len(legend_labels)
    
    for i, patch in enumerate(ax.patches):
        height = patch.get_height()
        if height == 0:
            continue
            
        # Calculate which group and hue this patch corresponds to
        # For seaborn barplot with dodge=True, patches are ordered by hue first, then by group
        hue_idx = i // n_groups
        group_idx = i % n_groups
        
        if hue_idx >= n_hues or group_idx >= n_groups:
            continue
            
        group_name = x_tick_labels[group_idx]
        hue_name = legend_labels[hue_idx]
        
        # Find the corresponding data row
        row = bar_df[(bar_df[group_by] == group_name) & (bar_df["Type"] == hue_name)]
        
        if not row.empty:
            # Dynamically build the label based on text_label_info
            label_parts = []
            if "count" in text_label_info:
                count_val = row["Count"].values[0]
                label_parts.append(f"{count_val}")
            if "percent" in text_label_info:
                percent_val = row["Percent"].values[0]
                label_parts.append(f"{percent_val:.2f}%")
            label = "\n".join(label_parts) if label_parts else ""
            
            # Position the annotation at the top of the bar
            ax.annotate(label,
                        (patch.get_x() + patch.get_width() / 2, height),
                        ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points',
                        fontsize='small')
    
    if xlabel is None:
        xlabel = group_by.title()
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.legend(title=legend_title, frameon=False)


    
    plt.tight_layout()
    bar_df["superpopulation_clean"] = bar_df["superpopulation"].str.replace("\n"," ")
    
    return {'fig':fig, 'axes':ax, 'data':bar_df}

def revert_haplotype_naming(names, sep="_"):
    """
    Revert haplotype naming from the compact form to a more readable mutation list.

    For example, converts "871P_L_1645R_T" to "871P>L,1645R>T".

    Args:
        names (str or iterable of str): Haplotype name(s) in the compact form.
        sep (str): Separator used in the compact form (default: "_").

    Returns:
        str or list of str: Reverted haplotype name(s) in the format "orig>mut,orig2>mut2,...".
            The return type matches the input type.

    Example:
        >>> revert_haplotype_naming("871P_L_1645R_T")
        '871P>L,1645R>T'
        >>> revert_haplotype_naming(["871P_L_1645R_T", "100A_G"])
        ['871P>L,1645R>T', '100A>G']
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
        return revert_one(os.path.basename(names))
    else:
        return [revert_one(name) for name in names]
    


def compute_highlight_scores(
    mat1,
    mat2,
    highlight_min_size,
    highlight_n_regions,
    torch_device,
    zero_thresh=0.001,
    use_zero_fraction=False,
    diff_mode="diff",  # "diff" (default, subtraction/abs diff), or "cosine"
    stride=1
):
    """
    Compute highlight region scores for two matrices to identify the most different regions.

    This function slides a square window of size `highlight_min_size` over both input matrices,
    and computes, for each window position, a score that reflects how different the two matrices
    are in that region. The scoring can either emphasize regions where one matrix is mostly zero
    and the other is not (if use_zero_fraction=True), or simply use the absolute difference
    between the two matrices in each window (if use_zero_fraction=False and diff_mode="diff"),
    or use cosine distance (if diff_mode="cosine").

    The function uses PyTorch for efficient convolution operations.

    Args:
        mat1 (np.ndarray): First input matrix (e.g., haplotype contact map), shape (L, L).
        mat2 (np.ndarray): Second input matrix (e.g., reference contact map), shape (L, L).
        highlight_min_size (int): Side length of the square window to scan for differences.
        highlight_n_regions (int): Number of top regions to return (more indices are returned to avoid overlap).
        torch_device (torch.device): PyTorch device to use for computation (e.g., "cpu" or "cuda").
        use_zero_fraction (bool): If True, score is high when one matrix is mostly zero and the other is not.
                                  If False, score is the sum of absolute differences in the window (if diff_mode="diff").
        diff_mode (str): "diff" (default, subtraction/abs diff) or "cosine" (cosine distance in window).
        stride (int): Stride for the sliding window (default: 1).

    Returns:
        score (np.ndarray): 2D array of scores for each window position, shape depends on stride.
        idx_sorted (np.ndarray): 1D array of flat indices of the top-scoring regions, sorted in descending order.

    Notes:
        - The function returns more indices than `highlight_n_regions` to allow for post-filtering of overlapping regions.
        - The threshold for "close to zero" is set to 0.05, but can be adjusted in the code.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    # Convert to torch tensors
    mat1_t = torch.from_numpy(mat1).float().to(torch_device)
    mat2_t = torch.from_numpy(mat2).float().to(torch_device)

    # Convolution kernel for windowed sum
    kernel = torch.ones((1, 1, highlight_min_size, highlight_min_size), dtype=torch.float32, device=torch_device)
    mat1_4d = mat1_t.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
    mat2_4d = mat2_t.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)

    if diff_mode == "cosine":
        # Compute cosine distance in each window in a memory-efficient way (avoid storing all windows at once)
        L = mat1.shape[0]
        out_size = (L - highlight_min_size) // stride + 1
        score = np.empty((out_size, out_size), dtype=np.float32)

        # Precompute denominator for normalization for each window in mat1 and mat2
        # We'll use torch.nn.functional.unfold, but process in small batches to save memory
        unfold = torch.nn.Unfold(kernel_size=(highlight_min_size, highlight_min_size), stride=stride)
        # Get number of windows
        num_windows = out_size * out_size
        batch_size = 1024  # Tune this for your memory constraints

        # Unfold mat1 and mat2, but process in batches
        # We'll process by rows to minimize memory
        mat1_windows = unfold(mat1_4d).squeeze(0).T  # (num_windows, window*window)
        mat2_windows = unfold(mat2_4d).squeeze(0).T  # (num_windows, window*window)

        # Compute cosine distance for each window pair in batches
        for start in range(0, num_windows, batch_size):
            end = min(start + batch_size, num_windows)
            m1 = mat1_windows[start:end]  # (batch, window*window)
            m2 = mat2_windows[start:end]  # (batch, window*window)
            # Normalize
            m1_norm = m1 / (m1.norm(dim=1, keepdim=True) + 1e-8)
            m2_norm = m2 / (m2.norm(dim=1, keepdim=True) + 1e-8)
            # Cosine similarity
            cos_sim = (m1_norm * m2_norm).sum(dim=1)
            cos_dist = 1.0 - cos_sim
            # Place in score array
            score_flat_idx = np.arange(start, end)
            score.ravel()[score_flat_idx] = cos_dist.cpu().numpy()
    elif use_zero_fraction:
        # Compute the fraction of values close to zero in each window for both matrices

        mat1_zero = (mat1_t < zero_thresh).float().unsqueeze(0).unsqueeze(0)
        mat2_zero = (mat2_t < zero_thresh).float().unsqueeze(0).unsqueeze(0)
        frac1_zero = F.conv2d(mat1_zero, kernel, stride=stride).squeeze().cpu().numpy() / (highlight_min_size**2)
        frac2_zero = F.conv2d(mat2_zero, kernel, stride=stride).squeeze().cpu().numpy() / (highlight_min_size**2)

        # Score: highlight where one is mostly zero and the other is not
        # For each window, score = max(frac1_zero * (1-frac2_zero), frac2_zero * (1-frac1_zero))
        # This is high when one is mostly zero and the other is not
        score = np.maximum(
            frac1_zero * (1 - frac2_zero),
            frac2_zero * (1 - frac1_zero)
        )
    elif diff_mode == "sum":
        # Score: absolute difference of sums in the window (not sum of differences)
        mat1_sum = F.conv2d(mat1_t.unsqueeze(0).unsqueeze(0), kernel, stride=stride).squeeze()
        mat2_sum = F.conv2d(mat2_t.unsqueeze(0).unsqueeze(0), kernel, stride=stride).squeeze()
        score = torch.abs(mat1_sum - mat2_sum).cpu().numpy()
    else:
        # Score: sum of absolute differences in the window
        abs_diff = torch.abs(mat1_t - mat2_t).unsqueeze(0).unsqueeze(0)
        score = F.conv2d(abs_diff, kernel, stride=stride).squeeze().cpu().numpy()

    score_flat = score.ravel()
    # Get more indices than needed to avoid overlap, then filter below
    idx_sorted = np.argpartition(-score_flat, highlight_n_regions*8)[:highlight_n_regions*8]
    idx_sorted = idx_sorted[np.argsort(-score_flat[idx_sorted])]  # sort top indices

    return score, idx_sorted


def plot_contact_map_subplots(
    contact_maps=None, 
    axes=None,
    split_ref_haplotype=False,
    show_diag=False, 
    pow=4,
    bin_size=10,
    cmap="gnuplot2",
    split_divider_params={"linestyle": "solid", "color": "white", "width": 1, "alpha": 1.0},
    ncols=3,
    axis_title_fontsize=10,
    highlight_diff_regions=False,
    highlight_linewidth=2,
    highlight_palette="Set3",
    highlight_min_size=100,
    highlight_n_regions=2,
    highlight_alpha=1,
    highlight_color_fill=False,
    highlight_color="white",
    highlight_postprocessed=False,
    device=None,
    hspace=0.05,
    wspace=0.001,
    agg_func=np.nanmax,
    ref_key=None,
    normalize=True,
    show_zoom_in=True,  # Option to show zoom-in of highlighted regions
    zoom_in_params=None,  # Dictionary of zoom-in options (see below)
):
    """
    Plot contact maps (reference and most different) as subplots.

    This function visualizes a set of contact maps (including a reference and the most different maps)
    as a grid of subplots. It supports optional features such as splitting each subplot along the diagonal
    (showing reference in the lower triangle and haplotype in the upper), highlighting the most different
    regions, and adding a border to the reference plot.

    Args:
        axes: Array of matplotlib axes to plot into.
        contact_maps (dict): Dictionary mapping keys to contact map numpy arrays.
        most_diff_contact_maps (dict): Dictionary of the most different contact maps to plot.
        max_subplots (int): Maximum number of subplots to display.
        split_ref_haplotype (bool): If True, each subplot is split along the diagonal (bottom=REF, top=haplotype).
        show_diag (bool): Whether to display the diagonal in the contact maps. 
        pow (int or float): Power to raise the contact map values before binning.
        bin_size (int): Bin size for binning the contact map.
        cmap (str or Colormap): Colormap to use for imshow.
        split_divider_params (dict): Parameters for the split diagonal divider (linestyle, color, width, alpha).
        ncols (int): Number of columns in the subplot grid.
        axis_title_fontsize (int): Font size for subplot titles and axis labels. 
        highlight_diff_regions (bool): If True, highlight the most different regions in each map.
        highlight_linewidth (float): Line width for highlight rectangles.
        highlight_palette (str): Colormap or palette name for highlight colors.
        highlight_min_size (int): Minimum size (in residues) of highlighted region (rectangle side).
        highlight_n_regions (int): Number of most different regions to highlight per map.
        highlight_alpha (float): Alpha (opacity) for highlight rectangles.
        highlight_color_fill (bool): Whether to fill the highlight rectangles.
        highlight_postprocessed (bool): Whether to use the postprocessed diff map for highlighting.
        device (str or torch.device): Device for torch operations (e.g., "cuda", "cpu"). 
        hspace (float): Height space between subplots.
        wspace (float): Width space between subplots.   
        agg_func (function): Function to aggregate the contact map values.
        normalize (bool): Whether to normalize the contact map values to [0, 1] range.
        show_zoom_in (bool): If True, show a zoom-in of the highlighted region in each subplot.
        zoom_in_params (dict): Dictionary of zoom-in options. Supported keys:
            - "size" (float): Fraction of the main axis width/height for the zoom-in square (default: 0.33)
            - "border_color" (str): Color for the border of the zoom-in inset (default: "lime")
            - "border_width" (float): Line width for the border of the zoom-in inset (default: 1.5)
            - "cmap" (str or Colormap): Colormap for the zoom-in (defaults to main cmap)

    Returns:
        None. The function modifies the provided axes and displays the plot.

    Notes:
        - If split_ref_haplotype is True, each subplot shows the haplotype in the upper triangle and the reference in the lower triangle.
        - If highlight_diff_regions is True, the most different regions between each map and the reference are highlighted.
        - A border is added to the reference subplot unless split_ref_haplotype is True.
        - If show_zoom_in is True, a zoom-in of the highlighted region is shown in the upper right of each subplot.
    """
    import matplotlib.patches as patches
    import matplotlib as mpl
    import torch     
    import math  

    # --- Set up zoom-in parameters ---
    _default_zoom_in_params = {
        "size": 0.33,
        "border_color": "lime",
        "border_width": 1.5,
        "cmap": None,
    }
    if zoom_in_params is None:
        zoom_in_params = {}
    zoom_in_cfg = {**_default_zoom_in_params, **zoom_in_params}

    # --- Calculate grid shape robustly ---
    n_plots = len(contact_maps)
    if ncols is None or ncols < 1:
        ncols = 3
    nrows = math.ceil(n_plots / ncols)

    if axes is None:
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(ncols * 4, nrows * 4)
        )
        # axes could be 2D or 1D depending on nrows/ncols
        axes = np.array(axes).reshape(-1)
    else:
        axes = np.array(axes).reshape(-1)

    if ref_key is None:
        ref_key = get_ref_key(contact_maps, error=False)
    ref_map = contact_maps[ref_key] if ref_key is not None else None

    # If no reference map, disable features that require it, but still plot
    if ref_map is None:
        split_ref_haplotype = False
        highlight_diff_regions = False
        show_zoom_in = False
 

    highlight_boxes = []  # List of (color, [(x, y, w), ...])

    # For zoom-in: store the highlight region coordinates for each subplot
    # We will store the *first* highlight region for each subplot, matching the highlight box
    zoom_in_regions = [None] * len(contact_maps)

    for i, (name, contact_map) in enumerate(contact_maps.items()):
        if name == ref_key:
            continue
        if i >= len(axes):
            break  # Prevent IndexError if more items than axes
        if not show_diag:
            np.fill_diagonal(contact_map, np.nan)
        # Prepare binned maps for both haplotype and REF
        contact_map_binned = mc.bin_matrix(X=contact_map**pow, bin_size=bin_size, agg_func=agg_func)
        if ref_map is not None:
            ref_map_binned = mc.bin_matrix(X=ref_map**pow, bin_size=bin_size, agg_func=agg_func)
        if normalize:
            contact_map_binned = mc.minmax_normalize_numpy(contact_map_binned)
            if ref_map is not None:
                ref_map_binned = mc.minmax_normalize_numpy(ref_map_binned)
        binned_shape = contact_map_binned.shape[0]

        # Compose the split image if requested and possible
        if split_ref_haplotype and ref_map is not None:
            # Create a new array for the split image
            split_img = np.zeros_like(contact_map_binned)
            # Fill upper triangle (excluding diagonal) with haplotype, lower triangle (including diagonal) with REF
            triu_idx = np.triu_indices(binned_shape, k=1)
            tril_idx = np.tril_indices(binned_shape, k=0)
            split_img[triu_idx] = contact_map_binned[triu_idx]
            split_img[tril_idx] = ref_map_binned[tril_idx]
            im = axes[i].imshow(split_img, cmap=cmap, interpolation="nearest", aspect='equal')

            # --- Split divider appearance controls ---
            if split_divider_params is not None:
                split_divider = split_divider_params.get("linestyle", "solid")
                split_divider_color = split_divider_params.get("color", "white")
                split_divider_width = split_divider_params.get("width", 1.5)
                split_divider_alpha = split_divider_params.get("alpha", 1.0)

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
            if i % ncols == 0:
                axes[i].annotate(
                    "REF",
                    xy=(0, 0.5),
                    xycoords='axes fraction',
                    fontsize=axis_title_fontsize,
                    ha='right',
                    va='center',
                    rotation=90,
                )
        else:
            im = axes[i].imshow(contact_map_binned, cmap=cmap, interpolation="nearest", aspect='equal')

        try:
            map_name = get_haplotype_ids(name)[0]
        except:
            map_name = name
        axes[i].set_title(f"{map_name}", fontsize=axis_title_fontsize, pad=2)
        axes[i].axis('off')
        axes[i].set_aspect('equal')

        # Highlight most different regions for non-REF subplots if requested
        if 'highlight_box_offset' not in locals() and 'highlight_box_offset' not in globals():
            highlight_box_offset = highlight_linewidth / 2.0  # Default: half the linewidth

        # For zoom-in: store the region for this subplot
        zoom_in_this = None

        if (
            highlight_diff_regions
            and ref_map is not None
            and name != ref_key
            and contact_map.shape == ref_map.shape
        ):
            if split_ref_haplotype:
                color = highlight_color
            else:
                color_idx = i - 1  # i=0 is REF, so subtract 1
                better_cmap = mpl.cm.get_cmap(highlight_palette, max(1, len(contact_maps)-1))
                color = mpl.colors.to_hex(better_cmap(color_idx % better_cmap.N))

            # Prepare the two matrices to compare
            if highlight_postprocessed:
                mat1 = np.nan_to_num(mc.expand_matrix(contact_map_binned, target_size=contact_map.shape[0]))
                mat2 = np.nan_to_num(mc.expand_matrix(ref_map_binned, target_size=ref_map.shape[0]))
            else:
                mat1 = np.nan_to_num(contact_map)
                mat2 = np.nan_to_num(ref_map)
            L = mat1.shape[0]
            coords = []
            lw = highlight_linewidth
            offset = highlight_box_offset

            if L <= highlight_min_size:
                print("small map")
                # For small maps, highlight the whole map, but draw both the original and symmetric rectangles
                for region_idx in range(highlight_n_regions):
                    # Rectangle 1: top-left to bottom-right
                    rect1 = patches.Rectangle(
                        (-offset, -offset),
                        L + 2*offset,
                        L + 2*offset,
                        linewidth=lw,
                        edgecolor=color,
                        facecolor=color,
                        alpha=highlight_alpha,
                        fill=highlight_color_fill,
                    )
                    axes[i].add_patch(rect1)
                    coords.append((0, 0, L))
                    # Rectangle 2: symmetric (bottom-left to top-right)
                    rect2 = patches.Rectangle(
                        (-offset, -offset),
                        L + 2*offset,
                        L + 2*offset,
                        linewidth=lw,
                        edgecolor=color,
                        facecolor=color,
                        alpha=highlight_alpha,
                        fill=highlight_color_fill,
                        transform=axes[i].transData + mpl.transforms.Affine2D().rotate_deg_around(L/2, L/2, 90)
                    )
                    axes[i].add_patch(rect2)
                    coords.append(("symmetric", 0, 0, L))
                # For zoom-in, just use the first region (the whole map)
                zoom_in_this = (0, 0, L)
            else:
                if device is None:
                    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    torch_device = torch.device(device) if isinstance(device, str) else device

                # Use the new function
                score, idx_sorted = compute_highlight_scores(
                    mat1, mat2, highlight_min_size, highlight_n_regions, torch_device
                )

                mask = np.zeros_like(score, dtype=bool)
                found = 0
                for idx in idx_sorted:
                    if found >= highlight_n_regions:
                        break
                    x = idx // score.shape[1]
                    y = idx % score.shape[1]
                    x_center = x + highlight_min_size // 2
                    y_center = y + highlight_min_size // 2
                    x0 = x_center - highlight_min_size // 2
                    y0 = y_center - highlight_min_size // 2
                    x0 = max(0, min(x0, L - highlight_min_size))
                    y0 = max(0, min(y0, L - highlight_min_size))
                    mask_x0 = x0
                    mask_x1 = x0 + highlight_min_size
                    mask_y0 = y0
                    mask_y1 = y0 + highlight_min_size
                    if not mask[mask_x0:mask_x1, mask_y0:mask_y1].any():
                        coords.append((x0, y0, highlight_min_size))
                        mask[mask_x0:mask_x1, mask_y0:mask_y1] = True
                        found += 1

                bin_scale = bin_size
                for (x, y, w) in coords:
                    # The offset should increase the size of the box, not just shift its position.
                    # So, we expand the box by 2*offset in both width and height, and shift the origin accordingly.
                    x_binned = x / bin_scale
                    y_binned = y / bin_scale
                    w_binned = w / bin_scale
                    # Rectangle 1: (y_binned, x_binned)
                    rect1 = patches.Rectangle(
                        (y_binned - offset, x_binned - offset),  # imshow: (col, row), offset outwards
                        w_binned + 2*offset,
                        w_binned + 2*offset,
                        linewidth=lw,
                        edgecolor=color,
                        facecolor=color,
                        alpha=highlight_alpha,
                        fill=highlight_color_fill,
                    )
                    axes[i].add_patch(rect1)
                    # Rectangle 2: symmetric across the diagonal
                    # For a square at (x, y, w), the symmetric is at (y, x, w)
                    rect2 = patches.Rectangle(
                        (x_binned - offset, y_binned - offset),
                        w_binned + 2*offset,
                        w_binned + 2*offset,
                        linewidth=lw,
                        edgecolor=color,
                        facecolor=color,
                        alpha=highlight_alpha,
                        fill=highlight_color_fill,
                    )
                    axes[i].add_patch(rect2)
                # Store both rectangles' coordinates for REF overlay
                highlight_boxes.append((color, [
                    (y / bin_scale - offset, x / bin_scale - offset, w / bin_scale + 2*offset) for (x, y, w) in coords
                ] + [
                    (x / bin_scale - offset, y / bin_scale - offset, w / bin_scale + 2*offset) for (x, y, w) in coords
                ]))
                # For zoom-in, use the first region (x, y, w) in coords
                if coords:
                    zoom_in_this = coords[0]

        # Store the zoom-in region for this subplot
        zoom_in_regions[i] = zoom_in_this

    # Optionally include all highlight boxes in the REF plot afterwards, color-coded
    if highlight_diff_regions and len(highlight_boxes) > 0 and not split_ref_haplotype and ref_map is not None:
        lw = highlight_linewidth
        offset = highlight_box_offset
        for color, coords in highlight_boxes:
            for (x, y, w) in coords:
                rect = patches.Rectangle(
                    (y, x),  # imshow: (col, row), already offset in coords
                    w,
                    w,
                    linewidth=lw,
                    edgecolor=color,
                    facecolor=color,
                    alpha=highlight_alpha,
                    fill=highlight_color_fill,
                )
                axes[0].add_patch(rect)

    # --- Add a grey border around the entire first subplot (REF) ---
    # Only if not using split_ref_haplotype, since there is no dedicated REF plot in split mode
    if not split_ref_haplotype and ref_map is not None:
        ref_binned_shape = mc.bin_matrix(X=ref_map**pow, bin_size=bin_size, agg_func=agg_func).shape[0]
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

    # --- ZOOM-IN INSET: for each subplot, if requested and region available ---
    if show_zoom_in and ref_map is not None:
        import matplotlib.colors as mcolors
        for i, (name, contact_map) in enumerate(contact_maps):
            region = zoom_in_regions[i]
            if region is None:
                continue
            # region: (x, y, w) in original (unbinned) coordinates
            x, y, w = region
            bin_scale = bin_size

            # Expand the region by X% on all sides
            expand_frac = 0.33
            expand_amt = w * expand_frac
            x_exp = x - expand_amt
            y_exp = y - expand_amt
            w_exp = w + 2 * expand_amt

            # Convert to binned coordinates
            x_binned = int(np.floor(x_exp / bin_scale))
            y_binned = int(np.floor(y_exp / bin_scale))
            w_binned = int(np.ceil(w_exp / bin_scale))

            # Clamp to map size
            if split_ref_haplotype and ref_map is not None:
                contact_map_binned = mc.bin_matrix(X=contact_maps[name]**pow, bin_size=bin_size, agg_func=agg_func)
                ref_map_binned = mc.bin_matrix(X=ref_map**pow, bin_size=bin_size, agg_func=agg_func)
                if normalize:
                    contact_map_binned = mc.minmax_normalize_numpy(contact_map_binned)
                    ref_map_binned = mc.minmax_normalize_numpy(ref_map_binned)
                binned_shape = contact_map_binned.shape[0]
                x_binned = max(0, x_binned)
                y_binned = max(0, y_binned)
                w_binned = min(binned_shape - max(x_binned, y_binned), w_binned)
                region_slice = (slice(x_binned, x_binned + w_binned), slice(y_binned, y_binned + w_binned))
                haplo_region = contact_map_binned[region_slice]
                ref_region = ref_map_binned[region_slice]
                # Compose the zoom-in image: upper right triangle from haplo_region, lower left from ref_region
                zoom_img = np.zeros_like(haplo_region)
                N = zoom_img.shape[0]
                # Upper right triangle (including diagonal) from haplo_region
                triu_idx = np.triu_indices(N, k=0)
                # Lower left triangle (excluding diagonal) from ref_region
                tril_idx = np.tril_indices(N, k=-1)
                zoom_img[triu_idx] = haplo_region[triu_idx]
                zoom_img[tril_idx] = ref_region[tril_idx]
                zoom_cmap = zoom_in_cfg["cmap"] if zoom_in_cfg["cmap"] is not None else cmap
            else:
                contact_map_binned = mc.bin_matrix(X=contact_map**pow, bin_size=bin_size, agg_func=agg_func)
                if normalize:
                    contact_map_binned = mc.minmax_normalize_numpy(contact_map_binned)
                binned_shape = contact_map_binned.shape[0]
                x_binned = max(0, x_binned)
                y_binned = max(0, y_binned)
                w_binned = min(binned_shape - max(x_binned, y_binned), w_binned)
                region_slice = (slice(x_binned, x_binned + w_binned), slice(y_binned, y_binned + w_binned))
                zoom_img = contact_map_binned[region_slice]
                zoom_cmap = zoom_in_cfg["cmap"] if zoom_in_cfg["cmap"] is not None else cmap

            # Place the zoom-in as an inset in the upper right of the main axis
            ax = axes[i]
            bbox = ax.get_position()
            inset_size = zoom_in_cfg["size"]
            inset_left = 1 - inset_size
            inset_bottom = 1 - inset_size
            inset_ax = ax.inset_axes([inset_left, inset_bottom, inset_size, inset_size], 
                                     transform=ax.transAxes, zorder=10)
            im_zoom = inset_ax.imshow(zoom_img, cmap=zoom_cmap, interpolation="nearest", aspect='equal')
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_xticklabels([])
            inset_ax.set_yticklabels([])
            inset_ax.set_aspect('equal')
            for spine in inset_ax.spines.values():
                spine.set_edgecolor(zoom_in_cfg["border_color"])
                spine.set_linewidth(zoom_in_cfg["border_width"])
            if split_ref_haplotype and ref_map is not None:
                N = zoom_img.shape[0]
                inset_ax.plot(
                    [-0.5, N-0.5],
                    [-0.5, N-0.5],
                    color=zoom_in_cfg["border_color"],
                    linewidth=zoom_in_cfg["border_width"] / 2,
                    zorder=15,
                )

            # --- Draw connections from highlight rectangle to zoom-in inset if requested ---
            if zoom_in_cfg.get("draw_connections", False):
                # Main rectangle coordinates in binned space
                main_rect_x = y_binned
                main_rect_y = x_binned
                main_rect_w = w_binned

                # Four corners of the rectangle in data coordinates (main axis)
                main_corners = [
                    (main_rect_x, main_rect_y),  # top-left
                    (main_rect_x + main_rect_w, main_rect_y),  # top-right
                    (main_rect_x + main_rect_w, main_rect_y + main_rect_w),  # bottom-right
                    (main_rect_x, main_rect_y + main_rect_w),  # bottom-left
                ]

                # Four corners of the zoom-in rectangle in data coordinates (inset axis)
                N_zoom = zoom_img.shape[0]
                zoom_corners = [
                    (0, 0),  # top-left
                    (N_zoom, 0),  # top-right
                    (N_zoom, N_zoom),  # bottom-right
                    (0, N_zoom),  # bottom-left
                ]

                # Transform main axis data coords to display coords
                main_disp = [ax.transData.transform((x, y)) for (x, y) in main_corners]
                # Transform inset axis data coords to display coords
                inset_disp = [inset_ax.transData.transform((x, y)) for (x, y) in zoom_corners]

                # Now, draw lines from each main corner to corresponding zoom-in corner
                import matplotlib.lines as mlines
                for (p1, p2) in zip(main_disp, inset_disp):
                    # Create a line in figure coordinates
                    line = mlines.Line2D(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        transform=None,  # display coordinates
                        color=zoom_in_cfg.get("border_color", "lime"),
                        linewidth=1.2,
                        linestyle="dashed",
                        zorder=20,
                        alpha=0.8,
                    )
                    ax.figure.add_artist(line)

                # Optionally, draw the rectangle on the main axis if not already present
                rect = patches.Rectangle(
                    (main_rect_x, main_rect_y),
                    main_rect_w,
                    main_rect_w,
                    linewidth=1.5,
                    edgecolor=zoom_in_cfg.get("border_color", "lime"),
                    facecolor='none',
                    linestyle="dashed",
                    zorder=15,
                )
                ax.add_patch(rect)

                # Optionally, draw a rectangle on the inset axis (should already be the border, but for clarity)
                rect_inset = patches.Rectangle(
                    (0, 0),
                    N_zoom,
                    N_zoom,
                    linewidth=1.2,
                    edgecolor=zoom_in_cfg.get("border_color", "lime"),
                    facecolor='none',
                    linestyle="dashed",
                    zorder=16,
                )
                inset_ax.add_patch(rect_inset)
 
    # Hide any unused axes (if there are more axes than items_to_plot)
    for j in range(len(contact_maps), len(axes)):
        axes[j].set_visible(False)

    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    plt.show()

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
    highlight_min_size=None,
    highlight_n_regions=1, 
    highlight_palette="Set3",
    highlight_color_fill=False,
    highlight_alpha=1,
    highlight_linewidth=2,
    highlight_box_offset=None,
    highlight_color="white",
    highlight_postprocessed=False,
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
    max_per_mutation=None,
    max_mutations=None,
    agg_func=np.nanmax,
    normalize=True,
    show_zoom_in=False,
    zoom_in_params=None,
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
        highlight_postprocessed (bool): Whether to use the postprocessed diff map for highlighting.
        device (str or torch.device, optional): Device for torch operations, e.g., "cuda:2".
        split_ref_haplotype (bool): If True, each subplot is split along the diagonal: bottom=REF, top=haplotype.
        hspace (float): Vertical space between subplots.
        wspace (float): Horizontal space between subplots.
        axis_title_fontsize (int): Font size for x and y axis titles.
        max_per_mutation (int): Maximum number of times a mutations can appear before moving onto the next haplotype. 
            This helps avoid plotting the same mutation driving a change in the contact map multiple times.
        max_mutations (int): Maximum number of mutations per haplotype.
        agg_func (function): Function to aggregate the contact map values.
        normalize (bool): Whether to normalize the contact map values to [0, 1] range.
        show_zoom_in (bool): Whether to show a zoom-in of the highlighted region in each subplot.
        zoom_in_params (dict): Dictionary of zoom-in options. Supported keys:
            - "size" (float): Fraction of the main axis width/height for the zoom-in square (default: 0.33)
            - "border_color" (str): Color for the border of the zoom-in inset (default: "lime")
            - "border_width" (float): Line width for the border of the zoom-in inset (default: 1.5)
            - "cmap" (str or Colormap): Colormap for the zoom-in (defaults to main cmap)
    """

    def get_most_different_contact_maps(
        contact_maps, 
        max_subplots=6,
        diff_mode="global",  # "global" or "local"
        conv_window=100,      # int, only used if diff_mode=="local"
        pow=4,
        bin_size=1,
        max_per_mutation=None,
        max_mutations=None,
        highlight_postprocessed=False,
        agg_func=np.nanmax,
        normalize=True
    ):
        """
        Selects the most different contact maps compared to the reference.

        Args:
            contact_maps (dict): Dictionary mapping keys to contact map numpy arrays.
            max_subplots (int): Maximum number of subplots to show (including reference).
            diff_mode (str): "global" (sum of all differences) or "local" (max difference in a window).
            conv_window (int or None): Window size for local difference (square side length).
            pow (int): Power to raise the contact map values before binning.
            bin_size (int): Bin size for binning the contact map.
            max_per_mutation (int): Maximum number of times a mutations can appear before moving onto the next haplotype. 
                This helps avoid plotting the same mutation driving a change in the contact map multiple times.
            highlight_postprocessed (bool): Whether to use the postprocessed diff map for highlighting.
            normalize (bool): Whether to normalize the contact map values to [0, 1] range.

        Returns:
            dict: Most different contact maps (excluding reference).
        """
        import numpy as np
        import torch 

        ref_key = get_ref_key(contact_maps)
        ref_map = contact_maps[ref_key]
        diff_scores = {}
        mutation_counter = {}
 
        # Prepare binned and normalized reference map if needed
        ref_key = get_ref_key(contact_maps)
        ref_map = contact_maps[ref_key]

        # Filter out maps that are not the same shape as the reference or are the reference itself
        filtered_maps = {
            k: v for k, v in contact_maps.items()
            if k != ref_key and v.shape == ref_map.shape
        }

        # Apply mutation count filtering
        mutation_counter = {}
        selected_maps = {}
        for k, v in filtered_maps.items():
            haplotype_id = get_haplotype_ids(k, revert_naming=True)[0]
            mutations = haplotype_id.split(",")
            if max_mutations is not None and len(mutations) > max_mutations:
                continue
            skip = False
            for mutation in mutations:
                if mutation not in mutation_counter:
                    mutation_counter[mutation] = 0
                mutation_counter[mutation] += 1
                if max_per_mutation is not None and mutation_counter[mutation] > max_per_mutation:
                    skip = True
                    break
            if skip:
                continue
            selected_maps[k] = v

        # Compute difference scores using compute_highlight_scores
        diff_scores = {}
        for k, v in selected_maps.items():
            v_clean = np.nan_to_num(v)
            ref_clean = np.nan_to_num(ref_map)

            # Bin the maps if requested
            if highlight_postprocessed:
                v_clean = mc.bin_matrix(X=v_clean**pow, bin_size=bin_size, agg_func=agg_func)
                ref_clean = mc.bin_matrix(X=ref_clean**pow, bin_size=bin_size, agg_func=agg_func)

            if normalize:
                v_clean = mc.minmax_normalize_numpy(v_clean)
                ref_clean = mc.minmax_normalize_numpy(ref_clean)

            # Use compute_highlight_scores to get a difference score
            # Use highlight_min_size and highlight_n_regions for window/region parameters
            # Use torch.device("cuda" if torch.cuda.is_available() else "cpu") as default device
            if 'device' in locals() and device is not None:
                torch_device = torch.device(device) if isinstance(device, str) else device
            else: 
                torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            score, idx_sorted = compute_highlight_scores(
                v_clean, ref_clean, conv_window if diff_mode == "local" else min(v_clean.shape[0], v_clean.shape[1]),  # window size
                1,  # just need the top region for scoring
                torch_device
            )
            # For "global", use the sum of the score matrix; for "local", use the max
            if diff_mode == "global":
                diff = np.nansum(np.abs(v_clean - ref_clean))
            elif diff_mode == "local":
                diff = np.nanmax(score)
            else:
                raise ValueError(f"Unknown diff_mode: {diff_mode}")

            diff_scores[k] = diff

        # Get the most different contact maps
        most_diff_keys = sorted(diff_scores, key=diff_scores.get, reverse=True)[:max_subplots]
        most_diff_contact_maps = {k: contact_maps[k] for k in most_diff_keys}
        return most_diff_contact_maps

    if highlight_min_size is None:
        highlight_min_size = conv_window

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
                                                             conv_window=conv_window,
                                                             pow=pow,
                                                             bin_size=bin_size,
                                                             max_per_mutation=max_per_mutation,
                                                             max_mutations=max_mutations,
                                                             highlight_postprocessed=highlight_postprocessed,
                                                             agg_func=agg_func,
                                                             normalize=normalize) 

    # Call the new function in place of the old code
    plot_contact_map_subplots(
        axes=axes,
        contact_maps=contact_maps, 
        split_ref_haplotype=split_ref_haplotype,
        show_diag=show_diag, 
        pow=pow,
        bin_size=bin_size,
        cmap=cmap,
        split_divider_params=split_divider_params,
        ncols=ncols,
        axis_title_fontsize=axis_title_fontsize, 
        highlight_diff_regions=highlight_diff_regions,
        highlight_linewidth=highlight_linewidth,
        highlight_palette=highlight_palette,
        highlight_min_size=highlight_min_size,
        highlight_n_regions=highlight_n_regions,
        highlight_alpha=highlight_alpha,
        highlight_color_fill=highlight_color_fill,
        highlight_color=highlight_color,
        highlight_postprocessed=highlight_postprocessed,
        device=device, 
        hspace=hspace,
        wspace=wspace,
        agg_func=agg_func,
        normalize=normalize,
        show_zoom_in=show_zoom_in,
        zoom_in_params=zoom_in_params
    )

def get_ref_size(contact_maps):
    ref_key = get_ref_key(contact_maps)
    ref_map = contact_maps[ref_key]
    return ref_map.shape[0]



def plot_stacked_contact_maps(
    contact_maps, 
    n_maps=6, 
    cmap='gnuplot2', 
    front_on_top=True, 
    direction='ul2lr',
    pow=4,
    bin_size=1,
    dpi=100,
    alpha=1,
    border_params={"linewidth": 10, "edgecolor": "white", "facecolor": "none", "alpha": 1},
    max_per_stack=None,
    stack_x_offset=None,  # horizontal offset between stacks
    stack_y_offset=None,  # vertical offset between stacks
    stack_x_inner_offset=None,  # NEW: horizontal offset within each stack
    stack_y_inner_offset=None,  # NEW: vertical offset within each stack
    ref_on_top=False      # ensure REF is the front-most plot if True
):
    """
    Plot contact maps stacked diagonally, with partial overlap.
    Optionally, split into multiple stacks if max_per_stack is set.

    Args:
        contact_maps (dict): Dictionary of {filename: contact_map_array}
        n_maps (int): Number of contact maps to display (default 6)
        cmap (str): Colormap for imshow
        front_on_top (bool): If True, the last map is on top (default). 
                             If False, the first map is on top.
        direction (str): Diagonal stacking direction. 
                         Options: 
                            'ul2lr' (upper-left to lower-right, default),
                            'ur2ll' (upper-right to lower-left),
                            'll2ur' (lower-left to upper-right),
                            'lr2ul' (lower-right to upper-left)
        max_per_stack (int or None): If set, create a new stack for every max_per_stack maps.
        stack_x_offset (int or None): Horizontal offset (in pixels) between stacks. 
                                      If None, defaults to int(w * 0.25)
        stack_y_offset (int or None): Vertical offset (in pixels) between stacks. 
                                      If None, defaults to 0 (no vertical shift)
        stack_x_inner_offset (int or None): Horizontal offset (in pixels) between maps within a stack.
                                            If None, defaults to int(w * 0.15)
        stack_y_inner_offset (int or None): Vertical offset (in pixels) between maps within a stack.
                                            If None, defaults to int(h * 0.15)
        ref_on_top (bool): If True, ensure the REF haplotype is the front-most plot.
    """
    # Select up to n_maps contact maps, but ensure REF is included if ref_on_top
    all_keys = list(contact_maps.keys())
    ref_key = get_ref_key(contact_maps=contact_maps)
    keys = all_keys[:n_maps]

    # If ref_on_top and REF is not in the first n_maps, add it (unless already present)
    if ref_on_top and ref_key is not None and ref_key not in keys:
        # Remove last key to keep total at n_maps, unless already less than n_maps
        if len(keys) == n_maps:
            keys = keys[:-1]
        keys.append(ref_key)

    maps = [contact_maps[k] for k in keys]
    n = len(maps)
    if n == 0:
        print("No contact maps to display.")
        return

    # Find the index of the REF haplotype, if present
    ref_idx = None
    if ref_key is not None and ref_key in keys:
        ref_idx = keys.index(ref_key)

    # If ref_on_top is True and REF is present, move it to the end (for front_on_top=True) or start (for front_on_top=False)
    if ref_on_top and ref_idx is not None:
        ref_key_val = keys.pop(ref_idx)
        ref_map = maps.pop(ref_idx)
        if front_on_top:
            keys.append(ref_key_val)
            maps.append(ref_map)
        else:
            keys.insert(0, ref_key_val)
            maps.insert(0, ref_map)

    # Assume all maps are square and same shape
    map_shape = maps[0].shape
    h, w = map_shape

    # Offset for each map within a stack (in pixels)
    if stack_x_inner_offset is None:
        x_offset = int(w * 0.15)
    else:
        x_offset = stack_x_inner_offset
    if stack_y_inner_offset is None:
        y_offset = int(h * 0.15)
    else:
        y_offset = stack_y_inner_offset

    # Direction multipliers
    dir_map = {
        'ul2lr': (1, 1),
        'ur2ll': (-1, 1),
        'll2ur': (1, -1),
        'lr2ul': (-1, -1)
    }
    if direction not in dir_map:
        raise ValueError(f"Unknown direction '{direction}'. Choose from {list(dir_map.keys())}")
    dx, dy = dir_map[direction]

    # Determine stack splitting
    if max_per_stack is not None and isinstance(max_per_stack, int) and max_per_stack > 0:
        n_stacks = math.ceil(n / max_per_stack)
        stack_indices = [
            (i * max_per_stack, min((i + 1) * max_per_stack, n))
            for i in range(n_stacks)
        ]
    else:
        n_stacks = 1
        stack_indices = [(0, n)]

    # Set stack-to-stack offsets (between stacks, not within)
    if stack_x_offset is None:
        stack_x_offset = int(w * 0.25)
    if stack_y_offset is None:
        stack_y_offset = 0

    # Calculate figure size for all stacks (arranged with partial overlap)
    stack_fig_w = w + abs(x_offset) * (max_per_stack-1 if max_per_stack else n-1)
    stack_fig_h = h + abs(y_offset) * (max_per_stack-1 if max_per_stack else n-1)
    # For multiple stacks, arrange them with user-defined offset
    total_fig_w = stack_fig_w + (n_stacks-1) * stack_x_offset
    total_fig_h = stack_fig_h + (n_stacks-1) * abs(stack_y_offset)

    fig, ax = plt.subplots(figsize=(total_fig_w/100, total_fig_h/100), dpi=dpi)
    fig.patch.set_alpha(0.0)  # Make the figure background transparent

    for stack_num, (start, end) in enumerate(stack_indices):
        stack_keys = keys[start:end]
        stack_maps = maps[start:end]
        stack_n = len(stack_maps)
        if stack_n == 0:
            continue

        # Determine plotting order within stack
        if front_on_top:
            plot_indices = range(stack_n)
        else:
            plot_indices = reversed(range(stack_n))

        # Offset for this stack
        stack_x_shift = stack_num * stack_x_offset
        stack_y_shift = stack_num * stack_y_offset

        for i in plot_indices:
            name = stack_keys[i]
            m = stack_maps[i]
            # Calculate offset for this map within the stack
            if front_on_top:
                offset_idx = i
            else:
                offset_idx = stack_n-1-i
            x0 = x_offset * offset_idx * dx if dx >= 0 else stack_fig_w - w + x_offset * offset_idx * dx
            y0 = y_offset * offset_idx * dy if dy >= 0 else stack_fig_h - h + y_offset * offset_idx * dy

            # Add stack shift
            x0 += stack_x_shift
            y0 += stack_y_shift

            m = mc.bin_matrix(m**pow, bin_size=bin_size)

            # Prevent flipping: set origin='upper' so (0,0) is top-left, and do not swap y-limits
            ax.imshow(
                m,
                extent=(x0, x0+w, y0, y0+h),
                cmap=cmap,
                alpha=alpha if (i != (stack_n-1 if front_on_top else 0)) else 1.0,
                zorder=stack_num*100 + i,
                origin='upper'
            )
            
            # Add border to each map
            rect = plt.Rectangle((x0, y0), w, h, **border_params, zorder=stack_num*100 + i)
            ax.add_patch(rect)
            
            # Label each map in the upper left of the heatmap
            protein_id = os.path.basename(name).split("_unrelaxed")[0].split("_")[0]
            haplotype_id = os.path.basename(name).split("_unrelaxed")[0].replace(protein_id + "_", "")
            ax.text(
                x0 + 2, y0 + 2 + 8,  # 2 px padding from top/left, 8 for font height
                f"{revert_haplotype_naming(haplotype_id)}",
                color='white', fontsize=40, zorder=stack_num*100 + 20, alpha=1,
                va='bottom', ha='left', fontweight='bold', bbox=dict(facecolor='black', alpha=0.9, pad=1, edgecolor='none')
            )

    ax.set_xlim(0, total_fig_w)
    ax.set_ylim(0, total_fig_h)  # y increases downward, so (0,0) is top-left
    ax.axis('off') 
    plt.show()


def pad_contact_map_to_msa_fast(contact_map, non_gap_positions, L_msa, pad_value=np.nan):
    """
    Pad a contact map to the full MSA length, inserting pad_value at gap positions.

    This function efficiently inserts a smaller contact map (corresponding to non-gap positions)
    into a larger square matrix of size (L_msa, L_msa), filling all other positions with pad_value.
    It uses advanced numpy indexing for speed.

    Parameters
    ----------
    contact_map : np.ndarray
        The contact map array of shape (L_seq, L_seq), where L_seq is the number of non-gap positions.
    non_gap_positions : array-like of int
        Indices of non-gap positions in the MSA (length L_seq).
    L_msa : int
        Length of the MSA (number of columns/rows in the padded matrix).
    pad_value : scalar, optional
        Value to use for padding at gap positions (default: np.nan).

    Returns
    -------
    np.ndarray
        A (L_msa, L_msa) array with the contact map values at non-gap positions and pad_value elsewhere.

    Raises
    ------
    ValueError
        If the shape of contact_map does not match the number of non-gap positions.
    """
    L_seq = len(non_gap_positions)
    if contact_map.shape[0] != L_seq or contact_map.shape[1] != L_seq:
        raise ValueError(f"Contact map shape {contact_map.shape} does not match non-gap positions {L_seq}")
    padded_map = np.full((L_msa, L_msa), pad_value, dtype=contact_map.dtype)
    idx = np.ix_(non_gap_positions, non_gap_positions)
    padded_map[idx] = contact_map
    return padded_map

def pad_all_contact_maps_to_msa(contact_maps, msa, pad_value=np.nan):
    """
    Pad all contact maps in a dictionary to the MSA length, inserting pad_value at gap positions.

    For each contact map, this function determines the non-gap positions in the corresponding
    MSA sequence and pads the contact map to the full MSA length using pad_value for gaps.

    Parameters
    ----------
    contact_maps : dict
        Dictionary mapping contact map IDs (e.g., filenames or haplotype IDs) to contact map arrays.
    msa : Bio.Align.MultipleSeqAlignment
        Multiple sequence alignment object.
    pad_value : scalar, optional
        Value to use for padding at gap positions (default: np.nan).

    Returns
    -------
    dict
        Dictionary mapping contact map IDs to padded contact map arrays of shape (L_msa, L_msa).

    Notes
    -----
    - Only contact maps with a corresponding sequence in the MSA are processed.
    - The mapping between contact map IDs and MSA sequence names is determined using
      `cf.get_haplotype_ids` and the sequence name format in the MSA.
    - Non-gap positions are those not equal to '-' or '.' in the MSA sequence.

    Example
    -------
    >>> padded_maps = pad_all_contact_maps_to_msa(contact_maps, msa)
    >>> padded_maps['haplotype1'].shape
    (L_msa, L_msa)
    """
    # Precompute all non-gap positions for each sequence in the MSA
    L_msa = msa.get_alignment_length()
    msa_seq_strs = [str(rec.seq) for rec in msa]
    msa_non_gap_positions = [
        np.fromiter((i for i, aa in enumerate(seq) if aa != '-' and aa != '.'), dtype=int)
        for seq in msa_seq_strs
    ]

    # Build a mapping from sequence name to MSA index for fast lookup
    msa_name_to_index = {rec.name.split(":")[1]: i for i, rec in enumerate(msa)}

    # Prepare contact_map_dict as before
    contact_map_dict = get_haplotype_ids(contact_maps.keys(), as_dict=-1)

    # Only process those in both contact_maps and MSA, and vectorize as much as possible
    padded_contact_maps = {}
    for seq_id, cmap_id in tqdm(contact_map_dict.items(),
                                total=len(contact_map_dict),
                                desc="Padding contact maps to MSA"):
        msa_idx = msa_name_to_index.get(seq_id)
        if msa_idx is not None:
            non_gap_positions = msa_non_gap_positions[msa_idx]
            padded_contact_maps[cmap_id] = pad_contact_map_to_msa_fast(
                contact_maps[cmap_id], non_gap_positions, L_msa, pad_value=pad_value
            )
    return padded_contact_maps



def plot_superpopulation_contact_diff(
    specific_binary_diff_figs,
    specific_binary_diff_maps, 
    pairs_per_colset=4,
    use_common_heatmap_scale=True,
    set_common_barplot_ylim=True,
    suptitle="Gained/Lost Contacts by Superpopulation Relative to REF",
    translate_superpop_names=True,
    width_ratios=[1, 0.2],
    figsize=(8, 4.5),
    wspace=0.05, 
    hspace=0.3,
    show_plot=True,
    suptitle_fontsize=None,
    annotation_fontsize=None,
):
    """
    Plot heatmaps and barplots of gained/lost contacts by superpopulation.

    Parameters
    ----------
    specific_binary_diff_figs : dict
        Dictionary mapping population name to matplotlib Figure (heatmap).
    specific_binary_diff_maps : dict
        Dictionary mapping population name to binary diff map (numpy array).
    cf : module or object
        Must provide plot_contact_map_diff_barplot(diff_map, ax=...).
    pairs_per_colset : int, optional
        Number of (heatmap+barplot) pairs per "row block" before starting a new set of columns.
    use_common_heatmap_scale : bool, optional
        Whether all heatmaps use the same color scale.
    set_common_barplot_ylim : bool, optional
        Whether to set a common y-axis max for all barplots.
    suptitle : str, optional
        Figure supertitle.
    show : bool, optional
        Whether to call plt.show(). 
    width_ratios : list, optional
        Width ratios for the columns.
    figsize : tuple, optional
        Figure size.
    wspace : float, optional
        Width of the space between columns.
    hspace : float, optional
        Height of the space between rows.
    suptitle_fontsize : float, optional
        Font size for the figure suptitle. If None, uses matplotlib rcParams default.
    annotation_fontsize : float, optional
        Font size for barplot annotations. If None, uses matplotlib rcParams default.

    Returns
    -------
    fig : matplotlib.figure.Figure, optional
        The figure, if return_fig is True.
    """
    populations = list(specific_binary_diff_figs.keys())
    n_pops = len(populations)


    # Translate superpopulation names to more readable names in the subplot titles
    if translate_superpop_names:
        import src.onekg as og
        superpop_names_dict = og.SUPERPOP_NAMES_DICT 

    # For this application, we know the min/max for the binary diff maps are -1 and 1 (proportion units)
    if use_common_heatmap_scale:
        global_vmin, global_vmax = -1, 1
    else:
        global_vmin, global_vmax = None, None

    # If using a common y-lim for barplots, determine the max value across all barplots
    if set_common_barplot_ylim:
        barplot_max = 0
        for pop in populations:
            diff_map = specific_binary_diff_maps[pop]
            triu_mask = np.triu(np.ones(diff_map.shape, dtype=bool), k=1)
            contacts_gained = np.sum((diff_map > 0) & triu_mask)
            contacts_lost = np.sum((diff_map < 0) & triu_mask)
            this_max = max(contacts_gained, contacts_lost)
            if this_max > barplot_max:
                barplot_max = this_max
        # Add a little headroom for labels
        barplot_ylim = (0, barplot_max * 1.15 if barplot_max > 0 else 1)
    else:
        barplot_ylim = None

    # Calculate how many column sets we need
    n_colsets = int(np.ceil(n_pops / pairs_per_colset))
    nrows = min(pairs_per_colset, n_pops)
    ncols = n_colsets * 2  # 2 columns per colset (heatmap, barplot)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(figsize[0] * n_colsets, figsize[1] * nrows),
        width_ratios=width_ratios * n_colsets
    )
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)  # ensure 2D array for consistent indexing

    # Reduce whitespace between columns (without changing width_ratios)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)  # wspace controls width between columns

    bar_figs = []
    bar_dfs = []

    for idx, pop in enumerate(populations):
        colset = idx // pairs_per_colset
        row = idx % pairs_per_colset
        col_heat = colset * 2
        col_bar = colset * 2 + 1

        # Left: heatmap
        ax_heat = axes[row, col_heat] if nrows > 1 else axes[0, col_heat]
        orig_fig = specific_binary_diff_figs[pop]
        orig_axes = orig_fig.get_axes()
        if len(orig_axes) > 0:
            orig_ax = orig_axes[0]
            for im in orig_ax.get_images():
                arr = im.get_array()
                if hasattr(arr, "data"):
                    arr = arr.data
                cmap = im.get_cmap()
                vmin = global_vmin if use_common_heatmap_scale else im.get_clim()[0]
                vmax = global_vmax if use_common_heatmap_scale else im.get_clim()[1]
                ax_heat.imshow(arr,
                               cmap=cmap,
                               interpolation='nearest',
                               vmin=vmin, vmax=vmax)
            ax_heat.set_title(f"{superpop_names_dict[pop]} ({pop})" if translate_superpop_names else pop)
            ax_heat.set_xlabel(orig_ax.get_xlabel())
            ax_heat.set_ylabel(orig_ax.get_ylabel())
        else:
            ax_heat.set_title(pop)
            ax_heat.axis('off')

        # Right: barplot
        ax_bar = axes[row, col_bar] if nrows > 1 else axes[0, col_bar]
        bar_fig, bar_df = plot_contact_map_diff_barplot(specific_binary_diff_maps[pop], 
                                                        ax=ax_bar,
                                                        annotation_fontsize=annotation_fontsize)
        bar_df["superpopulation"] = pop
        ax_bar.set_title(f"{superpop_names_dict[pop]} ({pop})" if translate_superpop_names else pop)
        if barplot_ylim is not None:
            ax_bar.set_ylim(barplot_ylim)
        bar_figs.append(bar_fig)
        bar_dfs.append(bar_df)

    bar_dfs = pd.concat(bar_dfs, axis=0)
    # Hide any unused subplots
    for idx in range(n_pops, nrows * n_colsets):
        colset = idx // pairs_per_colset
        row = idx % pairs_per_colset
        col_heat = colset * 2
        col_bar = colset * 2 + 1
        axes[row, col_heat].axis('off')
        axes[row, col_bar].axis('off')

    suptitle_kwargs = {'y': 1.01}
    if suptitle_fontsize is not None:
        suptitle_kwargs['fontsize'] = suptitle_fontsize
    fig.suptitle(suptitle, **suptitle_kwargs)
    plt.tight_layout()
    if show_plot:
        plt.show() 
    return {'fig':fig, 'axes':axes, 'data':bar_dfs}


def plot_superpopulation_maps(contact_maps, pow=1, nrows=2, cmap='viridis', bin_size=10):
    """
    Plot each superpopulation map as a subplot across 2 rows.

    Parameters
    ----------
    avg_maps : dict
        Dictionary of superpopulation names to contact maps (numpy arrays).
    pow : int or float, optional
        Power to which each map is raised before plotting.
    nrows : int, optional
        Number of rows in the subplot grid.
    cmap : str, optional
        Colormap to use for imshow.
    """
    n_maps = len(contact_maps)
    ncols = math.ceil(n_maps / nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                             figsize=(3 * ncols, 6 * nrows // 2), 
                             squeeze=False)

    axes_flat = axes.flatten()
    for ax, (pop, m) in zip(axes_flat, contact_maps.items()):
        m_binned = mc.bin_matrix(m, bin_size=bin_size)
        im = ax.imshow(m_binned**pow, cmap=cmap)
        ax.set_title(pop)
        ax.axis('off')
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots
    for ax in axes_flat[len(contact_maps):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def compress_npy_files(base_dir="~/projects/data/colabfold/",
                       rep_types=["_single_repr_", 
                                  "_pair_repr_"], 
                       force=False,
                       remove_original=False):

    import os
    import numpy as np
    import glob
    from tqdm import tqdm

    # Expand the user path and format with tx_id
    base_dir = os.path.expanduser(base_dir)
    for rep_type in rep_types:

        # List all folders (directories) within the base_dir
        tx_folders = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

        subfolders = [os.path.join(tx_folder, d) 
                for tx_folder in tx_folders 
                for d in os.listdir(tx_folder) 
                if os.path.isdir(os.path.join(tx_folder, d))]

        for folder in subfolders:

            # Get the files
            files = glob.glob(os.path.expanduser(f"{folder}/*{rep_type}*.npy"))
            if len(files) == 0:
                print(f"No files found for {folder}")
                continue
            else:
                print(f"{folder} has {len(files)} files")
            
            # Get total file sizes
            total_size = sum(os.path.getsize(f) for f in files)
            print(f"Total size of {len(files)} files in {folder}: {total_size/1024/1024/1024:.2f} GB")
            
            # Define the npz file
            npz_file = f"{folder}/{os.path.basename(files[0]).split("_")[0]}_{rep_type.strip('_')}.npz"
            
            # Check if the npz file exists
            if not os.path.exists(npz_file) or force:  
                # Collect the map arrays in a dictionary
                mmap_arrays = {}
                for i, file in tqdm(enumerate(files), total=len(files)):
                    mmap_arrays[os.path.basename(file)] = np.load(file, mmap_mode='r+')
                    
                # # You can then process and save chunks of this mmap_array
                # # For example, to compress a portion and save it as a new compressed NPY file:
                # # (This still involves loading a chunk into memory for compression)
                
                print(f"Saving {npz_file}")
                np.savez_compressed(npz_file, **mmap_arrays)

            # Get the size of the original files (already computed as total_size)
            # Get the size of the compressed npz file
            compressed_size = os.path.getsize(npz_file)
            print(f"Compressed file size: {compressed_size/1024/1024/1024:.2f} GB")
            if total_size > 0:
                percent_compression = 100 * (1 - compressed_size / total_size)
                print(f"Percent compression: {percent_compression:.2f}%")
            else:
                print("Original total size is zero, cannot compute percent compression.")
            
            # # Read back in the compressed npz file to check its contents
            # loaded = np.load(npz_file, allow_pickle=True)
            # print(f"Keys in {npz_file}: {list(loaded.keys())}")
            # for k in loaded.files:
            #     print(f"Shape of {k}: {loaded[k].shape}")
            
            if remove_original:
                for file in files:
                    os.remove(file)

def compute_population_specific_contact_maps(
    contact_maps,
    contact_maps_binary,
    freq_df, 
    superpopulation_specific_only=True,
    exclude_ref_from_weights=False,
    weight_binary_maps=False,
):
    """
    Compute population-specific contact map statistics and difference maps.

    This function performs the following steps for each population (as defined by frequency columns in the input DataFrame):

    1. **Map haplotype IDs to PDB files:**
       - For each haplotype in the contact maps, determine the corresponding PDB file and add this information to the frequency DataFrame.

    2. **Identify frequency columns:**
       - Select columns in the DataFrame that represent population frequencies (excluding 'top_superpopulation_freq').

    3. **For each population:**
       1. Identify the subset of haplotypes belonging exclusively to the current population by filtering the DataFrame.
       2. Gather the contact maps for these haplotypes.
       3. Compute the weighted average contact map for the population using the population-specific frequencies as weights.
       4. Compute the mean difference map for the population by subtracting the reference contact map from each population member's contact map, then averaging these differences.
       5. Compute binary gain/loss maps for the population by:
          - Creating a dictionary of binary contact maps for the population and the reference.
          - Calling `plot_contact_map_diff` to generate a difference map and a corresponding figure (without displaying the plot).
       6. Store the results for each population in dictionaries.

    4. **Return a dictionary containing:**
       - `specific_maps`: Weighted average contact maps for each population.
       - `specific_diff_maps`: Mean difference maps (relative to reference) for each population.
       - `specific_binary_diff_maps`: Binary gain/loss maps for each population.
       - `specific_binary_diff_figs`: Figures visualizing the binary gain/loss maps for each population.

    Args:
        contact_maps: dict of {haplotype_id: contact_map}
        contact_maps_binary: dict of {haplotype_id: binary_contact_map}
        freq_df: DataFrame with haplotype frequencies and metadata
        superpopulation_specific_only: bool, optional
            Whether to only include haplotypes that are exclusive to the superpopulation.
            If True (default), only include haplotypes that are exclusive to the superpopulation, and weight them according to their frequency in the superpopulation.
            If False, include all haplotypes but weight them according to their frequency in the superpopulation.
        exclude_ref_from_weights: bool, optional
            Whether to exclude the reference from the weights.
            If True, the reference will not be included in the weights.
            If False, the reference will be included in the weights.
            Default is False.
            
    Returns:
        dict with:
            - specific_maps: average contact maps per population
            - specific_diff_maps: mean difference maps per population
            - specific_binary_diff_maps: binary gain/loss maps per population
            - specific_binary_diff_figs: figures for binary gain/loss maps per population
    """
    # Get the pdb file for each haplotype
    hap_id_dict = get_haplotype_ids(contact_maps.keys(), as_dict=-1)
    freq_df = freq_df.copy()
    freq_df["pdb_file"] = freq_df['haplotype'].str.split(":").str[1].map(hap_id_dict)
    freq_df_pdb = freq_df.set_index("pdb_file")
    freq_df_pdb = freq_df_pdb[freq_df_pdb.index.notna()] 
    # reorder freq_df_pdb to match contact_maps (not strictly needed here)
    freq_df_pdb = freq_df_pdb.loc[[k for k in contact_maps.keys() if k in freq_df_pdb.index]]

    freq_cols = [col for col in freq_df.columns if "freq" in col and col != "top_superpopulation_freq"]
    specific_maps = {}
    specific_diff_maps = {}
    specific_binary_diff_maps = {}
    specific_binary_diff_figs = {}

    ref_key = get_ref_key(contact_maps)

    for col in tqdm(freq_cols):
        pop = col.split("_")[0] 

        ##### COMPUTE CONTINUOUS CONTACT MAPS #####
        
        if superpopulation_specific_only:
            # Subset to only haplotypes that are exclusive to the superpopulation
            df = freq_df_pdb.loc[freq_df_pdb["specific_superpopulation"]==pop].copy() 
        
        else:  
            # Include all haplotypes but weight them according to their frequency in the superpopulation
            df = freq_df_pdb.copy()
         
        # Exclude the reference from the weights
        if exclude_ref_from_weights:
            df = df.loc[df.index != ref_key]

        # Subset the contact maps to only include the haplotypes in the dataframe
        pop_map_keys, pop_maps = zip(*[(k,v) for k,v in contact_maps.items() if k in df.index])
        
        # Reorder the dataframe to match the contact maps
        df = df.loc[[k for k in pop_map_keys if k in df.index]]

        # Compute weight average population contact map
        specific_maps[pop] = mc.average_matrices(pop_maps, 
                                                weights=df[col].values,
                                                normalize_scale=False)  
        # Compute absolute difference, then mean
        if pop_maps:
            diffs = np.stack(pop_maps) - contact_maps[ref_key]
            mean_div = np.mean(diffs, axis=0)
        else:
            mean_div = 0
        specific_diff_maps[pop] = mean_div 

        ##### COMPUTE BINARY CONTACT MAPS #####
        # Compute superpopulation-specific binary gain/loss maps
        if superpopulation_specific_only:
            binary_fig_out = plot_contact_map_diff(
                {k: v for k, v in contact_maps_binary.items() if k in df.index or k == ref_key}, 
                ref_key=ref_key,
                weights=df[col].values if weight_binary_maps else None,

                show_plot=False,
                verbose=False,
                
            )  
        else:
            binary_fig_out = plot_contact_map_diff(
                {k: v for k, v in contact_maps_binary.items() if k in df.index or k == ref_key}, 
                ref_key=ref_key,
                weights=df[col].values if weight_binary_maps else None,

                show_plot=False,
                verbose=False
            )  
        specific_binary_diff_maps[pop] = binary_fig_out['data'] 
        specific_binary_diff_figs[pop] = binary_fig_out['fig']
        #---- END OF FOR LOOP ----#

    return {
        "specific_maps": specific_maps,
        "specific_diff_maps": specific_diff_maps,
        "specific_binary_diff_maps": specific_binary_diff_maps,
        "specific_binary_diff_figs": specific_binary_diff_figs,
    }



def plot_contact_map_and_zooms(
    contact_maps_wt,
    ridge_df, 
    main_plot_above=False,
    max_highlights=12,
    highlight_size=100,
    zoom_nrows=6,
    zoom_ncols=None,
    zoom_subplot_size=2,
    zoom_value_col="interaction_strength_signed",
    main_nrows=2,
    main_ncols=2,
    linewidth=1,
    pow=1/2,
    main_padding_v=0.05,  # Vertical padding around main plot (when stacked or in shared grid)
    main_padding_h=0.05,  # Horizontal padding around main plot (in shared grid layout)
    zoom_hspace=0.1,  # Vertical spacing between zoom plots and between main plot and zooms
    zoom_wspace=0.1,  # Horizontal spacing between zoom plots
    save_path=None,
    palette="gnuplot2_r",
    title="AlphaFold Distance Map", 
    highlight_colors=None,
    color_by_interaction_score=True,
    white_border=True,
    white_border_width=2,
    zoom_rect_white_outline=False,
    zoom_rect_white_outline_width=3,
    zoom_show_ticks=False,
    zoom_tick_density=5,
    zoom_show_ylabel=True,  # Show y-axis label for zoom plots (only on leftmost if True)
    zoom_show_xlabel=True,  # Show x-axis label for zoom plots
    zoom_highlight_marker_scale=1.0,           # NEW: scale factor for size of zoom highlight marker
    zoom_highlight_linewidth=3,                # NEW: color (non-border) marker linewidth in zoom
    zoom_highlight_white_linewidth=5,           # NEW: white border linewidth in zoom
    zoom_effect_label_prefix=r"$E_{\text{WT-Clin}}$",    # NEW: prefix label for interaction strength in zoom plot titles
    figsize=None  # Figure size tuple (width, height). If None, auto-calculated based on layout
):
    """
    Visualize a contact (distance) map with highlighted variant positions and detail zooms.

    This function creates a main contact map plot, overlays highlight rectangles with crosshairs
    for selected variant pairs, and provides detailed zoomed-in subplots for each highlighted region.
    It supports flexible coloring, grid arrangements, and customizable annotation.

    Args:
        contact_maps_wt (dict): Dictionary of contact maps, keyed by haplotype/ref identifier.
        ridge_df (pd.DataFrame): Dataframe with rows describing highlights, must contain
            'clinical_position', 'wt_position', 'wt_variant', 'clinical_variant', and
            'interaction_strength_signed' columns for each variant pair to highlight.
        main_plot_above (bool|int): If True, main plot appears above zooms; if -1, below. 
            Otherwise, main plot shares grid with zooms in left/top block.
        max_highlights (int): Maximum number of highlight regions/zooms to display.
        highlight_size (int): Size (in pixels) of side of the square region to highlight and zoom in.
        zoom_nrows (int): Number of rows for zoomed-in subplots.
        zoom_ncols (int or None): Number of columns for zoom subplots. If None, computed based on max_highlights/zoom_nrows.
        zoom_subplot_size (int): Width (in inches) of each zoomed-in subplot.
        main_nrows (int): Rows occupied by the main plot in grid (when sharing grid).
        main_ncols (int): Columns occupied by the main plot in grid (when sharing grid).
        linewidth (float): Linewidth for highlight rectangle/crosshair coloring (main plot).
        pow (float): Power (gamma) for colormap scaling (default: 1/2 for sqrt scaling).
        main_padding_v (float): Vertical padding around main plot (when stacked or in shared grid).
        main_padding_h (float): Horizontal padding around main plot (in shared grid layout).
        zoom_hspace (float): Vertical spacing between zoom plots and between main plot and zooms.
        zoom_wspace (float): Horizontal spacing between zoom plots.
        save_path (str or None): If not None, path to which the resulting figure is saved.
        palette (str or matplotlib colormap): Colormap to use for distance map visualization.
        title (str): Title for the main plot.
        highlight_colors (list/str/cmap/callable, optional): Colors for each highlight box/crosshair.
            If None, uses blue/red by sign of effect or 'lime' by default.
        color_by_interaction_score (bool): If True, colors highlights by sign of interaction score.
        white_border (bool): If True, draws a white border and white crosshairs around highlights for contrast.
        white_border_width (int): Width of the white border for highlights (main plot).
        zoom_rect_white_outline (bool): If True, draws a white outline around each zoomed-in square in zooms.
        zoom_rect_white_outline_width (int): Width of the zoomed rectangle outline in zoom subplots.
        zoom_show_ticks (bool): If True, shows tick labels and coordinates in zoom.
        zoom_tick_density (int): Number of tick marks (x & y) to show in each zoom.
        zoom_show_ylabel (bool or list): If bool, applies to all zoom plots. If list, one bool per zoom plot.
            Shows y-axis label on leftmost zoom plot(s) when True (default: True).
        zoom_show_xlabel (bool or list): If bool, applies to all zoom plots. If list, one bool per zoom plot.
            Shows x-axis label on zoom plot(s) when True (default: True).
        zoom_highlight_marker_scale (float): Relative scale for highlight marker size in zoom subplots (default: 1.0).
        zoom_highlight_linewidth (float): Line thickness for color highlight marker in zoom (default: 3).
        zoom_highlight_white_linewidth (float): Line thickness for white outline marker in zoom (default: 5).
        zoom_effect_label_prefix (str): Prefix label for interaction strength in zoom plot titles (default: "Joint Effect").
        figsize (tuple or None): Figure size (width, height) in inches. If None, auto-calculated based on layout.

    Returns:
        dict: A dictionary with one entry for the reference haplotype; each entry contains:
            - 'fig': the matplotlib Figure object
            - 'axes': the main Axes object (main contact map)
            - 'data': dict with 'contact_map' (the plotted matrix)

    Example:
        results = plot_contact_map_and_zooms(
            contact_maps_wt,
            ridge_df,
            max_highlights=9,
            highlight_size=90,
            palette='viridis'
        )
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import matplotlib as mpl

    half_size = highlight_size // 2

    if zoom_ncols is None:
        zoom_ncols = int(np.ceil(max_highlights / zoom_nrows))
    
    # Determine highlight colors
    if highlight_colors is None:
        if color_by_interaction_score:
            # Colors highlights by interaction score sign (blue/red)
            sorted_ridge_df = ridge_df.iloc[:max_highlights].sort_values('clinical_position')
            highlight_colors = []
            for idx, row in sorted_ridge_df.iterrows():
                interaction_score = row.get(zoom_value_col, 0)
                if interaction_score >= 0:
                    highlight_colors.append('blue')
                else:
                    highlight_colors.append('red')
            # Padding if fewer than max_highlights (shouldn't happen)
            while len(highlight_colors) < max_highlights:
                highlight_colors.append('blue')
        else:
            highlight_colors = ['lime'] * max_highlights
    else:
        # Accepts colormap, str, or iterable; pads as needed
        import seaborn as sns
        if callable(highlight_colors):
            cmap = highlight_colors
            highlight_colors = [mpl.colors.to_hex(cmap(i / max_highlights)) for i in range(max_highlights)]
        elif isinstance(highlight_colors, str):
            highlight_colors = [highlight_colors] * max_highlights
        elif hasattr(highlight_colors, "as_hex") and callable(getattr(highlight_colors, "as_hex", None)):
            palette_colors = highlight_colors.as_hex()
            highlight_colors = list(palette_colors) * (max_highlights // len(palette_colors) + 1)
            highlight_colors = highlight_colors[:max_highlights]
        elif hasattr(highlight_colors, "__iter__") and not isinstance(highlight_colors, dict):
            highlight_colors = list(highlight_colors) * (max_highlights // len(highlight_colors) + 1)
            highlight_colors = highlight_colors[:max_highlights]

    ref_key = get_ref_key(contact_maps_wt)

    def get_gamma_cmap(base_cmap, gamma):
        """
        Returns a new colormap that applies a gamma/power correction
        to the input colormap intensity scale.
        """
        base = mpl.cm.get_cmap(base_cmap)
        colors = base(np.linspace(0, 1, 256))
        x = np.linspace(0, 1, 256)
        x_gamma = x**gamma
        from scipy.interpolate import interp1d
        new_colors = np.empty_like(colors)
        for i in range(colors.shape[1]):
            channel = colors[:, i]
            f = interp1d(x, channel, kind='linear')
            new_colors[:, i] = f(x_gamma)
        return mpl.colors.ListedColormap(new_colors)

    gamma_cmap = get_gamma_cmap(palette, pow)

    def plot_contact_map_and_zooms_i(v, hap_id, main_plot_above=main_plot_above, main_padding_v=0.25, main_padding_h=0.25, figsize=figsize):
        """
        Core subroutine: plots one contact map (v) with highlight rects/crosshairs and zooms.
        """
        # Figure/grid setup depending on main_plot_above mode
        if main_plot_above == -1:
            # Main plot is below zooms
            default_figsize = (max(6, zoom_subplot_size * zoom_ncols), 6 + zoom_subplot_size * zoom_nrows + 0.5)
            fig = plt.figure(figsize=figsize if figsize is not None else default_figsize)
            height_ratios = [zoom_subplot_size] * zoom_nrows + [main_padding_v, 6]
            gs = fig.add_gridspec(
                zoom_nrows + 2, zoom_ncols,
                height_ratios=height_ratios,
                hspace=zoom_hspace,
                wspace=zoom_wspace
            )
            ax_main = fig.add_subplot(gs[zoom_nrows + 1, :])
            zoom_subplot_indices = []
            for i in range(max_highlights):
                row_idx = i // zoom_ncols
                col_idx = i % zoom_ncols
                if row_idx < zoom_nrows and col_idx < zoom_ncols:
                    zoom_subplot_indices.append((row_idx, col_idx))
        elif main_plot_above:
            # Main plot above zooms
            default_figsize = (max(6, zoom_subplot_size * zoom_ncols), 6 + zoom_subplot_size * zoom_nrows + 0.5)
            fig = plt.figure(figsize=figsize if figsize is not None else default_figsize)
            height_ratios = [6, main_padding_v] + [zoom_subplot_size] * zoom_nrows
            gs = fig.add_gridspec(
                zoom_nrows + 2, zoom_ncols,
                height_ratios=height_ratios,
                hspace=zoom_hspace,
                wspace=zoom_wspace
            )
            ax_main = fig.add_subplot(gs[0, :])
            zoom_subplot_indices = []
            for i in range(max_highlights):
                row_idx = (i // zoom_ncols) + 2
                col_idx = i % zoom_ncols
                if row_idx <= zoom_nrows + 1 and col_idx < zoom_ncols:
                    zoom_subplot_indices.append((row_idx, col_idx))
        else:
            # Shared grid; main plot occupies upper left, zooms fill remainder to the right
            # Zooms should start at the same vertical level as main plot (row 0)
            # Total rows = max of (main plot rows, zoom rows needed)
            total_nrows = max(main_nrows, zoom_nrows)
            # Add a padding column between main plot and zooms if main_padding_h > 0
            total_ncols = main_ncols + (1 if main_padding_h > 0 else 0) + zoom_ncols
            # Calculate figure size - main plot should be square-ish, zooms to the right
            # Use zoom_subplot_size for height calculation to ensure consistent zoom sizes
            main_plot_width = 6  # Fixed width per column for main plot to prevent squishing
            fig_width = main_plot_width * main_ncols + (main_padding_h if main_padding_h > 0 else 0) + zoom_subplot_size * zoom_ncols
            fig_height = zoom_subplot_size * total_nrows
            default_figsize = (fig_width, fig_height)
            fig = plt.figure(figsize=figsize if figsize is not None else default_figsize)
            if main_padding_h > 0:
                width_ratios = [main_plot_width] * main_ncols + [main_padding_h] + [zoom_subplot_size] * zoom_ncols
            else:
                width_ratios = [main_plot_width] * main_ncols + [zoom_subplot_size] * zoom_ncols
            # Height ratios: all rows should use zoom_subplot_size for consistent zoom plot sizing
            # The main plot will span multiple rows but that's fine - it will just be larger
            height_ratios = [zoom_subplot_size] * total_nrows
            gs = fig.add_gridspec(
                total_nrows, total_ncols,
                width_ratios=width_ratios,
                height_ratios=height_ratios,
                hspace=zoom_hspace,
                wspace=zoom_wspace
            )
            ax_main = fig.add_subplot(gs[0:main_nrows, 0:main_ncols])
            zoom_subplot_indices = []
            for i in range(max_highlights):
                zoom_idx = i
                # Zooms start at row 0, same vertical level as main plot
                row = zoom_idx // zoom_ncols
                # Account for padding column if it exists
                col_offset = main_ncols + (1 if main_padding_h > 0 else 0)
                col = zoom_idx % zoom_ncols + col_offset
                if row < total_nrows and col < total_ncols:
                    zoom_subplot_indices.append((row, col))

        im = ax_main.imshow(v, cmap=gamma_cmap, origin='upper')
        cbar = fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
        cbar.set_label("3D Distance (Ångstroms)")

        # Normalize zoom_show_xlabel and zoom_show_ylabel to lists if they're single bools
        # Use separate variable names to avoid UnboundLocalError
        if isinstance(zoom_show_xlabel, bool):
            zoom_show_xlabel_list = [zoom_show_xlabel] * max_highlights
        elif isinstance(zoom_show_xlabel, (list, tuple)):
            zoom_show_xlabel_list = list(zoom_show_xlabel)
            if len(zoom_show_xlabel_list) < max_highlights:
                # Pad with last value if list is shorter than max_highlights
                last_val = zoom_show_xlabel_list[-1] if zoom_show_xlabel_list else False
                zoom_show_xlabel_list = zoom_show_xlabel_list + [last_val] * (max_highlights - len(zoom_show_xlabel_list))
        else:
            # Fallback: treat as False
            zoom_show_xlabel_list = [False] * max_highlights
        
        if isinstance(zoom_show_ylabel, bool):
            zoom_show_ylabel_list = [zoom_show_ylabel] * max_highlights
        elif isinstance(zoom_show_ylabel, (list, tuple)):
            zoom_show_ylabel_list = list(zoom_show_ylabel)
            if len(zoom_show_ylabel_list) < max_highlights:
                # Pad with last value if list is shorter than max_highlights
                last_val = zoom_show_ylabel_list[-1] if zoom_show_ylabel_list else False
                zoom_show_ylabel_list = zoom_show_ylabel_list + [last_val] * (max_highlights - len(zoom_show_ylabel_list))
        else:
            # Fallback: treat as False
            zoom_show_ylabel_list = [False] * max_highlights

        # Sort highlights for display order
        sorted_ridge_df = ridge_df.iloc[:max_highlights].sort_values('clinical_position')
        
        for i, (idx, row) in enumerate(sorted_ridge_df.iterrows()):
            x = int(row['clinical_position'])
            y = int(row['wt_position'])
            color = highlight_colors[i] if i < len(highlight_colors) else 'lime'

            # Draw white border for contrast behind highlights (rectangle and crosshairs)
            if white_border:
                white_border_rect = patches.Rectangle(
                    (x - half_size, y - half_size),
                    highlight_size, highlight_size,
                    linewidth=white_border_width, edgecolor='white', facecolor='none',
                    zorder=9
                )
                ax_main.add_patch(white_border_rect)
            # Top highlight rectangle
            rect = patches.Rectangle(
                (x - half_size, y - half_size),
                highlight_size, highlight_size,
                linewidth=linewidth, edgecolor=color, facecolor='none',
                zorder=10
            )
            ax_main.add_patch(rect)

            # Draw white crosshair lines under colored crosshairs
            if white_border:
                crosshair_border_width = 3
                ax_main.plot(
                    [-0.5, x - half_size], [y, y],
                    color='white', linewidth=linewidth + crosshair_border_width, linestyle='-',
                    zorder=8, clip_on=False
                )
                ax_main.plot(
                    [x + half_size + 1, v.shape[1] - 0.5], [y, y],
                    color='white', linewidth=linewidth + crosshair_border_width, linestyle='-',
                    zorder=8, clip_on=False
                )
                ax_main.plot(
                    [x, x], [v.shape[0] - 0.5, y + half_size + 1],
                    color='white', linewidth=linewidth + crosshair_border_width, linestyle='-',
                    zorder=8, clip_on=False
                )
                ax_main.plot(
                    [x, x], [y - half_size, -0.5],
                    color='white', linewidth=linewidth + crosshair_border_width, linestyle='-',
                    zorder=8, clip_on=False
                )
            # Draw colored crosshair lines
            ax_main.plot(
                [-0.5, x - half_size], [y, y],
                color=color, linewidth=linewidth, linestyle='-',
                zorder=10, clip_on=False
            )
            ax_main.plot(
                [x + half_size + 1, v.shape[1] - 0.5], [y, y],
                color=color, linewidth=linewidth, linestyle='-',
                zorder=10, clip_on=False
            )
            ax_main.plot(
                [x, x], [v.shape[0] - 0.5, y + half_size + 1],
                color=color, linewidth=linewidth, linestyle='-',
                zorder=10, clip_on=False
            )
            ax_main.plot(
                [x, x], [y - half_size, -0.5],
                color=color, linewidth=linewidth, linestyle='-',
                zorder=10, clip_on=False
            )
            # Remove top and right spines (will be set once after loop)

            # Extract zoom region and pad if at edge
            x_start = x - half_size
            x_end = x + half_size + 1
            y_start = y - half_size
            y_end = y + half_size + 1
            pad_left = max(0, -x_start)
            pad_right = max(0, x_end - v.shape[1])
            pad_top = max(0, -y_start)
            pad_bottom = max(0, y_end - v.shape[0])
            x_start_valid = max(x_start, 0)
            x_end_valid = min(x_end, v.shape[1])
            y_start_valid = max(y_start, 0)
            y_end_valid = min(y_end, v.shape[0])
            zoomed = v[y_start_valid:y_end_valid, x_start_valid:x_end_valid]
            zoomed = np.pad(
                zoomed,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0
            )

            if i < len(zoom_subplot_indices):
                gs_idx = zoom_subplot_indices[i]
                ax_zoom = fig.add_subplot(gs[gs_idx])
                ax_zoom.imshow(zoomed, cmap=gamma_cmap, origin='upper')
                ax_zoom.set_title(
                    f"{row['wt_variant']} | {row['clinical_variant']}\n{zoom_effect_label_prefix}={row[zoom_value_col]:.2f}",
                    fontsize='medium'
                )
                center_x = highlight_size // 2
                center_y = highlight_size // 2

                # Compute size of marker symbol based on highlight size and scaling factor
                s_base = 200
                s = s_base * (zoom_highlight_marker_scale ** 2)

                if white_border:
                    ax_zoom.scatter(
                        [center_x], [center_y],
                        facecolors='none', edgecolors='white',
                        marker='s', s=s,
                        linewidths=zoom_highlight_white_linewidth,
                        zorder=10
                    )
                ax_zoom.scatter(
                    [center_x], [center_y],
                    facecolors='none', edgecolors=color,
                    marker='s', s=s,
                    linewidths=zoom_highlight_linewidth,
                    zorder=11
                )
                
                if zoom_show_ticks:
                    x_start_actual = x - half_size
                    x_end_actual = x + half_size + 1
                    y_start_actual = y - half_size
                    y_end_actual = y + half_size + 1

                    x_ticks = np.linspace(0, highlight_size - 1, zoom_tick_density)
                    x_tick_labels = [str(int(x_start_actual + (x_end_actual - x_start_actual) * tick / (highlight_size - 1))) 
                                   for tick in x_ticks]
                    ax_zoom.set_xticks(x_ticks)
                    ax_zoom.set_xticklabels(x_tick_labels, fontsize=6)

                    y_ticks = np.linspace(0, highlight_size - 1, zoom_tick_density)
                    y_tick_labels = [str(int(y_start_actual + (y_end_actual - y_start_actual) * tick / (highlight_size - 1))) 
                                   for tick in y_ticks]
                    ax_zoom.set_yticks(y_ticks)
                    ax_zoom.set_yticklabels(y_tick_labels, fontsize=6)
                    ax_zoom.tick_params(axis='y', which='major', pad=1)
                    # Use per-plot setting for xlabel
                    if zoom_show_xlabel_list[i]:
                        ax_zoom.set_xlabel("Clinical Variant Position", fontsize=6)
                    # Use per-plot setting for ylabel
                    if zoom_show_ylabel_list[i]:
                        ax_zoom.set_ylabel("WT Variant Position", fontsize=6)
                else:
                    ax_zoom.set_xticks([])
                    ax_zoom.set_yticks([])

                # Outline zoomed region in white, if requested
                if zoom_rect_white_outline:
                    outline_rect = patches.Rectangle(
                        (0, 0),
                        highlight_size, highlight_size,
                        linewidth=zoom_rect_white_outline_width,
                        edgecolor='white',
                        facecolor='none',
                        zorder=11
                    )
                    ax_zoom.add_patch(outline_rect)
                
                # Remove top and right spines from zoom plots
                ax_zoom.spines['top'].set_visible(False)
                ax_zoom.spines['right'].set_visible(False)

        # Remove top and right spines from main plot
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)
        
        ax_main.set_title(f"{title}")
        ax_main.set_xlabel("Clinical Variant Position")
        ax_main.set_ylabel("WT Variant Position")

        # Restrict axis limits/ticks to valid region in contact map
        ax_main.set_xlim(-0.5, v.shape[1] - 0.5)
        ax_main.set_ylim(v.shape[0] - 0.5, -0.5)
        xticks = [tick for tick in ax_main.get_xticks() if 0 <= tick < v.shape[1]]
        yticks = [tick for tick in ax_main.get_yticks() if 0 <= tick < v.shape[0]]
        ax_main.set_xticks(xticks)
        ax_main.set_yticks(yticks)

        # Hide unwanted axes in padding zone if using stack layouts
        if main_plot_above == -1:
            for col in range(zoom_ncols):
                ax_pad = fig.add_subplot(gs[zoom_nrows, col])
                ax_pad.axis('off')
        elif main_plot_above:
            for col in range(zoom_ncols):
                ax_pad = fig.add_subplot(gs[1, col])
                ax_pad.axis('off')
        else:
            # Hide padding column in shared grid layout (no padding row in side-by-side layout)
            if main_padding_h > 0:
                for row in range(total_nrows):
                    ax_pad = fig.add_subplot(gs[row, main_ncols])
                    ax_pad.axis('off')

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, **utils.FIG_SAVE_KWARGS)
        plt.show()
        return {'fig': fig, 'axes': ax_main, 'data': {'contact_map': v}}

    results = {}
    for k, v in contact_maps_wt.items():
        if k == ref_key:
            hap_id = get_haplotype_ids(k)[0]
            results[k] = plot_contact_map_and_zooms_i(
                v, hap_id,
                main_plot_above=main_plot_above,
                main_padding_v=main_padding_v,
                main_padding_h=main_padding_h,
                figsize=figsize
            )

    return results

def test_pae_variance_correlation(
    pae_matrix, 
    variance_map, 
    bin_size=1,
    fit_exponential=False,
    log_x=False,
    log_y=False,
    rasterize_points=False
):
    """
    Test correlation between pae_matrix and variance_map, accounting for NaNs.
    Optionally fit an exponential curve to the data and report pseudo-R2.
    The exponential fit is of the form y = a * exp(b * x), which starts small and shoots up.

    Args:
        pae_matrix (np.ndarray): The PAE matrix.
        variance_map (np.ndarray): The variance map.
        fit_exponential (bool): If True, fit an exponential curve to the data.
        log_x (bool): If True, log-transform the x (PAE) values before analysis.
        log_y (bool): If True, log-transform the y (variance) values before analysis.
        rasterize_points (bool): If True, rasterize the scatter points (not the rest of the plot).
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr, spearmanr

    if pae_matrix is not None and variance_map is not None:
        if bin_size > 1:
            pae_matrix = mc.bin_matrix(pae_matrix, bin_size=bin_size)
            variance_map = mc.bin_matrix(variance_map, bin_size=bin_size)

        # Flatten both matrices for correlation analysis
        pae_flat = pae_matrix.flatten()
        variance_flat = variance_map.flatten()

        # Remove pairs where either value is NaN
        valid_mask = ~np.isnan(pae_flat) & ~np.isnan(variance_flat)
        pae_flat_valid = pae_flat[valid_mask]
        variance_flat_valid = variance_flat[valid_mask]

        # Optionally log-transform x and/or y
        # Only log values > 0, otherwise mask out
        if log_x:
            mask_x = pae_flat_valid > 0
        else:
            mask_x = np.ones_like(pae_flat_valid, dtype=bool)
        if log_y:
            mask_y = variance_flat_valid > 0
        else:
            mask_y = np.ones_like(variance_flat_valid, dtype=bool)
        mask = mask_x & mask_y

        pae_flat_valid = pae_flat_valid[mask]
        variance_flat_valid = variance_flat_valid[mask]

        if log_x:
            pae_flat_valid = np.log(pae_flat_valid)
        if log_y:
            variance_flat_valid = np.log(variance_flat_valid)

        if len(pae_flat_valid) == 0:
            print("No valid (non-NaN) data points to compute correlation.")
        else:
            # Calculate Pearson correlation
            pearson_corr, pearson_p = pearsonr(pae_flat_valid, variance_flat_valid)

            # Calculate Spearman correlation (rank-based)
            spearman_corr, spearman_p = spearmanr(pae_flat_valid, variance_flat_valid)

            # Store the stats values for later display
            stats_text = (
                f"Pearson r: {pearson_corr:.3f} (p={pearson_p:.2e})\n"
                f"Spearman ρ: {spearman_corr:.3f} (p={spearman_p:.2e})"
            )

            # Create scatter plot
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(
                pae_flat_valid, 
                variance_flat_valid, 
                alpha=0.01, 
                s=1,
                rasterized=rasterize_points
            )
            xlabel = 'PAE Matrix Values'
            ylabel = 'Variance Map Values'
            if log_x:
                xlabel = 'log(' + xlabel + ')'
            if log_y:
                ylabel = 'log(' + ylabel + ')'
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # Print correlations in console as before (optional, but not in plot)
            print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
            print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")

            # Start of collecting more stats for in-plot display
            extra_stats_lines = []
            linear_r2_display = ""
            exponential_r2_display = ""
            exp_fit_params_display = ""

            # Add trend line (linear) using scikit-learn
            if len(pae_flat_valid) > 1:
                X = pae_flat_valid.reshape(-1, 1)
                y = variance_flat_valid
                linreg = LinearRegression()
                linreg.fit(X, y)
                y_pred = linreg.predict(X)
                # For a proper line, sort x for plotting
                sort_idx = np.argsort(pae_flat_valid)
                # Do not rasterize the trend line
                ax.plot(
                    pae_flat_valid[sort_idx], 
                    y_pred[sort_idx], 
                    "r--", 
                    alpha=0.8, 
                    label="Linear fit"
                )
                r2 = r2_score(y, y_pred)
                linear_r2_display = f"Linear R²: {r2:.4f}"
                print(f"Linear fit R2: {r2:.4f}")

            # Optionally fit and plot exponential curve, and compute pseudo-R2
            pseudo_r2 = None
            a = None
            b = None
            if fit_exponential and len(pae_flat_valid) > 1:
                # Only fit to positive y values for log-exp fit stability
                # If log_y is True, y is already log-transformed, so skip exponential fit
                if log_y:
                    print("Exponential fit is not meaningful when log_y=True. Skipping exponential fit.")
                else:
                    mask_exp = variance_flat_valid > 0
                    x_fit = pae_flat_valid[mask_exp].reshape(-1, 1)
                    y_fit = variance_flat_valid[mask_exp]
                    if len(x_fit) > 1:
                        # Fit y = a * exp(b * x) <=> log(y) = log(a) + b*x
                        log_y_fit = np.log(y_fit)
                        linreg_exp = LinearRegression()
                        linreg_exp.fit(x_fit, log_y_fit)
                        # Get parameters
                        b = linreg_exp.coef_[0]
                        log_a = linreg_exp.intercept_
                        a = np.exp(log_a)
                        # For plotting, use sorted x range
                        x_line = np.linspace(np.min(x_fit), np.max(x_fit), 200).reshape(-1, 1)
                        y_line = a * np.exp(b * x_line.flatten())
                        # Do not rasterize the exponential curve
                        ax.plot(
                            x_line, 
                            y_line, 
                            "g-", 
                            alpha=0.8, 
                            label=f"Exponential fit (y = a*exp(bx))"
                        )
                        # Calculate pseudo-R2 for the exponential fit
                        y_pred_exp = a * np.exp(b * x_fit.flatten())
                        ss_res = np.sum((y_fit - y_pred_exp) ** 2)
                        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                        pseudo_r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                        exponential_r2_display = f"Exp. fit R²: {pseudo_r2:.4f}"
                        exp_fit_params_display = f"a={a:.3g}, b={b:.3g}"
                        print(f"Exponential fit pseudo-R2: {pseudo_r2:.4f}")
                        print(f"Exponential fit parameters: a={a:.4g}, b={b:.4g}")
                    else:
                        print("Not enough positive y values for exponential fit.")
            
            # Compose annotation string for in-plot display
            lines = [stats_text]
            if linear_r2_display:
                lines.append(linear_r2_display)
            if exponential_r2_display:
                lines.append(exponential_r2_display)
            if exp_fit_params_display:
                lines.append(f"Exp: {exp_fit_params_display}")

            annotation_text = "\n".join(lines)
            # Place annotation box at upper left of axes (top left, with some margin)
            ax.annotate(
                annotation_text,
                xy=(0.01, 0.97),
                xycoords='axes fraction',
                va='top',
                ha='left',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec='grey', lw=0.8)
            )

            ax.set_title(
                f'PAE vs. Variance'
            )

            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.show()

            # Print interpretation
            if abs(pearson_corr) > 0.7:
                strength = "strong"
            elif abs(pearson_corr) > 0.3:
                strength = "moderate"
            else:
                strength = "weak"

            direction = "positive" if pearson_corr > 0 else "negative"
            print(f"\nInterpretation: There is a {strength} {direction} correlation between PAE matrix and variance map values.")

            return {"fig": fig, 'axes': ax, 'data':{'pae_matrix': pae_matrix, 'variance_map': variance_map}}
    else:
        print("Cannot test correlation: pae_matrix or variance_map not available")

def analyze_target_group_contact_loss(
    barplot_df,
    target_group="EAS",
    percent_col="Percent",
    group_col="superpopulation",
    type_col="Type",
    lost_type_value="Lost",
    verbose=True
):
    """
    Analyze contact loss percent for a specified target group vs others,
    with outlier detection via z-score.
    """
    # Select the percent of contacts lost for the target group and for all others
    target_mask = (
        barplot_df[group_col].str.contains(target_group, case=False, na=False)
        & (barplot_df[type_col] == lost_type_value)
    )
    other_mask = (
        ~barplot_df[group_col].str.contains(target_group, case=False, na=False)
        & (barplot_df[type_col] == lost_type_value)
    )

    target_lost = barplot_df[target_mask][percent_col].values
    other_lost = barplot_df[other_mask][percent_col].values

    target_val = float(target_lost[0]) if len(target_lost) > 0 else np.nan
    other_mean = np.mean(other_lost) if len(other_lost) > 0 else np.nan
    fold_change = target_val / other_mean if other_mean != 0 else np.nan

    if verbose:
        print(f"{target_group} lost contact %: {target_val:.8f}")
        print(f"Other populations lost contact % (mean): {other_mean:.8f}")
        print(f"Fold-change ({target_group}/others): {fold_change:.2f}")

    # Outlier detection (treat target group as a single value)
    if len(other_lost) > 1 and not np.isnan(target_val):
        mu = np.mean(other_lost)
        sigma = np.std(other_lost, ddof=1)
        if sigma == 0:
            zscore = 0
            pval = 1.0
        else:
            zscore = abs(target_val - mu) / sigma
            # p-value for a two-sided Gaussian (assuming normality)
            pval = 2 * stats.norm.sf(zscore)

        # If pval is exactly 0, set to sys.float_info.min for reporting.
        pval_to_report = pval if pval != 0 else sys.float_info.min
        p_print = f"{pval_to_report:.2e}" if pval_to_report < 1e-6 else f"{pval_to_report:.6f}"

        if verbose:
            print(f"{target_group} z-score (relative to others): {zscore:.2f}")
            print(f"Outlier p-value (two-sided, normal): {p_print}")

            # Interpret outlier status (using alpha=0.05)
            if pval_to_report < 0.05:
                print(f"{target_group} lost contact % is a statistical outlier compared to the other superpopulations (p < 0.05).")
            else:
                print(f"{target_group} lost contact % is not a statistical outlier compared to the other superpopulations (p >= 0.05).")
    else:
        if verbose:
            print("Not enough comparison groups to perform outlier test or value is missing.")

    if verbose:
        if np.isnan(fold_change):
            print("Unable to compute fold-change due to missing data.")
        else:
            if fold_change > 1.2:
                print(f"{target_group} superpopulation has a higher percent of lost contacts compared to the mean of other superpopulations.")
            elif fold_change < 0.8:
                print(f"{target_group} superpopulation has a lower percent of lost contacts compared to the mean of other superpopulations.")
            else:
                print(f"{target_group} superpopulation is similar to other superpopulations in percent of lost contacts.")
       

    # Optionally, return results for further use
    return {
        "target_val": target_val,
        "other_mean": other_mean,
        "fold_change": fold_change,
        "zscore": zscore if len(other_lost) > 1 and not np.isnan(target_val) else np.nan,
        "pval": pval if len(other_lost) > 1 and not np.isnan(target_val) else np.nan
    }
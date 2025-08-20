import os
import glob
import io
import re
from tkinter import N
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Optional
from contextlib import contextmanager
import sys


def is_pd(x):
    """
    Check if the input object is a pandas DataFrame.

    Parameters:
    x (object): The object to be checked.

    Returns:
    bool: True if the object is a pandas DataFrame, False otherwise.
    """
    return isinstance(x, pd.DataFrame)

def as_list(x,
            type_func=None):
    """
    Convert a string to a list when possible.

    Args:
        x: The input value to be converted to a list.
        type_func: (optional) A function to apply to each element of the list.

    Returns:
        A list containing the converted value(s).
    """
    if is_pd(x):
        return [x]  
    if isinstance(x, type({}.keys())):
        return list(x)
    if isinstance(x, type({}.values())):
        return list(x)
    if isinstance(x, set):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, pd.Series):
        return x.tolist()
    if type_func != None:
        return [type_func(y) for y in x]
    if x == None:
        return x
    if not isinstance(x, list):
        return [x]
        
    return x

def one_only(lst):
    lst = as_list(lst)
    return lst[0]

def intersect(x, y, return_list=True):
    """
    Intersect two lists or sets.

    """
    x = as_list(x)
    y = as_list(y)

    if return_list:
        return list(set(x) & set(y))
    else:
        return set(x) & set(y)

def list_vcf(dir='/grid/koo/home/schilder/projects/GenomeEncoder/data/1KG/vcf_formatted/*.vcf.gz', 
             as_dict=False):
    # List all VCF files in the 1KG directory 
    vcf_files = glob.glob(dir)
    print(len(vcf_files),"VCF files found.")
    if as_dict:
        chroms_vcf = [os.path.basename(vcf_file).split(".")[1] for vcf_file in vcf_files]
        chroms_dict = dict(zip(chroms_vcf, vcf_files))
        return chroms_dict
    else:
        return vcf_files



def add_codon_buffer(seq, codon_buffer='N'):
    if codon_buffer is not None:
        seq += codon_buffer * (3 - len(seq) % 3)
    return seq

def get_chr_map(save_path=None,#'"../data/chr_name_conv.txt"'
                chroms_only=False,
                ):
    import pandas as pd
    # Create map
    chr_map = """
    1 chr1
    2 chr2
    3 chr3
    4 chr4
    5 chr5
    6 chr6
    7 chr7
    8 chr8
    9 chr9
    10 chr10
    11 chr11
    12 chr12
    13 chr13
    14 chr14
    15 chr15
    16 chr16
    17 chr17
    18 chr18
    19 chr19
    20 chr20
    21 chr21
    22 chr22
    X chrX
    """
    # 23 chrX
    # 24 chrY
    # 25 chrXY
    # 26 chrM
    # write map to file
    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(chr_map)
        # Import chromosomes
        df = pd.read_csv(save_path, sep=" ", header=None)
    else:
        # Read directly from string
        df = pd.read_csv(chr_map, header=None)
    if chroms_only:
        chrs = df[1].tolist()
        chrs.reverse()
        return chrs
    else:
        return df
    
def download_url(args): 
    import time
    import requests
    t0 = time.time() 
    url, fn = args[0], args[1] 
    if os.path.exists(fn):
        print("Already exists:",fn)
    else:
        try: 
            r = requests.get(url) 
            with open(fn, 'wb') as f: 
                f.write(r.content) 
                print('url:', url, 'time (s):',time.time())
                return url 
        except Exception as e: 
            print('Exception in download_url():', e)

def download_parallel(args): 
    from os import cpu_count
    from multiprocessing.pool import ThreadPool
    cpus = cpu_count() 
    results = ThreadPool(cpus - 1).imap_unordered(download_url, args)  
    return results

def find_file(name='genomic.gff',
              dir='./'):
    import os
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file == name:
                return os.path.join(root, file)
    return None
        

def find_col(df, prefix, rename=False):
    idx = [i for i in range(0,df.shape[1]-1) if str(df.iloc[:,i].values[0]).startswith(prefix)][0]
    if rename:
        df.rename(columns={df.columns[idx]:prefix}, inplace=True)
        return prefix
    else:
        return idx
    
def name_enst(df, prefix="ENST"):
    find_col(df, prefix, rename=True)


# for each proteoform, come up with a unique ID based on the position and identity of each amino acid 
# range(len(protein_seq))
# map each amino acid to a number
def create_proteoform_id(transcript_id,
                         aa_seq,
                         alphabet=None,
                         extra_items=['*']):
    if alphabet is None:
        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    token_map = {aa:i for i,aa in enumerate(set(alphabet.all_toks+extra_items))}
    return f"{transcript_id}_proteoform{sum([token_map[x]*i for i,x in enumerate(str(aa_seq))])}"

def filter_matrix(X):
    print("Filtering matrix")
    import torch
    # Get indicies of rows in X that have Nan
    nan_indices = torch.isnan(X).any(dim=1)
    print("Matrix shape before filtering:",X.shape)
    # Remove rows with Nan
    X = X[~nan_indices]
    print("Matrix shape after filtering:",X.shape)
    return X, nan_indices

def run_umap(X,
             verbose = True,
             filter=True,
             random_state=42,
             **kwargs):
    if filter is True:
        X, nan_indices = filter_matrix(X)
    else:
        nan_indices = None
    if verbose:
        print("Running UMAP")
    import umap
    model = umap.UMAP(verbose=verbose,
                       random_state=random_state,
                       **kwargs)
    model.fit(X)
    embedding = model.transform(X)
    return model, embedding, nan_indices

def run_tsvd(X,
             n_components=2,
             filter=True,
             verbose=True,
             **kwargs):
    
    if filter is True:
        X, nan_indices = filter_matrix(X)
    else:
        nan_indices = None
    if verbose:
        print("Running TSVD")
    import sklearn
    model = sklearn.decomposition.TruncatedSVD(n_components=n_components,
                                              **kwargs)
    model.fit(X)
    embedding = model.transform(X)
    return model, embedding, nan_indices

def run_lda(X,
            n_components=2,
            filter=True,
            verbose=True,
            **kwargs):
    
    if filter is True:
        X, nan_indices = filter_matrix(X)
    else:
        nan_indices = None
    if verbose:
        print("Running LDA")
    import sklearn
    model = sklearn.decomposition.LatentDirichletAllocation(n_components=n_components,
                                                            **kwargs)
    model.fit(X)
    embedding = model.transform(X)
    return model, embedding, nan_indices

def run_ica(X,
            n_components=2,
            filter=True,
            verbose=True,
            **kwargs):
    if filter is True:
        X, nan_indices = filter_matrix(X)
    else:
        nan_indices = None
    if verbose:
        print("Running ICA")
    import sklearn
    model = sklearn.decomposition.FastICA(n_components=n_components,
                                            **kwargs)
    model.fit(X)
    embedding = model.transform(X)
    return model, embedding, nan_indices

### The rpca package is currently broken due to incompatibility an python 3.9.5 version of numpy version...
## Also causes A LOT of problems with the environment when you try to install it.
## Downgrading to numpy 1.22.4 fixes the issue:
#### pip uninstall numpy -y
#### pip install numpy==1.22.4

# def run_rpca(X,
#              n_components=2,
#              filter=True,
#              verbose=True,
#              **kwargs):
#     if filter is True:
#         X, nan_indices = filter_matrix(X)
#     else:
#         nan_indices = None
#     if verbose:
#         print("Running RPCA")
#     import rpca
#     model = rpca.RobustPCA(n_components=n_components,
#                            **kwargs)
#     model.fit(X)
#     embedding = model.transform(X)
#     return model, embedding, nan_indices

def plot_density(df,
                 x='UMAP_1',
                 y='UMAP_2',
                 alpha=0.5,
                 color='white',
                 s=1,
                 facecolor='#2F0154',
                 cmap='viridis',
                 fill=True,
                 add_density=True,
                 figsize=(10, 8),
                 **kwargs):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=figsize)  # Set figure size
    
    if facecolor is not None:
        plt.gca().set_facecolor(facecolor) # Set background to slightly darker than darkest viridis color
    
    if add_density:
        sns.kdeplot(data=df, x=x, y=y, 
                    fill=fill, 
                    cmap=cmap,
                    **kwargs
                    ) 
    # Then overlay scatter points with some transparency
    plt.scatter(data=df,
                x=x, y=y,
                alpha=alpha, # Make points semi-transparent
                s=s, # Small point size
                color=color) # White points
    
def is_VariantFile(x):
    import pysam
    return isinstance(x, pysam.VariantFile)

def save_pickle(obj, 
                save_path,
                verbose=True):
    """
    Save an object to a pickle file.
    """
    if save_path is not None:
        import pickle
        import os
        if verbose:
            print(f"Saving ==> {save_path}")
        if os.path.dirname(save_path) != "":
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(obj, f)

def load_pickle_progress(filename, 
                         **kwargs):
    """
    Load a pickle file with a progress bar.
    
    Parameters:
        filename (str): Path to pickle file
        **kwargs: Additional arguments passed to tqdm
        
    Returns:
        Unpickled object
    """
    import pickle
    from tqdm import tqdm
    
    with open(filename, "rb") as f:
        # Get file size for progress bar
        file_size = f.seek(0, 2)
        f.seek(0)
        
        # Read entire file content first
        with tqdm(total=file_size,
                 unit='B',
                 unit_scale=True,
                 desc=f"Loading '{filename}'",
                 **kwargs) as pbar:
            content = f.read()
            pbar.update(file_size)
            
        # Then unpickle the complete content
        return pickle.loads(content)

def load_pickle(save_path,
                force=False,
                verbose=True,
                progress=True,
                **kwargs):
    """
    Load an object from a pickle file with a real-time progress bar.

    Parameters:
        save_path (str): Path to the pickle file.
        force (bool): If True, forces loading even if the file exists.
        verbose (bool): If True, prints loading status.

    Returns:
        The unpickled object or None.
    """
    import pickle
    import os
    from tqdm import tqdm

    if save_path is not None:
        if not isinstance(save_path, str):
            raise ValueError(f"save_path must be a string, not {type(save_path)}")
        if os.path.exists(save_path) and not force:
            if verbose:
                print(f"Loading ==> {save_path}") 
            try:
                if progress:
                    obj = load_pickle_progress(save_path, 
                                                    **kwargs)
                else:
                    with open(save_path, 'rb') as f:
                        obj = pickle.load(f)
                return obj
            except Exception as e:
                print(f"Failed to load pickle file: {e}")
                return None
    return None

def save_json(obj,
              save_path,
              verbose=True,
              compress=True):
    """
    Save an object to a JSON file with options to minimize size by removing whitespace
    and compressing the file.
    
    Parameters:
        obj (dict): The dictionary to save as JSON.
        save_path (str): The file path where the JSON will be saved.
        verbose (bool): If True, prints the saving status.
        compress (bool): If True, compresses the JSON file using gzip.
    """    
    if save_path is not None:
        import json
        import os 
        # Validate the obj
        if not isinstance(obj, dict):
            raise ValueError(f"obj must be a dictionary, not {type(obj)}")
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if verbose:
            print(f"Saving ==> {save_path}{' with compression' if compress else ''}")
        # Ensure save_path ends with .json or .json.gz
        if not save_path.endswith('.json') and not save_path.endswith('.json.gz'):
            save_path = save_path + '.json'
        if compress:
            import gzip
            # Ensure save_path ends with .gz
            if not save_path.endswith('.gz'):
                save_path = save_path + '.gz'
            with gzip.open(save_path, 'wt', encoding='utf-8') as f:
                # Use separators to eliminate unnecessary whitespace
                json.dump(obj, f, separators=(',', ':'))
        else:
            with open(save_path, 'w') as f:
                # Use separators to eliminate unnecessary whitespace
                json.dump(obj, f, separators=(',', ':'))

def load_json(save_path,
              error=True,
              verbose=True,
              force=False):
    """
    Load an object from a JSON file, supporting both compressed (.gz) and uncompressed files.

    Parameters:
        save_path (str): Path to the JSON file. Can be a .json or .json.gz file.
        verbose (bool): If True, prints loading status.

    Returns:
        dict: The loaded JSON object.
    """
    import json
    import os
    import gzip
    if save_path is not None:
        if not os.path.exists(save_path):
            if verbose:
                print(f"File does not exist: {save_path}")
            return None 
        elif force:
            return None
        else:
            if verbose:
                print(f"Loading ==> {save_path}")
        # Determine if the file is compressed based on the file extension
        _, file_ext = os.path.splitext(save_path)
        if file_ext == '.gz':
            open_func = gzip.open
            mode = 'rt'  # Read text mode
        else:
            open_func = open
            mode = 'r'

        try:
            with open_func(save_path, mode, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            if error:
                raise e
            else:
                print(f"Failed to load JSON file: {e}")
                return None
    return None


def save_torch(obj, 
                save_path,
                verbose=True):
    """
    Save an object to a torch file (.pth).
    """
    if save_path:
        import torch
        import os
        if verbose:
            print(f"Saving ==> {save_path}")
        if os.path.dirname(save_path) != "":
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(obj, save_path)


def load_torch(save_path,
               verbose=True):
    """
    Load an object from a torch file (.pth).
    """
    import torch
    if save_path is not None:
        if not os.path.exists(save_path):
            if verbose:
                print(f"File does not exist: {save_path}")
            return None
        else:
            if verbose:
                print(f"Loading ==> {save_path}")
            return torch.load(save_path)
    else:
        return None

def save_vcf(recs, save_path, header, mode="wb", index=True):
    import pysam
    from tqdm.auto import tqdm
    if save_path is not None:
        # Create index file if it doesn't exist
        with pysam.VariantFile(save_path, 
                               mode=mode,
                               header=header) as vcf_out:
            for rec in tqdm(recs,
                            desc=f"Saving VCF ==> {save_path}"):
                vcf_out.write(rec)
        if index:
            index_vcf(save_path)
                
def sort_variants(recs,
                  **kwargs):
    from tqdm.auto import tqdm
    if len(recs) > 0:
        return sorted(tqdm(recs, desc="Sorting variants", **kwargs),
                      key=lambda rec: (rec.contig, rec.start))
    else:
        return recs

def index_vcf(save_path,
              force=True):
    """
    Index a VCF file.
    See: https://pysam.readthedocs.io/en/latest/api.html#pysam.tabix_index
    """
    import pysam
    pysam.tabix_index(filename=save_path, 
                      preset="vcf", force=force,
                      index=save_path+".tbi")
    
def invert_dict(d):
    return {v:k for k,v in d.items()}

def _create_polygon_marker(n_sides, equal_edges=True):
    """Create a custom polygon marker with n sides
    
    Args:
        n_sides: Number of sides for the polygon
        equal_edges: If True, creates a regular polygon with equal edge lengths.
                    If False, creates a polygon with vertices evenly spaced around circle.
    """
    from matplotlib.path import Path
    import matplotlib.path as mpath
    import numpy as np

    if equal_edges:
        # Create regular polygon vertices with equal edge lengths
        theta = np.linspace(0, 2*np.pi, n_sides+1)[:-1]  # n points + close
        radius = 1.0  # Unit circle
        verts = radius * np.column_stack((np.cos(theta), np.sin(theta)))
    else:
        # Create polygon vertices evenly spaced around circle
        theta = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
        radius = 1.0
        verts = radius * np.column_stack((np.cos(theta), np.sin(theta)))
    
    # Add closing vertex
    verts = np.vstack((verts, verts[0]))  # Close the polygon
    codes = [Path.MOVETO] + [Path.LINETO]*(n_sides-1) + [Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _create_donut_marker(thickness=0.6):
    """Create a donut marker
    
    Args:
        thickness: Float between 0 and 1 controlling the thickness of the donut.
                  Higher values = thicker donut (smaller inner circle).
                  Default is 0.6.
    """
    from matplotlib.path import Path
    import matplotlib.path as mpath
    import numpy as np
    
    # Create outer circle vertices
    theta = np.linspace(0, 2*np.pi, 100)
    outer_verts = np.column_stack((np.cos(theta), np.sin(theta)))
    
    # Create inner circle vertices (smaller radius)
    inner_verts = (1-thickness) * np.column_stack((np.cos(theta), np.sin(theta)))
    
    # Combine vertices - go around outer circle then inner circle in reverse
    verts = np.vstack((outer_verts, inner_verts[::-1]))
    
    # Create path codes
    codes = ([Path.MOVETO] + [Path.LINETO]*(len(outer_verts)-1) + 
            [Path.MOVETO] + [Path.LINETO]*(len(inner_verts)-1))
    
    return mpath.Path(verts, codes)

def _create_semicircle_marker():
    """Create a semicircle marker"""
    from matplotlib.path import Path
    import matplotlib.path as mpath
    import numpy as np

    theta = np.linspace(0, np.pi, 100)
    verts = np.column_stack((np.cos(theta), np.sin(theta)))
    codes = [Path.MOVETO] + [Path.LINETO]*(len(verts)-2) + [Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def get_marker_map(n=8, 
                   subset=None, 
                   n_plus_marker="*",
                   is_ref_marker=False):
    """Get a map of markers for a given range of integers"""
    if is_ref_marker:
        marker_map = {
            True: _create_donut_marker(),
            False: "o"
        }
    else:
        marker_map = {
            0:_create_donut_marker(), 
            1:'o', 
            2:'X',
            3:'^', 
            4:'D', 
            5:'p', 
            6:'H', 
            7:_create_polygon_marker(7), 
            8:'8'
            }
    if n > 8 and not is_ref_marker:
        for i in range(9, n+1):
            if n_plus_marker is not None:
                marker_map[i] = n_plus_marker
            else:
                marker_map[i] = _create_polygon_marker(i)
    if subset is not None:
        marker_map = {k:v for k,v in marker_map.items() if k in subset}
    return marker_map


def _count_edits_indel(variant_str,
                          string="del"):
    # Indels with numbers
    if "{" in variant_str:
        # Add numbers in del{#}/ins{#}
        import re
        numbers = [int(n) for n in re.findall(f'{string}' + r'\{(\d+)\}', variant_str)]
        return sum(numbers)
    # Indels without numbers
    else:
        return variant_str.count(string)

def count_edits(lst,
                tx_id_sep=":",
                search_strings=['>','del','ins'], 
                as_dict=False):
    """
    Count the number of edits (number of differences relative to the reference sequence) in a list of haplotype names.
    
    Args:
        lst (list): List of haplotype names.
        tx_id_sep (str): Separator between transcript ID and variant string.
        search_strings (list): List of strings to search for in the variant string.
        as_dict (bool): If True, return a dictionary with the haplotype names as keys and the edit distances as values.

    Returns:
        If as_dict is False, returns a list of edit distances.

    Example:
    >>> count_edits(['ENSP00000350283:723N>D,871P>L,1183K>R,1561T>I,1613S>G,123del{22}'])
    """
    
    lst = process_ids(lst, 
                      unique=as_dict)
    non_indel_strings = [x for x in search_strings if x not in ['del', 'ins']]
    counts = []
    for x in lst:
        variants = x.split(tx_id_sep)[1].split(",")
        total = 0
        for variant_str in variants:
            # Count each variant string 
            # INDELS
            if "del" in search_strings and "del" in variant_str:
                total += _count_edits_indel(variant_str, "del")
            elif "ins" in search_strings and "ins" in variant_str:
                total += _count_edits_indel(variant_str, "ins")
            # Truncating mutations (already counted in INDELS)
            elif "*" in variant_str:
                continue
            # SUBSTITUTIONS
            else:
                for c in non_indel_strings:
                    total += variant_str.count(c)
        counts.append(total)
    if as_dict:
        return {x:y for x,y in zip(lst, counts)}
    else:
        return counts
    
def add_edits(df,
              haplotype_col="haplotype",
              **kwargs):
    """
    Add the edit distance to the list of variants.
    """
    edits_dict = count_edits(df[haplotype_col], as_dict=True, **kwargs)
    df['edits'] = df[haplotype_col].map(edits_dict)
    return df

def get_aa_tokens(as_dict=False):
    import esm
    toks = esm.pretrained.esm.constants.proteinseq_toks['toks']
    if as_dict:
        return {tok:i for i,tok in enumerate(toks)}
    else:
        return toks

def as_checksum(text, algorithm='md5'):
    """
    Generates a checksum for a given string using the specified algorithm.

    Args:
        text: The string to generate the checksum for.
        algorithm: The hashing algorithm to use (e.g., 'md5', 'sha256'). Defaults to 'md5'.

    Returns:
        The checksum as a hexadecimal string.
    """
    import hashlib
    encoded_text = text.encode('utf-8')
    if algorithm == 'md5':
        checksum = hashlib.md5(encoded_text).hexdigest()
    elif algorithm == 'sha256':
        checksum = hashlib.sha256(encoded_text).hexdigest()
    else:
        raise ValueError("Unsupported algorithm. Choose 'md5' or 'sha256'.")
    return checksum


def ids_to_checksum_filename(ids: List[str],
                             dir: Optional[str] = None,
                             suffix: Optional[str] = None,
                             sep: str = "_"):
    """
    Get the checksum for a list of IDs.
    """
    checksum = as_checksum(sep.join(process_ids(ids)))
    if suffix is not None:
        checksum = checksum + suffix
    if dir is not None:
        checksum = os.path.join(dir, checksum)
    return checksum

def encode_haplotype_name(seq_name, 
                          encode_haplotype_name_threshold=None,
                          include_counts=True):
    """
    Encode haplotype names that are far too long.

    Example:
        seq_name=  'ENSP00000382423:906V>I,948T>N,949del{12},962del{20},983delLS,986del{17},1004del{13},1018del{14},1033C>G,1034del{5},1040del{16},1057del{4},1062delPS,1065P>Q,1068del{5},1074delGDP,1078del{27},1106del{7},1114delTPV,1118del{4},1123del{5},1129delN,1131S>F,1133del{6},1140delMP,1143del{6},1150del{8},1159E>I,1160delKAE,1164del{7},1172del{24},1197delA,1199delQDA,1203delPI,1206del{6},1213del{11},1225ET>FF,1227del{27},1255delSSC,1259del{45},1305delN,1307I>C,1308del{5},1314delCEK,1318del{7},1326del{7},1334del{96},1431del{19},1451del{31},1483del{5},1490del{23}'
        encode_haplotype_name(seq_name)
    """ 

    if encode_haplotype_name_threshold is not None:
        if seq_name.count(",")<=encode_haplotype_name_threshold:
            return seq_name
    # Encode entire sequence name
    checksum = as_checksum(seq_name)

    if include_counts:
        # Split the sequence name by ':'
        tx_id, mut_str = seq_name.split(":")
        # Create a dictionary to store the mutation positions
        del_count = mut_str.count("del")
        ins_count = mut_str.count("ins")
        sub_count = mut_str.count(">") 
    
        # Construct the encoded name
        return f"{tx_id}:{del_count}del|{ins_count}ins|{sub_count}sub|md5:{checksum}"
    else:
        return f"{tx_id}:md5:{checksum}"
    
def decode_haplotype_name(seq_name, original_names=None, verbose=False):
    """
    Attempt to reverse the encoding process and recover the original haplotype name.

    Args:
        seq_name: The encoded haplotype name string.
        original_names: Optional. A list or dict of original haplotype names to search for a matching checksum.
                        If a dict, keys should be checksums and values the original names.
                        If a list, will compute checksums for all and match.

    Returns:
        The original haplotype name if found in original_names, else None.
        If the input is not encoded, returns the input as is.
    """
    # If not encoded, just return as is
    if "md5:" not in seq_name:
        return seq_name

    # Extract checksum
    try:
        checksum = seq_name.split("md5:")[-1]
        checksum = checksum.strip()
    except Exception:
        return None

    # If original_names is provided, try to match
    if original_names is not None:
        # If dict, assume keys are checksums
        if isinstance(original_names, dict):
            return original_names.get(checksum, None)
        # If list, compute checksums and match
        else:
            for name in original_names:
                if as_checksum(name) == checksum:
                    return name
            return None
    else:
        print("No way to recover original name without a mapping")
        return None
 

def get_candidate_proteins():
    candidate_proteins = pd.read_csv(io.StringIO("""Gene	Protein Common Name	RefSeq Protein	RefSeq Transcript	ENSP
    G6PD	Glucose-6-phosphate 1-dehydrogenase	NP_001346945.1	NM_001360016.2	ENSP00000377192
    HFE	Homeostatic Iron Regulator	NP_000401.1	NM_000410.4	ENSP00000417404
    CFTR	Cystic Fibrosis Transmembrane Conductance Regulator	NP_000483.3	NM_000492.4	ENSP00000003084 
    BRCA1	Breast Cancer Type 1 Susceptibility Protein	NP_009225.1	NM_007294.4	ENSP00000350283
    BRCA2	Breast Cancer Type 2 Susceptibility Protein	NP_000050.3	NM_000059.4	ENSP00000369497
    F5	Coagulation Factor V	NP_000121.2	NM_000130.5	ENSP00000356771
    HBB	Hemoglobin Subunit Beta	NP_000509.1	NM_000518.5	ENSP00000333994
    SERPINA1	Alpha-1 Antitrypsin	NP_000286.3	NM_000295.5	ENSP00000376802
    LDLR	Low-Density Lipoprotein Receptor	NP_000518.1	NM_000527.5	ENSP00000454071
    KCNQ1	Potassium Voltage-Gated Channel Subfamily Q Member 1	NP_000209.2	NM_000218.3	ENSP00000155840
    KCNH2	Potassium Voltage-Gated Channel Subfamily H Member 2	NP_000229.1	NM_000238.4	ENSP00000262186
    """), sep='\t')
    candidate_proteins['RefSeq Protein Stable'] = candidate_proteins['RefSeq Protein'].str.split(".").str[0]
    candidate_proteins['Gene'] = candidate_proteins['Gene'].str.strip()
    return candidate_proteins
    
def make_palette(values,
                 palette,
                 n_colors=None):
    """
    Create a color palette mapping values to colors.
    
    Parameters
    ----------
    values : list
        List of values to map colors to
    palette : str
        Name of seaborn color palette to use
    n_colors : int, optional
        Number of colors to sample from palette. If None, uses len(values)
    
    Returns
    -------
    dict
        Dictionary mapping values to hex colors
    """
    values = as_list(values) 
    n = n_colors if n_colors is not None else len(values)
    return dict(zip(values, sns.color_palette(palette, n_colors=n).as_hex()))

def get_clinsig_palette(values=['path', 'likely_path', 'likely_benign', 'benign'],
                         palette='bwr_r'):
    return make_palette(values, palette) 


def get_superpop_palette(values=['AFR', 'AMR', 'EAS', 'EUR', 
                                 'SAS', "CSA", "MID", "OCE"],
                        palette='Set3'):
    cmap = make_palette(values, palette)
    cmap["REF"] = "grey"
    return cmap

def get_ref_nonref_palette():
    cmap = {"REF": "grey", 
            "non-REF": "mediumslateblue", 
            "All": "darkslateblue"}
    return cmap

def list_to_df(lst,
               cols=None):
    if cols is None:
        cols = ['item']
    return pd.DataFrame(lst, columns=cols)

def process_ids(ids: List[str],
                remove_na: bool = True,
                sort: bool = True,
                unique: bool = True):
    """
    Process a list of IDs.
    """
    ids = as_list(ids)
    if remove_na:
        ids = [x for x in ids if x is not None and pd.notna(x)]
    if unique:
        ids = list(set(ids))
    if sort:
        ids = sorted(ids)
    return ids

def ids_to_checksum(ids: List[str],
                    sep: str = ""):
    """
    Get the checksum for a list of IDs.
    """
    return as_checksum(sep.join(process_ids(ids)))


def most_startswith(lst: List[str],
                    prefix: str, 
                    threshold: float = 0.5):
    """
    Get the most common prefix of a list of strings.
    """
    lst = process_ids(lst)
    assert len(lst) > 0, "List is empty"
    starts_with = [x.startswith(prefix) for x in lst]
    assert len(starts_with) > 0, "List of starts_with is empty"
    return sum(starts_with) / len(starts_with) > threshold


def sort_by_reverse_string(df, 
                           column, 
                           ascending=False,
                           extra_sort_cols=[]):
    """Sort a dataframe by the reverse of strings in a column.
    
    Args:
        df (pd.DataFrame): DataFrame to sort
        column (str): Column name containing strings to sort by
        extra_sort_cols (list): Additional columns to sort by
        
    Returns:
        pd.DataFrame: Sorted dataframe
        
    Example:
        >>> df = pd.DataFrame({'col': ['abc', 'def', 'ghi']})
        >>> sort_by_reverse_string(df, 'col')
        # Returns dataframe sorted by ['cba', 'fed', 'ihg']
    """
    # Create temporary column with reversed strings
    df = df.copy()
    df['_temp_rev'] = df[column].apply(lambda x: str(x)[::-1])
    
    # Sort by reversed strings and drop temp column
    df = df.sort_values(['_temp_rev']+extra_sort_cols, ascending=ascending).drop('_temp_rev', axis=1)
    
    return df



def get_sequence_similarity(ref_seq, 
                            query_seq, 
                            method="Levenshtein.ratio"):
                        
    import numpy as np
    if method == "list":
        # Get max length and pad shorter sequence with spaces which will count as mismatches
        if isinstance(ref_seq, str):
            ref_seq = list(ref_seq)
        if isinstance(query_seq, str):
            query_seq = list(query_seq)
        max_len = max(len(ref_seq), len(query_seq))
        ref_seq = ref_seq + [' '] * (max_len - len(ref_seq))
        query_seq = query_seq + [' '] * (max_len - len(query_seq))
        # Convert sequences to NumPy arrays
        arr1 = np.array(ref_seq, dtype='U1')
        arr2 = np.array(query_seq, dtype='U1')
        matches = np.sum(arr1 == arr2)
        return matches / max_len 
    elif method == "str": 
        if isinstance(ref_seq, list):
            ref_seq = "".join(ref_seq)
        if isinstance(query_seq, list):
            query_seq = "".join(query_seq)
        ref_array = np.frombuffer(ref_seq.encode(), dtype='S1')
        query_array = np.frombuffer(query_seq.encode(), dtype='S1')
        matches = np.sum(ref_array == query_array)
        return matches / len(ref_seq)
    elif method == "Levenshtein.ratio":
        from Levenshtein import ratio
        return ratio(ref_seq, query_seq)
    elif method == "Levenshtein.setratio":
        from Levenshtein import setratio
        if isinstance(ref_seq, str):
            ref_seq = list(ref_seq)
        if isinstance(query_seq, str):
            query_seq = list(query_seq)
        return setratio(ref_seq, query_seq)
    else:
        raise ValueError(f"Invalid method: {method}")
    

def set_seeds_torch(seed, deterministic=True):
    """Set random seeds for reproducibility across different libraries.
    
    Args:
        seed (int): Random seed value
        deterministic (bool): Whether to enforce deterministic behavior in PyTorch
    """ 
    import torch
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def check_arg(func,
              arg,
              arg_index,
              max_args=None):
    options = func.__defaults__[arg_index]
    if max_args is not None:
        arg = as_list(arg)[:max_args]
        if max_args==1:
            arg = one_only(arg)
    assert arg in options, f"Invalid argument: '{arg}'. Must be one of {','.join(options)}"
    return arg

@contextmanager
def ignore_stderr():
    """
    Ignore stderr output.

    Example:
        with ignore_stderr():
            ## Call code that calls tqdm
    """
    devnull = open(os.devnull, "w")
    stderr = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = stderr

@contextmanager
def ignore_stdout(disable=False):
    """
    Ignore stdout output.
    """
    if disable:
        yield
    else:
        devnull = open(os.devnull, "w")
        stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = stdout    

def scan_pth_file(file_path):
    """
    Scan a PyTorch .pth file without loading it into memory.
    Args:
        file_path (str): Path to the PyTorch .pth file
    Returns:
        list: List of keys in the .pth file
    """
    try:
        import torch
        checkpoint = torch.load(file_path, 
                                map_location=torch.device('cpu'), 
                                mmap=True, 
                                weights_only=True)
        return checkpoint.keys()
    except Exception as e:
        print(f"Error scanning {file_path}: {e}")
        return None

def split_batches(lst,
                  batch_size,
                  verbose=True):
    """
    Split a list into batches of a given size.
    """
    
    batches = [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]
    if verbose:
        print(f"Split {len(lst)} samples into {len(batches)} batches of ~{batch_size}")
    return batches




def sort_by_clinsig(df,
                    clinsig_col='clinsig',
                    clinsig_order=get_clinsig_palette().keys(),
                    ascending=True
                    ):
    """
    Sort a DataFrame by clinical significance values in a specified order.
    
    Args:
        df (pd.DataFrame): Input DataFrame to sort
        clinsig_col (str): Name of column containing clinical significance values
        clinsig_order (list): List of clinical significance values in desired order
        ascending (bool): Whether to sort in ascending order
        
    Returns:
        pd.DataFrame: Sorted DataFrame
        
    Example:
        >>> df = pd.DataFrame({'clinsig': ['Pathogenic', 'Benign', 'VUS']})
        >>> sort_by_clinsig(df, clinsig_order=['Benign', 'VUS', 'Pathogenic'])
           clinsig
        1   Benign
        2      VUS
        0  Pathogenic
    """
    return df.sort_values(
        by=clinsig_col,
        key=lambda x: x.map({k: i for i, k in enumerate(list(clinsig_order))}),
        ascending=ascending
    )


def variants_to_positions(df,
                         variant_col='wt_variant',
                         position_col='wt_variant_position',
                         ref_col='REF',
                         alt_col='ALT'):
    """
    Extract position numbers from variant strings and add them as a new column.
    Handles two variant encoding schemas:
    1. Position-based (e.g., '123A>G', '456C>T')
    2. Amino acid change (e.g., 'P1150S', 'R1234G')
    
    Args:
        df (pd.DataFrame): Input DataFrame containing variant information
        variant_col (str): Name of column containing variant strings (default: 'variant')
        position_col (str): Name of column to store extracted positions (default: 'variant_position')
        
    Returns:
        pd.DataFrame: DataFrame with added position column
        
    Example:
        >>> df = pd.DataFrame({'variant': ['123A>G', 'P1150S', 'REF']})
        >>> variants_to_positions(df)
           variant  variant_position
        0  123A>G              123
        1  P1150S             1150
        2     REF              NaN
    """
    import re

    if isinstance(df, list):
        df = pd.DataFrame({variant_col: df})

    def extract_position(variant):
        if variant == 'REF':
            return None
        # Try position-based format first (e.g., '123A>G')
        pos_match = re.search(r'^(\d+)', variant)
        if pos_match:
            return int(pos_match.group(1))
        # Try amino acid change format (e.g., 'P1150S')
        aa_match = re.search(r'[A-Z](\d+)[A-Z*]', variant)
        if aa_match:
            return int(aa_match.group(1))
        return None

    def extract_ref_alt_with_stop(variant):
        if variant == 'REF':
            return None, None

        # Handle deletion format (e.g., "616delS")
        del_match = re.search(r'(\d+)del([A-Z]+)', variant)
        if del_match:
            return del_match.group(2), ''  # REF is the deleted sequence, ALT is empty

        # Handle standard substitution format (e.g., "123A>G")
        sub_match = re.search(r'(\d+)([A-Z]+)>([A-Z*]+)', variant)
        if sub_match:
            return sub_match.group(2), sub_match.group(3)

        # Handle amino acid change format (e.g., "P1150S" or "Q12*")
        aa_match = re.search(r'([A-Z])(\d+)([A-Z*])', variant)
        if aa_match:
            return aa_match.group(1), aa_match.group(3)

        return None, None

    # Extract positions for unique variants
    positions = {variant: extract_position(variant) 
                for variant in df[variant_col].unique()}
    
    # Sort positions dictionary by position value, handling None values
    positions = {k: v for k, v in sorted(positions.items(), 
                                       key=lambda item: (item[1] is None, item[1] or 0))}
    
    # Map positions to new column
    df[position_col] = pd.to_numeric(df[variant_col].map(positions), errors='coerce').astype('Int64')

    # Extract REF and ALT for unique variants, including stop-gain (e.g., "12Q>*")
    ref_alt_dict = {variant: extract_ref_alt_with_stop(variant) 
                   for variant in df[variant_col].unique()}
    
    # Map to DataFrame columns
    df[ref_col] = df[variant_col].map(lambda x: ref_alt_dict[x][0]).astype(str)
    df[alt_col] = df[variant_col].map(lambda x: ref_alt_dict[x][1]).astype(str)

    return df


def extract_ref_alt(variant):
        if variant == 'REF':
            return None, None
        
        # Handle deletion format (e.g., "616delS")
        del_match = re.search(r'(\d+)del([A-Z]+)', variant)
        if del_match:
            return del_match.group(2), ''  # REF is the deleted sequence, ALT is empty
        
        # Handle standard substitution format (e.g., "123A>G")
        sub_match = re.search(r'(\d+)([A-Z]+)>([A-Z]+)', variant)
        if sub_match:
            return sub_match.group(2), sub_match.group(3)
        
        # Handle amino acid change format (e.g., "P1150S")
        aa_match = re.search(r'[A-Z](\d+)([A-Z])', variant)
        if aa_match:
            return aa_match.group(2), aa_match.group(2)  # Same amino acid, no change
        
        return None, None


def add_variant_name(df,
                    chrom_col='chrom',
                    start_col='chromStart',
                    end_col='chromEnd',
                    ref_col='REF',
                    alt_col='ALT',
                    alias='name',
                    force=False):
    """Add a variant name column to a DataFrame.
    
    Args:
        df: Polars or Pandas DataFrame
        chrom_col: Column name for chromosome
        start_col: Column name for start position
        end_col: Column name for end position. 
            If None, the end position is calculated as the start position + the length of the reference allele.
        ref_col: Column name for reference allele
        alt_col: Column name for alternate allele
        alias: Name for the output column
        force: Whether to overwrite existing column
    Returns:
        DataFrame with added variant name column
    """
    import polars as pl
    import pandas as pd

    if alias in df.columns and not force:
        print(f"Column {alias} already exists in dataframe, skipping")
        return df
    
    was_pandas = isinstance(df, pd.DataFrame)
    if was_pandas:
        df = pl.DataFrame(df)

    if end_col not in df.columns:
        end_col = None
    
    result = df.with_columns(pl.concat_str([
        pl.lit('chr'),
        pl.col(chrom_col).cast(pl.Utf8).str.replace('chr', ''),
        pl.lit(':'),
        pl.col(start_col).cast(pl.Utf8),
        pl.lit('-'),
        # If end_col is null, calculate end position as start + length of reference allele
        # Otherwise, use end_col if provided, or fall back to start position
        pl.when(pl.lit(end_col).is_null())
        .then(pl.col(start_col).cast(pl.Int32) + pl.col(ref_col).str.len_chars().cast(pl.Int32))
        .otherwise(pl.col(end_col).cast(pl.Utf8) if end_col is not None else (pl.col(start_col).cast(pl.Int32)+1).cast(pl.Utf8)),
        pl.lit('_'),
        pl.col(ref_col),
        pl.lit('_'),
        pl.col(alt_col)
    ]).alias(alias))
    
    if was_pandas:
        result = result.to_pandas()
    
    return result



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

def vep_to_matrix(
    vep_df,
    sample_col="sample",
    site_col="site",
    ploid_col="ploid",
    value_col="VEP",
    fillna_method=None,#"mean",
    duplicate_ref_hap=True,
    sort_index=True,
    verbose=True
):
    """
    Convert a VEP (Variant Effect Predictor) DataFrame to a matrix format.

    This function pivots a long-form VEP DataFrame into a matrix (wide-form) where rows correspond to samples,
    columns correspond to variant sites, and values are the VEP scores. If there are multiple VEP scores for the
    same sample-site pair, their mean is taken. Missing values are filled with `fill_value`.

    Parameters
    ----------
    vep_df : pandas.DataFrame
        Input DataFrame in long format, containing at least the sample, site, and VEP score columns.
    sample_col : str, optional
        Name of the column in `vep_df` identifying samples. Default is "sample".
    site_col : str, optional
        Name of the column in `vep_df` identifying variant sites. Default is "site".
    value_col : str, optional
        Name of the column in `vep_df` containing VEP scores. Default is "VEP".
    ploid_col : str, optional
        Name of the column in `vep_df` identifying ploidy (i.e. which haplotype). Default is "ploid".
    fillna_method : str, optional
        Method to use for filling missing values. Default is None.
    duplicate_ref_hap : bool, optional
        Whether to duplicate the REF haplotype to avoid NAs. Default is True.
    sort_index : bool, optional
        Whether to sort the index and columns. Default is True.
    verbose : bool, optional
        Whether to print progress. Default is True.

    Returns
    -------
    pandas.DataFrame
        A matrix (DataFrame) with samples as rows, sites as columns, and VEP scores as values.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "sample": ["A", "A", "B", "B"],
    ...     "site": ["s1", "s2", "s1", "s2"],
    ...     "VEP": [0.1, 0.2, 0.3, 0.4]
    ... })
    >>> vep_to_matrix(df)
       s1   s2
    A  0.1  0.2
    B  0.3  0.4
    """
    vep_df = vep_df.copy()
    
    # Add ploid column if it exists
    if ploid_col is not None and ploid_col in vep_df.columns:
        
        # Add merged site column
        if verbose:
            print("Adding merged site column")
        new_site_col = site_col + "_" + ploid_col
        if new_site_col not in vep_df.columns:
            vep_df[new_site_col] = vep_df[site_col].astype(str) + "_" + vep_df[ploid_col].astype(str)
        site_col = new_site_col 
        
        # Duplicate the REF haplotype to avoid NAs
        if duplicate_ref_hap:
            if verbose:
                print("Duplicating REF haplotype to avoid NAs")
            ref_hap = vep_df.loc[vep_df[sample_col]=="REF"].copy()
            if not ref_hap.empty:
                ref_hap[ploid_col] = int(ref_hap[ploid_col].iloc[0]) + 1
                vep_df = pd.concat([vep_df, ref_hap], ignore_index=True, copy=False)
    
    # Convert to dtypes to save memory
    if verbose:
        print("Converting vep_df dtypes")
    vep_df[sample_col] = vep_df[sample_col].astype(object)
    vep_df[site_col] = vep_df[site_col].astype(object)
    vep_df[value_col] = vep_df[value_col].astype(float)

    if verbose:
        calc_percent_nas(vep_df)

    # Use much faster groupby.unstack with built-in mean (skipna by default)
    # takes 4.7s, as opposed to pivot_table which takes 90 seconds!
    X = vep_df.groupby([sample_col, site_col], sort=False, observed=True
                       )[value_col].mean().unstack()

    X = fill_na(X, fillna_method)

    # Ensure index and columns are sorted for consistency
    if sort_index:
        X = X.sort_index(axis=0).sort_index(axis=1)

    return X

def calc_percent_nas(df):
    percent_nas = df.isna().sum().sum() / df.size * 100
    print(f"Percent of cells in df that are NaN: {percent_nas:.2f}%") 

def fill_na(X, 
            fillna_method=None,
            verbose=True):

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


def vep_to_matrix_torch(vep_samples,
                        sample_col = "sample",
                        site_col = "site",
                        ploid_col = "ploid",
                        value_col = "VEP_norm",
                        device = "cuda:3",
                        fillna_method="colmean",
                        drop_allna_rows=True,
                        drop_allna_cols=True,
                        return_as = "pandas", # "pandas", "numpy", "torch"
                        verbose = True
                        ):
    import torch

    # Explicitly set device to cuda:3 for all allocations and tensor moves
    device = torch.device(device)

    if verbose:
        print("Cleaning and copying vep_samples...")
    shape1 = vep_samples.shape[0]
    vep_samples = vep_samples.dropna(subset=[sample_col, site_col, value_col]).copy()
    if verbose:
        print(f"Dropped {shape1 - vep_samples.shape[0]} rows with NaNs in required columns...")


    if ploid_col is not None and ploid_col in vep_samples.columns:
        new_site_col = site_col + "_" + ploid_col
        if new_site_col not in vep_samples.columns:
            vep_samples[new_site_col] = vep_samples[site_col].astype(str) + "_" + vep_samples[ploid_col].astype(str)
        if verbose:
            print(f"Setting site_col to {new_site_col}")
        site_col = new_site_col 

    if verbose:
        print("Preparing index mappings for samples and sites...")
    sample_codes, sample_idx = torch.unique(torch.tensor(pd.factorize(vep_samples[sample_col])[0]), return_inverse=True)
    site_codes, site_idx = torch.unique(torch.tensor(pd.factorize(vep_samples[site_col])[0]), return_inverse=True)

    if verbose:
        print(f"Preparing {value_col} values as torch tensor...")
    vep_values = torch.tensor(vep_samples[value_col].values, dtype=torch.float32)

    if verbose:
        print("Determining matrix shape...")
    n_samples = sample_codes.shape[0]
    n_sites = site_codes.shape[0]

    if verbose:
        print(f"Creating empty ({n_samples}, {n_sites}) matrix and count matrix on CUDA:3...")
    X_torch = torch.zeros((n_samples, n_sites), dtype=torch.float32, device=device)
    count_torch = torch.zeros((n_samples, n_sites), dtype=torch.float32, device=device)

    if verbose:
        print("Moving indices and values to GPU 3...")
    sample_idx = sample_idx.to(device)
    site_idx = site_idx.to(device)
    vep_values = vep_values.to(device)

    if verbose:
        print("Accumulating VEP values and counts...")
    X_torch.index_put_((sample_idx, site_idx), vep_values, accumulate=True)
    count_torch.index_put_((sample_idx, site_idx), torch.ones_like(vep_values), accumulate=True)

    if verbose:
        print("Averaging VEP values and handling missing data...")
    count_torch[count_torch == 0] = 1
    X_torch = X_torch / count_torch
    X_torch[count_torch == 1] = float('nan')

    if verbose:
        print(f"Filling missing values with {fillna_method}...")
    inds = torch.isnan(X_torch)
    num_total = X_torch.numel()
    num_filled = torch.sum(~torch.isnan(X_torch)).item()
    percent_filled = 100 * num_filled / num_total
    if verbose:
        print(f"Matrix fill: {num_filled}/{num_total} ({percent_filled:.2f}%)")

    if drop_allna_rows:
        if verbose:
            print("Removing rows with all NaNs...")
        rows_to_keep = ~torch.all(torch.isnan(X_torch), dim=1)
        if verbose:
            print(f"Dropped {(~rows_to_keep).sum().item()}/{X_torch.shape[0]} rows ({100 * (~rows_to_keep).sum().item() / X_torch.shape[0]:.2f}%) with all NaNs...")
        X_torch = X_torch[rows_to_keep, :]

    if drop_allna_cols:
        if verbose:
            print("Removing columns with all NaNs...")
        cols_to_keep = ~torch.all(torch.isnan(X_torch), dim=0)
        if verbose:
            print(f"Dropped {(~cols_to_keep).sum().item()}/{X_torch.shape[1]} columns ({100 * (~cols_to_keep).sum().item() / X_torch.shape[1]:.2f}%) with all NaNs...")
        X_torch = X_torch[:, cols_to_keep]

    if fillna_method == "colmean":
        col_means = torch.nanmean(X_torch, dim=0)
        X_torch[inds] = col_means[inds.nonzero(as_tuple=True)[1]]
    elif fillna_method == "rowmean":
        row_means = torch.nanmean(X_torch, dim=1)
        X_torch[inds] = row_means[inds.nonzero(as_tuple=True)[0]]
    elif isinstance(fillna_method, int) or isinstance(fillna_method, float):
        X_torch[inds] = fillna_method
    else:
        if verbose:
            print("NA filling will be skipped")

    if return_as == "torch":
        return X_torch

    if verbose:
        print("Moving matrix back to CPU and converting to numpy array...")
    X_np = X_torch.cpu().numpy()
    del X_torch

    if return_as == "numpy":
        return X_np

    if verbose:
        print("Building pandas DataFrame with correct index and columns, accounting for dropped columns...")
    # Get the unique samples and sites that remain after dropping columns
    unique_samples = vep_samples[sample_col].astype('category').cat.categories
    unique_sites = vep_samples[site_col].astype('category').cat.categories

    # If any samples or sites were dropped in the earlier merge/drop steps, ensure the DataFrame index/columns match X_np shape
    if X_np.shape[0] != len(unique_samples):
        if verbose:
            print(f"Warning: Number of samples in matrix ({X_np.shape[0]}) does not match unique samples ({len(unique_samples)}). Adjusting index.")
        unique_samples = unique_samples[:X_np.shape[0]]
    if X_np.shape[1] != len(unique_sites):
        if verbose:
            print(f"Warning: Number of sites in matrix ({X_np.shape[1]}) does not match unique sites ({len(unique_sites)}). Adjusting columns.")
        unique_sites = unique_sites[:X_np.shape[1]]

    X = pd.DataFrame(
        X_np,
        index=unique_samples,
        columns=unique_sites
    )
    del X_np

    if verbose:
        print("Matrix shape:", X.shape)

    return X

def vep_distance(
    X, 
    og_meta=None,
    site_cols=None, 
    groupby_cols=None,
    sample_col="sample",
    metric='euclidean',
    verbose=True
):
    """
    Compute a pairwise distance matrix between groups or samples based on VEP (variant effect predictor) data.

    This function calculates the Euclidean distance between group centroids (or samples) for the specified site columns.
    If `groupby_cols` is provided, the function computes the mean value of each site column for each group and then
    computes the pairwise distances between these group centroids. If `groupby_cols` is not provided, distances are
    computed directly between the rows of `X`.

    Parameters
    ----------
    X : pandas.DataFrame
        A sample (individual) x site (clinical variant) DataFrame containing VEP scores. 
        Each row should correspond to a sample, and columns should correspond to sites or features.
    og_meta : pandas.DataFrame, optional
        Sample metadata DataFrame. If not provided, it will be loaded from `src.onekg.get_sample_metadata()`.
        This is used to map samples to groups if `groupby_cols` is specified.
    site_cols : list of str, optional
        List of columns in `X` to use for distance calculation. If None, all columns in `X` are used.
    groupby_cols : list of str or str, optional
        Column(s) in the metadata to group by. If provided, distances are computed between group centroids.
    sample_col : str, optional
        Column in `X` to use as the sample identifier. Default is 'sample'.
    metric : str, optional
        The distance metric to use. Default is 'euclidean'. 
    verbose : bool, optional
        Whether to print progress. Default is True.

    Returns
    -------
    pandas.DataFrame
        A square DataFrame of pairwise Euclidean distances between groups (if `groupby_cols` is given)
        or between samples (if not). The rows and columns are labeled by group or sample.

    Notes
    -----
    - The function uses Euclidean distance for continuous data. For binary/categorical data, consider using
      'hamming' distance.
    - The function expects that the index of `X` contains sample identifiers matching the "Individual ID" in `og_meta`.
    """
    from scipy.spatial.distance import pdist, squareform

    if og_meta is None:
        import src.onekg as og
        og_meta = og.get_sample_metadata()
    
    if site_cols is None:
        site_cols = X.columns.tolist()

    if groupby_cols is not None:
        if verbose:
            print("Computing group centroids")
        group_centroids = (
            X.reset_index()
            .merge(og_meta, left_on=sample_col, right_on="Individual ID", how="left")
            .groupby(groupby_cols)[site_cols]
            .mean()
        )
    else:
        group_centroids = X
    
    if verbose:
        print("Computing distances")
    dists = pdist(group_centroids.values, metric=metric)
    dist_square = squareform(dists)
    
    group_labels = group_centroids.index.tolist()
    distVEP = pd.DataFrame(dist_square, index=group_labels, columns=group_labels)
    return distVEP


def sort_chromosomes(df, chrom_col="chrom"):
    """
    Sort chromosomes by natural order: "chr1", "chr2", ..., "chr22", "chrX", "chrY", "chrM"
    """
    import re

    def chrom_key(chrom):
        chrom = str(chrom)
        # Extract the part after 'chr'
        m = re.match(r"chr(\d+|X|Y|M)$", chrom)
        if m:
            val = m.group(1)
            if val.isdigit():
                return (0, int(val))
            elif val == "X":
                return (1, 23)
            elif val == "Y":
                return (1, 24)
            elif val == "M":
                return (1, 25)
        # If doesn't match, put at the end
        return (2, chrom)

    df_sorted = df.copy()
    df_sorted["_chrom_sort_key"] = df_sorted[chrom_col].map(chrom_key)
    df_sorted = df_sorted.sort_values(by="_chrom_sort_key")
    df_sorted = df_sorted.drop(columns=["_chrom_sort_key"])
    return df_sorted

# Use a topographic-inspired colorscale  
# Make the ocean (lowest values) much darker by using nearly black for the lowest stops
topo_colorscale = [
                    [0.00, "#0a0a23"],   # almost black (deepest water)
                    [0.02, "#142850"],   # very dark blue
                    [0.04, "#253494"],   # deep blue
                    [0.07, "#2c7fb8"],   # deeper blue 
                    [0.10, "#4575b4"],   # original deep blue (water)
                    [0.13, "#41b6c4"],   # deep blue-green (shallow water)
                    # [0.16, "#91bfdb"],   # light blue
                    # [0.20, "#e0f3f8"],   # very light blue
                    [0.30, "#ffffbf"],   # sand/yellow
                    [0.40, "#bfa06a"],   # tan (sandstone)
                    [0.55, "#a67c52"],   # light brown (limestone)
                    [0.70, "#8d5524"],   # medium brown (granite)
                    [0.80, "#7c4a02"],   # dark brown (basalt)
                    [0.90, "#5c3a21"],   # deep brown (shale)
                    [1.00, "#ffffff"],   # white (snow)
                ]




aa_3to1 = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
    'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
    'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val'
}

def add_hgvsp_id(
    vep_prot,
    variant_col="wt_variant",
    position_col="wt_p.position",
    ref_col="wt_p.REF",
    alt_col="wt_p.ALT",
    protein_col="protein",
    wt_HGVSp_col="wt_HGVSp"
):
    """
    Add an HGVSp (protein-level HGVS) identifier column to a DataFrame.

    This function generates a column containing HGVSp IDs in the format:
        <protein>:p.<REF><position><ALT>
    where <REF> and <ALT> are converted from one-letter to three-letter amino acid codes.

    Parameters
    ----------
    vep_prot : pandas.DataFrame
        DataFrame containing variant and protein information.
    variant_col : str, default="variant"
        Name of the column containing variant identifiers.
    position_col : str, default="wt_p.position"
        Name of the column containing the protein position.
    ref_col : str, default="wt_p.REF"
        Name of the column containing the reference amino acid (one-letter code).
    alt_col : str, default="wt_p.ALT"
        Name of the column containing the alternate amino acid (one-letter code).
    protein_col : str, default="protein"
        Name of the column containing the protein identifier.
    wt_HGVSp_col : str, default="wt_HGVSp"
        Name of the new column to be created with the HGVSp IDs.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with an additional column containing HGVSp IDs.

    Notes
    -----
    This function assumes that the input DataFrame contains the specified columns.
    The function also calls `variants_to_positions` to ensure position and amino acid columns are present.
    """
    # Add site info for WT variants (variant)
    vep_prot = variants_to_positions(
        vep_prot,
        variant_col=variant_col,
        position_col=position_col,
        ref_col=ref_col,
        alt_col=alt_col
    )

    vep_prot[wt_HGVSp_col] = (
        vep_prot[protein_col]
        + ":p."
        + vep_prot[ref_col].map(aa_3to1)
        + vep_prot[position_col].astype(str)
        + vep_prot[alt_col].map(aa_3to1)
    )
    return vep_prot



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
def is_pd(x):
    """
    Check if the input object is a pandas DataFrame.

    Parameters:
    x (object): The object to be checked.

    Returns:
    bool: True if the object is a pandas DataFrame, False otherwise.
    """
    import pandas as pd
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
    if x == None:
        return x
    if isinstance(x, type({}.keys())):
        return list(x)
    if type(x) != list:
        x = [x]
    if type(x) == set:
        x = list(x)
    if type_func != None:
        x = [type_func(y) for y in x]
    return x

def one_only(lst):
    lst = as_list(lst)
    return lst[0]

def intersect(x, y, as_list=True):
    if as_list:
        return list(set(x) & set(y))
    else:
        return set(x) & set(y)

def list_vcf(dir='/grid/koo/home/schilder/projects/GenomeEncoder/data/1KG/vcf_formatted/*.vcf.gz', 
             as_dict=False):
    # List all VCF files in the 1KG directory
    import glob
    import os
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
    import os
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
                 cmap='viridis',
                 figsize=(10, 8),
                 **kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)  # Set figure size
    plt.gca().set_facecolor('#2F0154') # Set background to slightly darker than darkest viridis color
    sns.kdeplot(data=df, x=x, y=y, 
                fill=True, 
                cmap=cmap,
                **kwargs
                ) 
    # Then overlay scatter points with some transparency
    plt.scatter(data=df,
                x=x, y=y,
                alpha=alpha, # Make points semi-transparent
                s=1, # Small point size
                color='white') # White points
    
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

def get_marker_map(n=8):
    """Get a map of markers for a given range of integers"""
    marker_map = {0:_create_donut_marker(), 
                  1:'o', 
                #   2:_create_semicircle_marker(), 
                2:'X',
                  3:'^', 
                  4:'D', 
                  5:'p', 
                  6:'H', 
                  7:_create_polygon_marker(7), 
                  8:'8'
                  }
    if n > 8:
        for i in range(10, n):
            marker_map[i] = _create_polygon_marker(i)
    return marker_map

def count_variants(lst,
                   tx_id_sep=":",
                   count=['<','>','del','ins']):
    
    lst = as_list(lst)
    return [sum([x.split(tx_id_sep)[1].count(c) for c in count]) for x in lst]


def as_seq(seq):
    from Bio.Seq import Seq
    if isinstance(seq, Seq):
        return seq
    if isinstance(seq, list):
        seq = "".join(seq)
    return Seq(seq)

def as_seqrecord(seq):
    from Bio.SeqRecord import SeqRecord
    if isinstance(seq, SeqRecord):
        return seq
    return SeqRecord(as_seq(seq))

def is_msa(seqs):
    from Bio.Align import MultipleSeqAlignment
    return isinstance(seqs, MultipleSeqAlignment)

def as_msa(seqs: list[str],
            **kwargs):
    """
    Convert a list of sequences to a MultipleSeqAlignment object.
    """
    from Bio.Align import MultipleSeqAlignment
    
    if is_msa(seqs):
        return seqs
    # Check if seqs is a list
    if not isinstance(seqs, list):
        raise ValueError("seqs must be a list")
    # Check if seqs contains at least 2 sequences
    if len(seqs) < 2:
        raise ValueError("seqs must contain at least 2 sequences")
    # Convert to Seq objects
    seqs = [as_seqrecord(seq) for seq in seqs]
    # Create MSA
    return MultipleSeqAlignment(seqs, **kwargs)

def query_msa(msa,
              pos,
              ref=0,
              query=1,
              join_str=None):
    """
    Query a MultipleSeqAlignment at a specific reference genome coordinates.
    
    Parameters
    ----------
    msa : Bio.Align.MultipleSeqAlignment
        Multiple sequence alignment object
    pos : int
        Position in the reference sequence to query (starts from 0)
    ref : int, optional
        Index of the reference sequence in the MSA, by default 0
    query : int, optional
        Index of the query sequence in the MSA to compare against reference, by default 1
        
    Returns
    -------
    list
        List of characters from the query sequence that align to the reference position.
        Returns None if no match is found at the specified position.
    """
    if ref > len(msa)-1:
        raise ValueError(f"Reference index out of range. Maximum index is {len(msa)-1}.")
    if query > len(msa)-1:
        raise ValueError(f"Query index out of range. Maximum index is {len(msa)-1}.")
    
    idx = msa.alignment.indices[ref] == (pos-1)
    if sum(idx) == 0:
        print(f"No matching sequence found at position {pos} for reference.")
        return None
    subseqs = [x for i,x in enumerate(msa[query].seq) if idx[i] == True]
    # Return 
    if join_str is not None:
        return join_str.join(subseqs)
    else:
        return subseqs

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


def encode_haplotype_name(seq_name, 
                          include_counts=True):
    """
    Encode haplotype names that are far too long.

    Example:
        seq_name=  'ENSP00000382423:906V>I,948T>N,949del{12},962del{20},983delLS,986del{17},1004del{13},1018del{14},1033C>G,1034del{5},1040del{16},1057del{4},1062delPS,1065P>Q,1068del{5},1074delGDP,1078del{27},1106del{7},1114delTPV,1118del{4},1123del{5},1129delN,1131S>F,1133del{6},1140delMP,1143del{6},1150del{8},1159E>I,1160delKAE,1164del{7},1172del{24},1197delA,1199delQDA,1203delPI,1206del{6},1213del{11},1225ET>FF,1227del{27},1255delSSC,1259del{45},1305delN,1307I>C,1308del{5},1314delCEK,1318del{7},1326del{7},1334del{96},1431del{19},1451del{31},1483del{5},1490del{23}'
        encode_haplotype_name(seq_name)
    """ 
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
 

def get_candidate_proteins():
    import io
    import pandas as pd
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
    return candidate_proteins



def _make_palette(values,
                  palette):
    # sample 4 colors from a palette that goes from hot to cold
    import seaborn as sns
    return dict(zip(values, sns.color_palette(palette, len(values)).as_hex()))

def get_clinsig_palette(values=['path', 'likely_path', 'likely_benign', 'benign'],
                         palette='bwr_r'):
    return _make_palette(values, palette) 
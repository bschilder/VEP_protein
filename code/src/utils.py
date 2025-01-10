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
    if type_func != None:
        x = [type_func(y) for y in x]
    return x

def intersect(x,y):
    return list(set(x) & set(y))

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
             **kwargs):
    if filter is True:
        X, nan_indices = filter_matrix(X)
    print("Running UMAP")
    import umap
    reducer = umap.UMAP(verbose=verbose,
                        **kwargs)
    reducer.fit(X)
    embedding = reducer.transform(X)
    return reducer, embedding, nan_indices

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
    

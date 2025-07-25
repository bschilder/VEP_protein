# %%
from curses import halfdelay
import os
import glob
import esm
import torch
from tqdm.auto import tqdm
from pathlib import Path
from typing import Optional
# %%
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# %%
import src.haplosaurus as hs
import src.utils as utils
import src.config as config

# %%
def get_mean_embeddings(outputs,
                        verbose: bool = True):

    # we'll summarize the embeddings using their mean across the sequence dimension
    # which allows us to compare embeddings for sequences of different lengths
    all_mean_embeddings = [
        torch.mean(output.hidden_states, dim=-2).squeeze() for output in outputs
    ]

    # now we have a list of tensors of [num_layers, hidden_size]
    if verbose:
        print("embedding shape [num_layers, hidden_size]:", 
              all_mean_embeddings[0].shape)
    return all_mean_embeddings

def get_max_embedding_dim(haplotype_embeddings,
                          limit=5,
                          verbose=True,
                          **kwargs):
    
    ### If haplotype_embeddings is a directory, get the maximum embedding dimension
    if isinstance(haplotype_embeddings, str):
        max_embedding_dim = 0
        for path in tqdm(glob.glob(os.path.join(haplotype_embeddings, '*.pth'))[:limit],
                        desc="Getting max embedding dimension", 
                        leave=False):
            embedding_dict = torch.load(path, **kwargs)
            for embedding in embedding_dict.values():
                if embedding.shape[0] > max_embedding_dim:
                    max_embedding_dim = embedding.shape[0]
        if verbose:
            print(f"Max embedding dimension: {max_embedding_dim}")

    ### If haplotype_embeddings is a dictionary, get the maximum embedding dimension
    else:
        max_embedding_dim = 0
        i = 0
        for tx_id, embedding_dict in haplotype_embeddings.items():
            for embedding_i in embedding_dict.values():
                if embedding_i.shape[0] > max_embedding_dim:
                    max_embedding_dim = embedding_i.shape[0]
            i += 1
            if i > limit:
                break

    ### Return the maximum embedding dimension
    if verbose:
        print(f"Max embedding dimension: {max_embedding_dim}")
    return max_embedding_dim

def _get_patient_embeddings_dir(embeddings_dir,
                                name='patient_embeddings'):
    starts_with_slash = embeddings_dir.startswith(os.sep)
    path_sep = embeddings_dir.strip(os.sep).split(os.sep)
    path_sep[-2] = name
    if starts_with_slash:
        return os.path.join(os.sep, *path_sep)
    else:
        return os.path.join(*path_sep)

def _get_sample_tensor_path(patient_embeddings_dir,
                            sample):
    return os.path.join(patient_embeddings_dir,
                        sample+'.pth')

def _get_sample_tensor_dir(patient_embeddings_dir,
                           sample):
    return os.path.join(patient_embeddings_dir,
                        sample)

def _get_sample_tx_tensor_path(patient_embeddings_dir,
                            tx_id,
                            sample):
    return os.path.join(
        _get_sample_tensor_dir(patient_embeddings_dir,
                               sample),
        tx_id+'.pth')

def _get_haplotype_embeddings_path(embeddings_dir,
                                   tx_id):
    return os.path.join(embeddings_dir,
                        tx_id+'.pth')

def _merge_sample_tx_tensors(samples,
                             tx_ids,
                             phases,
                             patient_embeddings_dir,
                             max_embedding_dim,
                             force=False,
                             verbose=True):
    # Initialize list of sample save paths
    sample_save_paths = []

    # Iterate over samples to create final per-sample merged tensor
    for sample in tqdm(samples,
                       desc="Merging sample tensors",
                       leave=False):
        
        sample_tensor_path = _get_sample_tensor_path(
            patient_embeddings_dir, 
            sample
        )
        sample_tensor_dir = _get_sample_tensor_dir(
            patient_embeddings_dir, 
            sample
        )
        
        if os.path.exists(sample_tensor_path) and not force:

            if verbose>1:
                print(f"Skipping {sample} because it already exists")
            
            if os.path.exists(sample_tensor_dir):
                import shutil
                shutil.rmtree(sample_tensor_dir)
            
            continue

        else:
            import glob
            # Get all transcript-specific pth files for this sample
            tx_files = glob.glob(os.path.join(sample_tensor_dir, 
                                                "*.pth"))
            
            # Ensure all tx_ids are present
            sample_tx_ids = [os.path.basename(file).replace('.pth', '') for file in tx_files]
            
            # Check if all tx_ids are present
            if set(sample_tx_ids) == set(tx_ids):

                # Initialize tensor
                sample_tensor = torch.zeros(
                    len(tx_ids), # Transcripts
                    len(phases), # Phases
                    max_embedding_dim # embeddings - using maximum dimension to accommodate all
                )
                
                # Populate tensor
                for file_path in tx_files:
                    sample_tx_id = os.path.basename(file_path).replace('.pth', '')
                    sample_tensor[tx_ids.index(sample_tx_id)] = utils.load_torch(file_path, verbose=verbose)
                
                # Save merged tensor
                utils.save_torch(obj=sample_tensor, 
                                    save_path=sample_tensor_path,
                                    verbose=verbose)
                
                # Remove temporary tensor folder
                import shutil
                shutil.rmtree(sample_tensor_dir)

            else:
                if verbose>1:
                    print(f"No valid tensors found for sample: {sample}")
                continue 
        
        # Record sample save path
        sample_save_paths.append(sample_tensor_path)

    return sample_save_paths

def load_haplosaurus_data(haplotypes=None,
                          tx_ids=None,
                          key='protein_haplotypes',
                          force=False,
                          samples=None,
                          verbose=True):
    
    def _subset_samples(samples1,
                        samples2,
                        verbose=True):
        
        if samples1 is None:
            samples = samples2
        else:
            samples = utils.intersect(samples1,
                                      samples2)
        if verbose>1:
            print(f"Subsetting samples to {len(samples)}")
        return samples
    
    if tx_ids is None:
        tx_ids = hs.list_haplotypes()

    checksum_path = utils.ids_to_checksum_filename(
            tx_ids, 
            dir=hs.DIR_DICT["haplotypes_merged"],
            suffix="_haplotype_data.pkl")        
    
    if os.path.exists(checksum_path) and not force:
        res = utils.load_pickle(checksum_path, 
                                 verbose=verbose)
        samples = _subset_samples(samples,
                                  res['samples'],
                                  verbose=verbose)
        return res['haplotype_df'], res['sample_maps'], samples
    
    if haplotypes is None:
        haplotypes = hs.get_haplotypes(tx_ids=tx_ids,
                                        cache_only=True, 
                                        leave=False)

    # Convert haplotypes to dataframe
    haplotype_df = hs.haplotypes_to_df(haplotypes,
                                       key=key)
    
    # Convert haplotypes to samples
    sample_maps = hs.haplotypes_to_samples(haplotypes,
                                            key=key,
                                            return_seqs=False)
    # Get all sample names
    samples_tmp = list(set().union(*[set(x.keys()) for x in sample_maps.values()]))

    # Save haplotype data
    utils.save_pickle(obj={'haplotype_df':haplotype_df,
                           'sample_maps':sample_maps,
                           'samples':samples_tmp},
                      save_path=checksum_path,
                      verbose=verbose)
    
    # Subset to selected samples
    samples = _subset_samples(samples,
                              samples_tmp,
                              verbose=verbose)
    return haplotype_df, sample_maps, samples


def list_haplotype_embeddings_tx_ids(haplotype_embeddings):
    """
    List the transcript IDs in a haplotype embeddings directory.

    Args:
        haplotype_embeddings: Can be any of the following:
            - Path to haplotype embeddings directory
            - Path to a merged haplotype embeddings file
            - Dictionary of haplotype embeddings

    Returns:
        List of transcript IDs.

    Examples:
        tx_ids = list_haplotype_embeddings_tx_ids("<path>/<to>/<haplotype_embeddings>")
    """
    if isinstance(haplotype_embeddings, str):
        if os.path.isdir(haplotype_embeddings):
            # Handle directory case
            files = glob.glob(os.path.join(haplotype_embeddings,"*.pth"))
            tx_ids = [os.path.basename(file).replace('.pth', '') for file in files]
            return tx_ids
        else:
            # Handle file case
            return torch.load(haplotype_embeddings, 
                             mmap=True, 
                             weights_only=True,
                             map_location=torch.device('cpu')).keys()
    else:
        return haplotype_embeddings.keys()


def load_haplotype_embeddings(haplotype_embeddings,
                              tx_id):
    """
    Load haplotype embeddings from a directory or file.
    """
    if isinstance(haplotype_embeddings, str):
        
        # Get path
        if os.path.isdir(haplotype_embeddings):
            path = os.path.join(haplotype_embeddings, tx_id+'.pth')
        elif haplotype_embeddings.endswith('.pth'):
                path = haplotype_embeddings
        else:
            raise ValueError(f"Haplotype embeddings file must end with .pth")
        
        # Load embeddings
        if os.path.exists(path):
            return torch.load(path, 
                                mmap=True, 
                                weights_only=True,
                                map_location=torch.device('cpu'))
        else:
            raise ValueError(f"Haplotype embeddings not found for {tx_id}")
    else:
        return haplotype_embeddings[tx_id]


def make_patient_tensor(
    haplotype_embeddings, 
    haplotypes=None,    
    tx_ids=None,
    samples=None,
    key='protein_haplotypes',
    max_embedding_dim=None,
    drop_nan=False,
    patient_embeddings_dir: Optional[Path] = None,
    save_path: Optional[Path] = None,
    force: bool = False,
    verbose: bool = True
):
    """
    Construct a tensor containing patient-specific protein haplotype embeddings.

    This function builds a 4D tensor of shape 
    [num_samples, num_transcripts, num_phases, max_embedding_dim], 
    where each entry contains the embedding for a given sample, transcript, and phase.
    The embeddings are loaded from a directory or dictionary, and are matched to 
    the corresponding haplotype sequences for each sample and transcript.

    Optionally, the tensor can be saved to disk, or split and saved as per-sample tensors.

    Args:
        haplotype_embeddings (str or dict): 
            Path to a directory or file containing haplotype embeddings, or a dictionary of embeddings.
        haplotypes (optional): 
            Haplotype data or path to haplotype data. If None, will use default from haplosaurus.
        tx_ids (list, optional): 
            List of transcript IDs to include. If None, will use intersection of available embeddings and haplotypes.
        samples (list, optional): 
            List of sample IDs to include. If None, will use all samples in haplotype data.
        key (str, optional): 
            Key to use for extracting haplotype data. Default: 'protein_haplotypes'.
        max_embedding_dim (int, optional): 
            Maximum embedding dimension. If None, will be determined automatically.
        drop_nan (bool, optional): 
            If True, skip embeddings containing NaN values. Default: False.
        patient_embeddings_dir (Path, optional): 
            Directory to save per-sample tensors. If provided, saves each sample's tensor separately.
        save_path (Path, optional): 
            Path to save the full tensor dictionary. If provided, saves the tensor to this path.
        force (bool, optional): 
            If True, overwrite existing saved tensors. Default: False.
        verbose (bool or int, optional): 
            Verbosity level. If >1, prints additional information.

    Returns:
        If patient_embeddings_dir is provided:
            List of paths to per-sample tensor files.
        Else:
            Dictionary with keys:
                'tensor': the 4D tensor,
                'samples': list of sample IDs,
                'transcripts': list of transcript IDs,
                'phases': list of phases.

    Example:
        tensor_dict = make_patient_tensor(
            haplotype_embeddings="/path/to/embeddings",
            haplotypes="/path/to/haplotypes",
            tx_ids=["ENST00000367770", "ENST00000448914"],
            samples=["S1", "S2"],
            save_path="patient_tensor.pth"
        )
    """
    if save_path is not None and os.path.exists(save_path) and not force:
        return utils.load_torch(save_path, verbose=verbose)
    
    # Get max embedding dimension if not provided
    if max_embedding_dim is None:
        max_embedding_dim = get_max_embedding_dim(haplotype_embeddings, verbose=verbose)

    # Get transcript IDs available in both embeddings and haplotypes
    tx_ids_embeddings = list_haplotype_embeddings_tx_ids(haplotype_embeddings)
    tx_ids_haplotypes = hs.list_haplotypes()
    tx_ids_tmp = utils.intersect(tx_ids_embeddings, tx_ids_haplotypes)
    if tx_ids is None:
        tx_ids = tx_ids_tmp
    else:
        tx_ids = utils.intersect(tx_ids, tx_ids_tmp)
    if len(tx_ids) == 0:
        raise ValueError("No tx_ids found in both embeddings and haplotypes")
    if verbose > 1:
        print(f"Subsetting tx_ids to {len(tx_ids)}")

    # Define phases
    phases = ['phase1', 'phase2']

    # Initialize tensor: [samples, transcripts, phases, embedding_dim]
    tensor = torch.zeros(
        len(samples),      # Samples
        len(tx_ids),       # Transcripts
        len(phases),       # Phases
        max_embedding_dim  # Embedding dimension
    )

    for transcript_idx, tx_id in tqdm(
        enumerate(tx_ids), 
        total=len(tx_ids),
        desc="Populating patient tensor",
        leave=False
    ):
        # Load haplotype data for this transcript
        hap_df, hap_samples, samples = load_haplosaurus_data(
            haplotypes=haplotypes,
            tx_ids=tx_id,
            key=key,
            samples=samples,
            verbose=verbose
        )
        # Load embeddings for this transcript
        haplotype_embeddings_tx = load_haplotype_embeddings(haplotype_embeddings, tx_id)

        # Iterate over samples
        for sample_idx, sample in tqdm(
            enumerate(samples),
            total=len(samples),
            desc="Iterating over samples",
            leave=False
        ):
            if sample not in samples:
                if verbose > 1:
                    print(f"Skipping {sample} because it is not in the selected samples")
                continue

            hap_df_tmp = hap_df.loc[hap_samples[tx_id][sample]]
            if verbose > 1:
                print(f"Found {len(hap_df_tmp)} haplotypes for {tx_id} in sample {sample}")

            if not hap_df_tmp.empty:
                for phase_idx, (hap_name, row) in enumerate(hap_df_tmp.iterrows()):
                    seq = row['sequence']
                    if seq in haplotype_embeddings_tx:
                        seq_rep = haplotype_embeddings_tx[seq]
                        if drop_nan and torch.isnan(seq_rep).any():
                            if verbose > 1:
                                print(f"Skipping {hap_name} because it has NaN values")
                            continue
                        tensor[sample_idx, transcript_idx, phase_idx, :] = seq_rep.detach().clone()

    # Save per-sample tensors if requested
    if patient_embeddings_dir:
        sample_tensor_paths = []
        for sample_idx, sample in tqdm(
            enumerate(samples),
            total=len(samples),
            desc="Saving per-sample tensors",
            leave=False
        ):
            sample_tensor_path = _get_sample_tensor_path(patient_embeddings_dir, sample)
            utils.save_torch(
                obj=tensor[sample_idx].detach().clone(),
                save_path=sample_tensor_path,
                verbose=verbose > 1
            )
            sample_tensor_paths.append(sample_tensor_path)
        return sample_tensor_paths
    else:
        # Construct tensor dictionary
        tensor_dict = {
            'tensor': tensor,
            'samples': samples,
            'transcripts': tx_ids,
            'phases': phases
        }
        if save_path:
            utils.save_torch(
                obj=tensor_dict,
                save_path=save_path,
                verbose=verbose
            )
        return tensor_dict


def merge_haplotype_embeddings(haplotype_embeddings_dir, 
                                unnest=False,
                                save_path=None,
                                return_save_path=False,
                                force=False,
                                verbose=True):
    """
    Merge all pth files in a directory into one large file.
    """
    
    if save_path is None:
        save_path = os.path.join(haplotype_embeddings_dir, 
                                   "haplotype_embeddings_merged{}.pth".format(
                                       "_unnested" if unnest else ""
                                   ))
    if os.path.exists(save_path) and not force:
        if return_save_path:
            return save_path
        else:
            return utils.load_torch(save_path, verbose=verbose)
    
    # Get all pth files
    pth_files = glob.glob(os.path.join(haplotype_embeddings_dir, "*.pth"))
    # Exclude the merged_embeddings.pth file if it exists
    pth_files = [file for file in pth_files if not file.find("haplotype_embeddings_merged") != -1]
    # Report
    print(f"Found {len(pth_files)} pth files to merge")
    
    # Create a dictionary to store all embeddings
    merged_dict = {}
    
    # Load and merge all embeddings
    for file_path in tqdm(pth_files, desc="Merging haplotype embeddings"):
        try:
            tx_id = os.path.basename(file_path).replace('.pth', '')
            embedding = torch.load(file_path, map_location='cpu')
            if unnest:
                merged_dict.update(embedding)
            else:
                merged_dict[tx_id] = embedding
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Save merged embeddings
    torch.save(merged_dict, save_path)
    print(f"Merged {len(merged_dict)} haplotype embeddings into {save_path}")
    print(f"Total size: {os.path.getsize(save_path) / (1024**3):.2f} GB")
    
    if return_save_path:
        return save_path
    else:
        return merged_dict



def embed_sequences(model,
                    save_dir,
                    batch_size=50,
                    max_len=None,
                    truncate=False,
                    full_embeddings=False,
                    embed_dtype=torch.float32,
                    pooling_types=['mean', 'cls'],
                    num_workers=0,
                    check_database = False,
                     
                    sql=False,
                    save=True,
                    tx_ids=None,
                    verbose=1):
    """
    Embed sequences using ESM++.
    Sequences will be grouped by transcript ID and embedded in batches.

    Args:
        model: ESM++ model
        save_dir: Directory to save embeddings
        batch_size: Batch size for embedding
        max_len: Maximum length of sequences to embed
        truncate: Whether to truncate sequences
        full_embeddings: Whether to return full embeddings
        embed_dtype: Data type for embeddings
        pooling_types: Pooling types to use
        num_workers: Number of workers for embedding
        check_database: Whether to check if embeddings already exist
        sql: Whether to save embeddings to SQLite database
        save: Whether to save embeddings to .pth file
        tx_ids: Transcript IDs to embed. If None, all cached Haplosaurus transcripts will be embedded.
        verbose: Verbosity level. 0: Silent, 1: Progress bar, 2: Debug.
    Examples:
        # Load model
        import torch
        from transformers import AutoModelForMaskedLM #AutoModel also works
        model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small',
                                                    trust_remote_code=True)
        # Embed all transcripts in the database
        embed_sequences(model,
                        save_dir='projects/data/1000_Genomes_on_GRCh38/embeddings/ESMplusplus_small',
                        )
    Note:
        - If sql=True, embeddings can only be stored in float32
        - sql is ideal if you need to stream a very large dataset for training in real-time
        - save=True is ideal if you can store the entire embedding dictionary in RAM
        - sql will be used if it is True and save is True or False
        - If your sql database or .pth file is already present, they will be scanned first for already embedded sequences
        - Sequences will be truncated to max_len and sorted by length in descending order for faster processing
        - If you stop the process while sql=True, the database will be locked.
            To unlock it, run:
            fuser  <path_to_db>/embeddings.db
            kill -9 <PID>
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    if tx_ids is None:
        tx_ids = hs.list_haplotypes()
    else:
        tx_ids = utils.as_list(tx_ids)

    # # Split tx_ids into batches of 1000
    # tx_batch_size = 1000
    # tx_id_batches = [tx_ids[i:i + tx_batch_size] for i in range(0, len(tx_ids), tx_batch_size)]
    # print(f"Split {len(tx_ids)} tx_ids into {len(tx_id_batches)} batches of {tx_batch_size}")

    for tx_id in tqdm(tx_ids, 
                    desc="Running batches"):
        sql_db_path=os.path.join(save_dir, tx_id+'.db')
        save_path=os.path.join(save_dir, tx_id+'.pth')
        if not check_database:
            if sql:
                if os.path.exists(sql_db_path):
                    continue
            else:
                if os.path.exists(save_path):
                    continue

        hap_df = hs.haplosaurus_dataloader(tx_ids=tx_id,
                                           key='protein_haplotypes',
                                           verbose=verbose)

        with utils.ignore_stdout(disable=verbose==2):
            embedding_dict = model.embed_dataset(
                sequences=hap_df["sequence"].tolist(),
                tokenizer=model.tokenizer,
                batch_size=batch_size, # adjust for your GPU memory
                max_len=max_len, # Maximum sequence length
                truncate=truncate, # Whether to truncate sequences to max_len
                full_embeddings=full_embeddings, # if True, no pooling is performed
                embed_dtype=embed_dtype, # cast to what dtype you want
                pooling_types=pooling_types, # more than one pooling type will be concatenated together
                num_workers=num_workers, # if you have many cpu cores, we find that num_workers = 4 is fast for large datasets
                sql=sql, # if True, embeddings will be stored in SQLite database
                sql_db_path=sql_db_path,
                save=save, # if True, embeddings will be saved as a .pth file
                save_path=save_path,  # .pth is PyTorch's standard file extension for saving model weights and tensors
            )
        

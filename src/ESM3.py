# %%
import os
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

def get_max_embedding_dim(embeddings_dir,
                          limit=5,
                          verbose=True,
                          **kwargs):
    # First determine the maximum embedding dimension
    import glob
    from tqdm.auto import tqdm

    max_embedding_dim = 0
    for path in tqdm(glob.glob(os.path.join(embeddings_dir, '*.pth'))[:limit],
                     desc="Getting max embedding dimension", 
                     leave=False):
        embedding_dict = torch.load(path, **kwargs)
        for embedding in embedding_dict.values():
            if embedding.shape[0] > max_embedding_dim:
                max_embedding_dim = embedding.shape[0]
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

def make_patient_tensor(embeddings_dir, 
                        tx_ids=None,
                        samples=None,
                        key='protein_haplotypes',
                        max_embedding_dim=None,
                        drop_nan = False,
                        save_path: Optional[Path] = None,
                        split_save: bool = False,
                        force: bool = False,
                        verbose: bool = True):
    
    if save_path is not None and os.path.exists(save_path) and not force:
        return utils.load_torch(save_path, verbose=verbose)
    
    if max_embedding_dim is None:
        max_embedding_dim = get_max_embedding_dim(embeddings_dir,
                                                  verbose=verbose)

    # Get tx ids available in embeddings_dir
    tx_ids_embeddings = hs.list_haplotypes(cache=embeddings_dir,
                                           suffix='.pth')
    tx_ids_haplotypes = hs.list_haplotypes()
    
    # Subset tx_ids to only include those in both embeddings and haplotypes
    tx_ids_tmp = utils.intersect(tx_ids_embeddings,
                                  tx_ids_haplotypes)
    
    # Subset selected tx_ids
    if tx_ids is None:
        tx_ids = tx_ids_tmp
    else:
        tx_ids = utils.intersect(tx_ids,
                                 tx_ids_tmp)
    if verbose>1:
        print(f"Subsetting tx_ids to {len(tx_ids)}")
        
    if split_save:
        patient_embeddings_dir = _get_patient_embeddings_dir(embeddings_dir)

    # Get phases
    phases = ['phase1', 'phase2']

    # initialize 3d tensor
    if not split_save:
        tensor = torch.zeros(
            len(samples), # Samples
            len(tx_ids), # Transcripts
            len(phases), # Phases
            max_embedding_dim # embeddings - using maximum dimension to accommodate all
        )
    
    for transcript_idx, tx_id in tqdm(enumerate(tx_ids), 
                                      total=len(tx_ids),
                                      desc="Populating patient tensor",
                                      leave=False):
        
        # Import haplotypes 
        haplotypes = hs.get_haplotypes(tx_ids=tx_id,
                                        cache_only=True, 
                                        leave=False)

        # Convert haplotypes to dataframe
        hap_df = hs.haplotypes_to_df(haplotypes,
                                     key=key)
        
        # Convert haplotypes to samples
        hap_samples = hs.haplotypes_to_samples(haplotypes,
                                                key=key,
                                                return_seqs=False)
        # Get all sample names
        samples_tmp = hs.get_haplotype_samples(haplotypes,
                                            unnest=True,
                                            cohort='1000GENOMES:phase_3',
                                            remove_prefix=True,
                                            key=key)
        
        # Subset to selected samples
        if samples is None:
            samples = samples_tmp
        else:
            samples = utils.intersect(samples,
                                      samples_tmp)
        if verbose>1:
            print(f"Subsetting samples to {len(samples)}")
            
        # Load embeddings
        embeddings_path = _get_haplotype_embeddings_path(embeddings_dir, tx_id)
        if os.path.exists(embeddings_path):
            embeddings_dict = torch.load(embeddings_path, 
                                         weights_only=True,
                                         mmap=True,
                                         map_location=torch.device('cpu'))
            
        else:
            if verbose>1:
                print(f"Skipping {tx_id} because it does not have embeddings")
            continue
        
        # Iterate over samples
        for sample_idx, sample in tqdm(enumerate(samples),
                                        total=len(samples),
                                        desc="Iterating over samples",
                                        leave=False):
            
            if sample not in samples:
                if verbose>1:
                    print(f"Skipping {sample} because it is not in the selected samples")
                continue
            
            # Split save mode
            if split_save:

                # Path to merged tensor for each sample
                sample_tensor_path = _get_sample_tensor_path(
                    patient_embeddings_dir, 
                    sample
                )
                # Skip if tensor already exists
                if os.path.exists(sample_tensor_path) and not force:
                    if verbose>1:
                        print(f"Skipping {sample} because it already exists")
                    continue
                
                # Path to temporary tensor for each sample-transcript
                sample_tx_tensor_path = _get_sample_tx_tensor_path(
                    patient_embeddings_dir, 
                    tx_id,
                    sample
                )
                # Skip if tensor already exists
                if os.path.exists(sample_tx_tensor_path) and not force:
                    if verbose>1:
                        print(f"Skipping {sample} because it already exists")
                    continue

                # Initialize tensor
                tensor = torch.zeros(
                    len(phases), # Phases
                    max_embedding_dim # embeddings - using maximum dimension to accommodate all
                )

            hap_df_tmp = hap_df.loc[hap_samples[tx_id][sample]] 
            if verbose>1:
                print(f"Found {len(hap_df_tmp)} haplotypes for {tx_id} in sample {sample}")

            if not hap_df_tmp.empty:
                
                for phase_idx, (hap_name, row) in enumerate(hap_df_tmp.iterrows()):
                    seq = row['sequence']
                
                    if seq in embeddings_dict:
                        seq_rep = embeddings_dict[seq]
                
                        if drop_nan and torch.isnan(seq_rep).any():
                            if verbose>1:
                                print(f"Skipping {hap_name} because it has NaN values")
                            continue

                        # Split save mode
                        if split_save:
                            tensor[phase_idx, :] = seq_rep
                        else:
                            tensor[sample_idx, transcript_idx, phase_idx, :] = seq_rep
            
            # Save tensor (split save mode)
            if split_save: 
                utils.save_torch(obj=tensor, 
                                 save_path=sample_tx_tensor_path,
                                 verbose=verbose>1)

    # Save a per-sample tensors
    if split_save:
        sample_save_paths = _merge_sample_tx_tensors(
            samples=samples,
            tx_ids=tx_ids,
            phases=phases,
            patient_embeddings_dir=patient_embeddings_dir,
            max_embedding_dim=max_embedding_dim,
            force=force,
            verbose=verbose
            )
        return sample_save_paths
            
    # Save a single tensor dictionary
    else:
        # Construct tensor dictionary
        tensor_dict = {'tensor':tensor,
                       'samples':samples,
                       'transcripts':tx_ids,
                       'phases':phases}
        
        utils.save_torch(obj=tensor_dict, 
                          save_path=save_path,
                          verbose=verbose)

        return tensor_dict


def embed_sequences(model,
                    save_dir,
                    batch_size=100,
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
        

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
        print("embedding shape [num_layers, hidden_size]:", all_mean_embeddings[0].shape)
    return all_mean_embeddings

def get_max_embedding_dim(embeddings_dir):
    # First determine the maximum embedding dimension

    import glob
    from tqdm.auto import tqdm

    max_embedding_dim = 0
    for path in tqdm(glob.glob(os.path.join(embeddings_dir, '*.pth')),
                     desc="Getting max embedding dimension", 
                     leave=False):
        embedding_dict = torch.load(path)
        for embedding in embedding_dict.values():
            if embedding.shape[0] > max_embedding_dim:
                max_embedding_dim = embedding.shape[0]
    return max_embedding_dim

def make_patient_tensor(embeddings_dir, 
                        haplotypes=None,
                        key='protein_haplotypes',
                        max_embedding_dim=None,
                        drop_nan = False,
                        save_path: Optional[Path] = None,
                        force: bool = False,
                        verbose: bool = True):
    
    if save_path is not None and os.path.exists(save_path) and not force:
        return utils.load_torch(save_path, verbose=verbose)
    
    if max_embedding_dim is None:
        max_embedding_dim = get_max_embedding_dim(embeddings_dir)

    # Get tx ids available in embeddings_dir
    tx_ids = hs.list_haplotypes(cache=embeddings_dir,
                                suffix='.pth')
    
    # Get haplotypes if not provided
    if haplotypes is None:
        haplotypes = hs.get_haplotypes(tx_ids=tx_ids,
                                       cache_only=True)
    tx_ids = utils.intersect(tx_ids, haplotypes.keys())
    # Filter haplotypes to only include those in embeddings_dir
    haplotypes = {tx_id: haplotypes[tx_id] for tx_id in tx_ids}
    tx_ids = list(haplotypes.keys())

    # Convert haplotypes to dataframe
    hap_df = hs.haplotypes_to_df(haplotypes,
                                 key=key)
    
    # Convert haplotypes to samples
    hap_samples = hs.haplotypes_to_samples(haplotypes,
                                            key=key,
                                            return_seqs=False)
    # Get all sample names
    samples = hs.get_haplotype_samples(haplotypes,
                                        unnest=True,
                                        cohort='1000GENOMES:phase_3',
                                        remove_prefix=True,
                                        key=key)

    
    phases = ['phase1', 'phase2']

    # initialize 3d tensor
    tensor = torch.zeros(
        len(samples), # Samples
        len(tx_ids), # Transcripts
        len(phases), # Phases
        max_embedding_dim # embeddings - using maximum dimension to accommodate all
        )
    
    for tx_id, tx_haps in tqdm(hap_samples.items(), 
                            desc="Populating patient tensor"):
        transcript_idx = list(hap_samples.keys()).index(tx_id)

        embeddings_path = os.path.join(embeddings_dir, tx_id+'.pth')
        if os.path.exists(embeddings_path):
            embeddings_dict = torch.load(embeddings_path)
        else:
            continue
        
        for sample, hap_names in tx_haps.items():
            sample_idx = list(tx_haps.keys()).index(sample)

            hap_df_tmp = hap_df.loc[hap_names] 
            if not hap_df_tmp.empty:
                
                for phase_idx, (hap_name, row) in enumerate(hap_df_tmp.iterrows()):
                    seq = row['sequence']
                
                    if seq in embeddings_dict:
                        seq_rep = embeddings_dict[seq]
                
                        if drop_nan and torch.isnan(seq_rep).any():
                            continue

                        tensor[sample_idx, transcript_idx, phase_idx, :] = seq_rep

    tensor_dict = {'tensor':tensor,
                   'samples':samples,
                   'transcripts':tx_ids,
                   'phases':phases}
    
    if save_path is not None:
        utils.save_torch(obj=tensor_dict, 
                          save_path=save_path,
                          verbose=verbose)

    return tensor_dict


def embed_sequences(model,
                    save_dir,
                    batch_size=200,
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
    Note:
    - If you stop the process while sql=True, the database will be locked.
    - To unlock it, run:
      fuser  projects/data/1000_Genomes_on_GRCh38/embeddings/ESMplusplus_small/embeddings.db
      kill -9 <PID>
    
    Note:
        - If sql=True, embeddings can only be stored in float32
        - sql is ideal if you need to stream a very large dataset for training in real-time
        - save=True is ideal if you can store the entire embedding dictionary in RAM
        - sql will be used if it is True and save is True or False
        - If your sql database or .pth file is already present, they will be scanned first for already embedded sequences
        - Sequences will be truncated to max_len and sorted by length in descending order for faster processing
    """
    
    os.makedirs(save_dir, exist_ok=True)
    tx_ids = hs.list_haplotypes()

    # # Split tx_ids into batches of 1000
    # tx_batch_size = 1000
    # tx_id_batches = [tx_ids[i:i + tx_batch_size] for i in range(0, len(tx_ids), tx_batch_size)]
    # print(f"Split {len(tx_ids)} tx_ids into {len(tx_id_batches)} batches of {tx_batch_size}")

    for tx_id in tqdm(tx_ids, 
                    desc="Running batches"):
        sql_db_path=os.path.join(save_dir,tx_id+'.db')
        save_path=os.path.join(save_dir, tx_id+'.pth')
        if not check_database:
            if sql:
                if os.path.exists(sql_db_path):
                    continue
            else:
                if os.path.exists(save_path):
                    continue

        haplotypes = hs.get_haplotypes(tx_ids=tx_id, 
                                        leave=False,
                                        cache_only=True)
        hap_seqs = hs.get_haplotype_seqs(haplotypes, 
                                        aligned=0,
                                        add_haplotype_names=True,
                                        key='protein_haplotypes')
        hap_df = hs.haplotypes_to_df(haplotypes,
                                     preprocess=True,
                                     key='protein_haplotypes')

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
        
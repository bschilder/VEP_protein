from tqdm.auto import tqdm
import warnings
import torch

import src.utils as utils
import src.biopython as bp

def check_sequence(sequence, 
                   pos, 
                   expected,
                   type="wildtype",
                   invert=False,
                   is_ref=True,
                   verbose=True,
                   error=True):
    
    if bp.is_msa(sequence):
        query = 0 if type == "wildtype" else 1
        subseq = bp.query_msa(sequence, 
                                pos=pos, 
                                ref=0, 
                                query=query,
                                join_str="",
                                return_as='subseq',
                                error=error)

        if invert:
            if subseq == expected:
                if verbose:
                    warnings.warn(f"The listed {type} is already present in the provided sequence at position {pos}: {subseq} == {expected} in {sequence}")
        else:
            txt = f"The listed {type} does not match the provided sequence at position {pos}: {subseq} != {expected} in {sequence}"
            if error:
                assert subseq == expected, txt
            else:
                if subseq != expected:
                    if verbose:
                        warnings.warn(txt)
    else:
        txt = f"The listed {type} does not match the provided sequence at position {pos}: {sequence[pos]} != {expected} in {sequence}"
        if is_ref:
            if error:
                assert bp.preprocess_sequence(sequence)[pos-1] == expected, txt
            else:
                if bp.preprocess_sequence(sequence)[pos-1] != expected:
                    if verbose:
                        warnings.warn(txt) 

def _parse_mutation_row(mutation_row):
    # parses "G195S" into wt="G", pos=195, mt="S"
    """
    Example:
        parse_mutation_row("G195S") -> ("G", 195, "S")
    """
    wt, pos, mt = mutation_row[0], int(mutation_row[1:-1]), mutation_row[-1]
    assert isinstance(wt, str)
    assert isinstance(pos, int)
    assert isinstance(mt, str)
    return wt, pos, mt

def check_ref_sequence(row,
                       sequence,
                       error=True,
                       verbose=2):
    if 'protein_sequence' in row.index:
        ref_sequence1 = bp.preprocess_sequence(row['protein_sequence'])
        ref_sequence2 = bp.preprocess_sequence(bp.get_sequence(sequence, 
                                                               i=0))
        txt = f"The protein sequence in the mutation row is not the same as the reference sequence in the MSA"
        if verbose>1:
                txt += f"\nMUT>> {ref_sequence1}\nMSA>> {ref_sequence2}" 
        # Error or warning
        if error:
            assert ref_sequence1 == ref_sequence2, txt
        else:
            if ref_sequence1 != ref_sequence2:
                if verbose>0:
                    warnings.warn(txt) 


def compute_mt_wt_score(row,
                        mutation_col,
                        sequence, 
                        token_probs, 
                        alphabet, 
                        is_ref,
                        offset_idx,
                        verbose=True):
    
    # Check that the protein sequence in the mutation row is the same as the sref equence in MSA
    check_ref_sequence(row=row,
                       sequence=sequence,
                       error=False)

    mutation_row = row[mutation_col]
    # parses "G195S" into wt="G", pos=195, mt="S"
    wt, pos, mt = _parse_mutation_row(mutation_row)

    # Get the index of the mutation in the non-reference sequence (even if there are indels in the non-reference sequence)
    if bp.is_msa(sequence):
        idx = bp.query_msa(msa=sequence,
                            pos=pos, 
                            ref=0, 
                            query=1, 
                            return_as='pos_idx')[0]  
    else:
        # Assumes there are no indels in non-reference sequence (i.e. an MSA with gaps)
        idx = pos - offset_idx

    # Check the wildtype sequence
    try:
        check_sequence(sequence=sequence, 
                       pos=pos, 
                       expected=wt, 
                       type="wildtype", 
                       is_ref=is_ref)
    except AssertionError as e:
        warnings.warn(e)
        return None
    
    # Token probabilites are computed using the preprocessed WT sequence (after removing gaps)
    # So we need to update the index to the preprocessed sequence
    idx_preprocessed = bp.get_preprocessed_index(sequence=sequence,
                                                 idx=idx,
                                                 verbose=False)

    # Get the indices of the wildtype and mutant alleles
    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # Check if the mutation position is out of bounds
    # Get preprocessed seq length
    seq_len = token_probs.size(1) - 1  # -1 for the BOS (Beginning of Sentence) token
    if idx_preprocessed is None:
        if verbose:
            wrn = f"Mutation position {idx} is out of bounds for sequence length {seq_len}"
            warnings.warn(wrn)
        return None
    
    # Compute the log probability of the mutation vs the wildtype
    # add 1 for the BOS (Beginning of Sentence) token
    score = token_probs[0, 1 + idx_preprocessed, mt_encoded] - token_probs[0, 1 + idx_preprocessed, wt_encoded]
    
    # Return the score
    return score.item()
    
def get_mutation_index(sequence,
                       pos,
                       ref=0, 
                       query=1, 
                       offset_idx=1):
    """
    Get the index of the mutation in the non-reference sequence (even if there are indels in the non-reference sequence)

    Args:
        sequence: str
            The sequence to get the mutation index of
        pos: int
            The position of the mutation
        ref: int
            The reference sequence index
        query: int
            The query sequence index
        offset_idx: int
            The offset index
    Example:
        sequence = "MALWMRLLPLLALLALWGPDPAAA"
        pos = 195
        ref = 0
        query = 1
        offset_idx = 1
    Returns:
        int
            The index of the mutation in the non-reference sequence
    """
    if bp.is_msa(sequence):
        idx = bp.query_msa(msa=sequence,
                            pos=pos, 
                            ref=ref, 
                            query=query, 
                            return_as='pos_idx')[0]  
    else:
        # Assumes there are no indels in non-reference sequence (i.e. an MSA with gaps)
        idx = pos - offset_idx
    return idx

def get_mutation_indices(df,
                         sequence,
                         mutation_col="mutant",
                         offset_idx=1):
    """
    Get the indices of each mutation in the non-reference sequence,
     even if there are indels in the non-reference sequence

    Args:
        df: pd.DataFrame
            The dataframe containing the mutation rows
        sequence: str
            The sequence to get the mutation indices of
        mutation_col: str
            The column containing the mutation rows
        offset_idx: int
            The offset index
    Returns:
        list
            The indices of each mutation in the non-reference sequence
    Example:
        df = pd.DataFrame({'mutation': ['G195S', 'G195S']})
        sequence = "MALWMRLLPLLALLALWGPDPAAA"
    """
    def func(mutation_row):
        wt, pos, mt = _parse_mutation_row(mutation_row)
        return get_mutation_index(sequence=sequence,
                                  pos=pos,
                                  offset_idx=offset_idx)
    # Get the mutation row
    mutation_idx = df[mutation_col].apply(func).to_list()
    return mutation_idx
    

def mutate_sequence(mutation_row,
                    sequence, 
                    offset_idx=1, 
                    is_ref=True):
    
    # Check the wildtype sequence
    wt, pos, mt = _parse_mutation_row(mutation_row)
    check_sequence(sequence=sequence, 
                   pos=pos, 
                   expected=wt, 
                   type="wildtype", 
                   is_ref=is_ref)
    
    # Get the index of the mutation in the non-reference sequence (even if there are indels in the non-reference sequence)
    if bp.is_msa(sequence):
        idx = bp.query_msa(msa=sequence,
                            pos=pos, 
                            ref=0, 
                            query=1, 
                            return_as='pos_idx')[0]  
    else:
        # Assumes there are no indels in non-reference sequence (i.e. an MSA with gaps)
        idx = pos - offset_idx

    # Get the haplotype sequence (before removing gaps)
    sequence_str = bp.get_sequence(sequence, 
                                   i=-1)
    # Mutate the sequence
    sequence_mut = sequence_str[:idx] + mt + sequence_str[(idx + 1) :] 

    # If the sequence was provided as an MSA, convert the mutated sequence back to an MSA
    if bp.is_msa(sequence):
        sequence_mut = bp.as_msa([sequence[0].seq,
                                  sequence_mut])
        
    # Check the mutant sequence
    check_sequence(sequence=sequence_mut, 
                   pos=pos, 
                   expected=mt, 
                   type="mutant", 
                   invert=True,
                   is_ref=is_ref)

    # Return 
    # - wt: wildtype allele
    # - pos: genomic position
    # - idx: 0-indexed mutation position within the sequence string
    # - mt: mutant allele
    # - sequence_mut: mutated sequence
    return wt, pos, idx, mt, sequence_mut

def get_available_gpus(verbose=False,
                       min_free_memory=1e9):
    """
    Get the available devices
    Returns a list of tuples (device_id, free_memory)
    Args:
        verbose: bool
            Whether to print the free memory of each GPU
        min_free_memory: float
            The minimum free memory (in bytes) to consider a GPU available

    Example:
        [(0, 10.0), (1, 8.0), (2, 6.0)]
    """
    # Get available GPUs (those with most free memory)
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        try:
            # Check if GPU is available by querying its memory
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            # Only include GPUs with sufficient free memory (e.g., 1GB)
            if free_memory > min_free_memory:
                available_gpus.append((i, free_memory))
                if verbose:
                    print(f"GPU {i}: {free_memory/1e9:.2f}GB free")
        except Exception as e:
            if verbose:
                print(f"Error checking GPU {i}: {e}")
            continue
    
    # Sort GPUs by available memory (descending)
    available_gpus.sort(key=lambda x: x[1], reverse=True)
    return available_gpus

def enable_data_parallel(model, 
                         check_available=True,
                         verbose=False):
    """
    Enables data parallelism for a PyTorch model across multiple GPUs.
    
    This function wraps the model with PyTorch's DataParallel module to distribute
    computation across available GPUs. When check_available is True, it prioritizes
    GPUs with the most available memory.
    
    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to parallelize.
    check_available : bool, default=True
        If True, checks available GPU memory and prioritizes GPUs with most free memory.
        If False, uses all available GPUs without checking memory.
    verbose : bool, default=True
        If True, prints information about the GPUs being used.
        
    Returns
    -------
    torch.nn.Module
        The model wrapped with DataParallel if multiple GPUs are available,
        otherwise the original model.
    """
    # Enable multiple GPUs
    if torch.cuda.device_count() > 1:
        # First move model to CUDA before wrapping with DataParallel
        model = model.cuda()
        
        if check_available:
            
            available_gpus = get_available_gpus(verbose=verbose)
            # Use GPUs with most available memory
            if available_gpus:
                device_ids = [gpu[0] for gpu in available_gpus]
                if verbose:
                    print(f"Using {len(device_ids)} GPUs: {device_ids}")
                # Ensure model is on the same device as device_ids[0]
                model = model.to(f"cuda:{device_ids[0]}")
                # No need to call model.to(device) after this - DataParallel handles device placement
                # The model is already on device_ids[0] from the previous line
                model = torch.nn.DataParallel(model, 
                                              device_ids=device_ids)
            else:
                if verbose:
                    print("No GPUs with sufficient memory found, using single GPU")
                model = model.cuda()
        else:
            if verbose:
                print(f"Using all {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
    else:
        # If only one GPU, just move to cuda
        if verbose:
            print("Only one GPU available")
        model = model.cuda()
    
    # Verify which devices are being used
    if isinstance(model, torch.nn.DataParallel):
        if verbose:
            print(f"Model is using DataParallel with devices: {model.device_ids}")
    else:
        if verbose:
            print(f"Model is using single device: {next(model.parameters()).device}")
    
    return model


 
def get_token_probs(model,
                    batch_tokens,
                    alphabet, 
                    sequence=None,
                    framework=["torch",
                               "tensorflow"],
                    method=["wt-marginals",
                            "masked-marginals",
                            "masked-marginals-msa",
                            "pseudo-ppl",
                            "pseudo-ppl-mlm"],
                    tokenizer=None,
                    token_indices=None,
                    progress_bar=True,  
                    leave=False):
    
    def _check_token_probs(token_probs,
                           batch_tokens,
                           alphabet):
        assert token_probs.shape[0] == 1, f"Token probabilities must have a batch dimension of 1. Got {token_probs.shape[0]}."
        assert token_probs.shape[1] == batch_tokens[0].shape[0], f"Token probabilities must have a sequence length of {batch_tokens[0].shape[0]}. Got {token_probs.shape[1]}."
        assert token_probs.shape[2] == len(alphabet), f"Token probabilities must have a vocabulary size of {len(alphabet)}. Got {token_probs.shape[2]}."

    # Get method options from function defaults
    method = utils.one_only(method)
    method = utils.check_arg(func=get_token_probs,
                              arg=method,
                              arg_index=2,
                              max_args=1)
    
    # Check that the framework is supported
    framework = utils.one_only(framework)
    if framework!="torch":
        raise ValueError(f"Only 'torch' is supported for now. Got '{framework}'.") 
    
    ##### wt-marginals #####
    # Compute the log probabilities of the wildtype sequence
    if method == "wt-marginals": 

        with torch.no_grad():
            token_probs = torch.log_softmax(
                model(batch_tokens.cuda())["logits"], 
                dim=-1
            )
            for _ in tqdm(
                range(1), 
                desc="Computing token probabilities: 'wt-marginals'",
                disable=not progress_bar, 
                leave=leave
            ):
                pass
        return token_probs
    
    ##### masked-marginals #####
    # Compute the log probabilities of the masked sequence
    elif method == "masked-marginals":
        
        # Get the device from batch_tokens
        device = batch_tokens.device
        
        all_token_probs = []
        for i in tqdm(range(batch_tokens.size(1)),
                        desc=f"Computing token probabilities: 'masked-marginals'",
                        disable=not progress_bar,
                        leave=leave):
            
            # Skip tokens that are not in the token_indices list
            if token_indices is not None:
                # Account for the BOS (Beginning of Sentence) token
                if i-1 not in token_indices:
                    # Create placeholder tensor on the same device as batch_tokens
                    token_probs = torch.zeros(1, len(alphabet), device=device)
                    all_token_probs.append(token_probs)
                    continue

            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i] = alphabet.mask_idx
            with torch.no_grad():
                token_probs = torch.log_softmax(
                    model(batch_tokens_masked.to(device))["logits"], 
                    dim=-1
                )
            all_token_probs.append(token_probs[:, i].to(device))  # vocab size
        # Concatenate all token probabilities along dimension 0 (sequence length)
        # and then add a batch dimension (unsqueeze at dim 0)
        # This creates a tensor of shape [1, sequence_length, vocab_size]
        token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)

        # Check the shape of the token probabilities
        _check_token_probs(token_probs=token_probs,
                           batch_tokens=batch_tokens,
                           alphabet=alphabet)
        
        return token_probs
    
    ##### masked-marginals-msa #####
    # Compute the log probabilities of the masked sequence for MSA models
    elif method == "masked-marginals-msa":

        all_token_probs = []
        for i in tqdm(range(batch_tokens.size(2)), 
                      desc="Computing token probabilities: 'masked-marginals-msa'",
                      disable=not progress_bar, 
                      leave=leave):
            
            # Skip tokens that are not in the token_indices list
            if token_indices is not None:
                if i not in token_indices:
                    all_token_probs.append(None)
                    continue

            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, 0, i] = alphabet.mask_idx  # mask out first sequence
            with torch.no_grad():
                token_probs = torch.log_softmax(
                    model(batch_tokens_masked.cuda())["logits"], 
                    dim=-1
                )
            all_token_probs.append(token_probs[:, 0, i])  # vocab size
        token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)

        # Check the shape of the token probabilities
        _check_token_probs(token_probs=token_probs,
                           batch_tokens=batch_tokens,
                           alphabet=alphabet)
        
        return token_probs
    
    ##### pseudo-ppl #####
    # Compute the log probabilities of the pseudo-ppl sequence
    elif method == "pseudo-ppl":
        
        if sequence is None:
            raise ValueError("Sequence must be provided for pseudo-ppl")
        sequence_str = bp.preprocess_sequence(sequence)
        
        # compute probabilities at each position
        log_probs = []
        for i in tqdm(range(1, len(sequence_str) - 1),
                      desc="Computing token probabilities: 'pseudo-ppl'",
                      disable=not progress_bar, 
                      leave=leave):
            
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i] = alphabet.mask_idx
            with torch.no_grad():
                token_probs = torch.log_softmax(
                    model(batch_tokens_masked.cuda())["logits"], 
                    dim=-1
                )
            log_probs.append(token_probs[0, i, alphabet.get_idx(sequence_str[i])].item())  # vocab size
        return log_probs
    
    elif method == "pseudo-ppl-mlm":

        if sequence is None:
            raise ValueError("Sequence must be provided for pseudo-ppl")
        sequence_str = bp.preprocess_sequence(sequence)
        
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for pseudo-ppl-mlm")
        
        # Tokenize the sequence
        token_ids = tokenizer.encode(sequence, return_tensors='pt')
        input_length = token_ids.size(1)
        log_probs = []

        # Compute the log likelihood of the sequence
        for i in tqdm(range(input_length),
                      desc="Computing token probabilities: 'pseudo-ppl-mlm'",
                      disable=not progress_bar, 
                      leave=leave):
            
            # Create a copy of the token IDs
            masked_token_ids = token_ids.clone()
            # Mask a token that we will try to predict back
            masked_token_ids[0, i] = tokenizer.mask_token_id
            
            with torch.no_grad():
                output = model(masked_token_ids)
                token_probs = torch.nn.functional.log_softmax(
                    output.logits, 
                    dim=-1
                )
            log_probs.append(token_probs[0, i, token_ids[0, i]])
        return log_probs
    
    else:
        raise ValueError(f"Invalid method: {method}")
    

def compute_pppl_mlm(sequence,
                     model_name=None, 
                     tokenizer=None, 
                     model=None):
    """
    Compute the pseudo-perplexity (PPPL) of a sequence using a masked language model (MLM).
    Sources:
        
        https://huggingface.co/blog/AmelieSchreiber/mutation-scoring#example-usage

    Args:
        model_name: str
            The name of the model to use
        tokenizer: transformers.Tokenizer
            The tokenizer to use
        sequence: str
            The sequence to compute the PPPL of 
    Returns:
        float
            The PPPL of the sequence
    Example:
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sequence = "Hello, world!"
    """
    # If the tokenizer is not provided, initialize it
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    if model is None:
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained(model_name)

   
    # Tokenize the sequence
    token_ids = tokenizer.encode(sequence, return_tensors='pt')
    input_length = token_ids.size(1)

    log_probs = get_token_probs(model=model,
                                batch_tokens=None,
                                alphabet=None,
                                method="pseudo-ppl-mlm",
                                tokenizer=tokenizer,
                                sequence=sequence)
    
    # Calculate the average log likelihood per token
    log_likelihood = sum(log_probs)
    avg_log_likelihood = log_likelihood / input_length

    # Compute and return the pseudo-perplexity
    pppl = torch.exp(-avg_log_likelihood)
    return pppl.item()




def compute_pppl(sequence, 
                 batch_tokens,
                 model, 
                 alphabet, 
                 agg_func=sum,
                 progress_bar=True): 
    """
    Compute the pseudo-perplexity (PPPL) of a sequence

    Args:
        sequence: str
            The sequence to compute the PPPL of
        batch_tokens: torch.Tensor
            The batch of tokens to compute the PPPL of
        model: torch.nn.Module
            The model to compute the PPPL of
        alphabet: Alphabet
            The alphabet to compute the PPPL of
        progress_bar: bool
            Whether to show a progress bar

    Returns:
        float
            The PPPL of the sequence

    Example:
        sequence = "MALWMRLLPLLALLALWGPDPAAA"
        batch_tokens = torch.randint(0, 20, (1, 100))
        model = model
        alphabet = alphabet
        progress_bar = True
    """
    # Compute the log probabilities of the mutated sequence
    log_probs = get_token_probs(model=model,
                                  batch_tokens=batch_tokens,
                                  sequence=sequence,
                                  alphabet=alphabet,
                                  method="pseudo-ppl",
                                  progress_bar=progress_bar)
    
    # Return the sum of the log probabilities
    return agg_func(log_probs)
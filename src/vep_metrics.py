from tqdm.auto import tqdm

from Bio import SeqIO
import torch

import src.utils as utils
import src.biopython as bp

def check_sequence(sequence, 
                   idx, 
                   expected,
                   type="wildtype",
                   invert=False,
                   is_ref=True,
                   verbose=True,
                   error=True):
    
    if bp.is_msa(sequence):
        query = 0 if type == "wildtype" else 1
        subseq = bp.query_msa(sequence, 
                                pos=idx+1, 
                                ref=0, 
                                query=query,
                                join_str="",
                                error=error)
        if invert:
            if subseq == expected:
                if verbose:
                    print(f"Warning: The listed {type} is already present in the provided sequence at position {idx+1}: {subseq} == {expected} in {sequence}")
        else:
            txt = f"The listed {type} does not match the provided sequence at position {idx+1}: {subseq} != {expected} in {sequence}"
            if error:
                assert subseq == expected, txt
            else:
                if subseq != expected:
                    if verbose:
                        print(txt)
    else:
        txt = f"The listed {type} does not match the provided sequence at position {idx+1}: {sequence[idx]} != {expected} in {sequence}"
        if is_ref:
            if error:
                assert sequence[idx] == expected, txt
            else:
                if sequence[idx] != expected:
                    print(txt) 


def _parse_mutation_row(mutation_row,
                        offset_idx):
    # parses "G195S" into wt="G", idx=195, mt="S"
    """
    Example:
        parse_mutation_row("G195S") -> ("G", 195, "S")
    """
    wt, idx, mt = mutation_row[0], int(mutation_row[1:-1]) - offset_idx, mutation_row[-1]
    assert isinstance(wt, str)
    assert isinstance(idx, int)
    assert isinstance(mt, str)
    return wt, idx, mt

def _check_ref_sequence(row,
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
                    print(txt)


def _get_apply_fun(df,
                   desc=None,
                   progress_bar=True):
    if progress_bar:
        tqdm.pandas(desc=desc)
        apply_fun = df.progress_apply
    else:
        apply_fun = df.apply
    return apply_fun


def compute_mt_wt_score(row,
                        mutation_col,
                        sequence, 
                        token_probs, 
                        alphabet, 
                        offset_idx,
                        is_ref):
    
    # Check that the protein sequence in the mutation row is the same as the sref equence in MSA
    _check_ref_sequence(row=row,
                        sequence=sequence,
                        error=False)

    mutation_row = row[mutation_col]
    # parses "G195S" into wt="G", idx=195, mt="S"
    wt, idx, mt = _parse_mutation_row(mutation_row, offset_idx)

    # Check the wildtype sequence
    try:
        check_sequence(sequence=sequence, 
                       idx=idx, 
                       expected=wt, 
                       type="wildtype", 
                       is_ref=is_ref)
    except AssertionError as e:
        print(e)
        return None

    # Get the indices of the wildtype and mutant alleles
    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # Check if the mutation position is out of bounds
    seq_len = token_probs.size(1) - 1  # -1 for BOS token
    if idx >= seq_len:
        print(f"Mutation position {idx} is out of bounds for sequence length {seq_len}")
        return None
    
    # Compute the log probability of the mutation vs the wildtype
    # add 1 for BOS token
    score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
    
    # Return the score
    return score.item()

def mutate_sequence(mutation_row,
                    sequence, 
                    offset_idx=1, 
                    is_ref=True):
    
    # Check the wildtype sequence
    wt, idx, mt = _parse_mutation_row(mutation_row, offset_idx)
    check_sequence(sequence=sequence, 
                   idx=idx, 
                   expected=wt, 
                   type="wildtype", 
                   is_ref=is_ref)

    # Get the haplotype sequence
    sequence_str = bp.get_sequence(sequence, i=-1)
    # Mutate the sequence
    sequence_mut = sequence_str[:idx] + mt + sequence_str[(idx + 1) :]
    
    # If the sequence was provided as an MSA, convert the mutated sequence back to an MSA
    if bp.is_msa(sequence):
        sequence_mut = bp.as_msa([sequence[0].seq,
                                  sequence_mut])
        
    # Check the mutant sequence
    check_sequence(sequence=sequence_mut, 
                   idx=idx, 
                   expected=mt, 
                   type="mutant", 
                   invert=True,
                   is_ref=is_ref)

    # Return 
    # - wt: wildtype allele
    # - idx: 0-indexed mutation position
    # - mt: mutant allele
    # - sequence_mut: mutated sequence
    return wt, idx, mt, sequence_mut
    
def compute_pppl(row,
                 mutation_col,
                 sequence, 
                 model, 
                 alphabet, 
                 offset_idx, 
                 is_ref,
                 progress_bar=True):
    
    # Check that the protein sequence in the mutation row is the same as the sref equence in MSA
    _check_ref_sequence(row=row,
                        sequence=sequence,
                        error=False)
    
    # Mutate the sequence
    mutation_row = row[mutation_col]
    wt, idx, mt, sequence_mut = mutate_sequence(mutation_row, sequence, offset_idx, is_ref)
 
    # Encode the mutatedsequence
    sequence_mut = bp.preprocess_sequence(sequence_mut)
    data = [
        ("protein1", sequence_mut),
    ] 
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Compute the log probabilities of the mutated sequence
    log_probs = get_token_probs(model=model,
                                batch_tokens=batch_tokens,
                                sequence=sequence_mut,
                                alphabet=alphabet,
                                method="pseudo-ppl",
                                progress_bar=progress_bar)
    
    # Return the sum of the log probabilities
    return sum(log_probs)
 

def get_token_probs(model,
                    batch_tokens,
                    alphabet, 
                    sequence=None,
                    framework=["torch",
                               "tensorflow"],
                    method=["wt-marginals",
                            "masked-marginals",
                            "masked-marginals-msa",
                            "pseudo-ppl"],
                    progress_bar=True, 
                    leave=False):
    
    # Get method options from function defaults
    method = utils.check_arg(func=get_token_probs,
                              arg=method,
                              arg_index=1,
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
                model(batch_tokens.cuda())["logits"], dim=-1
            )
            for _ in tqdm(
                range(1), 
                desc="Computing token probabilities:'wt-marginals'",
                disable=not progress_bar, 
                leave=leave
            ):
                pass
        return token_probs
    
    ##### masked-marginals #####
    # Compute the log probabilities of the masked sequence
    elif method == "masked-marginals":
        all_token_probs = []
        for i in tqdm(range(batch_tokens.size(1)),
                        desc=f"Computing token probabilities:'masked-marginals'",
                        disable=not progress_bar):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i] = alphabet.mask_idx
            with torch.no_grad():
                token_probs = torch.log_softmax(
                    model(batch_tokens_masked.cuda())["logits"], dim=-1
                )
            all_token_probs.append(token_probs[:, i])  # vocab size
        token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
        return token_probs
    
    ##### masked-marginals-msa #####
    # Compute the log probabilities of the masked sequence for MSA models
    elif method == "masked-marginals-msa":
        all_token_probs = []
        for i in tqdm(range(batch_tokens.size(2)), 
                      desc="Computing token probabilities:'masked-marginals-msa'",
                      disable=not progress_bar, 
                      leave=leave):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, 0, i] = alphabet.mask_idx  # mask out first sequence
            with torch.no_grad():
                token_probs = torch.log_softmax(
                    model(batch_tokens_masked.cuda())["logits"], dim=-1
                )
            all_token_probs.append(token_probs[:, 0, i])  # vocab size
        token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
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
                      desc="Computing token probabilities:'pseudo-ppl'",
                      disable=not progress_bar, 
                      leave=leave):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i] = alphabet.mask_idx
            with torch.no_grad():
                token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
            log_probs.append(token_probs[0, i, alphabet.get_idx(sequence_str[i])].item())  # vocab size
        return log_probs
    
    else:
        raise ValueError(f"Invalid method: {method}")
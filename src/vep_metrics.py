from tqdm.auto import tqdm
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
                    print(f"Warning: The listed {type} is already present in the provided sequence at position {pos}: {subseq} == {expected} in {sequence}")
        else:
            txt = f"The listed {type} does not match the provided sequence at position {pos}: {subseq} != {expected} in {sequence}"
            if error:
                assert subseq == expected, txt
            else:
                if subseq != expected:
                    if verbose:
                        print(txt)
    else:
        txt = f"The listed {type} does not match the provided sequence at position {pos}: {sequence[pos]} != {expected} in {sequence}"
        if is_ref:
            if error:
                assert bp.preprocess_sequence(sequence)[pos-1] == expected, txt
            else:
                if bp.preprocess_sequence(sequence)[pos-1] != expected:
                    if verbose:
                        print(txt) 

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
                    print(txt) 


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
        print(e)
        return None
    
    # Token probabilites are computed using the preprocessed sequence (after removing gaps)
    # So we need to update the index to the preprocessed sequence
    idx_preprocessed = bp.get_preprocessed_index(sequence=sequence,
                                                 idx=idx,
                                                 verbose=False)

    # Get the indices of the wildtype and mutant alleles
    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # Check if the mutation position is out of bounds
    # Get preprocessed seq length
    seq_len = token_probs.size(1) - 1  # -1 for BOS token
    if idx_preprocessed is None:
        if verbose:
            print(f"Mutation position {idx} is out of bounds for sequence length {seq_len}")
        return None
    
    # Compute the log probability of the mutation vs the wildtype
    # add 1 for BOS token
    score = token_probs[0, 1 + idx_preprocessed, mt_encoded] - token_probs[0, 1 + idx_preprocessed, wt_encoded]
    
    # Return the score
    return score.item()

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
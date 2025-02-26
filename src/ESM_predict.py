### --- From https://github.com/facebookresearch/esm/blob/main/examples/variant-prediction/predict.py --- ###
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import string

import torch

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import pandas as pd
from tqdm.auto import tqdm
from Bio import SeqIO
import itertools
from typing import List, Tuple 
import os

import src.biopython as bp

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None

    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)


def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions.
    
    The input file must be in a3m format (although we use the SeqIO fasta parser)
    for remove_insertions to work properly."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"MSA file not found: {filename}")
    msa = [
        (record.description, remove_insertions(str(record.seq)))
        for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
    ]
    return msa


def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from an ensemble of ESM-1v models."  # noqa
    )

    # fmt: off
    parser.add_argument(
        "--model-location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
        nargs="+",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Base sequence to which mutations were applied",
    )
    parser.add_argument(
        "--dms-input",
        type=pathlib.Path,
        help="CSV file containing the deep mutational scan",
    )
    parser.add_argument(
        "--mutation-col",
        type=str,
        default="mutant",
        help="column in the deep mutational scan labeling the mutation as 'AiB'"
    )
    parser.add_argument(
        "--dms-output",
        type=pathlib.Path,
        help="Output file containing the deep mutational scan along with predictions",
    )
    parser.add_argument(
        "--offset-idx",
        type=int,
        default=0,
        help="Offset of the mutation positions in `--mutation-col`"
    )
    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="wt-marginals",
        choices=["wt-marginals", "pseudo-ppl", "masked-marginals"],
        help=""
    )
    parser.add_argument(
        "--msa-path",
        type=pathlib.Path,
        help="path to MSA in a3m format (required for MSA Transformer)"
    )
    parser.add_argument(
        "--msa-samples",
        type=int,
        default=400,
        help="number of sequences to select from the start of the MSA"
    )
    parser.add_argument(
        "--is-ref",
        type=bool,
        default=False,
        help="Whether the sequence is a reference genome sequence"
    )
    parser.add_argument(
        "--force",
        type=bool,
        default=False,
        help="Rerun the prediction even if the results already exist"
    )
    # fmt: on
    parser.add_argument(
        "--nogpu", 
        action="store_true", 
        help="Do not use GPU even if available")
    return parser

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

def label_row(row,
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

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # Check if the mutation position is out of bounds
    seq_len = token_probs.size(1) - 1  # -1 for BOS token
    if idx >= seq_len:
        print(f"Mutation position {idx} is out of bounds for sequence length {seq_len}")
        return None
    # Compute the log probability of the mutation vs the wildtype
    # add 1 for BOS token
    score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
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

    # Mutate the sequence
    sequence_str = bp.get_sequence(sequence)
    sequence_mut = sequence_str[:idx] + mt + sequence_str[(idx + 1) :]
    if bp.is_msa(sequence):
        sequence_mut = bp.as_msa([sequence[0].seq,
                                  sequence_str])
        
    # Check the mutant sequence
    check_sequence(sequence=sequence_mut, 
                   idx=idx, 
                   expected=mt, 
                   type="mutant", 
                   invert=True,
                   is_ref=is_ref)

    return wt, idx, mt, sequence_mut
    
def compute_pppl(row,
                 mutation_col,
                 sequence, 
                 model, 
                 alphabet, 
                 offset_idx, 
                 is_ref):
    
    # Check that the protein sequence in the mutation row is the same as the sref equence in MSA
    _check_ref_sequence(row=row,
                        sequence=sequence,
                        error=False)
    
    mutation_row = row[mutation_col]
    wt, idx, mt, sequence = mutate_sequence(mutation_row, sequence, offset_idx, is_ref)
 
    sequence_str = bp.preprocess_sequence(sequence)
    # encode the sequence
    data = [
        ("protein1", sequence_str),
    ] 

    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # compute probabilities at each position
    log_probs = []
    for i in range(1, len(sequence_str) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
        log_probs.append(token_probs[0, i, alphabet.get_idx(sequence_str[i])].item())  # vocab size
    return sum(log_probs)

def main(
    dms_input: str,
    dms_output: str, 
    model_location: list,
    sequence: str,
    mutation_col: str,
    offset_idx: int,
    scoring_strategy: str,
    msa_path: str = None,
    msa_samples: int = None,
    nogpu: bool = False,
    is_ref: bool = False,
    force: bool = False
):
    """Run ESM model predictions on mutation data.
    
    Args:
        dms_input: Path to input CSV file containing mutations
            Example: "data/mutations.csv"
        dms_output: Path to save output CSV with predictions
            Example: "results/predictions.csv"
        model_location: List of ESM model paths to use
            Example: ["esm2_t33_650M_UR50D"]
        sequence: Wild-type protein sequence
            Example: "MVKVGVNG..."
        mutation_col: Column name containing mutations in dms_input
            Example: "mutant"
        offset_idx: Offset to apply to mutation positions
            Example: 0 if 1-indexed
        scoring_strategy: Prediction strategy - one of "wt-marginals", "masked-marginals", or "pseudo-ppl"
        msa_path: Path to MSA file for MSATransformer
            Example: "data/msa.a3m"
        msa_samples: Number of sequences to sample from MSA
            Example: 400
        nogpu: Whether to disable GPU usage even if available
        is_ref: Whether the sequence is a reference genome sequence
            Example: False
        force: Whether to force the prediction even if the model has already been downloaded
            Example: False
    """

    # Check if results already exist
    if os.path.exists(dms_output) and not force:
        print(f"Results already exist for {dms_output}. Use --force True to re-run.")
        return

    # Load the deep mutational scan or clinical variants
    df = pd.read_csv(dms_input)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not nogpu else "cpu")

    if isinstance(model_location, str):
        model_location = [model_location]
    if isinstance(scoring_strategy, list):
        scoring_strategy = scoring_strategy[0]
        print(f">1 scoring_strategy provided. Using only: '{scoring_strategy}'")


    # inference for each model
    for model_loc in model_location:
        
        # Avoid an infinite loop of trying to download the model (internal to esm)
        if model_loc == "esm1v_t33_650M_UR90S":
            model_loc = "esm1v_t33_650M_UR90S_1"
        
        # Load the model
        model, alphabet = pretrained.load_model_and_alphabet(model_loc)
        model.eval()
        
        # Move the model to the device
        model = model.to(device)
        if device.type == "cuda":
            print("Transferred model to GPU")

        batch_converter = alphabet.get_batch_converter()

        ## MSA models
        if isinstance(model, MSATransformer):
            data = [read_msa(msa_path, msa_samples)]
            assert (
                scoring_strategy == "masked-marginals"
            ), "MSA Transformer only supports masked marginal strategy"

            (batch_labels, 
             batch_strs, 
             batch_tokens) = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            all_token_probs = []
            for i in tqdm(range(batch_tokens.size(2))):
                batch_tokens_masked = batch_tokens.clone()
                batch_tokens_masked[0, 0, i] = alphabet.mask_idx  # mask out first sequence
                with torch.no_grad():
                    token_probs = torch.log_softmax(
                        model(batch_tokens_masked)["logits"], dim=-1
                    )
                all_token_probs.append(token_probs[:, 0, i])  # vocab size
            token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)

            df[model_loc] = df.apply(
                lambda row: label_row(
                    row, mutation_col, sequence, token_probs, alphabet, offset_idx
                ),
                axis=1,
            )

        ## Non-MSA models
        else:
            data = [
                ("protein1", bp.preprocess_sequence(sequence)),
            ]
            (batch_labels, 
             batch_strs, 
             batch_tokens) = batch_converter(data)
            

            if scoring_strategy == "wt-marginals":
                # batch_tokens = batch_tokens.to(device)
                # Compute the log probabilities of the wildtype sequence
                with torch.no_grad():
                    token_probs = torch.log_softmax(model(batch_tokens.cuda())["logits"], dim=-1)
                
                tqdm.pandas(desc=f"Computing 'wt-marginals' for {model_loc}")
                df[model_loc] = df.progress_apply(
                    lambda row: label_row(
                        row, 
                        mutation_col,
                        sequence,
                        token_probs,
                        alphabet,
                        offset_idx,
                        is_ref,
                    ),
                    axis=1,
                )
            elif scoring_strategy == "masked-marginals":

                all_token_probs = []
                for i in tqdm(range(batch_tokens.size(1)),
                              desc=f"Computing 'masked-marginals' for {model_loc}"):
                    batch_tokens_masked = batch_tokens.clone()
                    batch_tokens_masked[0, i] = alphabet.mask_idx
                    with torch.no_grad():
                        token_probs = torch.log_softmax(
                            model(batch_tokens_masked.cuda())["logits"], dim=-1
                        )
                    all_token_probs.append(token_probs[:, i])  # vocab size
                token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)

                df[model_loc] = df.apply(
                    lambda row: label_row(
                        row,
                        mutation_col,
                        sequence,
                        token_probs,
                        alphabet,
                        offset_idx,
                        is_ref,
                    ),
                    axis=1,
                )
            elif scoring_strategy == "pseudo-ppl":
                # Update compute_pppl to use device
                tqdm.pandas(desc=f"Computing 'pseudo-ppl' for {model_loc}")
                df[model_location] = df.progress_apply(
                    lambda row: compute_pppl(
                        row,
                        mutation_col,
                        sequence,
                        model,
                        alphabet,
                        offset_idx,
                        is_ref,
                    ),
                    axis=1,
                )

    df.to_csv(dms_output)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(
        dms_input=args.dms_input,
        dms_output=args.dms_output, 
        model_location=args.model_location,
        sequence=args.sequence,
        mutation_col=args.mutation_col,
        offset_idx=args.offset_idx,
        scoring_strategy=args.scoring_strategy,
        msa_path=args.msa_path,
        msa_samples=args.msa_samples,
        nogpu=args.nogpu,
        is_ref=args.is_ref
    )
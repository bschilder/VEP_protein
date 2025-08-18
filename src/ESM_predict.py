### --- From https://github.com/facebookresearch/esm/blob/main/examples/variant-prediction/predict.py --- ###
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
import argparse
import pathlib 
import pandas as pd
from tqdm.auto import tqdm  
import torch
from typing import List, Union, Tuple, Any

import src.utils as utils
import src.biopython as bp
import src.vep_metrics as vm

esm_old = True
try: 
    from src.ESM import load_model
    esm_old = True
except:
    pass
try: 
    from src.ESM3 import load_model
    esm_old = False
except:
    pass  



def seq_to_data(sequence,
                name="protein1",
                **kwargs):
    data = [
        (name, bp.preprocess_sequence(sequence, **kwargs)),
    ]
    return data

def seq_to_batch(sequence,
                 alphabet,
                 **kwargs):
    """Convert a sequence to a batch of data.
    
    Args:
        sequence: The sequence to convert.
        alphabet: The alphabet to use.
    Returns:
        batch_labels: The labels of the batch.
        batch_strs: The strings of the batch.
        batch_tokens: The tokens of the batch.
    """
    # Convert the sequence to data
    data = seq_to_data(sequence, **kwargs)

    # Get the batch converter
    batch_converter = alphabet.get_batch_converter()
    
    # Convert the data to a batch
    (batch_labels, 
     batch_strs, 
     batch_tokens) = batch_converter(data)
    return batch_labels, batch_strs, batch_tokens


def _get_model_name(model_loc):
    if isinstance(model_loc, str):
        model_name = model_loc
    elif isinstance(model_loc, tuple) and len(model_loc) == 2:
        model_name = model_loc[0]._get_name()
    else:
         model_name = model_loc._get_name()
    return model_name

def _assign_model_name(model, model_name):
    model.__class__.__name__ = model_name


def msa_to_batch(msa_path,
                 msa_samples,
                 alphabet):
    # Read the MSA
    data = bp.read_msa(msa_path, msa_samples)
    batch_converter = alphabet.get_batch_converter()
    (batch_labels, 
        batch_strs, 
        batch_tokens) = batch_converter(data)   
    return batch_labels, batch_strs, batch_tokens


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
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Print verbose output"
    )
    parser.add_argument(
        "--enable-data-parallel",
        type=bool,
        default=True,
        help="Enable data parallelization across multiple GPUs"
    )
    # fmt: on
    parser.add_argument(
        "--nogpu", 
        action="store_true", 
        help="Do not use GPU even if available")
    return parser

 
  
def compute_pppl(row,
                 mutation_col,
                 sequence, 
                 model, 
                 alphabet, 
                 offset_idx, 
                 is_ref,
                 progress_bar=True,
                 verbose=True):
    
    # Check that the protein sequence in the mutation row is the same as the ref sequence in MSA
    vm.check_ref_sequence(row=row,
                          sequence=sequence,
                          error=False,
                          verbose=verbose)
    
    # Mutate the sequence 
    wt, pos, idx, mt, sequence_mut = vm.mutate_sequence(mutation_row=row[mutation_col], 
                                                        sequence=sequence, 
                                                        offset_idx=offset_idx, 
                                                        is_ref=is_ref,
                                                        verbose=verbose)
    sequence_mut = bp.preprocess_sequence(sequence_mut)
    
    
    # Encode the mutated sequence using ESM alphabet
    (batch_labels, 
        batch_strs, 
        batch_tokens) = seq_to_batch(sequence_mut, alphabet)

        # Return the perplexity score using ESM model
    tuple_mean_sum_pppl = vm.compute_pppl(model=model,
                                          batch_tokens=batch_tokens,
                                          sequence=sequence_mut,
                                          alphabet=alphabet,
                                          progress_bar=progress_bar)
    return tuple_mean_sum_pppl 

def main(
    dms_input: str,
    dms_output: str, 
    model_location: List[Union[str, Tuple[Any, Any]]],  # List of strings or (model, alphabet) tuples
    sequence: str,
    mutation_col: str,
    offset_idx: int,
    scoring_strategy: str,
    msa_path: str = None,
    msa_samples: int = None,
    nogpu: bool = False,
    is_ref: bool = False,
    force: bool = False,
    progress_bar: bool = True,
    enable_data_parallel: bool = True,
    verbose: bool = True
):
    """Run ESM model predictions on mutation data.
    
    Args:
        dms_input: Path to input CSV file containing mutations
            Example: "data/mutations.csv"
        dms_output: Path to save output CSV with predictions
            Example: "results/predictions.csv"
        model_location: List of ESM model paths to use, or model names, or (model, alphabet) tuples
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
        enable_data_parallel: Whether to enable data parallelization
            Example: True
        verbose: Whether to print verbose output
            Example: True

    Returns:
        df: DataFrame with predictions
            Example: df.head()

    Example:
        df = main(
            dms_input="data/mutations.csv",
            dms_output="results/predictions.csv",
            model_location="esm2_t33_650M_UR50D",
            sequence="MVKVGVNG...",
            mutation_col="mutant", 
            scoring_strategy="wt-marginals",   
        )
        df.head()
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


    all_model_names = []


    # inference for each model
    for model_loc in model_location:
        
        # get model name
        model_name = _get_model_name(model_loc)
        all_model_names += [model_name]
        
        # Load the model
        model, alphabet = load_model(model_loc, 
                                     model_name=model_name, 
                                     verbose=verbose)
        
        # Move the model to the device
        if enable_data_parallel:
            model = vm.enable_data_parallel(model,
                                            verbose=verbose>1)
        else:
            model = model.to(device)
            
        # Set the model to evaluation mode, which disables dropout and other training-specific behaviors
        # This is important for inference to ensure consistent predictions
        # We do this after moving to device for optimal performance
        model.eval()

        ####-- MSA models --####
        # Original code from: 
        # https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/examples/variant-prediction/predict.py#L161
        if type(model).__name__ == "MSATransformer":
            #### masked-marginals-msa ####
            
            # Check that the scoring strategy is masked-marginals-msa
            if scoring_strategy == "masked-marginals":
                scoring_strategy = "masked-marginals-msa"
            assert (
                scoring_strategy == "masked-marginals-msa"
            ), "MSA Transformer only supports masked marginals strategy" 
            
            # Convert the data to a batch 
            (batch_labels, 
             batch_strs, 
             batch_tokens) = msa_to_batch(msa_path,
                                          msa_samples,
                                          alphabet)
            
            # Move batch tokens to device
            batch_tokens = batch_tokens.to(device)
            
            # Compute token probabilities
            token_probs = vm.get_token_probs(model=model,
                                             batch_tokens=batch_tokens,
                                             alphabet=alphabet,
                                             method="masked-marginals-msa",
                                             progress_bar=progress_bar)

            tqdm.pandas(desc=f"Computing 'masked-marginals-msa' for {model_name}", 
                        disable=not progress_bar,
                        leave=False)
            df.loc[:,model_name] = df.progress_apply(
                lambda row: vm.compute_mt_wt_score(
                    row, 
                    mutation_col, 
                    sequence, 
                    token_probs, 
                    alphabet, 
                    offset_idx,
                    is_ref=False
                ),
                axis=1,
            )

        ####-- Non-MSA models --####
        else: 
            
            #### wt-marginals ####
            # Original code from: 
            # https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/examples/variant-prediction/predict.py#L192
            if scoring_strategy == "wt-marginals":
                
                # Convert the sequence to a batch
                (batch_labels, 
                 batch_strs, 
                 batch_tokens) = seq_to_batch(sequence,
                                              alphabet)
                
                # Compute token probabilities
                token_probs = vm.get_token_probs(model=model,
                                                 alphabet=alphabet,
                                                 batch_tokens=batch_tokens, 
                                                 method="wt-marginals",
                                                 progress_bar=progress_bar)
                
                
                tqdm.pandas(desc=f"Computing 'wt-marginals' for {model_name}",
                            disable=not progress_bar,
                            leave=False)
                df.loc[:,model_name] = df.progress_apply(
                    lambda row: vm.compute_mt_wt_score(
                        row, 
                        mutation_col,
                        sequence,
                        token_probs,
                        alphabet,
                        offset_idx,
                        is_ref,
                        verbose=verbose
                    ),
                    axis=1,
                )
                
            #### masked-marginals ####
            # Original code from: 
            # https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/examples/variant-prediction/predict.py#L205
            elif scoring_strategy == "masked-marginals":

                if not esm_old:
                    scoring_strategy = "masked-marginals-esm3"

                # Convert the sequence to a batch
                (batch_labels, 
                 batch_strs, 
                 batch_tokens) = seq_to_batch(sequence,
                                              alphabet)
                
                # Get the mutation indices
                mutation_idx = vm.get_mutation_indices(df=df,
                                                       mutation_col=mutation_col,
                                                       sequence=sequence,
                                                       offset_idx=offset_idx)

                # Compute token probabilities
                token_probs = vm.get_token_probs(model=model,
                                                 alphabet=alphabet,
                                                 sequence=batch_strs[0], # Pass the processed sequence
                                                 batch_tokens=batch_tokens,
                                                 token_indices=mutation_idx,
                                                 method=scoring_strategy, 
                                                 progress_bar=progress_bar)

                tqdm.pandas(desc=f"Computing 'masked-marginals' for {model_name}", 
                            disable=not progress_bar,
                            leave=False)
                df.loc[:,model_name] = df.progress_apply(
                    lambda row: vm.compute_mt_wt_score(
                        row,
                        mutation_col,
                        sequence,
                        token_probs,
                        alphabet,
                        offset_idx,
                        is_ref,
                        verbose=verbose
                    ),
                    axis=1,
                )

            #### pseudo-ppl ####
            # Original code from: 
            # https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/examples/variant-prediction/predict.py#L226
            elif scoring_strategy == "pseudo-ppl":
                
                # Update compute_pppl to use device
                tqdm.pandas(desc=f"Computing 'pseudo-ppl' for {model_name}", 
                            disable=not progress_bar,
                            leave=False)
                
                df.loc[:,model_name] = df.progress_apply(
                    lambda row: compute_pppl(
                        row=row,
                        mutation_col=mutation_col,
                        sequence=sequence,
                        model=model,
                        alphabet=alphabet,
                        offset_idx=offset_idx,
                        is_ref=is_ref,
                        progress_bar=progress_bar>1,
                        verbose=verbose
                    ),
                    axis=1,
                )
    # Check if there are any predictions
    if  df.dropna(subset=all_model_names, how="all").empty:
        if verbose>1:
            print(f"No predictions generated for {" | ".join(all_model_names)}. Skipping file save.")
    # Save the results
    else:
        if verbose>1:
            print(f"Saving results to {dms_output}")
        if dms_output.endswith(".parquet"):
            df.to_parquet(dms_output, 
                            compression="gzip") 
        else:
            df.to_csv(dms_output)

### MAIN SCRIPT ###
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
        is_ref=args.is_ref,
        enable_data_parallel=args.enable_data_parallel,
        verbose=args.verbose
    )
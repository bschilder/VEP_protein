from typing import Optional, Union, List, Tuple
import string
import os
import itertools
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature
from Bio.Align import MultipleSeqAlignment
from Bio import SeqIO
import src.utils as utils


def to_stop(sequence: str,
            stop_str: str = '*'):
    """
    Truncate a sequence at the first occurrence of a stop codon.

    Args:
        sequence: The sequence to truncate
        stop_str: The stop codon to truncate at (default: '*')

    """
    return sequence if sequence.find(stop_str)==-1 else sequence[:sequence.find(stop_str)]

def get_sequence(sequence: MultipleSeqAlignment,
                  i: int = -1,
                  copy: bool = False):
    """
    Get a sequence from a MultipleSeqAlignment object or return the input sequence.

    Args:
        sequence: A sequence string or MultipleSeqAlignment object
        i: Index of sequence to get from MSA (default: -1 for last sequence)
        copy: Whether to return a deep copy of the sequence (default: True)

    Returns:
        str: The sequence as a string
    """

    if is_msa(sequence):
        if copy:
            from copy import deepcopy
            sequence = str(deepcopy(sequence[i].seq))
        else:
            sequence = str(sequence[i].seq)
    elif isinstance(sequence, SeqRecord):
        return str(sequence.seq)
    elif isinstance(sequence, Seq):
        return str(sequence)
    elif isinstance(sequence, list):
        return sequence[i]
    elif isinstance(sequence, str):
        return sequence
    else:
        raise ValueError(f"Invalid sequence type: {type(sequence)}")
    # Return the sequence
    return sequence

def preprocess_sequence(sequence: str,
                        strip: list = ['*'],
                        replace: list = ['.', '-'],
                        truncate: bool = True,
                        i: int = -1, 
                        copy: bool = True) -> str:
    """
    Preprocess a sequence by stripping characters, replacing characters, and truncating at a stop codon.

    Args:
        sequence: The sequence to preprocess
        strip: List of characters to strip from the sequence
        replace: List of characters to replace in the sequence
        truncate: Whether to truncate the sequence at a stop codon
        i: Index of the sequence to preprocess
        copy: Whether to return a deep copy of the sequence

    Returns:
        str: The preprocessed sequence
    """
    processed = sequence
    processed = get_sequence(processed, i, copy)
    if strip is not None:
        for s in strip:
            processed = processed.strip(s)
    if replace is not None:
        for r in replace:
            processed = processed.replace(r, '')
    # Truncate protein if "*" is in the sequence
    if truncate:
        processed = to_stop(processed)
    return processed

def as_seq(seq: Union[str, list[str]]) -> Seq:
    """
    Convert a sequence to a Seq object.

    Args:
        seq: The sequence to convert

    Returns:
        Seq: The converted sequence
    """
    if isinstance(seq, Seq):
        return seq
    if isinstance(seq, list):
        seq = "".join(seq)
    return Seq(seq)

def as_seqrecord(seq: str,
                 **kwargs) -> SeqRecord: 
    """
    Convert a sequence to a SeqRecord object.

    Args:
        seq: The sequence to convert

    Returns:
        SeqRecord: The converted sequence
    """
    if isinstance(seq, SeqRecord):
        return seq
    return SeqRecord(as_seq(seq), **kwargs)

def is_msa(seqs: Union[str, list[str]]) -> bool:
    """
    Check if the input is a MultipleSeqAlignment object.

    Args:
        seqs: The input to check

    Returns:
        bool: True if the input is a MultipleSeqAlignment object, False otherwise
    """
    return isinstance(seqs, MultipleSeqAlignment)

def as_msa(seqs: list[str],
           names=['REF','NONREF'],
           **kwargs):
    """
    Convert a list of sequences to a MultipleSeqAlignment object.

    Args:
        seqs: The list of sequences to convert
        **kwargs: Additional arguments to pass to the MultipleSeqAlignment constructor

    Returns:
        MultipleSeqAlignment: The converted MultipleSeqAlignment object
    """ 
    if is_msa(seqs):
        return seqs
    
    # Check if seqs is a list    
    assert isinstance(seqs, list), "seqs must be a list"
    # Check if seqs contains at least 2 sequences
    assert len(seqs) >= 2, "seqs must contain at least 2 sequences"
    
    # Check if each sequence has the same length after alignment
    for i, seq in enumerate(seqs):
        # Remove insertions and truncate at stop codon
        original_len = len(preprocess_sequence(seqs[i]))
        msa_len = len(preprocess_sequence(seq))
        assert msa_len == original_len, f"Sequence {i} has length {msa_len} but should have length {original_len}"
    
    # Convert to Seq objects
    seqs = [as_seqrecord(seq, name=names[i]) for i,seq in enumerate(seqs)]
    # Create MSA
    return MultipleSeqAlignment(seqs, **kwargs)
 

def query_msa(msa: MultipleSeqAlignment,
              pos: int,
              ref: int = 0,
              query: int = 1,
              join_str: Optional[str] = None,
              return_as = ['subseq','idxs','idxs_bool', 'pos_idx'],
              error: bool = True,
              verbose: bool = True) -> Optional[str]:
    """
    Query a MultipleSeqAlignment at a specific reference genome coordinates.
    
    Args:
        msa : Bio.Align.MultipleSeqAlignment
            Multiple sequence alignment object
        pos : int
        Position in the reference sequence to query (starts from 0)
        ref : int, optional
            Index of the reference sequence in the MSA, by default 0
        query : int, optional
            Index of the query sequence in the MSA to compare against reference, by default 1
        
    Returns:
        list: List of characters from the query sequence that align to the reference position.
        Returns None if no match is found at the specified position.
    """

    return_as = utils.one_only(return_as)

    if ref > len(msa)-1:
        raise ValueError(f"Reference index out of range. Maximum index is {len(msa)-1}.")
    if query > len(msa)-1:
        raise ValueError(f"Query index out of range. Maximum index is {len(msa)-1}.")
    
    # Get the true index of the reference sequence
    idxs = msa.alignment.indices[ref] 
    idxs_bool = idxs == (pos-1)

    max_ref_index = max(idxs)
    assert max_ref_index >= (pos-1), f"Query position out of range for reference. Maximum index is {max_ref_index} and position is {pos}."
    
    if sum(idxs_bool) == 0:
        txt = f"No matching sequence found at position {pos} for reference: '{msa[ref].seq}'"
        if error:
            raise ValueError(txt)
        else:
            if verbose:
                print("Warning:",txt)
            return None
        
    subseqs = np.array(list(msa[query].seq))[idxs_bool].tolist()
    # Return the subseqs
    if join_str is not None:
        subseqs = join_str.join(subseqs)
    
    assert len(subseqs)>0
    assert isinstance(subseqs[0], str)
    assert subseqs is not None

    if return_as == 'subseq':
        return subseqs
    elif return_as == 'idxs':
        return idxs
    elif return_as == 'idxs_bool':
        return idxs_bool
    elif return_as == 'pos_idx':
        # np.where returns a tuple of arrays for each dimension, but since idxs_bool is 1D,
        # we can just return the first element
        return np.where(idxs_bool)[0]
    else:
        raise ValueError(f"Invalid return_as: {return_as}")


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

def get_preprocessed_index(sequence: str,
                            idx: int,
                            stop_str: str = '*',
                            replace: list = ['.', '-'],
                            truncate: bool = True,
                            return_map: bool = False,
                            verbose: bool = False) -> Optional[int]:
    """
    Maps an index from the original sequence to its position in the preprocessed sequence.
    Returns None if the index corresponds to a removed character or is after truncation.

    Args:
        sequence: The original sequence
        idx: The index in the original sequence
        strip: List of characters to strip from the sequence
        replace: List of characters to replace in the sequence
        truncate: Whether to truncate the sequence at a stop codon
        verbose: Whether to print warning messages

    Returns:
        Optional[int]: The corresponding index in the preprocessed sequence, or None if the position was removed
    """
    sequence = get_sequence(sequence)
    if idx < 0 or idx >= len(sequence):
        if verbose:
            print(f"Index {idx} is out of bounds for sequence of length {len(sequence)}")
        return None

    # First handle truncation if enabled
    if truncate:
        stop_pos = sequence.find(stop_str)
        if stop_pos != -1 and idx >= stop_pos:
            if verbose:
                print(f"Index {idx} is after truncation point at position {stop_pos}")
            return None
        working_seq = sequence[:stop_pos] if stop_pos != -1 else sequence
    else:
        working_seq = sequence

    # Create a mapping from original to new positions
    new_pos = 0
    pos_map = {}
    
    for old_pos, char in enumerate(working_seq):
        # Skip characters that would be replaced
        if replace and char in replace:
            continue
        pos_map[old_pos] = new_pos
        new_pos += 1

    if return_map:
        return pos_map
    else:
        return pos_map.get(idx, None)


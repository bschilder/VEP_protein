import os
import pandas as pd
from tqdm.auto import tqdm
from Bio.Align import Alignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import AlignIO
from Bio.motifs import Motif


import src.utils as utils
import src.config as config
import src.haplosaurus as hs
from src.Align._ClustalOmega import ClustalOmegaCommandline
from src.Align._Clustalw import ClustalwCommandline

def align_haplotypes(tx_ids, 
                     aligner=ClustalOmegaCommandline,
                     save_dir=os.path.join(config.DATA_DIR,"1KG","fasta","clustalo"),
                     force=False, 
                     error=True):
    """
    Align haplotypes using ClustalOmega.

    Parameters:
    -----------
    tx_ids: list
        List of transcript IDs to align.
    aligner: ClustalOmegaCommandline or ClustalwCommandline
        The aligner to use.
    save_dir: str
        The directory to save the alignments.
    force: bool
        If True, overwrite existing alignments.

    Returns:
    --------
    alignment_paths: dict
        Dictionary of alignment paths.
    """

    os.makedirs(save_dir, exist_ok=True)

    alignment_paths = {}
    for tx_id in tqdm(tx_ids, 
                      desc="Aligning haplotypes"):
        
        # Check if the alignment already exists
        out_file = os.path.join(save_dir, f"{tx_id}.fasta.gz")
        
        if os.path.exists(out_file) and not force:
            alignment_paths[tx_id] = out_file
            continue

        # Create the input FASTA file (unaligned)
        fasta_paths = hs.haplotypes_to_fasta(tx_ids=tx_id,
                                             save_dir=save_dir,
                                             merge=False, 
                                             verbose=False)
        fasta_path = fasta_paths[list(fasta_paths.keys())[0]] 

        # Create the ClustalOmega command
        cline = aligner(infile=fasta_path, 
                            outfile=out_file, 
                            verbose=False, 
                            auto=True)
        
        # Run the ClustalOmega command
        try:
            cline()
        except Exception as e:
            if error:
                raise e
            else:
                print(e)
                continue

        alignment_paths[tx_id] = out_file
    
    return alignment_paths


def get_weighted_alignment(alignment, 
                           hap_df, 
                           total_samples=2504,
                           freq_col='1000GENOMES:phase_3:ALL'):
    """
    Create a new alignment object that is weighted by the population frequencies, 
    by adding that haplotype equal to the number of times it appears in the population (frequency * 2600)

    Parameters:
    -----------
    alignment: Alignment
        The alignment to weight.
    hap_df: pd.DataFrame
        The haplotype dataframe.
    total_samples: int
        The total number of samples in the population.
    freq_col: str
        The column name of the population frequency.

    Returns:
    --------
    weighted_alignment: Alignment
        The weighted alignment.
    """

    # Get the global population frequency
    global_freqs = hap_df[freq_col].astype(float).fillna(0)

    # Create a new weighted alignment
    weighted_records = []
    for idx, row in hap_df.iterrows():
        haplotype = row['haplotype']
        freq = global_freqs[idx]
        # Calculate how many times to add this haplotype (frequency * 2600)
        count = int(freq * total_samples)
        if count > 0:
            # Get the corresponding sequence from the alignment
            for record in alignment:
                if record.id == haplotype:
                    # Add this sequence multiple times based on frequency
                    for i in range(count):
                        new_record = SeqRecord(
                            Seq(str(record.seq)),
                            id=f"{record.id}_{i}",
                            description=""
                        )
                        weighted_records.append(new_record)
                    break

    # Create an Alignment object instead of MultipleSeqAlignment
    weighted_alignment = Alignment(weighted_records)
    return weighted_alignment


def get_consensus_sequences(alignment_paths,
                             weighted = True,
                             identity_threshold=0):
    """
    Get the consensus sequence from each alignment file.

    Parameters:
    -----------
    alignment_paths: dict
        The alignment paths.
    weighted: bool
        Whether to weight the alignment by the population frequencies.
    identity_threshold: float
        The identity threshold for the consensus sequence.

    Returns:
    --------
    consensus_seqs: dict
        The consensus sequences.
    consensus_in_cohort: dict
        Whether the consensus sequence is in the population frequencies.
    """

    consensus_seqs = {}
    consensus_in_cohort = {}
    for tx_id, alignment_path in tqdm(alignment_paths.items(), 
                                       desc="Generating consensus sequences"):
        
        #### Haplotypes ####
        # Get the haplotypes
        haplotypes = hs.get_haplotypes(tx_ids=tx_id, verbose=False, leave=False)
        # Convert the haplotypes to a dataframe
        hap_df = hs.haplotypes_to_df(haplotypes).reset_index(drop=False)

        # Merge all sets of amino acids across all sequences
        all_amino_acids = set().union(*hap_df['sequence'].apply(lambda x: set(x)))
        all_amino_acids

        # Get the population frequencies
        pop_freqs, populations, cohorts = hs.get_haplotype_freqs(haplotypes)
        pop_freqs_df = hs.pop_freqs_to_df(pop_freqs)

        # Merge the haplotype dataframe with the population frequencies dataframe
        hap_df = hap_df.merge(pop_freqs_df, on='haplotype', how='left')

        #### Alignment ####
        # Read the alignment using Bio.Align instead of Bio.AlignIO
        alignment = AlignIO.read(alignment_path, format="fasta")

        # Convert to new-style Alignment object
        if weighted:
            new_alignment = get_weighted_alignment(alignment, hap_df)
        else:
            new_alignment = alignment.alignment

        # Create a Motif object from the alignment with protein alphabet
        motif = Motif(alphabet="".join(sorted(all_amino_acids)), 
                    alignment=new_alignment) 

        # Get the consensus sequence using the counts property
        consensus_seqs[tx_id] = motif.counts.calculate_consensus(identity=identity_threshold)

        #### Consensus in cohort ####
        # Check if the consensus sequence is in the population frequencies 
        consensus_in_cohort[tx_id] = str(consensus_seqs[tx_id]) in hap_df['haplotype'].tolist()

    return consensus_seqs, consensus_in_cohort
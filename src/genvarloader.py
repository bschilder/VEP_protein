import src.utils as utils
import src.pyensembl as PYE

import os
import pandas as pd
import genvarloader as gvl
import numba as nb
import numpy as np
import polars as pl
import seqpro as sp
import pooch
from tqdm.auto import tqdm

def prepare_example(save_dir="/grid/koo/home/schilder/projects/GenomeEncoder/data/gvl",
                    bgzip_exec = "~/.conda/envs/genome-loader/bin/bgzip"):
    
    os.chdir(save_dir)
    # GRCh38 chromosome 22 sequence
    reference = pooch.retrieve(
        url="https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz",
        known_hash="sha256:974f97ac8ef7ffae971b63b47608feda327403be40c27e391ee4a1a78b800df5",
        progressbar=True,
    ) 
    os.system(f"gzip -dc {reference} | {bgzip_exec} > {reference[:-3]}.bgz")
    reference = reference[:-3] + ".bgz"
    
    # Set up pooch to retry downloads on timeout
    pooch.HTTPDownloader.timeout = 60  # Increase timeout to 60 seconds
    pooch.HTTPDownloader.max_retries = 3  # Retry failed downloads up to 3 times

    # PLINK 2 files
    variants = pooch.retrieve(
        url="doi:10.5281/zenodo.13656224/1kGP.chr22.pgen",
        known_hash="md5:31aba970e35f816701b2b99118dfc2aa",
        progressbar=True,
        fname="1kGP.chr22.pgen",
    )
    pooch.retrieve(
        url="doi:10.5281/zenodo.13656224/1kGP.chr22.psam",
        known_hash="md5:eefa7aad5acffe62bf41df0a4600129c",
        progressbar=True,
        fname="1kGP.chr22.psam",
    )
    pooch.retrieve(
        url="doi:10.5281/zenodo.13656224/1kGP.chr22.pvar",
        known_hash="md5:5f922af91c1a2f6822e2f1bb4469d12b",
        progressbar=True,
        fname="1kGP.chr22.pvar",
    ) 
    # BED
    bed_path = pooch.retrieve(
        url="doi:10.5281/zenodo.13656224/chr22_egenes.bed",
        known_hash="md5:ccb55548e4ddd416d50dbe6638459421",
        progressbar=True,
    )

    return reference, variants, bed_path


def create_db(reference=None, 
              bed=None, 
              variants=None, 
              save_path="gvl/geuvadis.chr22.gvl", 
              force = False):

    if not os.path.exists(save_path) or force is True:
        print("Creating database...")
        gvl.write(
            path=save_path,
            bed=bed.filter(pl.col("chrom")=="chr22"),
            variants=variants,
            # bigwigs=gvl.BigWigs.from_table(name="depth", table=bigwig_table),
            length=2**15, # <-- required to select sequence subsets afterwards
    #         max_jitter=128,
            max_mem=16*2**30,
            overwrite=True,
        )
    print("Connecting to database")
    ds = gvl.Dataset.open(save_path, reference=reference)
    return ds

def read_bed(bed_path="/grid/koo/home/schilder/projects/GenomeEncoder/data/ucsc/ucsc_knownGene_CDS.bed"):
    bed = gvl.read_bedlike(bed_path)
    print(bed.shape)
    return bed

def get_spliced_seqs(db, bed, sample = 0):
    from itertools import chain
    
    seqs = {}
    for row in range(bed.shape[0])[:3]:
        transcript_id = bed[row]['name'][0]
        nuc = db.sel(regions=bed[row],
                     samples=db.samples[sample]
                     )
        blockStarts = [int(x) for x in bed[row]['blockStarts'][0].strip(',').split(",")]
        blockSizes = [int(x) for x in bed[row]['blockSizes'][0].strip(',').split(",")] 
        nuc_spliced = list(chain.from_iterable([nuc[blockStarts[i]:(blockStarts[i]+blockSizes[i])] for i in range(len(blockStarts))]))[0]
        seqs[transcript_id] = nuc_spliced
    return seqs

    # ds.sel(regions=ds.get_bed()[:1], samples=ds.samples[:5]).shape

def map_bed(ds, bed, suffix = "_tx"):
    """
    Map a user-defined input BED file to the regions (windows) of the GVL database.
    Returns a table containing the regions that overlap with the input BED file.
    """
    regions = ds.get_bed().with_row_index(name="region_idx")
    regions_tx = regions.join(bed, how="cross", suffix=suffix)
    regions_tx = regions_tx.filter(
        (pl.col("chromStart") <= pl.col(f"chromEnd{suffix}")) & 
        (pl.col("chromEnd") >= pl.col(f"chromStart{suffix}"))
    )
    return regions_tx

def get_tx_offset(regions_tx):
    tx_len = regions_tx['chromEnd_tx'][0] - regions_tx['chromStart_tx'][0]
    tx_start = regions_tx['chromStart_tx'][0] - regions_tx['chromStart'][0]
    tx_end = regions_tx['chromEnd_tx'][0] - regions_tx['chromStart'][0] 
    return tx_start, tx_end, tx_len

def get_tx_seqs(ds, regions_tx, sample = 0, **kwargs):
    tx_start, tx_end, tx_len = get_tx_offset(regions_tx)
    seqs_block = ds.isel(regions=regions_tx['region_idx'],
                         samples=sample, 
                         **kwargs) 
    seqs = seqs_block[:,:,tx_start:tx_end] 
    if seqs.shape[2] != tx_len:
        raise ValueError(f"Sequence length does not match: {seqs.shape[2]} != {tx_len}")
    return seqs

def get_tx_seqs_spliced(ds, 
                        tx, 
                        regions_tx, 
                        sample = 0,
                        error = [True,False],
                        **kwargs):
    
    tx_id = regions_tx['name'][0].split(".")[0]
    tx = PYE.get_transcript(tx_id)
    seqs = get_tx_seqs(ds, regions_tx, sample, **kwargs)
    start = tx.first_start_codon_spliced_offset
    end = tx.last_stop_codon_spliced_offset 
    seqs_spliced = seqs[:,:,start:end+1]
    
    # Check sequence divisibility
    if seqs_spliced.shape[-1] % 3 != 0:
        msg = f"Sequence length not divisible by 3: {seqs_spliced.shape[-1]}"
        if error[0] is True:
            raise ValueError(msg)
        else:
            print(msg)
    # Check sequence length
    tx_len = len(tx.coding_sequence)
    if seqs_spliced.shape[-1] != tx_len:
        msg = f"Sequence length does not match: {seqs_spliced.shape[-1]} != {tx_len}"
        if error[1] is True:
            raise ValueError(msg)
        else:
            print(msg)
    # Return sequence
    return seqs_spliced


def string_to_bytearray(str):
    return np.array(list(str)).astype('|S1')

def bytearray_to_string(byte_arr):
    return byte_arr.tobytes().decode()

def bytearray_to_bioseq(byte_arr):

    from Bio.Seq import Seq 
    # Sample, Ploid, Sequence
    if byte_arr.ndim == 1:
        return [Seq(bytearray_to_string(byte_arr))]
    elif byte_arr.ndim == 2:
        ploid_idx = range(byte_arr.shape[-2])
        return [Seq(bytearray_to_string(byte_arr[idx,:])) for idx in ploid_idx]
    elif byte_arr.ndim == 3:
        ploid_idx = range(byte_arr.shape[-2])
        return [Seq(bytearray_to_string(byte_arr[:,idx,:])) for idx in ploid_idx]
    else:
        raise ValueError(f"Invalid number of dimensions: {byte_arr.ndim}")

def calculate_sequence_similarities(ds, 
                                    tx_sample_seqs, 
                                    add_exonic=False,
                                    level = ["nt","aa"],
                                    tx_ids=None):
    """
    Calculate pairwise sequence similarities between GVL and Haplosaurus sequences.
    
    Args:
        ds: GVL dataset object
        tx_sample_seqs: Dictionary of Haplosaurus sequences by transcript and sample
        tx_ids: List of transcript IDs to process. 
            If None, all transcripts in the GVL that are present in the Haplosaurus sequences will be processed.
        level: "nt" for nucleotide level, "aa" for amino acid level
        add_exonic: Whether to also run comparisons for the ds using ds.with_settings(var_filter='exonic'). 
            This will be designated as GVLex[0] and GVLex[1]
    Returns:
        DataFrame containing all pairwise sequence similarities
    """
    level = utils.one_only(level)
    
    all_seq_sim_data = []
    if tx_ids is None:
        tx_ids = utils.intersect(ds.get_bed()['name'],
                                 tx_sample_seqs.keys())
    
    for tx_id in tqdm(tx_ids, desc="Processing transcripts", leave=True):
        samples = utils.intersect(ds.samples, tx_sample_seqs[tx_id].keys())

        tx_metadata = ds.spliced_regions.filter(pl.col("splice_id")==tx_id)

        for sample in tqdm(samples, desc="Processing samples", leave=False):
            gvl_seqs = ds[tx_id, sample][0]

            # Haplosaurus
            hap_seqs = tx_sample_seqs[tx_id][sample]
            if len(hap_seqs)==0:
                continue

            # Define sequence sources with labels
            sequences = {
                "GVL[0]": gvl_seqs[0],
                "GVL[1]": gvl_seqs[1],
                "HS[0]": string_to_bytearray(hap_seqs[0]),
                "HS[1]": string_to_bytearray(hap_seqs[1])
            }

            # This method only injects variants that are fully within exons (not just overlapping)
            if add_exonic is True:
                dse = ds.with_settings(var_filter='exonic')
                gvlex_seqs = dse[tx_id, sample][0]
                sequences["GVLex[0]"] = gvlex_seqs[0]
                sequences["GVLex[1]"] = gvlex_seqs[1]
             
            # Calculate similarity for all unique pairs
            for i, (name1, seq1) in enumerate(sequences.items()):
                for i2, (name2, seq2) in enumerate(sequences.items()):
                    # Skip self-comparisons
                    if i2 == i:
                        continue
                    all_seq_sim_data.append({
                        'transcript_id': tx_id,
                        'gene_name': tx_metadata['gene_name'][0][0],
                        'exon_count': len(tx_metadata['regions'][0]),
                        'sample': sample,
                        'seq1': name1,
                        'seq2': name2,
                        'seq1_len': len(seq1),
                        'seq2_len': len(seq2),
                        'seq_sim': utils.get_sequence_similarity(seq1, seq2)
                    })
    
    # Create the final dataframe with all results
    seq_sim = pd.DataFrame(all_seq_sim_data)
    # Add extra columns
    seq_sim['group1'] = seq_sim['seq1'].str.split('[').str[0]
    seq_sim['group2'] = seq_sim['seq2'].str.split('[').str[0]
    seq_sim['seq1_phase'] = seq_sim['seq1'].str.split('[').str[1].str.split(']').str[0]
    seq_sim['seq2_phase'] = seq_sim['seq2'].str.split('[').str[1].str.split(']').str[0]
    # Only apply phase_match when comparing sequences from the same group
    seq_sim['phase_match'] = np.where(
        seq_sim['group1'] == seq_sim['group2'],
        seq_sim['seq1_phase'] == seq_sim['seq2_phase'],
        np.nan
    )
    return seq_sim


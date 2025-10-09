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
import zipfile  
from IPython.display import clear_output 

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
                    gene_cols = utils.intersect(tx_metadata.columns, 
                                                ['gene_name','hgnc'])
                    all_seq_sim_data.append({
                        'transcript_id': tx_id,
                        'gene_name': tx_metadata[gene_cols[0]][0],
                        'exon_count': len(tx_metadata['index'][0]),
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



def load_gvl_datasets(gvl_zip_paths, suffix=".gvl"):
    """
    Unzips and loads GVL datasets from a list of zip file paths.
    Returns a dictionary of {key: gvl.Dataset}.
    """
    datasets = {}
    for zip_path in gvl_zip_paths:
        extract_dir = zip_path.replace(".zip", "")
        # Unzip if not already unzipped
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                print(f"Extracted zip to {extract_dir}")

        # Find all folders matching chr*_dataset.gvl/ in the extracted directory
        chr_gvl_folders = [
            f for f in os.listdir(extract_dir)
            if f.endswith(suffix) and f.startswith('chr')
        ]

        for folder in chr_gvl_folders:
            folder_path = os.path.join(extract_dir, folder)
            try:
                ds = gvl.Dataset.open(folder_path)
                # Use a unique key for each dataset, e.g. include the parent directory name
                key = f"{os.path.basename(extract_dir)}_{folder}"
                datasets[key] = ds
                print(f"Loaded dataset for {key}")
            except Exception as e:
                print(f"Error loading dataset for {folder}: {e}")
            clear_output(wait=True)
    return datasets

def get_n_variants_df(datasets):
    """
    Given a dictionary of {key: gvl.Dataset}, returns a DataFrame with
    columns: cohort, chrom, n_variants.
    """ 
    n_variants = []
    for key, ds in tqdm(datasets.items()):
        regions = ds.regions.to_pandas()
        regions["region_id"] = regions["chrom"] +":" +regions["chromStart"].astype(str) + "-" + regions["chromEnd"].astype(str)

        n_variants.append(pd.DataFrame(
            {
                "cohort": key.split("_")[0].upper(),
                "chrom": ds.contigs[0],
                "region": np.repeat(regions["region_id"].values, ds.n_samples*2),
                "sample": np.repeat(np.repeat(ds.samples, 2), ds.regions.shape[0]),
                "n_variants": ds.n_variants().flatten()
            }
        ))
    n_variants_df = pd.concat(n_variants)
    return n_variants_df



def filter_region_to_site(region_to_site,
                          site_filters=None,
                          verbose=True):
    """
    Filter the region_to_site to only include one region per site.
    This avoids unncessary iterations over multiple regions per site.
    This function also filters the region_to_site to only include sites that are in the site_filters.

    NOTE:
    sites_ds.sites contains all the sites provided to DatasetWithSites.
    site_ds.rows contains all the sites provided to DatasetWithSites mapped onto each region in the BED file input to the GVL dataset, 
        resulting in a many:many mapping between regions and sites

    Args:
        region_to_site (pl.DataFrame): A polars DataFrame with columns "region_idx" and "site_idx"
        site_filters (dict): A dictionary of site filters.
            Keys are column names and values are lists of values to filter on.
            Values can be lists, integers, or strings.
        verbose (bool): Whether to print verbose output.

    Returns:
        pl.DataFrame: A polars DataFrame with one row per site

    Example:
        >>> region_to_site = pl.DataFrame({"region_idx": [0, 0, 1, 1, 2, 2], "site_idx": [0, 1, 0, 1, 0, 1]})
        >>> filter_region_to_site(region_to_site)
        # Returns a DataFrame with one row per site
        #   region_idx  site_idx
        # 0          0        0
        # 1          1        1
    """ 
    # print(region_to_site.to_pandas())
    shape0 = region_to_site.shape 

    region_to_site = (
        region_to_site
        .with_row_index()
        .filter(pl.col("region_idx") == pl.col("site_idx")) 
    )
    
     # Filter the sites without affecting the structure of the GVL/xarray datasets
    if site_filters is not None:
        for fk, fv in site_filters.items():
            if verbose>1:
                print(f"Filtering {fk} with {fv}")
            if isinstance(fv, list):
                region_to_site = region_to_site.filter(pl.col(fk).is_in(fv))
            elif isinstance(fv, pd.core.arrays.string_.StringArray):
                region_to_site = region_to_site.filter(pl.col(fk).is_in(list(fv)))
            elif isinstance(fv, int):
                region_to_site = region_to_site.filter(pl.col(fk)>=fv)
            elif isinstance(fv, str):
                region_to_site = region_to_site.filter(pl.col(fk).str.contains(fv)) 
            else:
                if verbose:
                    print(f"Filtering with type {type(fv)} is not implemented") 
                    
    shape1 = region_to_site.shape 

    # Report if no sites are left after filtering
    if verbose or region_to_site.shape[0]==0:
        print("Sites before filter_region_to_site: ", shape0)
        print("Sites after site_filters: ", shape1)
        print("Sites after region_idx==site_idx filter: ", region_to_site.shape)
    
    return region_to_site
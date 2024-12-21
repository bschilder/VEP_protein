import code.src.genvarloader as gvl
import numba as nb
import numpy as np
import polars as pl
import seqpro as sp
import pooch
import os
os.chdir("/grid/koo/home/schilder/projects/GenomeEncoder/data/gvl")
from tqdm.auto import tqdm

def prepare_example():
    # GRCh38 chromosome 22 sequence
    reference = pooch.retrieve(
        url="https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz",
        known_hash="sha256:974f97ac8ef7ffae971b63b47608feda327403be40c27e391ee4a1a78b800df5",
        progressbar=True,
    )
    bgzip_exec = "~/.conda/envs/genome-loader/bin/bgzip"
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


def create_db(reference, bed, variants):
    import os
    import polars as pl
    ds_path = "geuvadis.chr22.gvl"
    force = False
    if not os.path.exists(ds_path) or force is True:
        gvl.write(
            path=ds_path,
            bed=bed.filter(pl.col("chrom")=="chr22"),
            variants=variants,
            # bigwigs=gvl.BigWigs.from_table(name="depth", table=bigwig_table),
            length=2**15, # <-- required to select sequence subsets afterwards
    #         max_jitter=128,
            max_mem=16*2**30,
            overwrite=True,
        )
    ds = gvl.Dataset.open(ds_path, reference=reference)
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
                    samples=db.samples[sample])
        blockStarts = [int(x) for x in bed[row]['blockStarts'][0].strip(',').split(",")]
        blockSizes = [int(x) for x in bed[row]['blockSizes'][0].strip(',').split(",")] 
        nuc_spliced = list(chain.from_iterable([nuc[blockStarts[i]:(blockStarts[i]+blockSizes[i])] for i in range(len(blockStarts))]))[0]
        seqs[transcript_id] = nuc_spliced
    return seqs

    # ds.sel(regions=ds.get_bed()[:1], samples=ds.samples[:5]).shape

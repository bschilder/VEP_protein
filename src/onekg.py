import sys
sys.path.append("code")
from src.utils import download_parallel


def get_chr_1kg(vcf_file):
    import os
    return os.path.basename(vcf_file).split(".")[-3]

def list_remote_fasta(url="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/20130502.phase3.analysis.sequence.index",
                       **kwargs):
    import pandas as pd
    seqin = pd.read_csv(url, sep="\t", **kwargs)
    return seqin

def list_remote_vcf(vcf_ftp = "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL/"):
    import pandas as pd
    import os
    manifest = pd.read_csv(vcf_ftp+"20190312_biallelic_SNV_and_INDEL_MANIFEST.txt", sep="\t", header=None)
    manifest = manifest.loc[manifest[0].str.contains("ALL.chr")]
    manifest['url'] = vcf_ftp+manifest[0].str.replace(r'^\./', '', regex=True)
    manifest['local'] = os.path.abspath("../data/1KG/")+manifest[0].str.replace(r'^\.', '', regex=True)
    return manifest

def download_vcfs(manifest=None):
    if manifest is None:
        manifest = list_remote_vcf()
    url_path_pairs = list(zip(manifest['url'], manifest['local']))
    download_parallel(url_path_pairs)


def get_pop(url="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/20131219.populations.tsv"):
    import pandas as pd
    pops = pd.read_csv(url, sep="\t")
    pops.index = pops['Population Code'].tolist()
    return pops

def get_ped(url="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20200731.ALL.ped"):
    import pandas as pd
    ped = pd.read_csv(url, sep="\t")
    ped.index = ped['Individual ID'].tolist()
    return ped

def get_sample_metadata():
    ped = get_ped()
    pop = get_pop()
    sample_metadata = ped.merge(pop, left_on='Population', right_index=True)
    return sample_metadata

def get_annotation_vcf(chrom,
                       base_url="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/functional_annotation/filtered/"):
    import pysam
    chrom = "chr"+str(chrom).replace("chr", "")
    url = f"{base_url}ALL.{chrom}.phase3_shapeit2_mvncall_integrated_v5.20130502.sites.annotation.vcf.gz"  
    return pysam.VariantFile(url)

def query_annotation_vcf(vcf,
                         rec,
                         start=None,
                         stop=None):
    chrom = rec.contig.replace("chr", "")
    if start is None:
        start = rec.pos
    if stop is None:
        stop = rec.pos+1
    return vcf.fetch(chrom, start, stop)
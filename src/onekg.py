import os
import pandas as pd
import pooch
from tqdm import tqdm

import src.utils as utils
import src.config as config


DEFAULT_KEY = "1000_Genomes_30x_on_GRCh38"

def get_chr_1kg(vcf_file):
    return os.path.basename(vcf_file).split(".")[-3]

def list_remote_fasta(url="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/20130502.phase3.analysis.sequence.index",
                       **kwargs):
    """
    Retrieve the 1000 Genomes Project FASTA sequence index from a remote URL.
    
    Parameters:
    -----------
    url : str, optional
        URL to the 1000 Genomes Project sequence index file.
        Default is the phase3 analysis sequence index.
    **kwargs : dict
        Additional keyword arguments to pass to pd.read_csv().
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the sequence index information.
    """
    seqin = pd.read_csv(url, sep="\t", **kwargs)
    return seqin

def get_ftp_dict():
    """
    There are many different collections within the 1000 Genomes Project data: 
    https://www.internationalgenome.org/data-portal/data-collection
    This function returns a dictionary of the FTP URLs and manifests for the 1000 Genomes Project.
    
    Returns:
    --------
    dict
        Dictionary containing the FTP URLs and manifests for the 1000 Genomes Project.
    """
    return {
        '1000_Genomes_30x_on_GRCh38':{
            'description': 'VCFs from high-coverage WGS with SNVs, INDELs, and SVs. Details: https://www.internationalgenome.org/data-portal/data-collection/30x-grch38',
            'url': "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/",
            'manifest': 'https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/20220804_manifest.txt',
            'manifest_cols': ['fname', 'md5'],
            'manifest_sep': ' ',
            'pop': 'https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/20131219.populations.tsv',
            'ped': 'https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20200731.ALL.ped',
            'ref': 'https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa'
        },
        '1000_Genomes_on_GRCh38':{
            'description': 'VCFs from low-coverage WGS with SNVs and INDELs. Details: https://www.internationalgenome.org/data-portal/data-collection/grch38',
            'url': "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL/",
            'manifest': 'https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL/20190312_biallelic_SNV_and_INDEL_MANIFEST.txt',
            'manifest_cols': ['fname', 'size', 'md5'],
            'manifest_sep': '\t',
            'pop': 'https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/20131219.populations.tsv',
            'ped': 'https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20200731.ALL.ped',
            'ref': 'https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa'
        },
        'Human_Genome_Diversity_Project':{
            'description': 'VCFs from WGS with SNVs and INDELs. Details: https://www.internationalgenome.org/data-portal/data-collection/hgdp',
            'url': "https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/",
            'manifest': None,
            'manifest_cols': None,
            'manifest_sep': None,
            'pop': None,
            'ped': 'https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/metadata/hgdp_wgs.20190516.metadata.txt',
            'ref': 'https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa'
        }
    }

def _get_cache_dir(key):
    return os.path.join(config.DATA_DIR, key, "vcf")

def list_remote_vcf(key=DEFAULT_KEY, 
                    cache=None, 
                    add_key_subdir=True):
    """
    Create a manifest of 1000 Genomes Project VCF files available for download.
    
    Parameters:
    -----------
    key : str
        The key to the 1000 Genomes Project data collection (with underscores instead of spaces). 
        See `get_ftp_dict()` for options or go to: 
        https://www.internationalgenome.org/data-portal/data-collection
    cache : str
        The path to the cache directory.
    add_key_subdir : bool
        Whether to add the key subdirectory to the local path.
        
    Returns:
    --------
    pandas.DataFrame
        Manifest DataFrame containing file information with columns:
        - fname: Filename
        - size: File size
        - md5: MD5 checksum
        - url: Complete URL for downloading
        - local: Local path where file will be saved
    """

    if cache is None:
        cache = _get_cache_dir(key)

    # Get ftp dict
    ftp_dict = get_ftp_dict()
    ftp = ftp_dict[key]['url']

    if key == 'Human_Genome_Diversity_Project':
        manifest = _get_hgdp_manifest()
    else:
        # Get manifest file
        manifest = pd.read_csv(ftp_dict[key]['manifest'], 
                               sep = ftp_dict[key]['manifest_sep'], 
                               names = ftp_dict[key]['manifest_cols'],
                               header = None)
    if key == "1000_Genomes_on_GRCh38":
        manifest = manifest.loc[manifest['fname'].str.contains("ALL.chr")]
        manifest.insert(0, "chrom", manifest["fname"].str.extract(r'(chr[0-9XYM]+)', expand=False))

    elif key == "Human_Genome_Diversity_Project":
        # Use regex to extract 'chr#' pattern from the filename for the chrom column
        manifest.insert(0, "chrom", manifest["fname"].str.extract(r'(chr[0-9XYM]+)', expand=False))
    manifest['url'] = ftp+manifest['fname'].str.replace(r'^\./', '', regex=True)
    
    # Add key subdirectory if requested
    if add_key_subdir:
        # print(cache)
        manifest['local'] = cache+"/"+manifest['fname'].str.replace(r'^\.', '', regex=True)
    
    manifest["key"] = key

    return manifest

def download_vcfs(key=None,
                  manifest=None,
                  skip_checks=False,
                  cache=None,
                  timeout=60*30,
                  as_dict=True,
                  verbose=False):
    """
    Download VCF files from the 1000 Genomes Project using pooch.
    
    Parameters:
    -----------
    manifest : pandas.DataFrame, optional
        Manifest DataFrame containing file information.
        If None, will be generated using list_remote_vcf().
        
    Returns:
    --------
    list
        List of local file paths where the VCF files were saved.
    """
    # Set pooch timeout for downloading files
    pooch.HTTPDownloader.timeout = timeout
    pooch.HTTPDownloader.max_retries = 3  # Retry failed downloads up to 3 times
    
    # Get manifest
    if manifest is None:
        manifest = list_remote_vcf(key=key, 
                                   cache=cache)
    else:
        key = manifest["key"].iloc[0]

    # Get cache directory
    cache = _get_cache_dir(key)
    os.makedirs(cache, exist_ok=True)
    
    local_files = {}
    for _, row in tqdm(manifest.iterrows(), 
                       total=len(manifest),
                       desc="Downloading VCF files"):
        if os.path.exists(row['local']) and skip_checks:
            if verbose:
                print(f"Skipping {row['local']} because it already exists")
            local_file = row['local']
        else:
            local_file = pooch.retrieve(
                url=row['url'],
                fname=os.path.basename(row['local']),
                path=os.path.dirname(row['local']),
                known_hash="md5:"+row['md5'] if 'md5' in row else None,
                progressbar=True
            )
        # Construct file key
        fkey = row["chrom"]
        if local_file.endswith(".tbi"):
            fkey = fkey+"_idx"
        else:
            fkey = fkey+"_vcf"
        local_files[fkey] = local_file
    
    if as_dict:
        return local_files
    else:
        return list(local_files.values())


def get_pop(key=DEFAULT_KEY):
    """
    Retrieve population information from the 1000 Genomes Project.
    
    Parameters:
    -----------
    url : str, optional
        URL to the populations TSV file.
        Default is the official 1000 Genomes Project populations file.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing population information with 'Population Code' as index.
    """
    ftp_dict = get_ftp_dict()
    url = ftp_dict[key]['pop']
    if url is None:
        return None
    pops = pd.read_csv(url, sep="\t")
    pops.index = pops['Population Code'].tolist()
    return pops

def get_ped(key=DEFAULT_KEY):
    """
    Retrieve pedigree information from the 1000 Genomes Project.
    
    Parameters:
    -----------
    url : str, optional
        URL to the pedigree (PED) file.
        Default is the official 1000 Genomes Project integrated call samples file.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing pedigree information with 'Individual ID' as index.
    """

    # Get the file URL
    ftp_dict = get_ftp_dict()
    url = ftp_dict[key]['ped']
    if url is None:
        return None
    ped = pd.read_csv(url, sep="\t")
    if key == 'Human_Genome_Diversity_Project':
        ped.rename(columns={'sample': 'Individual ID'}, inplace=True)
        # Using mappings from https://doi.org/10.1101/2023.01.23.525248
        # Map each region/population string to a superpopulation code (EAS, CSA, MID, EUR, AFR, AMR, OCE, SAS, or REF)
     
        ped['superpopulation'] = ped['region'].map(SUPERPOP_DICT)

    # Set index to sample column
    ped.index = ped['Individual ID'].tolist() 
    
    return ped

SUPERPOP_DICT = {
    # Self
    "AFR": "AFR",
    "AMR": "AMR",
    "CSA": "CSA",
    "EAS": "EAS",
    "EUR": "EUR",
    "MID": "MID",
    "OCE": "OCE",
    "SAS": "SAS",
    "EUR,AFR": "AFR", # If mixed, assign to AFR 

    # REF
    "REF": "REF",

    # Spelled out
    "AFRICA": "AFR",
    "AMERICA": "AMR",
    "CENTRAL_SOUTH_ASIA": "CSA",
    "EAST_ASIA": "EAS",
    "EUROPE": "EUR",
    "MIDDLE_EAST": "MID",
    "OCEANIA": "OCE",
    "SOUTH_ASIA": "SAS",

    # Human Genome Diversity Project
    "Africa (HGDP)": "AFR",
    "Africa (SGDP)": "AFR",
    "Africa (SGDP),Africa (HGDP)": "AFR",
    "African Ancestry": "AFR",
    "African Ancestry,Africa (SGDP)": "AFR",
    "America (HGDP)": "AMR",
    "America (SGDP)": "AMR",
    "America (SGDP),America (HGDP)": "AMR",
    "American Ancestry": "AMR",
    "Central Asia and Siberia (SGDP)": "CSA",
    "Central South Asia (HGDP)": "CSA",
    "East Asia (HGDP)": "EAS",
    "East Asia (SGDP)": "EAS",
    "East Asia (SGDP),East Asia (HGDP)": "EAS",
    "East Asia (SGDP),East Asian Ancestry": "EAS",
    "East Asian Ancestry": "EAS",
    "Europe (HGDP)": "EUR",
    "European Ancestry": "EUR",
    "European Ancestry,West Eurasia (SGDP)": "EUR",
    "European Ancestry,African Ancestry": "AFR",  # If mixed, assign to AFR
    "Middle East (HGDP)": "MID",
    "Middle East (HGDP),Africa (SGDP)": "AFR",  # Assign to AFR, could also be MID
    "Oceania (HGDP)": "OCE",
    "Oceania (SGDP)": "OCE",
    "Oceania (SGDP),Oceania (HGDP)": "OCE",
    "South Asia (SGDP)": "SAS",
    "South Asia (SGDP),Central South Asia (HGDP)": "CSA",
    "South Asia (SGDP),South Asian Ancestry": "SAS",
    "South Asian Ancestry": "SAS",
    "West Eurasia (SGDP)": "EUR",
}
 
MISSING_SAMPLE_METADATA = {
    'HGDP00927': {
        'population': 'YRI',
        'population_code': 'YRI',
        'population_name': 'Yoruba',
        'superpopulation': 'AFR',
        'sex': 'male',
        'source': 'https://www.cellosaurus.org/CVCL_I927'
    },
    'HGDP01284': {
        'population': 'GWD',
        'population_code': 'GWD',
        'population_name': 'Gambian Mandinka',
        'superpopulation': 'AFR',
        'sex': 'male', 
        'source': 'https://www.cellosaurus.org/CVCL_I420'
    },
    'HGDP01307': {
        'population': 'CDX',
        'population_code': 'CDX',
        'population_name': 'Dai Chinese',
        'superpopulation': 'EAS',
        'sex': 'male', 
        'source': 'https://www.cellosaurus.org/CVCL_I481'
    },
    'HGDP00665': {
        'population': 'Sardinian',
        'population_code': 'Sardinian',
        'population_name': 'Sardinian',
        'superpopulation': 'EUR',
        'sex': 'male', 
        'source': 'https://www.cellosaurus.org/CVCL_I665'
    },
    'HGDP00998': {
        'population': 'Karitiana',
        'population_code': 'Karitiana',
        'population_name': 'Karitiana',
        'superpopulation': 'AMR',
        'sex': 'male',
        'source': 'https://www.cellosaurus.org/CVCL_I998'
    },
    'HGDP00521': {
        'population': 'French',
        'population_code': 'French',
        'population_name': 'French',
        'superpopulation': 'EUR',
        'sex': 'male',
        'source': 'https://www.cellosaurus.org/CVCL_I521'
    },
    'HGDP00778': {
        'population': 'Han Chinese',
        'population_code': 'CHB',
        'population_name': 'Han Chinese',
        'superpopulation': 'EAS',
        'sex': 'male',
        'source': 'https://www.cellosaurus.org/CVCL_I778'
    }
}

def get_sample_metadata(key=DEFAULT_KEY,
                        harmonized=True,
                        prohap_format=False,
                        ):
    """
    Retrieve and merge sample metadata from the 1000 Genomes Project.
    
    This function combines pedigree information with population data to create
    a comprehensive sample metadata DataFrame.
    
    Parameters:
    -----------
    key : str
        The key to the 1000 Genomes Project data collection.
    prohap_format : bool
        Whether to return the metadata in the format required by 
        [ProHap](https://github.com/ProGenNo/ProHap/wiki/Input-&-Usage#prohap).
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing merged sample metadata with individual information
        and population details.
    """

    if harmonized:
        sample_metadata = pd.read_csv("metadata/igsr_samples.tsv", sep="\t")
        sample_metadata = sample_metadata.rename(columns={"Sample name":"sample",
                                                        "Population elastic ID":"population",
                                                        "Population name":"population_name",
                                                        "Superpopulation code":"superpopulation",
                                                        "Superpopulation name":"superpopulation_name",
                                                        "Sex":"sex"})
        # Reassign non-standard superpopulation codes
        sample_metadata.loc[sample_metadata['superpopulation']=="EUR,AFR", "superpopulation"] = "AFR"
        
        # Some samples are missing metadata, so we add them manually
        for sample, metadata in MISSING_SAMPLE_METADATA.items():
            sample_metadata.loc[sample_metadata["sample"]==sample, "population"] = metadata["population"]
            sample_metadata.loc[sample_metadata["sample"]==sample, "population_code"] = metadata["population_code"]
            sample_metadata.loc[sample_metadata["sample"]==sample, "population_name"] = metadata["population_name"]
            sample_metadata.loc[sample_metadata["sample"]==sample, "superpopulation"] = metadata["superpopulation"]
            sample_metadata.loc[sample_metadata["sample"]==sample, "sex"] = metadata["sex"]
            
        sample_metadata.loc[sample_metadata['superpopulation'].isna(), "superpopulation"] = sample_metadata.loc[sample_metadata['superpopulation'].isna()]["superpopulation_name"].map(SUPERPOP_DICT)

        print(sample_metadata['superpopulation'].isna().sum(), "samples are missing superpopulation metadata")
    else:
        ped = get_ped(key=key)
        
        if key == 'Human_Genome_Diversity_Project':
            if prohap_format:
                ped.rename(columns={'Individual ID': 'Sample name',
                                    'sex': 'Sex',
                                    'population': 'Population code',
                                    'superpopulation': 'Superpopulation code'},
                                    inplace=True)
            return ped
        
        pop = get_pop(key=key, )
        if ped is None or pop is None:
            return None
        sample_metadata = ped.merge(pop, left_on='Population', right_index=True)
   
    # Convert to ProHap format
    if prohap_format:
        if harmonized:
            # The default 'Population code' has NAs
            sample_metadata = sample_metadata.drop(['Population code'], axis=1)
        
        # Rename cols
        sample_metadata.rename(columns={'Individual ID': 'Sample name',
                                        'sample': 'Sample name',
                                'Gender': 'Sex',
                                'sex': 'Sex', 
                                'Population Code': 'Population code',
                                'population': 'Population code',
                                'Population elastic ID': 'Population code',
                                'Super Population': 'Superpopulation code',
                                'superpopulation': 'Superpopulation code'
                                },
                                inplace=True)
    return sample_metadata

def get_annotation_vcf(chrom,
                       base_url="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/functional_annotation/filtered/"):
    """
    Retrieve a variant annotation VCF file for a specific chromosome.
    
    Parameters:
    -----------
    chrom : str or int
        Chromosome identifier (with or without 'chr' prefix).
    base_url : str, optional
        Base URL for the annotation VCF files.
        Default points to the 1000 Genomes Project functional annotation directory.
        
    Returns:
    --------
    pysam.VariantFile
        A pysam VariantFile object for the requested chromosome annotation file.
    """
    import pysam
    chrom = "chr"+str(chrom).replace("chr", "")
    url = f"{base_url}ALL.{chrom}.phase3_shapeit2_mvncall_integrated_v5.20130502.sites.annotation.vcf.gz"  
    return pysam.VariantFile(url)

def query_annotation_vcf(vcf,
                         rec,
                         start=None,
                         stop=None):
    """
    Query a variant annotation VCF file for a specific genomic region.
    
    Parameters:
    -----------
    vcf : pysam.VariantFile
        The variant annotation file to query.
    rec : object
        A record object with contig and pos attributes.
    start : int, optional
        Start position for the query. If None, uses rec.pos.
    stop : int, optional
        Stop position for the query. If None, uses rec.pos+1.
        
    Returns:
    --------
    iterator
        An iterator over the variants in the specified region.
    """
    chrom = rec.contig.replace("chr", "")
    if start is None:
        start = rec.pos
    if stop is None:
        stop = rec.pos+1
    return vcf.fetch(chrom, start, stop)


def _get_hgdp_manifest():
    """
    No manifest file is made available for the HGDP data.
    This function created a manifest file using info copied and pasted from: 
    https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/
    """
    import pandas as pd
    import io
    
    txt = """hgdp_wgs.20190516.full.chr1.vcf.gz       2019-06-06 14:28   26G  
    hgdp_wgs.20190516.full.chr1.vcf.gz.tbi   2019-06-06 14:29  221K  
    hgdp_wgs.20190516.full.chr10.vcf.gz      2019-06-06 14:46   15G  
    hgdp_wgs.20190516.full.chr10.vcf.gz.tbi  2019-06-06 14:47  128K  
    hgdp_wgs.20190516.full.chr11.vcf.gz      2019-06-06 14:48   15G  
    hgdp_wgs.20190516.full.chr11.vcf.gz.tbi  2019-06-06 14:49  128K  
    hgdp_wgs.20190516.full.chr12.vcf.gz      2019-06-06 14:50   15G  
    hgdp_wgs.20190516.full.chr12.vcf.gz.tbi  2019-06-06 14:51  128K  
    hgdp_wgs.20190516.full.chr13.vcf.gz      2019-06-06 14:52   11G  
    hgdp_wgs.20190516.full.chr13.vcf.gz.tbi  2019-06-06 14:53   94K  
    hgdp_wgs.20190516.full.chr14.vcf.gz      2019-06-06 14:55  9.8G  
    hgdp_wgs.20190516.full.chr14.vcf.gz.tbi  2019-06-06 14:56   86K  
    hgdp_wgs.20190516.full.chr15.vcf.gz      2019-06-06 14:57  9.1G  
    hgdp_wgs.20190516.full.chr15.vcf.gz.tbi  2019-06-06 14:58   82K  
    hgdp_wgs.20190516.full.chr16.vcf.gz      2019-06-06 14:59   10G  
    hgdp_wgs.20190516.full.chr16.vcf.gz.tbi  2019-06-06 15:00   80K  
    hgdp_wgs.20190516.full.chr17.vcf.gz      2019-06-06 15:01  9.4G  
    hgdp_wgs.20190516.full.chr17.vcf.gz.tbi  2019-06-06 15:02   81K  
    hgdp_wgs.20190516.full.chr18.vcf.gz      2019-06-06 15:03  8.5G  
    hgdp_wgs.20190516.full.chr18.vcf.gz.tbi  2019-06-06 15:04   77K  
    hgdp_wgs.20190516.full.chr19.vcf.gz      2019-06-06 15:05  7.8G  
    hgdp_wgs.20190516.full.chr19.vcf.gz.tbi  2019-06-06 15:06   53K  
    hgdp_wgs.20190516.full.chr2.vcf.gz       2019-06-06 14:30   26G  
    hgdp_wgs.20190516.full.chr2.vcf.gz.tbi   2019-06-06 14:31  230K  
    hgdp_wgs.20190516.full.chr20.vcf.gz      2019-06-06 15:07  7.7G  
    hgdp_wgs.20190516.full.chr20.vcf.gz.tbi  2019-06-06 15:08   61K  
    hgdp_wgs.20190516.full.chr21.vcf.gz      2019-06-06 15:09  4.6G  
    hgdp_wgs.20190516.full.chr21.vcf.gz.tbi  2019-06-06 15:10   36K  
    hgdp_wgs.20190516.full.chr22.vcf.gz      2019-06-06 15:11  5.1G  
    hgdp_wgs.20190516.full.chr22.vcf.gz.tbi  2019-06-06 15:12   37K  
    hgdp_wgs.20190516.full.chr3.vcf.gz       2019-06-06 14:32   21G  
    hgdp_wgs.20190516.full.chr3.vcf.gz.tbi   2019-06-06 14:33  189K  
    hgdp_wgs.20190516.full.chr4.vcf.gz       2019-06-06 14:34   21G  
    hgdp_wgs.20190516.full.chr4.vcf.gz.tbi   2019-06-06 14:35  182K  
    hgdp_wgs.20190516.full.chr5.vcf.gz       2019-06-06 14:36   19G  
    hgdp_wgs.20190516.full.chr5.vcf.gz.tbi   2019-06-06 14:37  171K  
    hgdp_wgs.20190516.full.chr6.vcf.gz       2019-06-06 14:38   18G  
    hgdp_wgs.20190516.full.chr6.vcf.gz.tbi   2019-06-06 14:39  163K  
    hgdp_wgs.20190516.full.chr7.vcf.gz       2019-06-06 14:40   18G  
    hgdp_wgs.20190516.full.chr7.vcf.gz.tbi   2019-06-06 14:41  152K  
    hgdp_wgs.20190516.full.chr8.vcf.gz       2019-06-06 14:42   16G  
    hgdp_wgs.20190516.full.chr8.vcf.gz.tbi   2019-06-06 14:43  139K  
    hgdp_wgs.20190516.full.chr9.vcf.gz       2019-06-06 14:44   14G  
    hgdp_wgs.20190516.full.chr9.vcf.gz.tbi   2019-06-06 14:45  117K  
    hgdp_wgs.20190516.full.chrX.vcf.gz       2020-02-12 11:17  9.8G  
    hgdp_wgs.20190516.full.chrX.vcf.gz.tbi   2020-02-12 11:17  146K  
    hgdp_wgs.20190516.full.chrY.vcf.gz       2020-02-12 11:22  262M  
    hgdp_wgs.20190516.full.chrY.vcf.gz.tbi   2020-02-12 11:22   16K"""
    
    # Clean up the data by splitting on whitespace and creating a proper dataframe
    lines = [line.strip() for line in txt.strip().split('\n')]
    data = []
    
    for line in lines:
        parts = line.split()
        # The filename is the first part
        filename = parts[0]
        # Date and time are the next two parts
        date = parts[1]
        time = parts[2]
        # Size is the last part
        size = parts[3]
        
        data.append([filename, f"{date} {time}", size])
    
    return pd.DataFrame(data, columns=["fname", "datetime", "size"])


def _rm_subplot_prefixes(g):
    g.set_titles(row_template='{row_name}', 
                 col_template='{col_name}')  # Only show model name without prefix
    
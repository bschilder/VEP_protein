import sys
sys.path.append("code")
from src.utils import as_list, intersect

# https://ensemblrest.readthedocs.io/en/latest/

def get_ensembl_client(client=None,
                       force=False):
    import ensembl_rest
    if client is None or force:
        client = ensembl_rest.EnsemblClient()
    return client

def get_ensembl_figshare(fname=["haplotype_analysis_database.zip",
                        "SupplementaryData1.tar.gz",
                        "SupplementaryData2-all_protein_haplotypes_GRCh37.fa.gz"
                        ],
                **kwargs):
    """
    The protein haplotype database we built from the 1000 Genomes Phase 3 
    dataset is available via [https://doi.org/10.6084/m9.figshare.5545084].
    
    Supplementary Data 4: Data used to test/validate Haplosaurus. 
    Data are available via [https://doi.org/10.6084/m9.figshare.6834083.v1]. 
    
    Supplementary Data 5: Fasta protein sequences of all protein haplotypes in 1000 Genomes. 
    Data are available via [https://doi.org/10.6084/m9.figshare.6834191.v1].
    """
    import pooch
    fname = as_list(fname)[0]
    if fname=="haplotype_analysis_database.zip": 
        # From: https://doi.org/10.6084/m9.figshare.5545084
        fname2 = pooch.retrieve(
                    url="doi:10.6084/m9.figshare.5545084.v1/haplotype_analysis_database.zip",
                    known_hash="md5:6a32ff8147e31019dda55338c222c938",
                    progressbar=True,
                    fname="haplotype_analysis_database.zip",
                    **kwargs
                )
        # Unzip the file
        import zipfile
        import os
        fname2_unzipped = os.path.join(os.path.dirname(fname2), "haplotype_analysis_database")
        with zipfile.ZipFile(fname2, 'r') as zip_ref:
            zip_ref.extractall(fname2_unzipped)
        return fname2_unzipped
    elif fname=="SupplementaryData1.tar.gz":
        # From: https://doi.org/10.6084/m9.figshare.6834083.v1
        fname2 = pooch.retrieve(
                    url="doi:10.6084/m9.figshare.5545084.v1/SupplementaryData1.tar.gz",
                    known_hash="md5:3eb8334e9ed8e26602b4b991748342b4",
                    progressbar=True,
                    fname="SupplementaryData1.tar.gz",
                    **kwargs
                )
    elif fname=="SupplementaryData2-all_protein_haplotypes_GRCh37.fa.gz":
        fname2 = pooch.retrieve(
                    url="doi:10.6084/m9.figshare.6834191.v1/SupplementaryData2-all_protein_haplotypes_GRCh37.fa.gz",
                    known_hash="md5:d1a4257bc75314891c5ee502169988d8",
                    progressbar=True,
                    fname="SupplementaryData2-all_protein_haplotypes_GRCh37.fa.gz",
                    **kwargs
                ) 
    else:
        raise ValueError(f"Unknown file name: {fname}")
    return fname2 

def filter_variants(variants_tx, 
                    params=None,
                    exact=False,
                    reverse=False,
                    verbose=True,
                    **kwargs):
    from tqdm.auto import tqdm 
    if params is None:
        return variants_tx
    variants_tx_filtered = []
    if len(variants_tx) == 0:
        if verbose:
            print(f"No variants to filter")
        return variants_tx
    for variant in tqdm(variants_tx, 
                        desc="Filtering variants", 
                        **kwargs):
        condition_met = []
        for k,v in params.items():
            if k in variant.keys():
                opts = as_list(variant[k]) 
                opts_selected = as_list(v)
                if exact:
                    condition_met.append(set(opts) == set(opts_selected))
                else:
                    opt_overlap = intersect(opts, opts_selected)
                    condition_met.append(len(opt_overlap) > 0)
        if reverse:
            condition_met = [not x for x in condition_met]
        if all(condition_met):
            variants_tx_filtered += [variant] 
    if verbose:
        print(f"{len(variants_tx_filtered)}/{len(variants_tx)} variants retained")
    return variants_tx_filtered

def map_tx2prot_ensembl(tx_df, 
                        batch_size=500,
                        verbose=True):
    """
    Map Ensembl Transcript ID to Ensembl Protein ID
    """
    import ensembl_rest
    client = ensembl_rest.EnsemblClient()
    # Map Protein ID to Transcript ID
    from tqdm.auto import tqdm 
    batches = [tx_df['Ensembl Protein ID'].tolist()[i:i+batch_size] for i in range(0, len(tx_df['Ensembl Protein ID'].tolist()), batch_size)]
    prot_to_tx = {}
    for batch in tqdm(batches[:3]):
        prot_to_tx.update(client.lookup_post(params={'ids':batch})) 
    return prot_to_tx


def get_variants(tx_id,
                client=None,
                params_query={'feature':'variation',
                                'so_term': 'SO:0001583',
                                'variant_set': 'clin_assoc'
                                },
                params_filter={'clinical_significance':['pathogenic']},
                error=False,
                verbose=False):
    # https://ensemblrest.readthedocs.io/en/latest/#ensembl_rest.EnsemblClient.overlap_id
    # Ensembl overlap API: https://rest.ensembl.org/documentation/info/overlap_id
    # Variant sets: https://useast.ensembl.org/info/genome/variation/species/sets.html
    client = get_ensembl_client(client=client)
    try:
        variants_tx = client.overlap_id(id=tx_id, 
                                        params=params_query
                                        )
    except Exception as e:
        if error:
            raise e
        else:
            if verbose:
                print(f"Error getting variants for {tx_id}: {e}")
            variants_tx = []
    variants_tx = filter_variants(variants_tx, 
                                  params=params_filter,
                                  verbose=verbose,
                                  leave=False
                                            )
    if verbose:
        print(len(variants_tx),"variants returned")
    return variants_tx

def get_proteoform_freq(hap_tx):
    population_frequencies = {x['name']:x['population_frequencies'] for x in hap_tx['protein_haplotypes']}
    return population_frequencies
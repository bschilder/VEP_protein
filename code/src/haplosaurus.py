import sys
sys.path.append("code")
from src.utils import as_list, intersect, load_pickle, save_pickle, load_json, save_json

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
                        batch_size=500):
    """
    Map Ensembl Transcript ID to Ensembl Protein ID
    """
    import ensembl_rest
    client = ensembl_rest.EnsemblClient()
    # Map Protein ID to Transcript ID
    from tqdm.auto import tqdm 
    batches = [tx_df['Ensembl Protein ID'].tolist()[i:i+batch_size] for i in range(0, len(tx_df['Ensembl Protein ID'].tolist()), batch_size)]
    prot_to_tx = {}
    for batch in tqdm(batches):
        prot_to_tx.update(client.lookup_post(params={'ids':batch})) 
    return prot_to_tx


def get_variants(tx_id,
                client=None,
                params_query={},
                params_filter={},
                error=False,
                verbose=False,
                **kwargs):
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
                                  leave=False,
                                  **kwargs
                                  )
    if verbose:
        print(len(variants_tx),"variants returned")
    return variants_tx

def get_variant_sets(tx_ids,
                     so_term = 'SO:0001583',
                     consequence_type = ['missense_variant'],
                     nagative_variant_set = '1kg_3',  # '1kg_3_com'
                     save_dir="data/haplosaurus/variant_sets",
                     force=False,
                     client=None,
                     cache_only=False,
                     verbose=True):
    from tqdm.auto import tqdm
    # Gather pathogenic/benign variants
    variant_sets = {}
    client = get_ensembl_client(client=client)
    # Iterate over transcripts
    for tx_id in tqdm(tx_ids,"Gathering variant sets"):
        save_path = f"{save_dir}/{'.'.join(consequence_type)}/{tx_id}.json.gz"
        variant_sets_tx = load_json(save_path, 
                                    force=force, 
                                    verbose=verbose>1)
        if variant_sets_tx is not None:
            variant_sets[tx_id] = variant_sets_tx
            continue
        elif cache_only:
            continue
        else:
            variant_sets_tx = {}
        ## Get pathogenic variants
        varp = get_variants(
            tx_id=tx_id,
            client=client,
            params_query={'feature':'variation',
                          'so_term': so_term,
                          'variant_set': 'clin_assoc'
                        },
            params_filter={'clinical_significance':['pathogenic'],
                            'consequence_type': consequence_type}, 
            exact=True
            )
        if len(varp)==0: 
            continue
        else:
            variant_sets_tx['variants_pathogenic'] = varp
        ## Get benign variants
        varb = get_variants(
            tx_id=tx_id,
            params_query={'feature':'variation',
                        'so_term': so_term,
                        'variant_set': 'ClinVar'
                        },
            params_filter={'clinical_significance':['benign'], 
                           'consequence_type': consequence_type},
            exact=True
            )
        if len(varb)==0:
            continue
        else:
            variant_sets_tx['variants_benign'] = varb
        ## Get negative variant set (e.g. 1KG variants)
        varn = get_variants(
            tx_id=tx_id,
            client=client,
            params_query={'feature':'variation',
                          'so_term': so_term,
                          'variant_set': nagative_variant_set
                        }
            )
        if len(varn)>0:
            variant_sets_tx['variants_1kg'] = varn
            ids_neg = [x['id'] for x in varn]
            # Remove pathogenic variants present in 1KG
            varp = filter_variants(varp, 
                                    params={'id':ids_neg},
                                    reverse=True,
                                    verbose=False,
                                    leave=False
                                    )
            if len(varp)==0:
                continue
            # Remove benign variants present in 1KG
            varb = filter_variants(varb, 
                                    params={'id':ids_neg},
                                    reverse=True,
                                    verbose=False,
                                    leave=False
                                    )
            if len(varb)==0:
                continue
        # Append transcript if it made it through all the filters
        variant_sets[tx_id] = variant_sets_tx
        # save results as pickle 
        save_json(obj=variant_sets_tx,
                  save_path=save_path,
                  verbose=verbose>1)
    return variant_sets
    
def get_vep(ids,
            species='homo_sapiens',
            params={'AlphaMissense':1,
                'CADD':1,
                'ClinPred':1,
                'EVE':1,
                'Enformer':1,
                    'MaveDB':1,
                    'REVEL':1,
                    'pick':1
                    },
            client=None):
    """
    Get variant info from Ensembl VEP
    """
    from tqdm.auto import tqdm
    client = get_ensembl_client(client=client)
    variant_info = {}
    ids = as_list(ids)
    for id in tqdm(ids,
                   desc="Getting variant info"):
        variant_info[id] = client.vep_id_get(
            species=species,
            id=id, 
            params=params
            )
    return variant_info


def get_variant_ids(variant_sets):
    from tqdm.auto import tqdm
    if isinstance(variant_sets, list):
        ids = [x['id'] for x in variant_sets]
    elif isinstance(variant_sets, dict):
        ids = {}
        for tx_id in tqdm(variant_sets.keys()):
            print(tx_id)
            for k in variant_sets[tx_id].keys():
                if k.startswith('variants_'):
                    ids[tx_id][k] = [variant['id'] for variant in variant_sets[tx_id][k]]
    return ids

def add_variants(haplotypes,
                 variant_sets,
                 params={}):
    from tqdm.auto import tqdm
    for tx_id in tqdm(haplotypes.keys()):

        variants_vep = get_vep(variant_sets[tx_id], 
                               params=params)
    return haplotypes


def get_haplotypes(tx_ids,
                   save_dir="data/haplosaurus/haplotypes",
                   species="homo_sapiens",
                   params={'samples':1,
                            'sequence':1,
                            'aligned_sequences':1},
                   client=None,
                   force = False,
                   cache_only=False,
                   verbose=True):
    from tqdm.auto import tqdm
    client = get_ensembl_client(client=client)
    haplotypes = {} 
    for tx_id in tqdm(tx_ids,
                    desc="Getting haplotypes"):
        save_path = f"{save_dir}/{tx_id}.json.gz"
        hap_tx_id = load_json(save_path, 
                              force=force, 
                              verbose=verbose>1)
        if hap_tx_id is not None:
            haplotypes[tx_id] = hap_tx_id
        if cache_only:
            continue
        else:
            try:
                haplotypes[tx_id] = client.transcript_haplotypes_get(
                    id=tx_id,
                    species=species,
                    params=params
                    ) 
                save_json(obj=haplotypes[tx_id],
                          save_path=save_path,
                          verbose=verbose>1)
            except Exception as e:
                if verbose:
                    print(f"Error getting haplotypes for {tx_id}: {e}")
                continue 
    return haplotypes

def get_haplotype_seqs(haplotypes,
                       aligned=True):
    from tqdm.auto import tqdm
    hap_seqs = {}
    for tx_id in tqdm(haplotypes.keys()):
        hap_seqs[tx_id] = [x['aligned_sequences'][1] if aligned else x['seq'] for x in haplotypes[tx_id]['protein_haplotypes']]
    return hap_seqs

def get_haplotype_freqs(haplotypes):
    from tqdm.auto import tqdm
    pop_freqs = {}
    for tx_id in tqdm(haplotypes.keys()):
        pop_freqs[tx_id] = {x['name']:x['population_frequencies'] for x in haplotypes[tx_id]['protein_haplotypes']}
    return pop_freqs

def get_haplotype_names(haplotypes):
    """
    Get the haplotype name from the haplotype entry,
        or a dict of haplotype entries indexed by transcript ID.
    """
    if 'protein_haplotypes' in haplotypes.keys():
        return [x['name'].split('_')[0] for x in haplotypes['protein_haplotypes']]
    from tqdm.auto import tqdm
    hap_names = {}
    for tx_id in tqdm(haplotypes.keys()):
        hap_names[tx_id] = [x['name'].split('_')[0] for x in haplotypes[tx_id]['protein_haplotypes']]
    return hap_names

def get_haplotype_protein_ids(haplotypes):
    """
    Get the protein ID from the haplotype entry,
        or a dict of haplotype entries indexed by transcript ID.
    """
    if 'protein_haplotypes' in haplotypes.keys():
        return haplotypes['protein_haplotypes'][0]['name'].split(':')[0]
    from tqdm.auto import tqdm
    hap_protein_ids = {}
    for tx_id in tqdm(haplotypes.keys()):
        hap_protein_ids[tx_id] = haplotypes[tx_id]['protein_haplotypes'][0]['name'].split(':')[0]
    return hap_protein_ids


def haplotypes_to_batches(haplotypes,
                          strip='*'):
    from tqdm.auto import tqdm
    batches = {}
    for tx_id in tqdm(haplotypes.keys(),
                       desc="Preparing batches"):
        batches[tx_id] = [(x['name'], x['seq'].strip(strip)) for x in haplotypes[tx_id]['protein_haplotypes']]
    return batches

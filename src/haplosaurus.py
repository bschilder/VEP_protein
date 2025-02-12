import os
import sys
sys.path.append("code")
from src.utils import as_list, intersect, load_pickle, save_pickle, load_json, save_json, invert_dict
from src.config import PARAMS_VEP, PARAMS_HAPLOTYPES, set_params_vep, set_params_haplotypes, get_params_vep, get_params_haplotypes, PARAMS_VARIATION
from src.onekg import get_sample_metadata
from src.ontologies import get_sequence_ontology, get_descendants
from src.config import DATA_DIR
# https://ensemblrest.readthedocs.io/en/latest/


DIR_DICT = {
    "haplosaurus": os.path.join(DATA_DIR, "haplosaurus",""),
    "vep": os.path.join(DATA_DIR, "haplosaurus","vep",""),
    "haplotypes": os.path.join(DATA_DIR, "haplosaurus","haplotypes",""),
    "variants": os.path.join(DATA_DIR, "haplosaurus","variants",""),
    "variant_sets": os.path.join(DATA_DIR, "haplosaurus","variant_sets",""),
    "variation": os.path.join(DATA_DIR, "haplosaurus","variation",""),
}
 
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
                    max_variants=None,
                    exact=False,
                    reverse=False,
                    verbose=True,
                    **kwargs):
    from tqdm.auto import tqdm 
    if params is None or len(params)==0:
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
    if max_variants is not None:
        variants_tx_filtered = variants_tx_filtered[:max_variants]
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


def _params_to_filename(params,
                        suffix='.json.gz'):
    if params is None:
        return 'file' + suffix
    return ";".join([f"{k}:{v}" for k,v in sorted(params.items())]) + suffix


def get_variants(tx_id,
                 save_dir=DIR_DICT["variants"],
                 client=None,
                 params_query={'feature':'variation'},
                 params_filter={},
                 force=False, 
                 error=True,
                 verbose=True,
                 **kwargs):
    # https://ensemblrest.readthedocs.io/en/latest/#ensembl_rest.EnsemblClient.overlap_id
    # Ensembl overlap API: https://rest.ensembl.org/documentation/info/overlap_id
    # Variant sets: https://useast.ensembl.org/info/genome/variation/species/sets.html
    # Check if variants are already cached

    # Customize save subdir based on params_query
    variants_tx = None
    if save_dir is not None:
        import os
        filename = _params_to_filename(params_query)
        save_path = os.path.join(save_dir,tx_id,filename)
        variants_tx = load_json(save_path, 
                                force=force, 
                                verbose=verbose>1)
        variants_tx = list(variants_tx.values())[0] if isinstance(variants_tx, dict) else variants_tx
    if variants_tx is None:
        client = get_ensembl_client(client=client)
        try:
            variants_tx = client.overlap_id(id=tx_id, 
                                            params=params_query
                                            )
            # Cache variants
            if len(variants_tx)>0:
                save_json({tx_id:variants_tx}, 
                          save_path, 
                          verbose=verbose>1)
        except Exception as e:
            if error:
                raise e
            else:
                if verbose:
                    print(f"Error getting variants for {tx_id}: {e}")
                variants_tx = []
    # Filter variants
    variants_tx = filter_variants(variants_tx, 
                                  params=params_filter,
                                  verbose=verbose,
                                  leave=False,
                                  **kwargs
                                  )
    if verbose:
        print(len(variants_tx),"variants returned")
    return variants_tx

def get_variation(variant_id,
                  save_dir=DIR_DICT["variation"],
                  species='homo_sapiens',
                  params=PARAMS_VARIATION,
                  verbose=True,
                  client=None,
                  force=False,
                  error=True):
    """
    Uses a variant identifier (e.g. rsID) to return the variation features
      including optional genotype, phenotype and population data
    """
    # https://rest.ensembl.org/documentation/info/variation_id
    if save_dir is not None:
        import os
        filename = _params_to_filename(params)
        save_path = os.path.join(save_dir,variant_id,filename)
        variation = load_json(save_path, 
                            force=force, 
                            verbose=verbose>1)
        if variation is not None:
            return variation
    client = get_ensembl_client(client=client)
    try:
        variation = client.variation_id(id=variant_id, 
                                        species=species,
                                        params=params
                                        )
        save_json(obj=variation,
                  save_path=save_path,
                  verbose=verbose>1)
    except Exception as e:
        if error:
            raise e
        else:
            if verbose:
                print(f"Error getting info for {variant_id}: {e}")
            variation = None
    return variation

def variation_to_dfs(variation,
                    fields=['populations', # population-level variant frequencies
                            'population_genotypes', # population-level genotype frequencies
                            'genotypes', # individual-level genotype frequencies
                            'phenotypes', # associated phenotypes
                            'mappings', # genomic coordinates
                            ]):
    import pandas as pd
    fields_valid = [x for x in fields if x in variation.keys()]
    return {field:pd.DataFrame(variation[field]) for field in fields_valid}

def _check_vep_i(var,
                save_dir_vep,
                client,
                consequence_type,
                desc_get,
                desc_filter,
                verbose,
                exact=True,
                params_vep=PARAMS_VEP):
    
    vep = get_vep(ids=[x['id'] for x in var],
                save_dir=save_dir_vep,
                client=client,
                desc=desc_get,
                leave=False,
                verbose=verbose>1,
                params=params_vep
                )
    vep = filter_vep(vep,
                    consequence_terms=consequence_type,
                    exact=exact,
                    desc=desc_filter,
                    leave=False,
                    verbose=verbose>1)
    return vep

def _check_vep(varp,
               varb,
               tx_id,
               save_dir_vep,
               client,
               consequence_type,
               verbose,
               params_vep=PARAMS_VEP):
    varp_vep = _check_vep_i(varp,
                            save_dir_vep=save_dir_vep,
                            client=client,
                            consequence_type=consequence_type,
                            desc_get=f"{tx_id}: Annotating pathogenic variants",
                            desc_filter=f"{tx_id}: Filtering pathogenic variants",
                            verbose=verbose,
                            params_vep=params_vep) 
    # Annotate benign variants with VEP 
    varb_vep = _check_vep_i(varb,
                            save_dir_vep=save_dir_vep,
                            client=client,
                            consequence_type=consequence_type,
                            desc_get=f"{tx_id}: Annotating benign variants",
                            desc_filter=f"{tx_id}: Filtering benign variants",
                            verbose=verbose,
                            params_vep=params_vep)
    return varp_vep, varb_vep



def get_variant_sets(tx_ids,
                     so_term = 'SO:0001583',
                     consequence_type = ['missense_variant'],
                     include_descendants = True,
                     pathogenic_variant_set = 'clin_assoc',
                     benign_variant_set = 'ClinVar',
                     nagative_variant_set = '1kg_3',  # '1kg_3_com'
                     check_vep=True,
                     add_vep=True,
                     params_vep=PARAMS_VEP,
                     save_dir_variants=DIR_DICT["variants"],
                     save_dir_variant_sets=DIR_DICT["variant_sets"],
                     save_dir_vep=DIR_DICT["vep"],
                     force=False,
                     exact={'pathogenic':False,
                            'benign':False,
                            'negative':True},
                     client=None,
                     cache_only=False,
                     error=True,
                     verbose=True):

    from tqdm.auto import tqdm
    # Gather pathogenic/benign variants
    variant_sets = {}
    consequence_type = as_list(consequence_type)
    client = get_ensembl_client(client=client)
    if not check_vep:
        add_vep = False
    # Include descendants of consequence type
    consequence_type_path = '.'.join(consequence_type)
    if include_descendants:
        so = get_sequence_ontology()
        consequence_type = get_descendants(label_or_id=consequence_type,
                                            ont=so, 
                                            return_as="label",
                                            verbose=verbose)
        so_term = get_descendants(label_or_id=so_term,
                                  ont=so,
                                  return_as="id",
                                  verbose=verbose) 
    if cache_only:
        force = False
    # Iterate over transcripts
    for tx_id in tqdm(tx_ids,"Gathering variant sets"):
        save_path = f"{save_dir_variant_sets}/{consequence_type_path}/{tx_id}.json.gz"
        variant_sets_tx = load_json(save_path, 
                                    force=force, 
                                    verbose=verbose>1)
        if variant_sets_tx is not None: 
            # Annotate pathogenic/benign variants with VEP
            if check_vep:
                varp_vep, varb_vep = _check_vep(varp=variant_sets_tx['variants_pathogenic'],
                                                varb=variant_sets_tx['variants_benign'],
                                                tx_id=tx_id,
                                                params_vep=params_vep,
                                                save_dir_vep=save_dir_vep,
                                                client=client,
                                                consequence_type=consequence_type,
                                                verbose=verbose>1
                                                )  
                if varb_vep is None or varp_vep is None:
                    if verbose:
                        print(f"No VEP annotations for {tx_id}")
                    continue
            if add_vep:
                variant_sets_tx['variants_pathogenic_vep'] = varp_vep
                variant_sets_tx['variants_benign_vep'] = varb_vep
            variant_sets[tx_id] = variant_sets_tx
            continue
        elif cache_only:
            continue
        ## Get pathogenic variants
        try:
            variant_sets_tx = {}
            varp = get_variants(
                tx_id=tx_id,
                client=client,
                params_query={'feature':'variation',
                              'so_term': so_term,
                              'variant_set': pathogenic_variant_set
                            },
                params_filter={'clinical_significance':['pathogenic'],
                               'consequence_type': consequence_type}, 
                exact=exact['pathogenic'],
                verbose=verbose>1
                )
            if len(varp)==0: 
                continue
            else:
                variant_sets_tx['variants_pathogenic'] = varp
            ## Get benign variants
            varb = get_variants(
                tx_id=tx_id,
                save_dir=save_dir_variants,
                params_query={'feature':'variation',
                              'so_term': so_term,
                              'variant_set': benign_variant_set
                            },
                params_filter={'clinical_significance':['benign'], 
                               'consequence_type': consequence_type},
                exact=exact['benign'],
                verbose=verbose>1
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
                            },
                exact=exact['negative'],
                verbose=verbose>1
                )
            if len(varn)>0:
                variant_sets_tx['variants_negative'] = varn
                ids_neg = [x['id'] for x in varn]
                # Remove pathogenic variants present in 1KG
                varp = filter_variants(varp, 
                                        params={'id':ids_neg},
                                        reverse=True,
                                        verbose=verbose>1,
                                        leave=False
                                        )
                if len(varp)==0:
                    continue
                # Remove benign variants present in 1KG
                varb = filter_variants(varb, 
                                        params={'id':ids_neg},
                                        reverse=True,
                                        verbose=verbose>1,
                                        leave=False
                                        )
                if len(varb)==0:
                    continue
            # Annotate pathogenic/benign variants with VEP
            if check_vep:
                varp_vep, varb_vep = _check_vep(varp,
                                                varb,
                                                tx_id=tx_id,
                                                save_dir_vep=save_dir_vep,
                                                client=client,
                                                consequence_type=consequence_type,
                                                verbose=verbose>1)  
                if varb_vep is None or varp_vep is None:
                    continue
            if add_vep:
                variant_sets_tx['variants_pathogenic_vep'] = varp_vep
                variant_sets_tx['variants_benign_vep'] = varb_vep
            # Only add to variant_sets if it made it through all the filters
            variant_sets[tx_id] = variant_sets_tx
            # save results as pickle 
            save_json(obj=variant_sets_tx,
                    save_path=save_path,
                    verbose=verbose>1)
        except Exception as e:
            if error:
                raise e
            else:
                if verbose:
                    print(f"Error getting variant sets for {tx_id}: {e}")
                continue
    return variant_sets
    
def filter_vep(variant_vep,
               consequence_terms=None, 
               exact=False,
               desc="Filtering variants VEP",
               leave=False,
               verbose=True):
    """
    Filter VEP results to only include variants with HIGH impact
    """
    if consequence_terms is None:
        return variant_vep
    from tqdm.auto import tqdm
    consequence_terms = as_list(consequence_terms)
    variant_vep_filtered = {}
    for id, variant_vep_id in tqdm(variant_vep.items(),
                   desc=desc,
                   leave=leave): 
        cterms = set()
        for i in variant_vep_id:
            if 'transcript_consequences' not in i.keys():
                continue
            cterms.update([consequence_terms for tc in i['transcript_consequences'] for consequence_terms in tc['consequence_terms']])
        if exact:
            if set(cterms) == set(consequence_terms):
                variant_vep_filtered[id] = variant_vep_id
        else:
            if len(set(cterms) & set(consequence_terms)) > 0:
                variant_vep_filtered[id] = variant_vep_id
    if verbose:
        print(f"{len(variant_vep_filtered)}/{len(variant_vep)} variants retained")
    if len(variant_vep_filtered)==0:
        return None
    return variant_vep_filtered

def get_vep(ids,
            species='homo_sapiens',
            params=PARAMS_VEP,
            save_dir=DIR_DICT["vep"],
            client=None,
            desc="Getting variant info",
            leave=True,
            force=False,
            verbose=True,
            cache_only=False):
    """
    Get variant info from Ensembl VEP
    """
    # protein_id = hs.get_haplotype_protein_ids(haplotypes[tx_id])
    # protein_info = client.lookup(protein_id)
    # offeset = variant['start'] - protein_info['start']
    # variant_info = client.variation_id(species='homo_sapiens',
    #                                   id=variant['id'])
    # client.variant_recoder(species='homo_sapiens',
    #                         id=variant['id'], 
    #                         params={'fields':'hgvsp',
    #                                 'gencode_basic':1,
    #                                 'gencode_primary':1,
    #                                 }
    #                                 )

    from tqdm.auto import tqdm
    if cache_only:
        force = False
    client = get_ensembl_client(client=client)
    variant_vep = {}
    ids = as_list(ids)
    for id in tqdm(ids,
                   desc=desc,
                   leave=leave):
        # Check if variant info is already cached
        save_path = f"{save_dir}/{id}.json.gz"
        variant_vep_id = load_json(save_path, 
                                    force=force, 
                                    verbose=verbose>1)
        if variant_vep_id is not None:
            variant_vep[id] = list(variant_vep_id.values())[0]
            continue
        if cache_only:
            continue
        # Get variant info from Ensembl VEP
        variant_vep_id = client.vep_id_get(
            species=species,
            id=id, 
            params=params
            )
        # Save variant info to cache
        save_json(obj={id:variant_vep_id},
                  save_path=save_path,
                  verbose=verbose>1)
        variant_vep[id] = variant_vep_id
    return variant_vep

def get_variant_ids(variant_sets):
    from tqdm.auto import tqdm
    if isinstance(variant_sets, list):
        ids = [x['id'] for x in variant_sets]
    elif isinstance(variant_sets, dict):
        ids = {}
        for tx_id in tqdm(variant_sets.keys()):
            ids[tx_id] = {}
            for k in variant_sets[tx_id].keys(): 
                ids[tx_id][k] = [variant['id'] for variant in variant_sets[tx_id][k]]
    return ids

def list_haplotypes(save_dir=DIR_DICT["haplotypes"],
                    verbose=True):
    import glob
    files = glob.glob(f"{save_dir}/*.json.gz")
    if len(files)==0:
        raise ValueError(f"No haplotypes found in {save_dir}")
    else:
        if verbose:
            print(f"Found {len(files)} haplotypes in {save_dir}")
        return [x.split('/')[-1].split('.')[0] for x in files]
    

def get_haplotypes(tx_ids=None,
                   save_dir=DIR_DICT["haplotypes"],
                   species="homo_sapiens",
                   params=PARAMS_HAPLOTYPES,
                   client=None,
                   force = False,
                   cache_only=False,
                   verbose=True):
    from tqdm.auto import tqdm
    client = get_ensembl_client(client=client)
    haplotypes = {}
    # Get tx_ids if not provided
    if tx_ids is None:
        tx_ids = list_haplotypes(save_dir=save_dir,
                                 verbose=verbose)
    # If cache only, don't search for new tx_ids
    if cache_only:
        force = False
    # Get haplotypes
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
                       aligned=True,
                       return_missing=False,
                       use_protein_ids=False,
                       add_haplotype_names=False,
                       verbose=False):
    from tqdm.auto import tqdm
    hap_seqs = {}
    missing_seqs = []
    for tx_id in tqdm(haplotypes.keys(),
                       desc="Getting haplotype sequences"):
        if isinstance(haplotypes[tx_id], dict) and 'protein_haplotypes' in haplotypes[tx_id].keys():
            hap_seqs[tx_id] = [x['aligned_sequences'][1] if aligned else x['seq'] for x in haplotypes[tx_id]['protein_haplotypes']]
        else:
            if len(haplotypes[tx_id])>0:
                if aligned:
                    hap_seqs[tx_id] = haplotypes[tx_id][0]['aligned_sequences'][1] 
                else:
                    hap_seqs[tx_id] = haplotypes[tx_id][0]['seq']
            else:
                if verbose:
                    print(f"No seqs found for {tx_id}")
                missing_seqs += [tx_id]
                continue
                
    if add_haplotype_names:
        haplotype_names = get_haplotype_names(haplotypes) # list of haplotype names
        hap_seqs = {tx_id:list(zip(haplotype_names[tx_id], tx_seqs)) for tx_id,tx_seqs in hap_seqs.items()}
    if use_protein_ids:
        hap_seqs = dict(zip(get_haplotype_protein_ids(haplotypes).values(), hap_seqs.values()))
    if return_missing:
        return hap_seqs, missing_seqs
    else:
        return hap_seqs

def get_haplotype_counts(haplotypes,
                         key='protein_haplotypes'):
    return {k:len(haplotypes[k][key]) for k in haplotypes.keys()}

def get_haplotype_freqs(haplotypes,
                        tx_ids=None):
    from tqdm.auto import tqdm
    pop_freqs = {}
    populations = set()
    tx_ids = as_list(tx_ids)
    for tx_id in tqdm(haplotypes.keys(),
                      desc="Getting haplotype frequencies",
                      leave=False):
        if tx_ids is not None:
            if tx_id not in tx_ids:
                continue
        pop_freqs[tx_id] = {x['name']:x['population_frequencies'] for x in haplotypes[tx_id]['protein_haplotypes']}
        for hap_name in pop_freqs[tx_id]:
            populations.update(pop_freqs[tx_id][hap_name].keys())
    cohorts = set([':'.join(p.split(':')[:-1]) for p in populations])
    return pop_freqs, populations, cohorts

def add_haplotype_freqs(df, 
                        haplotypes, 
                        cohorts=['1000GENOMES:phase_3'],
                        haplotype_col="label_base",
                        add_top_pop=True,
                        add_top_superpop=True,
                        force=False,
                        verbose=True): 
    from tqdm.auto import tqdm
    if 'tx_id' not in df.columns:
        df = add_txid(df, haplotypes, verbose=verbose)
    # Get population frequencies dict
    (pop_freqs, 
     populations,
     cohorts_all) = get_haplotype_freqs(haplotypes=haplotypes)
    # Select cohorts
    if cohorts is None:
        if verbose:
            print(f"Using all {len(cohorts_all)} cohorts.")
        cohorts = cohorts_all
    else:
        cohorts = as_list(cohorts)
        cohort_intersect = set(cohorts) & set(cohorts_all)
        if len(cohort_intersect)==0:
            raise ValueError(f"No cohorts found in populations")
        else:
            if verbose:
                print(f"Using cohorts: {cohort_intersect}")
        cohorts = list(cohort_intersect)
    # Filter populations to only include the specified cohorts
    populations = sorted([p for p in populations if any(p.startswith(cohort) for cohort in cohorts)])
    if verbose:
        print(f"Using {len(populations)} populations")
    # Map population frequencies to embedding_df
    freq_cols = [f'freq_{pop}' for pop in populations]
    if not all(col in df.columns for col in freq_cols) or force:  
        for i,pop in tqdm(enumerate(populations),
                    desc="Adding haplotype frequencies",
                    total=len(populations),
                    leave=True):
            df[freq_cols[i]] = df.apply(
                lambda row: pop_freqs[row['tx_id']][row[haplotype_col]][pop] 
                    if row['tx_id'] in pop_freqs 
                    and row[haplotype_col] in pop_freqs[row['tx_id']] 
                    and pop in pop_freqs[row['tx_id']][row[haplotype_col]] 
                    else None,
                axis=1
            )
    ## Get top population
    if add_top_pop:
        if len(freq_cols)>0 or force:
            # Check if there are any non-NA values
            has_freqs = ~df[freq_cols].isna().all(axis=1)
            if has_freqs.any():
                # Only compute max for rows with frequencies
                max_freq_idx = df.loc[has_freqs, freq_cols].idxmax(axis=1)
                df.loc[has_freqs, 'top_pop'] = max_freq_idx.str.replace('freq_', '')
            else:
                print("Warning: All frequency columns contain NA values")
        else:
            print("Warning: No frequency columns found")

    if add_top_superpop:
        if 'top_pop' in df.columns:
            if 'top_superpop' not in df.columns or force:
                pops = get_sample_metadata()
                pop_map = dict(zip(pops['Population Code'], pops['Super Population']))
                print(pop_map)
                df['top_superpop'] = df['top_pop'].str.replace(f'{cohorts[0]}:', '').str.split('_').str[0].map(pop_map)
                df['top_superpop'].fillna('N/A', inplace=True) 
    return df


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

def get_haplotype_ref(haplotypes):
    from tqdm.auto import tqdm
    if isinstance(haplotypes, list):
        hap_ref = [x for x in haplotypes if ":REF" in tqdm(x['name'])]
    elif 'protein_haplotypes' in haplotypes.keys():
        hap_ref = [x for x in haplotypes['protein_haplotypes'] if ":REF" in tqdm(x['name'])]
    else:
        hap_ref = {}
        for tx_id in tqdm(haplotypes.keys(),"Processing transcripts"):
            hap_ref[tx_id] = [x for x in haplotypes[tx_id]['protein_haplotypes'] if ":REF" in x['name']]
            if isinstance(hap_ref[tx_id], list) and len(hap_ref[tx_id])>0:
                hap_ref[tx_id] = hap_ref[tx_id]
    return hap_ref

def to_stop(seq,
            replace='-'):
    if replace is not None:
        seq = seq.replace(replace, '')
    return seq if seq.find('*')==-1 else seq[:seq.find('*')]

def _add_variant_to_haplotypes_tx(hap_tx, 
                                  variants, 
                                  variants_vep,  
                                  include_all_fields,
                                  # Turn off for now since ref genome not gauranteed to be the same as the haplotype
                                  ref_checks=True,
                                  desc="Adding variants to haplotypes",
                                  leave=False
                                 ):
        from tqdm.auto import tqdm
        import copy
        variant_ids = [x['id'] for x in variants]
        variant_id = variant_ids[0]
        # Create a copy of the haplotypes to modify
        hap_tx_modified = {}
        if include_all_fields:
            hap_tx_modified['protein_haplotypes'] = copy.deepcopy(hap_tx['protein_haplotypes'])
        else:
            hap_tx_modified['protein_haplotypes'] = copy.deepcopy([{k: x[k] for k in ['name', 'aligned_sequences']} for x in hap_tx['protein_haplotypes']   ])
        # Get variant info once instead of for each haplotype
        try:
            variant = variants_vep[variant_id][0]['transcript_consequences'][0]
            protein_start = variant['protein_start']
            protein_end = variant['protein_end'] 
            ref_aa, alt_aa = variant['amino_acids'].split('/')
        except Exception as e:
            print(f"variant_id: {variant_id}")
            print(f"variant: {variant}")
            raise ValueError(f"Error getting variant info for {variant_id}: {e}")
        
        def _add_variant_to_haplotypes_tx_i(haplotype_entry):
            # Work directly with strings instead of converting to list
            aligned_seqs = haplotype_entry['aligned_sequences']
            ref_aa_seq = aligned_seqs[0][protein_start-1:protein_end]
            
            if ref_checks and ref_aa != ref_aa_seq:
                print(
                    f"aligned_seqs length: {len(aligned_seqs)}","\n",
                    f"Variant ID: {variant_id}","\n",
                    f"Variant: {variant}","\n",
                    f"Haplotype: {haplotype_entry['name']}","\n",
                    f"Haplotype sequence 1: {haplotype_entry['aligned_sequences'][0]}","\n",
                    f"Haplotype sequence 2: {haplotype_entry['aligned_sequences'][1]}","\n",
                    # Haplotype sequence windows
                    f"Haplotype sequence 1 preview: {aligned_seqs[0][protein_start-1:protein_end+5]}","\n",
                    f"Haplotype sequence 2 preview: {aligned_seqs[1][protein_start-1:protein_end+5]}","\n",
                    )
                raise ValueError(
                    f"Reference amino acid {ref_aa} does not match haplotype sequence {ref_aa_seq}",
                    )
                
            # Modify sequence directly
            haplotype_entry['aligned_sequences'][1] = (
                aligned_seqs[1][:protein_start-1] + 
                alt_aa + 
                aligned_seqs[1][protein_end:]
            )
            haplotype_entry['seq'] = haplotype_entry['aligned_sequences'][1].replace('-', '')
            return haplotype_entry

        return [_add_variant_to_haplotypes_tx_i(h) for h in 
                tqdm(hap_tx_modified['protein_haplotypes'], 
                     desc=desc,
                     leave=leave)]

def add_variant_to_haplotypes(haplotypes, 
                              variant_sets,
                              client=None,
                              tx_ids=None,
                              max_transcripts=None,
                              add_variant_info=True,
                              include_all_fields=False,
                              params_vep=PARAMS_VEP,
                              consequence_terms={},
                              save_dir_vep=DIR_DICT["vep"],
                              ref_checks=False,
                              error=True,
                              exact={'pathogenic':True,
                                     'benign':True},
                              verbose=True,
                              **kwargs
                              ):
    
    max_pathogenic_variants=1
    max_benign_variants=1
    
    from tqdm.auto import tqdm
    haplotypes_pathogenic = {}
    haplotypes_benign = {}
    haplotypes_wt = subset_haplotypes(
        haplotypes,
        tx_ids=tx_ids,
        max_transcripts=max_transcripts)
    for tx_id,hap_tx in tqdm(haplotypes_wt.items(),
                             desc="Adding variants to transcript haplotypes"):
        if tx_id not in variant_sets.keys():
            continue
        try:
            # Add pathogenic variants
            varp = filter_variants(
                max_variants=max_pathogenic_variants,
                variants_tx = variant_sets[tx_id]['variants_pathogenic'], 
                params={'consequence_terms':consequence_terms} if len(consequence_terms)>0 else None,
                exact=exact['pathogenic'],
                leave=False,
                verbose=verbose>1
                )
            if len(varp)==0:
                if verbose:
                    print(f"No pathogenic variants found for {tx_id}")
                continue
            varp_vep = get_vep(
                    ids=[x['id'] for x in varp], 
                    params=params_vep,
                    client=client,
                    save_dir=save_dir_vep,
                    desc="Getting pathogenic variant VEP",
                    leave=False,
                    **kwargs
                    ) 
            varp_vep = filter_vep(
                varp_vep,
                consequence_terms=consequence_terms,
                exact=exact['pathogenic'],
                leave=False,
                verbose=verbose>1
                )
            if varp_vep is None:
                if verbose:
                    print(f"No pathogenic variants found for {tx_id}")
                continue
            hap_tx_pathogenic = _add_variant_to_haplotypes_tx(
                hap_tx=hap_tx, 
                variants=varp, 
                variants_vep=varp_vep, 
                include_all_fields=include_all_fields,
                ref_checks=ref_checks,
                desc="Adding pathogenic variants to haplotypes",
                leave=False)
            # Add benign variants 
            varb = filter_variants(
                max_variants=max_benign_variants,
                variants_tx = variant_sets[tx_id]['variants_benign'], 
                params={'consequence_terms':consequence_terms} if len(consequence_terms)>0 else None,
                exact=exact['benign'],
                leave=False,
                verbose=verbose>1
                )
            if len(varb)==0:
                if verbose:
                    print(f"No benign variants found for {tx_id}")
                continue
            varb_vep = get_vep(
                    ids=[x['id'] for x in varb], 
                    params=params_vep,
                    client=client,
                    save_dir=save_dir_vep,
                    desc="Getting benign variant VEP",
                    leave=False,
                    **kwargs
                    )
            varb_vep = filter_vep(
                varb_vep,
                consequence_terms=consequence_terms,
                exact=exact['benign'],
                leave=False,
                verbose=verbose>1
                )
            if varb_vep is None:
                if verbose:
                    print(f"No benign variants found for {tx_id}")
                continue
            hap_tx_benign = _add_variant_to_haplotypes_tx(
                hap_tx=hap_tx, 
                variants=varb, 
                variants_vep=varb_vep, 
                include_all_fields=include_all_fields,
                ref_checks=ref_checks,
                desc="Adding benign variants to haplotypes",
                leave=False)
            # Add pathogenic variants only if they are present
            haplotypes_pathogenic[tx_id] = {}
            haplotypes_pathogenic[tx_id]['protein_haplotypes'] = hap_tx_pathogenic
            
            # Add benign variants only if they are present
            haplotypes_benign[tx_id] = {}
            haplotypes_benign[tx_id]['protein_haplotypes'] = hap_tx_benign
            # Store variant info in dict
            if add_variant_info:
                haplotypes_pathogenic[tx_id]['variants'] = varp
                haplotypes_benign[tx_id]['variants'] = varb
        except Exception as e:
            print(f"Error adding variants to {tx_id}")
            if error:
                raise e
            else:
                if verbose:
                    print(f"Error adding variants to {tx_id}: {e}")
                continue
    # Filter haplotypes_wt to only include tx_ids that have pathogenic/benign variants
    tx_ids = list(set(haplotypes_pathogenic.keys()) & set(haplotypes_benign.keys()))
    haplotypes_wt = subset_haplotypes(
        haplotypes,
        tx_ids=tx_ids,
        max_transcripts=max_transcripts)
    # Filter once more to ensure max_transcripts is respected
    haplotypes_wt = subset_haplotypes(
        haplotypes_wt,
        max_transcripts=max_transcripts)
    haplotypes_pathogenic = subset_haplotypes(
        haplotypes_pathogenic,
        tx_ids=tx_ids,
        max_transcripts=max_transcripts)
    haplotypes_benign = subset_haplotypes(
        haplotypes_benign,
        tx_ids=tx_ids,
        max_transcripts=max_transcripts)
    
    return haplotypes_wt, haplotypes_pathogenic, haplotypes_benign

def subset_haplotypes(haplotypes,
                      tx_ids=None,
                      max_transcripts=None):
    if tx_ids is not None:
        # Limit to tx_ids
        haplotypes = {k: haplotypes[k] for k in tx_ids if k in haplotypes.keys()}
    if max_transcripts is not None:
        # Limit to max_transcripts
        tx_ids = list(haplotypes.keys())[:max_transcripts]
        haplotypes = {k: haplotypes[k] for k in tx_ids if k in haplotypes.keys()}
    return haplotypes


def haplotypes_to_batches(haplotypes,
                          suffix='',
                          strip='*',
                          add_variant_ids=True,
                          leave=False,
                          verbose=True):
    """
    Prepare batches for ESM2
    """
    if verbose:
        print("Preparing batches for ESM")
    from tqdm.auto import tqdm
    batches = {}
    for tx_id, hap_tx in tqdm(haplotypes.items(),
                       desc="Preparing batches",
                       leave=leave):
        suffix_final = suffix
        if 'variants' in hap_tx.keys() and add_variant_ids:
            suffix_final += ":"+",".join(set([x['id'] for x in hap_tx['variants']]))
        batches[tx_id] = [(x['name']+suffix_final, to_stop(x['seq'].strip(strip))) for x in hap_tx['protein_haplotypes']]
    return batches

def haplotypes_to_batches_grouped(haplotypes_grouped,
                                  merge=True,
                                  suffix=None,
                                  **kwargs): 
        batches_grouped = {} 
        # Get batches
        for group, haplotypes_group in haplotypes_grouped.items():
            batches_grouped[group] = haplotypes_to_batches(
                haplotypes=haplotypes_group, 
                suffix=f"_{group}" if suffix is None else suffix,
                **kwargs
                )
        if merge:
            # Merge batches across groups into one dict entry per tx_id
            # Ensure that lists get appended instead of overwritten
            batches = {}
            for group in batches_grouped.values():
                for tx_id, v in group.items():
                    if tx_id in batches.keys():
                        batches[tx_id].extend(v)
                    else:
                        batches[tx_id] = v
            return batches
        else:
            return batches_grouped


def reconstruct_data(tx_ids=None,
                     save_dir_haplotypes=DIR_DICT["haplotypes"],
                     save_dir_variants=DIR_DICT["variants"],
                     save_dir_variant_sets=DIR_DICT["variant_sets"],
                     save_dir_vep=DIR_DICT["vep"],
                     species="homo_sapiens",
                     so_term='SO:0001583',
                     consequence_terms=['missense_variant'],
                     params_haplotypes=PARAMS_HAPLOTYPES,
                     params_vep=PARAMS_VEP,
                     add_vep=True,
                     max_transcripts=None,
                     ref_checks=False,
                     merge_batches=True,
                     exact={'pathogenic':False,
                            'benign':False,
                            'negative':True},
                     cache_only={'get_haplotypes':True,
                                 'get_variant_sets':True},
                     force={'get_haplotypes':False,
                            'get_variant_sets':False},
                     check_vep=True,
                     verbose=True):
    
    # Get tx_ids
    if tx_ids is None:
        tx_ids = list_haplotypes(save_dir=save_dir_haplotypes,
                                 verbose=verbose)
    tx_ids = list(set(as_list(tx_ids)))
    # Get haplotypes
    haplotypes = get_haplotypes(
        tx_ids=tx_ids, 
        save_dir=save_dir_haplotypes,
        species=species, 
        params=params_haplotypes,
        cache_only=cache_only['get_haplotypes'],
        force=force['get_haplotypes'],
        verbose=verbose>1
        )
    # Get variant sets
    variant_sets = get_variant_sets(
        tx_ids=tx_ids,
        so_term = so_term,
        consequence_type = consequence_terms,
        params_vep=params_vep,
        add_vep=add_vep,
        save_dir_variants=save_dir_variants,
        save_dir_variant_sets=save_dir_variant_sets,
        save_dir_vep=save_dir_vep,
        exact=exact,
        check_vep=check_vep,
        cache_only=cache_only['get_variant_sets'],
        force=force['get_variant_sets'],
        verbose=verbose>1
        )
    # Inject variant sets into haplotypes
    (haplotypes_wt, 
     haplotypes_pathogenic, 
     haplotypes_benign) = add_variant_to_haplotypes(
        haplotypes=haplotypes, 
        variant_sets=variant_sets, 
        tx_ids=tx_ids,
        consequence_terms=consequence_terms,
        max_transcripts=max_transcripts,
        ref_checks=ref_checks,
        verbose=verbose>1
        )
    # Group haplotypes
    haplotypes_grouped = {'WT':haplotypes_wt, 
                          'Pathogenic':haplotypes_pathogenic, 
                          'Benign':haplotypes_benign
                          }     
    batches = haplotypes_to_batches_grouped(
        haplotypes_grouped = haplotypes_grouped,
        merge=merge_batches,
        verbose=verbose>1
        )
    return haplotypes, haplotypes_grouped, variant_sets, batches 

def get_suffix_dict(haplotypes,
                    prefix='_',
                    by_tx_id=True):
    if by_tx_id:
        # Return dict of tx_ids to variant IDs
        return {tx_id:prefix+hap_tx['variants'][0]['id'] for tx_id, hap_tx in haplotypes.items() if 'variants' in hap_tx.keys()}
    else:
        # Return dict of variant IDs to tx_ids
        return {hap_tx['variants'][0]['id']:prefix+tx_id for tx_id, hap_tx in haplotypes.items() if 'variants' in hap_tx.keys()}
    
def get_txid_map(haplotypes,
                 to="protein_id",
                 invert=False,
                 check_all=False,
                 as_df=False): 
    if to == "protein_id":
        if check_all:
            if invert:
                raise ValueError("Cannot invert dict if check_all is True")
            tx_id_map  = {tx_id:list(set([x['name'].split(':')[0] for x in hap_tx['protein_haplotypes']])) for tx_id, hap_tx in haplotypes.items()}
        else:
            tx_id_map = {tx_id:hap_tx['protein_haplotypes'][0]['name'].split(':')[0] for tx_id, hap_tx in haplotypes.items()}
        if invert:
            tx_id_map  = invert_dict(tx_id_map)
        if as_df:
            import pandas as pd
            return pd.DataFrame(tx_id_map, index=['id']).T
        else:
            return tx_id_map
    else:
        raise ValueError("to must be 'protein_id'")
    
def add_txid(df,
             haplotypes,
             protein_id_col="protein_id",
             tx_id_col="tx_id",
             verbose=True):
    if verbose:
            print("Adding tx_id column")
    tx_id_map = get_txid_map(haplotypes, invert=True)
    df[tx_id_col] = df[protein_id_col].map(tx_id_map)
    return df

def add_genesymbol(df,
                   on=['protein_id'],
                   map_file=os.path.join(DATA_DIR, "41467_2018_6542_MOESM4_ESM.xlsx")):
    import pandas as pd
    if 'gene_symbol' not in df.columns:
        tx_df = pd.read_excel(map_file).rename(columns={'Ensembl Gene ID':'gene_id',
                                                         'Ensembl Protein ID':'protein_id',
                                                         'Gene Name':'gene_symbol'})
        df = df.merge(tx_df[['gene_id','gene_symbol']+on], 
                                on=on,
                                how='left')
    return df

def add_variant_freqs(df,
                      variant_col='variants', 
                      field='populations',
                      verbose=True):
    import pandas as pd
    variation = {id:get_variation(id) for id in df[variant_col].unique()}
    variation_df = pd.concat([variation_to_dfs(x)[field].assign(variant_id=k) for k,x in variation.items()], axis=0)
    return df.merge(variation_df, on=variant_col, how='left')

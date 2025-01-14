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

def _check_vep_i(var,
                tx_id,
                save_dir_vep,
                client,
                consequence_type,
                desc_get,
                desc_filter,
                verbose):
    vep = get_vep(ids=[x['id'] for x in var],
                save_dir=save_dir_vep,
                client=client,
                desc=desc_get,
                leave=False,
                verbose=verbose>1)
    vep = filter_vep(vep,
                    consequence_terms=consequence_type,
                    exact=True,
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
               verbose):
    varp_vep = _check_vep_i(varp,
                            tx_id=tx_id,
                            save_dir_vep=save_dir_vep,
                            client=client,
                            consequence_type=consequence_type,
                            desc_get=f"{tx_id}: Annotating pathogenic variants",
                            desc_filter=f"{tx_id}: Filtering pathogenic variants",
                            verbose=verbose) 
    # Annotate benign variants with VEP 
    varb_vep = _check_vep_i(varb,
                            tx_id=tx_id,
                            save_dir_vep=save_dir_vep,
                            client=client,
                            consequence_type=consequence_type,
                            desc_get=f"{tx_id}: Annotating benign variants",
                            desc_filter=f"{tx_id}: Filtering benign variants",
                            verbose=verbose)
    return varp_vep, varb_vep


def get_variant_sets(tx_ids,
                     so_term = 'SO:0001583',
                     consequence_type = ['missense_variant'],
                     pathogenic_variant_set = 'clin_assoc',
                     benign_variant_set = 'ClinVar',
                     nagative_variant_set = '1kg_3',  # '1kg_3_com'
                     check_vep=True,
                     save_dir="data/haplosaurus/variant_sets",
                     save_dir_vep="data/haplosaurus/vep",
                     force=False,
                     client=None,
                     cache_only=False,
                     verbose=True):
    from tqdm.auto import tqdm
    # Gather pathogenic/benign variants
    variant_sets = {}
    client = get_ensembl_client(client=client)
    if cache_only:
        force = False
    # Iterate over transcripts
    for tx_id in tqdm(tx_ids,"Gathering variant sets"):
        save_path = f"{save_dir}/{'.'.join(consequence_type)}/{tx_id}.json.gz"
        variant_sets_tx = load_json(save_path, 
                                    force=force, 
                                    verbose=verbose>1)
        if variant_sets_tx is not None: 
            # Annotate pathogenic/benign variants with VEP
            if check_vep:
                varp_vep, varb_vep = _check_vep(varp=variant_sets_tx['variants_pathogenic'],
                                                varb=variant_sets_tx['variants_benign'],
                                                tx_id=tx_id,
                                                save_dir_vep=save_dir_vep,
                                                client=client,
                                                consequence_type=consequence_type,
                                                verbose=verbose)  
                if varb_vep is None or varp_vep is None:
                    continue
            variant_sets[tx_id] = variant_sets_tx
            continue
        elif cache_only:
            continue
        ## Get pathogenic variants
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
                          'variant_set': benign_variant_set
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
            variant_sets_tx['variants_negative'] = varn
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
        # Annotate pathogenic/benign variants with VEP
        if check_vep:
            varp_vep, varb_vep = _check_vep(varp,
                                            varb,
                                            tx_id=tx_id,
                                            save_dir_vep=save_dir_vep,
                                            client=client,
                                            consequence_type=consequence_type,
                                            verbose=verbose)  
            if varb_vep is None or varp_vep is None:
                continue
        # Only add to variant_sets if it made it through all the filters
        variant_sets[tx_id] = variant_sets_tx
        # save results as pickle 
        save_json(obj=variant_sets_tx,
                  save_path=save_path,
                  verbose=verbose>1)
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
            params={'AlphaMissense':1,
                'CADD':1,
                'ClinPred':1,
                'EVE':1,
                'Enformer':1,
                    'MaveDB':1,
                    'REVEL':1,
                    'pick':1
                    },
            save_dir="data/haplosaurus/vep",
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
    if cache_only:
        force = False
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
                       verbose=False):
    from tqdm.auto import tqdm
    hap_seqs = {}
    missing_seqs = []
    for tx_id in tqdm(haplotypes.keys()):
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
    if return_missing:
        return hap_seqs, missing_seqs
    else:
        return hap_seqs

def get_haplotype_counts(haplotypes,
                         key='protein_haplotypes'):
    return {k:len(haplotypes[k][key]) for k in haplotypes.keys()}

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

def haplotypes_to_batches(haplotypes,
                          suffix='',
                          strip='*',
                          verbose=True):
    """
    Prepare batches for ESM2
    """
    if verbose:
        print("Preparing batches for ESM")
    from tqdm.auto import tqdm
    batches = {}
    for tx_id in tqdm(haplotypes.keys(),
                       desc="Preparing batches"):
        batches[tx_id] = [(x['name']+suffix, to_stop(x['seq'].strip(strip))) for x in haplotypes[tx_id]['protein_haplotypes']]
    return batches

def _add_variant_to_haplotypes_tx(hap_tx, 
                                  variants, 
                                  variants_vep, 
                                  add_variant_id,
                                  include_all_fields,
                                  include_variant_info,
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
            if add_variant_id:
                haplotype_entry['name'] = f"{haplotype_entry['name']}_{variant_id}"
            if include_variant_info:
                haplotype_entry['variant_info'] = variant
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
                              max_pathogenic_variants=1,
                              max_benign_variants=1,
                              add_variant_id=False,
                              include_variant_info=True,
                              include_all_fields=False,
                              params_vep={'AlphaMissense':1,
                                        'CADD':1,
                                        'ClinPred':1,
                                        'EVE':1,
                                        'Enformer':1,
                                        'MaveDB':1,
                                        'REVEL':1,
                                        'pick':1
                                        },
                              consequence_terms={},
                              save_dir_vep="data/haplosaurus/vep",
                              ref_checks=False,
                              error=True,
                              verbose=True,
                              **kwargs
                              ):
    from tqdm.auto import tqdm
    haplotypes_pathogenic = {}
    haplotypes_benign = {}
    haplotypes_wt = subset_haplotypes(
        haplotypes,
        tx_ids=tx_ids,
        max_transcripts=max_transcripts)
    for tx_id,hap_tx in tqdm(haplotypes_wt.items(),
                             desc="Adding variants to transcript haplotypes"):
        try:
            # Add pathogenic variants
            variants = filter_variants(
                max_variants=max_pathogenic_variants,
                variants_tx = variant_sets[tx_id]['variants_pathogenic'], 
                params={'consequence_terms':consequence_terms} if len(consequence_terms)>0 else None,
                exact=True,
                leave=False,
                verbose=verbose>1
                )
            if len(variants)==0:
                if verbose:
                    print(f"No pathogenic variants found for {tx_id}")
                continue
            variants_vep = get_vep(
                    ids=[x['id'] for x in variants], 
                    params=params_vep,
                    client=client,
                    save_dir=save_dir_vep,
                    desc="Getting pathogenic variant VEP",
                    leave=False,
                    **kwargs
                    ) 
            variants_vep = filter_vep(
                variants_vep,
                consequence_terms=consequence_terms,
                exact=True,
                leave=False,
                verbose=verbose>1
                )
            if variants_vep is None:
                if verbose:
                    print(f"No pathogenic variants found for {tx_id}")
                continue
            hap_tx_pathogenic = _add_variant_to_haplotypes_tx(
                hap_tx=hap_tx, 
                variants=variants, 
                variants_vep=variants_vep, 
                add_variant_id=add_variant_id,
                include_all_fields=include_all_fields,
                include_variant_info=include_variant_info,
                ref_checks=ref_checks,
                desc="Adding pathogenic variants to haplotypes",
                leave=False)
            # Add benign variants 
            variants = filter_variants(
                max_variants=max_benign_variants,
                variants_tx = variant_sets[tx_id]['variants_benign'], 
                params={'consequence_terms':consequence_terms} if len(consequence_terms)>0 else None,
                exact=True,
                leave=False,
                verbose=verbose>1
                )
            if len(variants)==0:
                if verbose:
                    print(f"No benign variants found for {tx_id}")
                continue
            variants_vep = get_vep(
                    ids=[x['id'] for x in variants], 
                    params=params_vep,
                    client=client,
                    save_dir=save_dir_vep,
                    desc="Getting benign variant VEP",
                    leave=False,
                    **kwargs
                    )
            variants_vep = filter_vep(
                variants_vep,
                consequence_terms=consequence_terms,
                exact=True,
                leave=False,
                verbose=verbose>1
                )
            if variants_vep is None:
                if verbose:
                    print(f"No benign variants found for {tx_id}")
                continue
            hap_tx_benign = _add_variant_to_haplotypes_tx(
                hap_tx=hap_tx, 
                variants=variants, 
                variants_vep=variants_vep, 
                add_variant_id=add_variant_id,
                include_all_fields=include_all_fields,
                include_variant_info=include_variant_info,
                ref_checks=ref_checks,
                desc="Adding benign variants to haplotypes",
                leave=False)
            # Add pathogenic/benign variants only if they are present
            haplotypes_pathogenic[tx_id] = {}
            haplotypes_pathogenic[tx_id]['protein_haplotypes'] = hap_tx_pathogenic
            haplotypes_benign[tx_id] = {}
            haplotypes_benign[tx_id]['protein_haplotypes'] = hap_tx_benign
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
    return haplotypes_wt, haplotypes_pathogenic, haplotypes_benign

def subset_haplotypes(haplotypes,
                      tx_ids=None,
                      max_transcripts=None):
    if tx_ids is not None:
        # Limit to tx_ids
        haplotypes = {k: haplotypes[k] for k in tx_ids}
    if max_transcripts is not None:
        # Limit to max_transcripts
        tx_ids = list(haplotypes.keys())[:max_transcripts]
        haplotypes = {k: haplotypes[k] for k in tx_ids}
    return haplotypes
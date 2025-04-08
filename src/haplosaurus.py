import os
import warnings
from typing import Dict, List, Optional, Union, Tuple, Set
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import ensembl_rest

import src.utils as utils
import src.config as config
import src.biopython as bp
import src.ontologies as on
import src.ensembl_rest as er
import src.onekg as onekg

DIR_DICT = er.DIR_DICT
DIR_DICT.update({
    "variants": os.path.join(config.DATA_DIR, "haplosaurus","variants",""),
    "variant_sets": os.path.join(config.DATA_DIR, "haplosaurus","variant_sets","")
})

def get_figshare(fname: Union[str, List[str]] = ["haplotype_analysis_database.zip",
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
    fname = utils.as_list(fname)[0]
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

def filter_variants(variants_tx: List[Dict], 
                    params: Optional[Dict] = None,
                    max_variants: Optional[int] = None,
                    exact: bool = False,
                    reverse: bool = False,
                    verbose: bool = True,
                    **kwargs):
    """
    Filter variants based on the provided parameters.

    Args:
        variants_tx (List[Dict]): A list of variant dictionaries.
        params (Optional[Dict]): A dictionary of parameters to filter the variants.
        max_variants (Optional[int]): The maximum number of variants to return.
    """
    assert isinstance(variants_tx, list), "variants_tx must be a list"
    assert isinstance(exact, bool), "exact must be a boolean"
    assert isinstance(reverse, bool), "reverse must be a boolean"
    assert isinstance(verbose, bool), "verbose must be a boolean"
    
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
                opts = utils.as_list(variant[k]) 
                opts_selected = utils.as_list(v)
                if exact:
                    condition_met.append(set(opts) == set(opts_selected))
                else:
                    opt_overlap = utils.intersect(opts, opts_selected)
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

def get_variants(tx_id: str,
                 save_dir: Optional[Path] = Path(DIR_DICT["variants"]),
                 client: Optional[ensembl_rest.EnsemblClient] = None,
                 params_query: Dict = {'feature':'variation'},
                 params_filter: Dict = {},
                 force: bool = False, 
                 error: bool = True,
                 verbose: bool = True,
                 **kwargs):
    """
    Get variants for a given transcript ID.
    See the following for more information:
        https://ensemblrest.readthedocs.io/en/latest/#ensembl_rest.EnsemblClient.overlap_id
        https://rest.ensembl.org/documentation/info/overlap_id
        https://useast.ensembl.org/info/genome/variation/species/sets.html

    Args:
        tx_id (str): The transcript ID.
        save_dir (Optional[Path]): The directory to save the variants.
        client (Optional[ensembl_rest.EnsemblClient]): The Ensembl client.
        params_query (Dict): The parameters for the query.
        params_filter (Dict): The parameters for the filter.
        force (bool): Whether to force the retrieval of the variants.
        error (bool): Whether to raise an error if the variants are not found.
        verbose (bool): Whether to print verbose output.
    """ 
    variants_tx = None
    # Check if variants are already cached 
    if save_dir is not None:
        filename = er._params_to_filename(params_query)
        # Customize save subdir based on params_query
        save_path = os.path.join(save_dir,tx_id,filename)
        variants_tx = utils.load_json(save_path, 
                                      force=force, 
                                      verbose=verbose>1)
        variants_tx = list(variants_tx.values())[0] if isinstance(variants_tx, dict) else variants_tx
    if variants_tx is None:
        client = er.get_ensembl_client(client=client)
        try:
            variants_tx = client.overlap_id(id=tx_id, 
                                            params=params_query
                                            )
            # Cache variants
            if len(variants_tx)>0:
                utils.save_json({tx_id:variants_tx}, 
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


def variation_to_dfs(variation: Dict,
                    fields: List[str] = ['populations', # population-level variant frequencies
                            'population_genotypes', # population-level genotype frequencies
                            'genotypes', # individual-level genotype frequencies
                            'phenotypes', # associated phenotypes
                            'mappings', # genomic coordinates
                            ]):
    """
    Convert a variation dictionary to a dictionary of pandas DataFrames.

    Args:
        variation (Dict): The variation dictionary.
        fields (List[str]): The fields to convert to DataFrames.
    """
    fields_valid = [x for x in fields if x in variation.keys()]
    return {field:pd.DataFrame(variation[field]) for field in fields_valid}

def _check_vep_i(var: List[Dict],
                save_dir_vep: Path,
                client: ensembl_rest.EnsemblClient,
                consequence_type: List[str],
                desc_get: str,
                desc_filter: str,
                verbose: bool,
                exact: bool = True,
                params_vep: Dict = config.PARAMS_VEP):
    """
    Check VEP for a list of variants.

    Args:
        var (List[Dict]): The list of variants.
        save_dir_vep (Path): The directory to save the VEP results.
    """
    vep = er.get_vep(ids=[x['id'] for x in var],
                save_dir=save_dir_vep,
                client=client,
                desc=desc_get,
                leave=False,
                verbose=verbose>1,
                params=params_vep
                )
    vep = er.filter_vep(vep,
                    consequence_terms=consequence_type,
                    exact=exact,
                    desc=desc_filter,
                    leave=False,
                    verbose=verbose>1)
    return vep

def _check_vep(varp: List[Dict],
                varb: List[Dict],
                tx_id: str,
                save_dir_vep: Path,
                client: ensembl_rest.EnsemblClient,
                consequence_type: List[str],
                verbose: bool,
                params_vep: Dict = config.PARAMS_VEP):
    """
    Check VEP for a list of variants.

    Args:
        varp (List[Dict]): The list of pathogenic variants.
        varb (List[Dict]): The list of benign variants.
        tx_id (str): The transcript ID.
        save_dir_vep (str): The directory to save the VEP results.
        client (ensembl_rest.EnsemblClient): The Ensembl client.
        consequence_type (List[str]): The consequence types to filter on.
        verbose (bool): Whether to print verbose output.
    """ 
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

def get_variant_sets(tx_ids: List[str],
                     so_term: str = 'SO:0001583',
                     consequence_type: List[str] = ['missense_variant'],
                     include_descendants: bool = True,
                     pathogenic_variant_set: str = 'clin_assoc',
                     benign_variant_set: str = 'ClinVar',
                     nagative_variant_set: str = '1kg_3',  # '1kg_3_com'
                     check_vep: bool = True,
                     add_vep: bool = True,
                     params_vep: Dict = config.PARAMS_VEP,
                     save_dir_variants: Path = Path(DIR_DICT["variants"]),
                     save_dir_variant_sets: Path = Path(DIR_DICT["variant_sets"]),
                     save_dir_vep: Path = Path(DIR_DICT["vep"]),
                     force: bool = False,
                     exact: Dict = {'pathogenic':False,
                                  'benign':False,
                                  'negative':True},
                     client: Optional[ensembl_rest.EnsemblClient] = None,
                     cache_only: bool = False,
                     error: bool = True,
                     verbose: bool = True):
    """
    Get variant sets for a list of transcripts.

    Args:
        tx_ids (List[str]): The list of transcript IDs.
    """
    # Gather pathogenic/benign variants
    variant_sets = {}
    consequence_type = utils.as_list(consequence_type)
    client = er.get_ensembl_client(client=client)
    if not check_vep:
        add_vep = False
    # Include descendants of consequence type
    consequence_type_path = '.'.join(consequence_type)
    if include_descendants:
        so = on.get_sequence_ontology()
        consequence_type = on.get_descendants(label_or_id=consequence_type,
                                            ont=so, 
                                            return_as="label",
                                            verbose=verbose)
        so_term = on.get_descendants(label_or_id=so_term,
                                  ont=so,
                                  return_as="id",
                                  verbose=verbose) 
    if cache_only:
        force = False
    # Iterate over transcripts
    for tx_id in tqdm(tx_ids,"Gathering variant sets"):
        save_path = f"{save_dir_variant_sets}/{consequence_type_path}/{tx_id}.json.gz"
        variant_sets_tx = utils.load_json(save_path, 
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
            utils.save_json(obj=variant_sets_tx,
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
    

def get_variant_ids(variant_sets: Union[List[Dict], Dict[str, Dict[str, List[Dict]]]]):
    """Extract variant IDs from variant sets data structure.

    Args:
        variant_sets: Either a list of variant dictionaries, or a nested dictionary
            mapping transcript IDs to variant categories containing variant lists.
            Each variant must have an 'id' field.

    Returns:
        If variant_sets is a list:
            list: List of variant IDs extracted from the variants.
        If variant_sets is a dict:
            dict: Nested dictionary mapping transcript IDs and variant categories to lists of variant IDs.
            Structure matches input variant_sets but with just the IDs extracted.

    Example:
        # For list input:
        >>> variants = [{'id': 'rs1'}, {'id': 'rs2'}]
        >>> get_variant_ids(variants)
        ['rs1', 'rs2']

        # For dict input:
        >>> variants = {'tx1': {'pathogenic': [{'id': 'rs1'}], 'benign': [{'id': 'rs2'}]}}
        >>> get_variant_ids(variants)
        {'tx1': {'pathogenic': ['rs1'], 'benign': ['rs2']}}
    """
    
    if isinstance(variant_sets, list):
        ids = [x['id'] for x in variant_sets]
    elif isinstance(variant_sets, dict):
        ids = {}
        for tx_id in tqdm(variant_sets.keys()):
            ids[tx_id] = {}
            for k in variant_sets[tx_id].keys(): 
                ids[tx_id][k] = [variant['id'] for variant in variant_sets[tx_id][k]]
    return ids

def list_haplotypes(cache: Path = Path(DIR_DICT["haplotypes"]),
                    verbose: bool = True):
    import glob
    files = glob.glob(os.path.join(cache, "*.json.gz"))
    if len(files)==0:
        raise ValueError(f"No haplotypes found in: '{cache}'")
    else:
        if verbose:
            print(f"Found {len(files)} haplotypes in: '{cache}'")
        return [x.split('/')[-1].split('.')[0] for x in files]
    
def get_haplotypes(tx_ids: Optional[List[str]] = None,
                   max_tx_ids: Optional[int] = None, 
                   species: str = "homo_sapiens",
                   params: dict = config.PARAMS_HAPLOTYPES,
                   use_protein_ids: bool = False,
                   client: Optional[ensembl_rest.EnsemblClient] = None,
                   force: bool = False,
                   cache: Path = Path(DIR_DICT["haplotypes"]),
                   cache_only: bool = False,
                   error: bool = False,
                   timeout: int = er.TIMEOUT,
                   verbose: bool = True) -> dict:
    """Get haplotype information for transcript IDs from Ensembl REST API.
    For more information, see:
        Ensembl REST API: https://rest.ensembl.org/documentation/info/transcript_haplotypes_get
        ensembl-rest module: https://ensemblrest.readthedocs.io/en/latest/#ensembl_rest.EnsemblClient.transcript_haplotypes_get
        Haplosaurus: https://useast.ensembl.org/info/docs/tools/vep/haplo/index.html

    Args:
        tx_ids (list, optional): List of transcript IDs to get haplotypes for. If None, will use all cached haplotypes. Defaults to None.
        max_tx_ids (int, optional): Maximum number of transcript IDs to process. Defaults to None.
        save_dir (Path, optional): Directory to cache haplotype data. Defaults to DIR_DICT["haplotypes"].
        species (str, optional): Species name for Ensembl API. Defaults to "homo_sapiens".
        params (dict, optional): Parameters for Ensembl API request. Defaults to PARAMS_HAPLOTYPES.
        use_protein_ids (bool, optional): Whether to convert transcript IDs to protein IDs. Defaults to False.
        client (EnsemblClient, optional): Ensembl REST API client. If None, will create new client. Defaults to None.
        force (bool, optional): Whether to force API request even if cached data exists. Defaults to False.
        cache (Path, optional): Directory to cache haplotype data. Defaults to DIR_DICT["haplotypes"].
        cache_only (bool, optional): Whether to only return cached data without making API requests. Defaults to False.
        error (bool, optional): Whether to raise exceptions on API errors. Defaults to False.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.

    Returns:
        dict: Dictionary mapping transcript IDs to their haplotype information.
            Each value contains protein haplotype sequences and metadata from Ensembl.

    Raises:
        Exception: If error=True and API request fails for a transcript.
    """
    client = er.get_ensembl_client(client=client)
    haplotypes = {}

    # Get tx_ids if not provided
    if tx_ids is None:
        tx_ids = list_haplotypes(cache=cache,
                                 verbose=verbose)
    if max_tx_ids is not None:
        tx_ids = tx_ids[:max_tx_ids]

    # Convert IDs to Ensembl transcript IDs
    tx_ids = utils.process_ids(ids=tx_ids)
    # non_enst = [id for id in tx_ids if not id.startswith('ENST')]
    # if len(non_enst)>0:
    #     enst_map = er.xref_external(non_enst, 
    #                                 species=species, 
    #                                 client=client)
    #     tx_ids = [enst_map[id]['ENST'] for id in non_enst]

    # Query Ensembl REST API
    haplotypes = er.transcript_haplotypes_get(ids=tx_ids,
                                            species=species,
                                            params=params,
                                            client=client,
                                            force=force,
                                            cache=cache,
                                            cache_only=cache_only,
                                            error=error,
                                            timeout=timeout,
                                            verbose=verbose)
    # Convert to protein IDs if requested
    if use_protein_ids:
        haplotypes = _as_protein_ids(haplotypes)
    # Return
    return haplotypes

def _as_protein_ids(haplotypes: Dict[str, Dict]) -> Dict[str, Dict]:
    return dict(zip(get_haplotype_protein_ids(haplotypes, add_self=True).values(), haplotypes.values()))

def get_haplotype_seqs(haplotypes: Union[Dict[str, Dict], Dict[str, List[Dict]]],
                       key: str = 'protein_haplotypes',
                       aligned: int = 1,
                       as_msa: bool = False,
                       return_missing: bool = False,
                       use_protein_ids: bool = False,
                       add_haplotype_names: bool = False,
                       verbose: bool = False):
    """Get haplotype sequences from haplotype information.

    Args:
        haplotypes (Union[Dict[str, Dict], Dict[str, List[Dict]]]): The haplotype information.
        key (str, optional): The key to use to get the haplotype sequences. 
        Options include: 'protein_haplotypes' or 'cds_haplotypes'.
            Defaults to 'protein_haplotypes'.
            aligned (int, optional): The alignment type. Defaults to 1.
            If 0, will return the unaligned sequence of the haplotype (without gaps).
            If 1, will return the aligned sequence of just the haplotype (with gaps).
            If 2, will return the aligned sequence of the reference and the haplotype (with gaps).
        return_missing (bool, optional): Whether to return missing sequences. Defaults to False.
        use_protein_ids (bool, optional): Whether to use protein IDs. Defaults to False.
        add_haplotype_names (bool, optional): Whether to add haplotype names. Defaults to False. 
            If 0 (or False), will only return unnamed sequences.
            If 1 (or True), will add haplotype names as first element in tuple, with the sequence as the second element.
            If 2, will add haplotype names as keys, with the sequence as the value.
            If 3, will unnest the haplotype names and sequences into a single dictionary with the haplotype name as the key and the sequence as the value.
        verbose (bool, optional): Whether to print progress messages. Defaults to False.

    Returns:
        Dict[str, List[str]]: A dictionary mapping transcript IDs to their haplotype sequences.
    """
    
    hap_seqs = {}
    missing_seqs = []
    if aligned!=2 and as_msa:
        warnings.warn('Aligned must be 2 to convert to MSA (`as_msa=True`)')
        as_msa = False
    for tx_id in tqdm(haplotypes.keys(),
                      desc="Getting haplotype sequences"):
        
        if isinstance(haplotypes[tx_id], dict) and key in haplotypes[tx_id].keys():
            if aligned==1:
                hap_seqs[tx_id] = [x['aligned_sequences'][1] for x in haplotypes[tx_id][key]]
            elif aligned==2:
                hap_seqs[tx_id] = [x['aligned_sequences'] for x in haplotypes[tx_id][key]]
            elif aligned==0:
                hap_seqs[tx_id] = [x['seq'] for x in haplotypes[tx_id][key]]
            else:
                raise ValueError(f"Invalid alignment type: {aligned}")
        else:
            if len(haplotypes[tx_id])>0:
                if aligned==1:
                    # Return aligned haplotype sequence only
                    hap_seqs[tx_id] = haplotypes[tx_id][0]['aligned_sequences'][1] 
                elif aligned==2:
                    # Return both sequences (aligned reference and haplotype)
                    hap_seqs[tx_id] = haplotypes[tx_id][0]['aligned_sequences']
                else:
                    # Return unaligned haplotype sequence only
                    hap_seqs[tx_id] = haplotypes[tx_id][0]['seq']
            else:
                if verbose:
                    warnings.warn(f"No seqs found for {tx_id}")
                missing_seqs += [tx_id]
                continue
    if as_msa:
        hap_seqs = {tx_id:[bp.as_msa(seqs) for seqs in tx_seqs] for tx_id,tx_seqs in hap_seqs.items()}
                
    # Add haplotype names
    if add_haplotype_names:
        haplotype_names = get_haplotype_names(haplotypes, 
                                              key=key) # list of haplotype names
        hap_seqs = {tx_id:list(zip(haplotype_names[tx_id], tx_seqs)) for tx_id,tx_seqs in hap_seqs.items()}
        # Add haplotype names as keys
        if add_haplotype_names>1:
             hap_seqs = {tx_id:{x[0]:x[1] for x in tx_seqs} for tx_id,tx_seqs in hap_seqs.items()}
             # Unnest if >2
             if add_haplotype_names>2:
                 hap_seqs = {k:v for d in hap_seqs.values() for k,v in d.items()}
    
    # Use protein IDs as names
    if use_protein_ids:
        enst_to_ensp = get_haplotype_protein_ids(haplotypes=haplotypes, 
                                                 add_self=True)
        hap_seqs = {enst_to_ensp[k]:v for k,v in hap_seqs.items()}
    
       
    
    # Return 
    if return_missing:
        return hap_seqs, missing_seqs
    else:
        return hap_seqs

def get_haplotype_counts(haplotypes: Dict[str, Dict],
                         key: str = 'protein_haplotypes') -> Dict[str, int]:
    """Get the number of haplotypes for each transcript.

    Args:
        haplotypes (Dict[str, Dict]): The haplotype information.
        key (str, optional): The key to use to count the haplotypes. Defaults to 'protein_haplotypes'.

    Returns:
        Dict[str, int]: A dictionary mapping transcript IDs to the number of haplotypes.
    """

    return {k:len(haplotypes[k][key]) for k in haplotypes.keys()}

def get_haplotype_freqs(haplotypes: Dict[str, Dict],
                        tx_ids: Optional[List[str]] = None) -> Tuple[Dict[str, Dict], Set[str], Set[str]]:
    pop_freqs = {}
    populations = set()
    tx_ids = utils.as_list(tx_ids)
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

def add_haplotype_freqs(df: pd.DataFrame, 
                        haplotypes: Dict[str, Dict], 
                        cohorts: Optional[List[str]] = ['1000GENOMES:phase_3'],
                        haplotype_col: str = "label_base",
                        protein_id_col: str = "ENSP_haplosaurus",
                        tx_id_col: str = "ENST_haplosaurus",
                        add_top_pop: bool = True,
                        add_top_superpop: bool = True,
                        force: bool = False,
                        verbose: bool = True): 
    """Add haplotype frequencies to a dataframe.

    Args:
        df (pd.DataFrame): The dataframe to add the haplotype frequencies to.
        haplotypes (Dict[str, Dict]): The haplotype information.
        cohorts (Optional[List[str]], optional): The cohorts to use. Defaults to ['1000GENOMES:phase_3'].
    """

    df = add_txid(df=df, 
                    haplotypes=haplotypes, 
                    protein_id_col=protein_id_col,
                    tx_id_col=tx_id_col,
                    force=force,
                    verbose=verbose)
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
        cohorts = utils.as_list(cohorts)
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
                lambda row: pop_freqs[row[tx_id_col]][row[haplotype_col]][pop] 
                    if row[tx_id_col] in pop_freqs 
                    and row[haplotype_col] in pop_freqs[row[tx_id_col]] 
                    and pop in pop_freqs[row[tx_id_col]][row[haplotype_col]] 
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
                warnings.warn("All frequency columns contain NA values")
        else:
            warnings.warn("No frequency columns found")

    if add_top_superpop:
        if 'top_pop' in df.columns:
            if 'top_superpop' not in df.columns or force:
                pops = onekg.get_sample_metadata()
                pop_map = dict(zip(pops['Population Code'], pops['Super Population']))
                df['top_superpop'] = df['top_pop'].str.replace(f'{cohorts[0]}:', '').str.split('_').str[0].map(pop_map)
                df['top_superpop'].fillna('N/A', inplace=True) 
    return df

def filter_haplotype_freqs(df: pd.DataFrame,
                           freq_filters: Optional[Dict] = {}) -> pd.DataFrame:
    """Filter haplotype frequencies in a dataframe.

    Args:
        df (pd.DataFrame): The dataframe to filter.
        freq_filters (Optional[Dict], optional): The filters to apply to the haplotype frequencies. Defaults to {}.
    """
    if freq_filters is not None and len(freq_filters)>0:
        for freq_col, freq_val in freq_filters.items():
            if freq_val is not None:
                df = df.loc[df[freq_col] >= freq_val]
    return df


def get_haplotype_names(haplotypes: Dict[str, Dict],
                        key: str = 'protein_haplotypes') -> Dict[str, List[str]]:
    """
    Get the haplotype name from the haplotype entry,
        or a dict of haplotype entries indexed by transcript ID.
    """
    if key in haplotypes.keys():
        return [x['name'].split('_')[0] for x in haplotypes[key]]
    hap_names = {}
    for tx_id in tqdm(haplotypes.keys()):
        hap_names[tx_id] = [x['name'].split('_')[0] for x in haplotypes[tx_id][key]]
    return hap_names

def get_haplotype_protein_ids(haplotypes: Dict[str, Dict], 
                              add_self: bool = False,
                              invert: bool = False) -> Dict[str, str]:
    """
    Get the protein ID from the haplotype entry or a dict of haplotype entries.

    Args:
        haplotypes: A single haplotype entry containing 'protein_haplotypes' key,
            or a dict of haplotype entries indexed by transcript ID.
        add_self: If True, add the protein ID to the dict of haplotype entries in addition to the transcript ID.

    Returns:
        str: The protein ID if input is a single haplotype entry
        dict: Mapping of transcript IDs to protein IDs if input is a dict
    """
    if 'protein_haplotypes' in haplotypes.keys():
        return haplotypes['protein_haplotypes'][0]['name'].split(':')[0]
    
    hap_protein_ids = {}
    for tx_id in haplotypes.keys():
        protein_id = haplotypes[tx_id]['protein_haplotypes'][0]['name'].split(':')[0]
        hap_protein_ids[tx_id] = protein_id
        if add_self:
            hap_protein_ids[protein_id] = protein_id 
    if invert:
        return utils.invert_dict(hap_protein_ids)
    return hap_protein_ids

def get_haplotype_ref(haplotypes: Union[List[Dict], Dict[str, Dict]],
                      verbose: bool = True) -> Union[List[Dict], Dict[str, Dict]]:
    """Get the reference haplotype from a list of haplotypes.

    Args:
        haplotypes (Union[List[Dict], Dict[str, Dict]]): The haplotypes to get the reference from.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.
    """
    if isinstance(haplotypes, list):
        if isinstance(haplotypes[0], dict):
            hap_ref = [x for x in haplotypes if ":REF" in tqdm(x['name'],
                                                            disable=not verbose,
                                                            leave=False)]
        elif isinstance(haplotypes[0], tuple):
            hap_ref = [x for x in haplotypes if ":REF" in tqdm(x[0],
                                                            disable=not verbose,
                                                            leave=False)]
        elif isinstance(haplotypes[0], str):
            hap_ref = [x for x in haplotypes if ":REF" in tqdm(x,
                                                            disable=not verbose,
                                                            leave=False)]
    elif 'protein_haplotypes' in haplotypes.keys():
        hap_ref = [x for x in haplotypes['protein_haplotypes'] if ":REF" in tqdm(x['name'])]
    elif isinstance(haplotypes, dict):
        hap_ref = {}
        for tx_id in tqdm(haplotypes.keys(),
                          desc="Processing transcripts",
                          disable=not verbose,
                          leave=False):
            hap_ref[tx_id] = [x for x in haplotypes[tx_id]['protein_haplotypes'] if ":REF" in x['name']]
            if isinstance(hap_ref[tx_id], list) and len(hap_ref[tx_id])>0:
                hap_ref[tx_id] = hap_ref[tx_id]
    else:
        raise ValueError(f"Invalid haplotype type: {type(haplotypes)}")
    return hap_ref

def to_stop(seq,
            replace='-'):
    if replace is not None:
        seq = seq.replace(replace, '')
    return seq if seq.find('*')==-1 else seq[:seq.find('*')]

def _add_variant_to_haplotypes_tx(hap_tx: Dict[str, Dict], 
                                  variants: List[Dict], 
                                  variants_vep: Dict[str, List[Dict]],  
                                  include_all_fields: bool,
                                  # Turn off for now since ref genome not gauranteed to be the same as the haplotype
                                  ref_checks: bool = True,
                                  desc: str = "Adding variants to haplotypes",
                                  leave: bool = False
                                 ) -> Dict[str, Dict]:
        """Add a variant to a haplotype.

        Args:
            hap_tx (Dict[str, Dict]): The haplotype to add the variant to.
            variants (List[Dict]): The variants to add.
            variants_vep (Dict[str, List[Dict]]): The VEP results for the variants.

        Returns:
            Dict[str, Dict]: The haplotypes with the variants added.
        """
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

def add_variant_to_haplotypes(haplotypes: Dict[str, Dict], 
                              variant_sets: Dict[str, Dict],
                              client: Optional[ensembl_rest.EnsemblClient] = None,
                              tx_ids: Optional[List[str]] = None,
                              max_tx_ids: Optional[int] = None,
                              add_variant_info: bool = True,
                              include_all_fields: bool = False,
                              params_vep: Dict = config.PARAMS_VEP,
                              consequence_terms: Dict = {},
                              save_dir_vep: Path = Path(DIR_DICT["vep"]),
                              ref_checks: bool = False,
                              error: bool = True,
                              exact={'pathogenic':True,
                                     'benign':True},
                              verbose: bool = True,
                              **kwargs
                              ) -> Dict[str, Dict]:
    """Add variants to haplotypes.

    Args:
        haplotypes (Dict[str, Dict]): The haplotypes to add the variants to.
        variant_sets (Dict[str, Dict]): The variant sets to add the variants to.
        client (Optional[ensembl_rest.EnsemblClient], optional): The Ensembl REST API client. Defaults to None.

    Returns:
        Dict[str, Dict]: The haplotypes with the variants added.
    """
    
    max_pathogenic_variants=1
    max_benign_variants=1
    
    haplotypes_pathogenic = {}
    haplotypes_benign = {}
    haplotypes_wt = subset_haplotypes(
        haplotypes,
        tx_ids=tx_ids,
        max_tx_ids=max_tx_ids)
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
            varp_vep = er.get_vep(
                    ids=[x['id'] for x in varp], 
                    params=params_vep,
                    client=client,
                    save_dir=save_dir_vep,
                    desc="Getting pathogenic variant VEP",
                    leave=False,
                    **kwargs
                    ) 
            varp_vep = er.filter_vep(
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
            varb_vep = er.get_vep(
                    ids=[x['id'] for x in varb], 
                    params=params_vep,
                    client=client,
                    save_dir=save_dir_vep,
                    desc="Getting benign variant VEP",
                    leave=False,
                    **kwargs
                    )
            varb_vep = er.filter_vep(
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
        max_tx_ids=max_tx_ids)
    # Filter once more to ensure max_tx_ids is respected
    haplotypes_wt = subset_haplotypes(
        haplotypes_wt,
        max_tx_ids=max_tx_ids)
    haplotypes_pathogenic = subset_haplotypes(
        haplotypes_pathogenic,
        tx_ids=tx_ids,
        max_tx_ids=max_tx_ids)
    haplotypes_benign = subset_haplotypes(
        haplotypes_benign,
        tx_ids=tx_ids,
        max_tx_ids=max_tx_ids)
    
    return haplotypes_wt, haplotypes_pathogenic, haplotypes_benign

def subset_haplotypes(haplotypes: Dict[str, Dict],
                      tx_ids: Optional[List[str]] = None,
                      max_tx_ids: Optional[int] = None) -> Dict[str, Dict]:
    """Subset haplotypes by tx_ids and max_tx_ids.

    Args:
        haplotypes (Dict[str, Dict]): The haplotypes to subset.
        tx_ids (Optional[List[str]], optional): The tx_ids to subset to. Defaults to None.

    Returns:
        Dict[str, Dict]: The subsetted haplotypes.
    """
    if tx_ids is not None:
        # Limit to tx_ids
        haplotypes = {k: haplotypes[k] for k in tx_ids if k in haplotypes.keys()}
    if max_tx_ids is not None:
        # Limit to max_tx_ids
        tx_ids = list(haplotypes.keys())[:max_tx_ids]
        haplotypes = {k: haplotypes[k] for k in tx_ids if k in haplotypes.keys()}
    return haplotypes


def haplotypes_to_batches(haplotypes: Dict[str, Dict],
                          suffix: str = '',
                          strip: str = '*',
                          add_variant_ids: bool = True,
                          leave: bool = False,
                          verbose: bool = True) -> Dict[str, List[Tuple[str, str]]]:
    """
    Prepare batches for ESM2.

    Args:
        haplotypes (Dict[str, Dict]): The haplotypes to prepare batches for.
        suffix (str, optional): The suffix to add to the batch names. Defaults to ''.
        strip (str, optional): The character to strip from the haplotype sequences. Defaults to '*'.
        add_variant_ids (bool, optional): Whether to add variant IDs to the batch names. Defaults to True.
        leave (bool, optional): Whether to leave the progress bar. Defaults to False.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.

    Returns:
        Dict[str, List[Tuple[str, str]]]: The batches.
    """
    if verbose:
        print("Preparing batches for ESM")
    batches = {}
    for tx_id, hap_tx in tqdm(haplotypes.items(),
                              desc="Preparing batches",
                              leave=leave):
        suffix_final = suffix
        if 'variants' in hap_tx.keys() and add_variant_ids:
            suffix_final += ":"+",".join(set([x['id'] for x in hap_tx['variants']]))
        batches[tx_id] = [(x['name']+suffix_final, bp.to_stop(x['seq'].strip(strip))) for x in hap_tx['protein_haplotypes']]
    return batches

def haplotypes_to_batches_grouped(haplotypes_grouped: Dict[str, Dict],
                                  merge: bool = True,
                                  suffix: Optional[str] = None,
                                  **kwargs) -> Dict[str, List[Tuple[str, str]]]: 
                                  
    """
    Prepare batches for ESM2.

    Args:
        haplotypes_grouped (Dict[str, Dict]): The haplotypes to prepare batches for.
        merge (bool, optional): Whether to merge the batches. Defaults to True.
        suffix (str, optional): The suffix to add to the batch names. Defaults to ''.
        **kwargs: Additional keyword arguments to pass to haplotypes_to_batches.

    Returns:
        Dict[str, List[Tuple[str, str]]]: The batches.
    """
    batches_grouped = {} 
    # Get batches
    for group, haplotypes_group in haplotypes_grouped.items():
        batches_grouped[group] = haplotypes_to_batches(haplotypes=haplotypes_group, 
                                                       suffix=f"_{group}" if suffix is None else suffix,
                                                       **kwargs
                                                       )
        
    # Merge batches across groups into one dict entry per tx_id
    # Ensure that lists get appended instead of overwritten
    if merge:
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


def reconstruct_data(tx_ids: Optional[List[str]] = None,
                     cache_haplotypes: Path = Path(DIR_DICT["haplotypes"]),
                     cache_variants: Path = Path(DIR_DICT["variants"]),
                     cache_variant_sets: Path = Path(DIR_DICT["variant_sets"]),
                     cache_vep: Path = Path(DIR_DICT["vep"]),
                     species: str = "homo_sapiens",
                     so_term: str = 'SO:0001583',
                     consequence_terms: List[str] = ['missense_variant'],
                     params_haplotypes: Dict = config.PARAMS_HAPLOTYPES,
                     params_vep: Dict = config.PARAMS_VEP,
                     add_vep: bool = True,
                     max_tx_ids: Optional[int] = None,
                     ref_checks: bool = False,
                     merge_batches: bool = True,
                     exact: Dict = {'pathogenic':False,
                                  'benign':False,
                                  'negative':True},
                     cache_only={'get_haplotypes':True,
                                 'get_variant_sets':True},
                     force={'get_haplotypes':False,
                            'get_variant_sets':False},
                     check_vep: bool = True,
                     verbose: bool = True):
    """Reconstruct data from haplotypes, variant sets, and ESM2 batches.

    Args:
        tx_ids (Optional[List[str]], optional): The tx_ids to reconstruct data for. Defaults to None.
        save_dir_haplotypes (Path, optional): The directory to save the haplotypes. Defaults to DIR_DICT["haplotypes"].
        save_dir_variants (Path, optional): The directory to save the variants. Defaults to DIR_DICT["variants"].
        save_dir_variant_sets (Path, optional): The directory to save the variant sets. Defaults to DIR_DICT["variant_sets"].
        save_dir_vep (Path, optional): The directory to save the VEP results. Defaults to DIR_DICT["vep"].
        species (str, optional): The species to reconstruct data for. Defaults to "homo_sapiens".
        so_term (str, optional): The SO term to use for variant sets. Defaults to 'SO:0001583'.
        consequence_terms (List[str], optional): The consequence terms to use for variant sets. Defaults to ['missense_variant'].
        params_haplotypes (Dict, optional): The parameters to use for haplotypes. Defaults to PARAMS_HAPLOTYPES.
        params_vep (Dict, optional): The parameters to use for VEP. Defaults to PARAMS_VEP.
        add_vep (bool, optional): Whether to add VEP results to the variant sets. Defaults to True.
        max_tx_ids (int, optional): The maximum number of tx_ids to reconstruct data for. Defaults to None.
        ref_checks (bool, optional): Whether to check the reference sequences. Defaults to False.
        merge_batches (bool, optional): Whether to merge the batches. Defaults to True.
        exact (Dict, optional): The exact parameters to use for variant sets. Defaults to {'pathogenic':False, 'benign':False, 'negative':True}.
        cache_only (Dict, optional): The cache only parameters to use for haplotypes and variant sets. Defaults to {'get_haplotypes':True, 'get_variant_sets':True}.
        force (Dict, optional): The force parameters to use for haplotypes and variant sets. Defaults to {'get_haplotypes':False, 'get_variant_sets':False}.
        check_vep (bool, optional): Whether to check the VEP results. Defaults to True.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict], Dict[str, List[Tuple[str, str]]]]: The haplotypes, variant sets, and batches.
    """

    # Get tx_ids
    if tx_ids is None:
        tx_ids = list_haplotypes(cache=cache_haplotypes,
                                 verbose=verbose)
    tx_ids = list(set(utils.as_list(tx_ids)))
    # Get haplotypes
    haplotypes = get_haplotypes(
        tx_ids=tx_ids, 
        cache=cache_haplotypes,
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
        save_dir_variants=cache_variants,
        save_dir_variant_sets=cache_variant_sets,
        save_dir_vep=cache_vep,
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
        max_tx_ids=max_tx_ids,
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

def get_suffix_dict(haplotypes: Dict[str, Dict],
                    prefix: str = '_',
                    by_tx_id: bool = True) -> Dict[str, str]:
    
    """
    Get a dictionary of tx_ids to variant IDs or variant IDs to tx_ids.

    Args:
        haplotypes (Dict[str, Dict]): The haplotypes to get the suffix dict for.
        prefix (str, optional): The prefix to add to the variant IDs. Defaults to '_'.
        by_tx_id (bool, optional): Whether to return a dict of tx_ids to variant IDs or variant IDs to tx_ids. Defaults to True.
    """
    if by_tx_id:
        # Return dict of tx_ids to variant IDs
        return {tx_id:prefix+hap_tx['variants'][0]['id'] for tx_id, hap_tx in haplotypes.items() if 'variants' in hap_tx.keys()}
    else:
        # Return dict of variant IDs to tx_ids
        return {hap_tx['variants'][0]['id']:prefix+tx_id for tx_id, hap_tx in haplotypes.items() if 'variants' in hap_tx.keys()}
    
def get_txid_map(haplotypes: Dict[str, Dict],
                 to: str = "protein_id",
                 invert: bool = False,
                 check_all: bool = False,
                 as_df: bool = False) -> Dict[str, str]:
    """
    Get a dictionary of tx_ids to protein IDs or protein IDs to tx_ids.

    Args:
        haplotypes (Dict[str, Dict]): The haplotypes to get the tx_id map for.
        to (str, optional): The type of ID to get. Defaults to "protein_id".
        invert (bool, optional): Whether to invert the dictionary. Defaults to False.
        check_all (bool, optional): Whether to check all protein IDs. Defaults to False.
        as_df (bool, optional): Whether to return a dataframe. Defaults to False.
    """
    if to == "protein_id":
        if check_all:
            if invert:
                raise ValueError("Cannot invert dict if check_all is True")
            tx_id_map  = {tx_id:list(set([x['name'].split(':')[0] for x in hap_tx['protein_haplotypes']])) for tx_id, hap_tx in haplotypes.items()}
        else:
            tx_id_map = {tx_id:hap_tx['protein_haplotypes'][0]['name'].split(':')[0] for tx_id, hap_tx in haplotypes.items()}
        if invert:
            tx_id_map  = utils.invert_dict(tx_id_map)
        if as_df:
            return pd.DataFrame(tx_id_map, index=['id']).T
        else:
            return tx_id_map
    else:
        raise ValueError("to must be 'protein_id'")
    
def add_txid(df: pd.DataFrame,
             haplotypes: Dict[str, Dict],
             protein_id_col: str = "ENSP_haplosaurus",
             tx_id_col: str = "ENST_haplosaurus",
             force: bool = False,
             verbose: bool = True):
    """
    Add a tx_id column to a dataframe.

    Args:
        df (pd.DataFrame): The dataframe to add the tx_id column to.
        haplotypes (Dict[str, Dict]): The haplotypes to get the tx_id map for.
        protein_id_col (str, optional): The column to add the tx_id to. Defaults to "ENSP_haplosaurus".
        tx_id_col (str, optional): The column to add the tx_id to. Defaults to "ENST_haplosaurus".
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        pd.DataFrame: The dataframe with the tx_id column added.
    """
    if tx_id_col not in df.columns or force:
        if verbose:
            print(f"Adding '{tx_id_col}' column")
        tx_id_map = get_txid_map(haplotypes, invert=True)
        df[tx_id_col] = df[protein_id_col].map(tx_id_map)
    return df

def add_genesymbol(df: pd.DataFrame,
                   on: List[str] = ['protein_id'],
                   map_file: str = os.path.join(config.DATA_DIR, "41467_2018_6542_MOESM4_ESM.xlsx")):
    """
    Add a gene symbol column to a dataframe.

    Args:
        df (pd.DataFrame): The dataframe to add the gene symbol column to.
        on (List[str], optional): The columns to merge on. Defaults to ['protein_id'].
        map_file (str, optional): The file to map the gene symbols to. Defaults to os.path.join(DATA_DIR, "41467_2018_6542_MOESM4_ESM.xlsx").

    Returns:
        pd.DataFrame: The dataframe with the gene symbol column added.
    """
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
    variation = {id:er.get_variation(id) for id in df[variant_col].unique()}
    variation_df = pd.concat([variation_to_dfs(x)[field].assign(variant_id=k) for k,x in variation.items()], axis=0)
    return df.merge(variation_df, on=variant_col, how='left')

def align_pairwise(hap_seqs):
    # convert hap_seqs to a MSA
    from Bio.Align import PairwiseAligner  
    msa_dict = {}
    for protein_id, tx_hap_seqs in tqdm(hap_seqs.items(),
                                        desc="Processing proteins",
                                        leave=False):
        
        ref_seq = get_haplotype_ref(tx_hap_seqs, verbose=False)[0][1][0]
        msa_dict[protein_id]  = []
        for seq_name, seqs in tx_hap_seqs:
            aligner = PairwiseAligner() 
            alignments = aligner.align(ref_seq.replace("-", ""),
                                       seqs[1].replace("-", ""))
            msa_dict[protein_id].append(alignments[0])

    return msa_dict

def get_offset_length(seq_name, 
                      base_offset=1):
    """Calculate the offset needed to account for insertions and deletions in a sequence.
    
    Args:
        seq_name (str): Name of the sequence containing mutation information.
            Format: "{protein_id}:{mutation1},{mutation2},...". 
            Example: "ENSP00000261556:565S>N,696insTA,699delELQ"
        base_offset (int, optional): Base offset to start from. Defaults to 1.
    
    Returns:
        int: Total offset after accounting for insertions and deletions.
            For reference sequences (ending in "REF"), returns base_offset.
            For variant sequences, returns base_offset + total_insertions - total_deletions.
    
    Example:
        get_offset("ENSP00000261556:565S>N,696insTA,699delELQ") 
    """
    if not seq_name.endswith("REF"):
        mutations = seq_name.split(":")[1].split(",")
        insertion_offsets = []
        deletion_offsets = []
        for mut in mutations:
            if 'ins' in mut:
                insertion_offsets += [len(mut.split('ins')[1])]
            elif 'del' in mut:
                deletion_offsets += [len(mut.split('del')[1])]

        return base_offset + sum(insertion_offsets) - sum(deletion_offsets)
    else:
        return base_offset

def haplotypes_to_fasta(haplotypes, 
                        aligned=1,
                        add_haplotype_names=2,
                        save_dir=os.path.join(config.DATA_DIR,"1KG","fasta"),
                        strip='*',
                        force=False,
                        verbose=True):
    """Convert haplotype sequences to FASTA format.
    
    Args:
        haplotypes (Dict[str, Dict]): The haplotypes to convert to FASTA format.
        aligned (int, optional): The alignment type. Defaults to 1.
        add_haplotype_names (int, optional): The type of haplotype names to add. Defaults to 2.
        save_dir (str, optional): The directory to save the FASTA files. Defaults to os.path.join(config.DATA_DIR,"1KG","fasta").
        force (bool, optional): Whether to force the conversion. Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
    
    Returns:
        Dict[str, str]: A dictionary of tx_ids to FASTA file paths.
    """ 
    hap_seqs = get_haplotype_seqs(haplotypes, 
                                  aligned=aligned,
                                  add_haplotype_names=add_haplotype_names)
    # Convert haplotype sequences to FASTA format
    fasta_paths = {}
    for tx_id in tqdm(hap_seqs.keys(),
                      desc="Converting haplotypes to FASTA",
                      leave=False,
                      disable=not verbose):
        fasta_path = os.path.join(save_dir, f"{tx_id}.fasta")
        fasta_paths[tx_id] = fasta_path
        if not os.path.exists(fasta_path) or force:
            with open(fasta_path, 'w') as f:
                for name, seq in hap_seqs[tx_id].items():
                    if strip is not None:
                        seq = seq.replace(strip, '')
                    f.write(f'>{name}\n{seq}\n')
    return fasta_paths
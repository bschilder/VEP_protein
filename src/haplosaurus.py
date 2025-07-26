import os
import warnings
from typing import Dict, List, Optional, Union, Tuple, Set
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from traitlets import default
import ensembl_rest
import glob

import src.utils as utils
import src.config as config
import src.biopython as bp
import src.ontologies as on
import src.ensembl_rest as er
import src.onekg as og
import src.Align.align_utils as au

from src.ensembl_rest import rename_haplotypes_keys as rename_haplotypes_keys


DIR_DICT = er.DIR_DICT
DIR_DICT.update({
    "haplotypes_merged": er.DIR_DICT['haplotypes'].replace('haplotypes', 'haplotypes_merged'),
    "variants": os.path.join(config.DATA_DIR, "haplosaurus","variants",""),
    "variant_sets": os.path.join(config.DATA_DIR, "haplosaurus","variant_sets",""),
    "tx_id_map": os.path.join(config.DATA_DIR, "haplosaurus","tx_id_map.pkl"),
    "HGDP_haplotypes": os.path.join(config.DATA_DIR, "Human_Genome_Diversity_Project","haplosaurus",""),
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
                    suffix: str = '.json.gz',
                    verbose: bool = True):
    import glob
    files = glob.glob(os.path.join(cache, f"*{suffix}"))
    if len(files)==0:
        raise ValueError(f"No haplotypes found in: '{cache}'")
    else:
        if verbose:
            print(f"Found haplotypes of {len(files)} transcripts in: '{cache}'")
        return [x.split(os.sep)[-1].split('.')[0] for x in files]
    
 
def get_haplotypes(tx_ids: Optional[List[str]] = None,
                   max_tx_ids: Optional[int] = None, 
                   species: str = "homo_sapiens",
                   params: dict = config.PARAMS_HAPLOTYPES,
                   use_protein_ids: bool = False,
                   client: Optional[ensembl_rest.EnsemblClient] = None,
                   force: bool = False,
                   cache: Path = Path(DIR_DICT["haplotypes"]),
                   cache_only: bool = False,
                   cache_merged: Optional[Path] = None,
                   check_names: bool = True,
                #    cache_merged: Path = Path(DIR_DICT["haplotypes_merged"]),
                   error: bool = False,
                   timeout: int = er.TIMEOUT,
                   leave: bool = True,
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
        cache_merged (Path, optional): Directory to cache merged haplotype data. 
            Defaults to DIR_DICT["haplotypes_merged"].
            If None, will not cache merged haplotypes.
        error (bool, optional): Whether to raise exceptions on API errors. Defaults to False.
        check_names (bool, optional): Whether to check and rename the keys of the haplotypes. Defaults to True.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.

    Returns:
        dict: Dictionary mapping transcript IDs to their haplotype information.
            Each value contains protein haplotype sequences and metadata from Ensembl.

    Raises:
        Exception: If error=True and API request fails for a transcript.
    """
    
    haplotypes = {}

    # Get tx_ids if not provided
    if tx_ids is None:
        tx_ids = list_haplotypes(cache=cache,
                                 verbose=verbose)
    elif cache_only:
        tx_ids_all = list_haplotypes(cache=cache,
                                     verbose=verbose)
        
        tx_ids = utils.intersect(tx_ids, tx_ids_all)
    if len(tx_ids)==0:
        raise ValueError(f"No haplotypes found for {tx_ids}")

    if max_tx_ids is not None:
        tx_ids = tx_ids[:max_tx_ids]

    # Convert IDs to Ensembl transcript IDs
    tx_ids = utils.process_ids(ids=tx_ids)

    # Check if cached merged haplotypes exists
    if cache_merged is not None:
        checksum_path = utils.ids_to_checksum_filename(
            tx_ids, 
            dir=DIR_DICT["haplotypes_merged"],
            suffix=".pkl")

        checksum_loaded = utils.load_pickle(checksum_path, 
                                            force=force,
                                            verbose=verbose)
        if checksum_loaded is not None:
            return checksum_loaded



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
                                            leave=leave,
                                            check_names=check_names,
                                            verbose=verbose)
    
    # Save haplotypes
    if cache_merged is not None:
        utils.save_pickle(haplotypes, 
                          checksum_path, 
                          verbose=verbose)
    
    # Convert to protein IDs if requested
    if use_protein_ids:
        haplotypes = _as_protein_ids(haplotypes)

    # Return
    return haplotypes

def _as_protein_ids(haplotypes: Dict[str, Dict]) -> Dict[str, Dict]:
    return dict(zip(get_haplotype_protein_ids(haplotypes, add_self=True).values(), haplotypes.values()))

def add_missing_ref_seqs(haplotypes: Dict[str, Dict],
                         keys: List[str] = ['protein_haplotypes', 'cds_haplotypes'],
                         verbose: bool = False) -> Dict[str, Dict]:
    """
    Add the seqeunce data for missing reference haplotypes.

    Args:
        haplotypes (Dict[str, Dict]): The haplotype information.
        verbose (bool, optional): Whether to print progress messages. Defaults to False.

    Returns:
        Dict[str, Dict]: The haplotype information with reference sequences added.
    """

    # Iterate over transcripts
    for tx_id in tqdm(haplotypes.keys(),
                      desc="Adding reference haplotype",
                      leave=False):
        # Iterate over keys
        for key in keys:
            hap_names = get_haplotype_names(haplotypes[tx_id], key=key)
            template_entry = haplotypes[tx_id][key][0]
            if not any(["REF" in x for x in hap_names]):
                if verbose:
                    print(f"Adding reference haplotype for {tx_id}")
                enst_id = template_entry['name'].split(':')[0]
                # By definition, the ref sequence has no gaps when aligned to itself
                ref_seq = template_entry['aligned_sequences'][0].replace('-', '')
                ref_entry = {
                    'name':enst_id+':REF',
                    'seq':ref_seq,
                    'aligned_sequences':[ref_seq]*2
                    }
                haplotypes[tx_id][key] = [ref_entry] + haplotypes[tx_id][key]

    return haplotypes
    

def get_haplotype_seqs(haplotypes: Union[Dict[str, Dict], Dict[str, List[Dict]]],
                       key: str = 'protein_haplotypes',
                       aligned: int = 1,
                       as_msa: bool = False,
                       return_missing: bool = False,
                       use_protein_ids: bool = False,
                       add_haplotype_names: bool = False,
                       add_missing_ref: bool = True,
                       add_consensus: bool = False,
                       encode_haplotype_name_threshold: int = None,
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
        add_missing_ref (bool, optional): Whether to automatically add in the reference sequence when it is missing from the data.
            Defaults to True.
        verbose (bool, optional): Whether to print progress messages. Defaults to False.

    Returns:
        Dict[str, List[str]]: A dictionary mapping transcript IDs to their haplotype sequences.
    """

    
    if haplotypes is None:
        haplotypes = get_haplotypes(cache_only=True, 
                                    verbose=verbose)
    
    hap_seqs = {}
    missing_seqs = []
    if aligned!=2 and as_msa:
        warnings.warn('Aligned must be 2 to convert to MSA (`as_msa=True`)')
        as_msa = False

    if add_missing_ref:
        haplotypes = haplotypes.copy()
        haplotypes =  add_missing_ref_seqs(haplotypes, verbose=verbose) 

    # Iterate over transcripts
    for tx_id in tqdm(haplotypes.keys(),
                      desc="Getting haplotype sequences",
                      leave=False):
        # Initialize haplotype sequences entry
        hap_seqs[tx_id] = {}
         
        # Get haplotype sequences when haplotypes is a dict
        if isinstance(haplotypes[tx_id], dict) and key in haplotypes[tx_id].keys(): 

            # Add haplotype sequences
            if aligned==1:
                hap_seqs[tx_id] = [x['aligned_sequences'][1] for x in haplotypes[tx_id][key]]
            elif aligned==2:
                hap_seqs[tx_id] = [x['aligned_sequences'] for x in haplotypes[tx_id][key]]
            elif aligned==0:
                hap_seqs[tx_id] = [x['seq'] for x in haplotypes[tx_id][key]]
            else:
                raise ValueError(f"Invalid alignment type: {aligned}")
            
            # Add consensus sequence
            if add_consensus:
                alignment_paths = au.align_haplotypes(tx_ids=[tx_id],
                                                      key=key,
                                                      error=False)
                (consensus_seqs, 
                 consensus_in_cohort) = au.get_consensus_sequences(
                    alignment_paths=alignment_paths
                    )
                if tx_id in consensus_seqs.keys():
                    hap_seqs[tx_id] = hap_seqs[tx_id] + [str(consensus_seqs[tx_id])]
            
        # Get haplotype sequences when haplotypes is a list 
        else:
    
            # Add haplotype sequences
            if len(haplotypes[tx_id])>0:
                if aligned==1:
                    # Return aligned haplotype sequence only
                    hap_seqs[tx_id] = [haplotypes[tx_id][0]['aligned_sequences'][1]] 
                elif aligned==2:
                    # Return both sequences (aligned reference and haplotype)
                    hap_seqs[tx_id] = [haplotypes[tx_id][0]['aligned_sequences']]
                else:
                    # Return unaligned haplotype sequence only
                    hap_seqs[tx_id] = [haplotypes[tx_id][0]['seq']]
            else:
                if verbose:
                    warnings.warn(f"No seqs found for {tx_id}")
                missing_seqs = [tx_id]
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
    
    if encode_haplotype_name_threshold is not None:
        if verbose:
            print(f"Encoding haplotype names with threshold {encode_haplotype_name_threshold}")
        hap_seqs = {k:utils.encode_haplotype_name(v, 
                                                  encode_haplotype_name_threshold=encode_haplotype_name_threshold) 
                    for k,v in hap_seqs.items()}

    # Return 
    if return_missing:
        return hap_seqs, missing_seqs
    else:
        return hap_seqs
    
def haplotypes_to_df(haplotypes: Optional[Dict[str, Dict]] = None,
                     preprocess: bool = True,
                     max_tx_ids: Optional[int] = None,
                     add_consensus: bool = False,
                     add_missing_ref: bool = True,
                     key: str = 'protein_haplotypes',
                     verbose: bool = False) -> pd.DataFrame:
    """Convert haplotype sequences to a dataframe.
    
    Args:
        hap_seqs (Dict[str, List[str]]): A dictionary mapping transcript IDs to their haplotype sequences.
        add_tx_id (bool, optional): Whether to add the transcript ID. Defaults to True.
        add_ensp (bool, optional): Whether to add the ENSP. Defaults to True.
        add_ensg (bool, optional): Whether to add the ENSG. Defaults to True.
        add_gene_symbol (bool, optional): Whether to add the gene symbol. Defaults to True.

    Returns:
        pd.DataFrame: A dataframe of the haplotype sequences.
    """

    # Get haplotypes
    if haplotypes is None:
        haplotypes = get_haplotypes(max_tx_ids=max_tx_ids, 
                                    leave=False,
                                    verbose=verbose>1)

    # Get haplotype sequences
    hap_seqs = get_haplotype_seqs(haplotypes, 
                                 aligned=0,
                                 add_haplotype_names=2,
                                 key=key,
                                 add_consensus=add_consensus,
                                 add_missing_ref=add_missing_ref,
                                 verbose=verbose>1)
    # Initialize dataframe
    df = pd.DataFrame({'ENST':[],
                       'ENSP':[],
                       'sequence':[]
                   })
    
    # Add sequences to dataframe
    # Create a list of records to build dataframe at once instead of concatenating repeatedly
    records = []
    for tx_id, seqs in tqdm(hap_seqs.items(),
                            desc="Converting haplotypes to dataframe",
                            leave=False, 
                            disable=verbose<0):
        for hap_name, seq in seqs.items():
            records.append({'haplotype': hap_name, 'sequence': seq, 'ENST': tx_id})
    
    # Create dataframe from records in one operation
    df = pd.DataFrame.from_records(records, index='haplotype')
    
    # Extract ENSP if needed (only once after dataframe is built)
    if key=='protein_haplotypes':
        df['ENSP'] = df.index.str.split(':').str[0]
    
    # Preprocess sequences
    if preprocess:
        df['sequence'] = df['sequence'].apply(bp.preprocess_sequence)
    
    return df

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


def pop_freqs_to_df(pop_freqs: Dict[str, Dict],
                    tx_id_col: str = 'ENST',
                    prefix: str = 'freq_', 
                    add_top_pop: bool = True, 
                    add_top_superpop: bool = True, 
                    add_specific_superpop: bool = True,
                    verbose: bool = True,
                    force: bool = False,
                    ) -> pd.DataFrame:
    """Convert population frequencies to a dataframe.
    """
    # Flatten the pop_freqs dictionary by adding tx_id to each haplotype's frequency data
    flattened_pop_freqs = {}
    for tx_id, haplotypes in pop_freqs.items():
        for haplotype_name, frequencies in haplotypes.items():
            # Create a copy of the frequencies dictionary and add tx_id
            haplotype_data = frequencies.copy()
            haplotype_data[tx_id_col] = tx_id
            
            # Store in the flattened dictionary with haplotype name as key
            flattened_pop_freqs[haplotype_name] = haplotype_data

    df = pd.DataFrame(flattened_pop_freqs).T
    # Move tx_id to the first column
    df = df.reset_index(drop=False)
    df = df.rename(columns={'index': 'haplotype'})
    df = df.loc[:, [tx_id_col, 'haplotype'] + [col for col in df.columns if col != tx_id_col and col != 'haplotype']]
    # Remname each column with the prefix
    df.rename(columns={col: f"{prefix}{col}".replace("__", "_") for col in df.columns if col != tx_id_col and col != 'haplotype'}, inplace=True)
    
    freq_cols = _get_freq_cols(df=df,
                              verbose=verbose)
    # Add top population
    if add_top_pop:
        _add_top_pop(df=df,
                          freq_cols=freq_cols,
                          verbose=verbose,
                          force=force)
    
    # Add top superpopulation
    if add_top_superpop:
        _add_top_superpop(df=df,
                         verbose=verbose,
                         force=force)
    
    # Add specific superpopulation
    if add_specific_superpop:
        _add_specific_superpops(df, verbose=verbose)
    
    # Return dataframe
    return df   


def get_haplotype_freqs(haplotypes: Optional[Dict[str, Dict]] = None,
                        key: str = 'protein_haplotypes',
                        tx_ids: Optional[List[str]] = None,
                        return_df: bool = False,
                        prefix: str = 'freq_',
                        tx_id_col: str = 'ENST',
                        verbose: bool = False
                        ) -> Union[Tuple[Dict[str, Dict], Set[str], Set[str]], pd.DataFrame]:
    pop_freqs = {}
    populations = set()
    tx_ids = utils.as_list(tx_ids)

    if haplotypes is None:
        haplotypes = get_haplotypes(tx_ids=tx_ids)

    for tx_id in tqdm(haplotypes.keys(),
                      desc="Getting haplotype frequencies",
                      leave=False):
        if tx_ids is not None:
            if tx_id not in tx_ids:
                if verbose:
                    print(f"Skipping {tx_id} because it is not in tx_ids")
                continue
                
        pop_freqs[tx_id] = {x['name']:x['population_frequencies'] for x in haplotypes[tx_id][key] if 'population_frequencies' in x.keys()}
        for hap_name in pop_freqs[tx_id]:
            populations.update(pop_freqs[tx_id][hap_name].keys())
    cohorts = set([':'.join(p.split(':')[:-1]) for p in populations])
    
    # Return dataframe if requested
    if return_df:
        return pop_freqs_to_df(pop_freqs, tx_id_col=tx_id_col, prefix=prefix)
    else:
        return pop_freqs, populations, cohorts


def _add_specific_superpops(df: pd.DataFrame, 
                           verbose: bool = True):
    """Add a column indicating which superpopulation is specific to a variant.
    
    Args:
        df (pd.DataFrame): The dataframe to add the specific superpopulation to.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        pd.DataFrame: The dataframe with the specific superpopulation added.
    """ 
    if 'specific_superpopulation' in df.columns:
        if verbose:
            print("Specific superpopulation column already exists")
        return df

    if 'top_superpop' not in df.columns:
        raise ValueError("top_superpop column not found in dataframe")
    
    # Add a new boolean col whether the haplotypes is specific to a population (all other populations are 0 or)
    superpops = og.get_sample_metadata()["Super Population"].unique()
    freq_cols = [col for col in df.columns if col.startswith('freq_') and col.endswith(tuple(superpops))]

    if len(freq_cols)==0:
        raise ValueError("No frequency columns found in dataframe")

    # Add a column indicating how many frequency columns are non-zero and non-nan for each row
    df['nonzero_superpop_freqs'] = df[freq_cols].apply(
        lambda row: (row.notna() & (row > 0)).sum(), axis=1
    )
    # For rows where only one frequency column is non-zero, identify which population it is
    def get_nonzero_population(row):
        if row['nonzero_superpop_freqs'] == 1:
            # Find the single non-zero frequency column
            for col in freq_cols:
                if pd.notna(row[col]) and row[col] > 0:
                    # Extract the population name from the column name (after 'freq_')
                    return col.replace('freq_', '')
        return None

    # Add a column indicating which population has the non-zero frequency (for population-specific variants)
    df['specific_superpopulation'] = df.apply(get_nonzero_population, axis=1)

    # Count how many variants are specific to each population
    if verbose:
        print("\nNumber of variants specific to each population:")
        print(df[df['nonzero_superpop_freqs'] == 1]['specific_superpopulation'].value_counts())
        # Calculate the percentage of variants that are population-specific
        print(f"\nPercentage of population-specific haplotypes: {df.loc[df['nonzero_superpop_freqs'] == 1]['haplotype'].nunique()/df['haplotype'].nunique():.2%}")

    # return df

def _get_freq_cols(df: Optional[pd.DataFrame] = None,
                   populations: Optional[List[str]] = None,
                   prefix: str = 'freq_',
                   verbose: bool = True):
    """Get the frequency columns from a dataframe.
    """
    if populations is not None:
        freq_cols = [f'{prefix}{pop}' for pop in populations]
    else:
        freq_cols = [col for col in df.columns if col.startswith(prefix)]
    return freq_cols

def _add_top_pop(df: pd.DataFrame,
                 freq_cols: List[str],
                 verbose: bool = True, 
                 force: bool = False):
    """
    Add the top population to a dataframe.

    Args:
        df (pd.DataFrame): The dataframe to add the top population to.
        freq_cols (List[str]): The frequency columns to use.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        force (bool, optional): Whether to force the addition of the top population. Defaults to False.

    Returns:
        pd.DataFrame: The dataframe with the top population added.
    """
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


def _add_top_superpop(df: pd.DataFrame,
                     verbose: bool = True,
                     force: bool = False):
    """
    Add the top superpopulation to a dataframe.
    """
    if 'top_pop' in df.columns:
        if 'top_superpop' not in df.columns or force:
            if verbose:
                print("Adding top superpopulation")
            ### Approach 1:
            # This approach does not consider the aggregate of freqs across populations,
            # but only the max freq in each population.
            pops = og.get_sample_metadata()
            superpops = pops["Super Population"].unique()
            # pop_map = dict(zip(pops['Population Code'], pops['Super Population']))
            # df.loc[:, 'top_superpop'] = df['top_pop'].str.replace(f'{cohorts[0]}:', '').str.split('_').str[0].map(pop_map)
            # df['top_superpop'].fillna('N/A', inplace=True) 
            
            ### Approach 2:
            # This approach considers the aggregate of freqs across populations,
            # as long as the superpop is present as a column.
            superpop_cols = [col for col in df.columns if any([col.endswith(sp) for sp in superpops])]
            has_freqs = ~df[superpop_cols].isna().all(axis=1)
            df.loc[has_freqs, 'top_superpop']  = df.loc[has_freqs, superpop_cols].idxmax(axis=1)

            # Extract the frequency value from the top_pop column for each row
            df.loc[has_freqs,'top_superpop_freq'] = df.loc[has_freqs,:].apply(lambda row: row[row['top_superpop']], axis=1)

def add_haplotype_freqs(df: pd.DataFrame, 
                        haplotypes: Optional[Dict[str, Dict]] = None, 
                        cohorts: Optional[List[str]] = ['1000GENOMES:phase_3'],
                        haplotype_col: str = "haplotype",
                        protein_id_col: str = "ENSP",
                        tx_id_col: str = "ENST",
                        add_top_pop: bool = True,
                        add_top_superpop: bool = True,
                        add_specific_superpop: bool = True,
                        force: bool = False,
                        verbose: bool = True,
                        leave: bool = False): 
    """Add haplotype frequencies to a dataframe.

    Args:
        df (pd.DataFrame): The dataframe to add the haplotype frequencies to.
        haplotypes (Optional[Dict[str, Dict]], optional): The haplotype information. Defaults to None.
        cohorts (Optional[List[str]], optional): The cohorts to use. Defaults to ['1000GENOMES:phase_3'].
        haplotype_col (str, optional): The column name of the haplotype. Defaults to "haplotype".
        protein_id_col (str, optional): The column name of the protein ID. Defaults to "ENSP".
        tx_id_col (str, optional): The column name of the transcript ID. Defaults to "ENST".
        add_top_pop (bool, optional): Whether to add the top population. Defaults to True.
        add_top_superpop (bool, optional): Whether to add the top superpopulation. Defaults to True.
        add_specific_superpop (bool, optional): Whether to add the specific superpopulation. Defaults to True.
        force (bool, optional): Whether to force the addition of the haplotype frequencies. Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        leave (bool, optional): Whether to leave the haplotype frequencies in the dataframe. Defaults to False.

    Returns:
        pd.DataFrame: The dataframe with the haplotype frequencies added.
    """
    df = df.copy()
    df = add_txid(df=df, 
                    haplotypes=haplotypes, 
                    protein_id_col=protein_id_col,
                    tx_id_col=tx_id_col,
                    force=force,
                    verbose=verbose)
    if haplotypes is None:
        haplotypes = get_haplotypes(tx_ids=df[tx_id_col].unique().tolist(), 
                                    leave=leave,
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
    freq_cols = _get_freq_cols(df=df,
                               populations=populations,
                               verbose=verbose)
    if not all(col in df.columns for col in freq_cols) or force:  
        if verbose:
            print("Adding haplotype frequencies per population")
        # Drop any existing frequency columns
        df.drop(columns=freq_cols+['top_pop', 'top_superpop','top_superpop_freq', 'nonzero_superpop_freqs','specific_superpopulation'], 
                errors='ignore', inplace=True)
        ## New method: faster
        pop_freqs_df = pop_freqs_to_df(pop_freqs, tx_id_col=tx_id_col, 
                                      add_top_pop=add_top_pop, 
                                      add_top_superpop=add_top_superpop, 
                                      add_specific_superpop=add_specific_superpop,
                                      verbose=verbose,
                                      force=force)
        df = df.merge(pop_freqs_df, on=['haplotype', tx_id_col], how='left')
        ## Old method: slower
        # for i,pop in tqdm(enumerate(populations),
        #             desc="Adding haplotype frequencies per population",
        #             total=len(populations),
        #             leave=leave):
        #     df.loc[:,freq_cols[i]] = df.apply(
        #         lambda row: pop_freqs[row[tx_id_col]][row[haplotype_col]][pop] 
        #             if row[tx_id_col] in pop_freqs 
        #             and row[haplotype_col] in pop_freqs[row[tx_id_col]] 
        #             and pop in pop_freqs[row[tx_id_col]][row[haplotype_col]] 
        #             else None,
        #         axis=1
        #     )

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
                        key: str = 'protein_haplotypes',
                        as_df: bool = False,
                        split_variants: bool = False) -> Dict[str, List[str]]:
    """
    Get the haplotype name from the haplotype entry,
        or a dict of haplotype entries indexed by transcript ID.

    Args:
        haplotypes (Dict[str, Dict]): The haplotypes to get the names from.
        key (str, optional): The key to use to get the names from. Defaults to 'protein_haplotypes'.
        as_df (bool, optional): Whether to return a dataframe. Defaults to False.
        split_variants (bool, optional): Whether to split the variants into separate rows. Defaults to False.

    Returns:
        Dict[str, List[str]]: The haplotype names.
        pd.DataFrame: The haplotype names as a dataframe if as_df is True.
    """
    if key in haplotypes.keys():
        return [x['name'].split('_')[0] for x in haplotypes[key]]
    hap_names = {}
    for tx_id in tqdm(haplotypes.keys(), 
                      desc="Getting haplotype names",
                      leave=False):
        hap_names[tx_id] = [x['name'].split('_')[0] for x in haplotypes[tx_id][key]]

    if as_df:
        df = pd.DataFrame([(tx_id, hap) for tx_id, haps in hap_names.items() for hap in haps], 
                            columns=['ENST', 'haplotype'])
        if split_variants:
            df.loc[:,["variant"]] = df['haplotype'].str.split(':').str[1].str.split(",")
            df = df.explode("variant")
            
        return df
    
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
                      key: str = 'protein_haplotypes',
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
    elif key in haplotypes.keys():
        hap_ref = [x for x in haplotypes[key] if ":REF" in tqdm(x['name'], leave=False)]
    elif isinstance(haplotypes, dict):
        hap_ref = {}
        for tx_id in tqdm(haplotypes.keys(),
                          desc="Processing transcripts",
                          disable=not verbose,
                          leave=False):
            hap_ref[tx_id] = [x for x in haplotypes[tx_id][key] if ":REF" in x['name']]
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
    
def get_txid_map(haplotypes: Optional[Dict[str, Dict]] = None,
                 to: str = "protein_id",
                 invert: bool = False,
                 check_all: bool = False,
                 as_df: bool = False,
                 cache: Path = DIR_DICT["tx_id_map"], 
                 force: bool = False) -> Dict[str, str]:
    """
    Get a dictionary of tx_ids to protein IDs or protein IDs to tx_ids.

    Args:
        haplotypes (Dict[str, Dict]): The haplotypes to get the tx_id map for.
        to (str, optional): The type of ID to get. Defaults to "protein_id".
        invert (bool, optional): Whether to invert the dictionary. Defaults to False.
        check_all (bool, optional): Whether to check all protein IDs. Defaults to False.
        as_df (bool, optional): Whether to return a dataframe. Defaults to False.
        cache (Path, optional): The cache to save the tx_id map to. Defaults to Path(DIR_DICT["haplotypes"]).
        force (bool, optional): Whether to force the tx_id map to be recalculated. Defaults to False.
    """
    if os.path.exists(cache) and not force:
            # load from cache
        tx_id_map = utils.load_pickle(cache)
        if invert:
            tx_id_map = utils.invert_dict(tx_id_map)
        return tx_id_map 
    
    if haplotypes is None:
        raise ValueError("Haplotypes are required to get the tx_id map is not cached")

    if to == "protein_id":
        if check_all:
            if invert:
                raise ValueError("Cannot invert dict if check_all is True")
            tx_id_map  = {tx_id:list(set([x['name'].split(':')[0] for x in hap_tx['protein_haplotypes']])) for tx_id, hap_tx in haplotypes.items()}
            
            # Save to cache
            if cache is not None:
                utils.save_pickle(tx_id_map, cache)
        else:
            tx_id_map = {tx_id:hap_tx['protein_haplotypes'][0]['name'].split(':')[0] for tx_id, hap_tx in haplotypes.items()}
            
            # Save to cache
            if cache is not None:
                utils.save_pickle(tx_id_map, cache)

        # Invert the dictionary if requested
        if invert:
            tx_id_map  = utils.invert_dict(tx_id_map)

        # Return as a dataframe if requested
        if as_df:
            return pd.DataFrame(tx_id_map, index=['id']).T
        else:
            return tx_id_map
    else:
        raise ValueError("to must be 'protein_id'")
    
def add_txid(df: pd.DataFrame,
             haplotypes: Optional[Dict[str, Dict]] = None,
             protein_id_col: str = "ENSP",
             tx_id_col: str = "ENST",
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

def haplotypes_to_fasta(haplotypes=None, 
                        tx_ids=None,
                        key='protein_haplotypes',
                        aligned=1,
                        add_haplotype_names=2,
                        add_missing_ref=True,
                        add_consensus=False,
                        save_dir=os.path.join(config.DATA_DIR,"1KG","fasta"),
                        strip='*',
                        merge=False,
                        force=False,
                        compress=True,
                        verbose=True):
    """Convert haplotype sequences to FASTA format.
    
    Args:
        haplotypes (Dict[str, Dict]): The haplotypes to convert to FASTA format.
        key (str, optional): The key to use to get the haplotype sequences. Defaults to "protein_haplotypes".
        aligned (int, optional): The alignment type. Defaults to 1.
        add_haplotype_names (int, optional): The type of haplotype names to add. Defaults to 2.
        add_missing_ref (bool, optional): Whether to automatically add in the reference sequence when it is missing from the data.
            Defaults to True.
        add_consensus (bool, optional): Whether to add the consensus sequence to the FASTA file. Defaults to False.
        save_dir (str, optional): The directory to save the FASTA files. 
            Defaults to os.path.join(config.DATA_DIR,"1KG","fasta").
        force (bool, optional): Whether to force the conversion. Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
    
    Returns:
        Dict[str, str]: A dictionary of tx_ids to FASTA file paths.
    """ 

    if tx_ids is not None:
        tx_ids = utils.as_list(tx_ids)

    # Convert haplotype sequences to merged FASTA format
    if merge: 
        os.makedirs(save_dir, exist_ok=True)

        # Define the path to the merged FASTA file
        fasta_path = os.path.join(save_dir, f"{key}.fasta")
        if compress:
            fasta_path = f"{fasta_path}.gz"
        if os.path.exists(fasta_path) and not force:
            if verbose:
                print(f"Returning merged FASTA file: {fasta_path}")
            return fasta_path

        # Get haplotypes if not provided
        if haplotypes is None:
            haplotypes = get_haplotypes(tx_ids=tx_ids,
                                        cache_only=True, 
                                        verbose=verbose)
            
        # Get haplotype sequences
        hap_seqs = get_haplotype_seqs(haplotypes, 
                                      key=key,
                                      aligned=aligned,
                                      add_haplotype_names=add_haplotype_names,
                                      add_missing_ref=add_missing_ref,
                                      add_consensus=add_consensus,
                                      verbose=verbose>1)
            
        # Check if the merged FASTA file exists
        if not os.path.exists(fasta_path) or force:

            # Open file with gzip if needed
            if compress:
                import gzip
                f = gzip.open(fasta_path, 'wt')
            else:
                f = open(fasta_path, 'w')
                
            # Write the merged FASTA file
            try:
                for tx_id in tqdm(hap_seqs.keys(),
                            desc="Converting haplotypes to merged FASTA",
                            leave=False,
                            disable=not verbose):
                    for name, seq in hap_seqs[tx_id].items():
                        if strip is not None:
                            seq = seq.replace(strip, '')
                        f.write(f'>{tx_id}\t{name}\n{seq}\n')
            finally:
                f.close()
        return fasta_path

    # Convert haplotype sequences to split FASTA format
    else:
        os.makedirs(os.path.join(save_dir, "split"), exist_ok=True)
        fasta_paths = {}
        if haplotypes is None:
            tx_ids = utils.intersect(tx_ids, 
                                     list_haplotypes(verbose=verbose)) 
        else:
            tx_ids = utils.intersect(tx_ids, 
                                     list(haplotypes.keys()))

        for tx_id in tqdm(tx_ids,
                            desc="Converting haplotypes to split FASTAs",
                            leave=False,
                            disable=not verbose):

            # Define the path to the split FASTA file
            fasta_path = os.path.join(save_dir, "split", f"{tx_id}.fasta")
            if compress:
                fasta_path = f"{fasta_path}.gz"
                
            fasta_paths[tx_id] = fasta_path

            # Check if the split FASTA file exists
            if not os.path.exists(fasta_path) or force:

                # Get haplotypes if not provided
                if haplotypes is None:
                    haplotypes_tx = get_haplotypes(tx_ids=[tx_id],
                                                   cache_only=True,
                                                   leave=False,
                                                   verbose=verbose>1)
                else:
                    haplotypes_tx = haplotypes
                    
                # Get haplotype sequences
                hap_seqs = get_haplotype_seqs(haplotypes_tx, 
                                              key=key,
                                              aligned=aligned,
                                              add_haplotype_names=add_haplotype_names,
                                              add_missing_ref=add_missing_ref,
                                              add_consensus=add_consensus,
                                              verbose=verbose>1)
                

                # Open file with gzip if needed
                if compress:
                    import gzip
                    f = gzip.open(fasta_path, 'wt')
                else:
                    f = open(fasta_path, 'w')
                    
                # Write the split FASTA file
                try:
                    for name, seq in hap_seqs[tx_id].items():
                        if strip is not None:
                            seq = seq.replace(strip, '')
                        f.write(f'>{name}\n{seq}\n')
                finally:
                    f.close()
        return fasta_paths


def get_haplotype_samples(haplotypes=None, 
                          tx_ids=None,
                          max_tx_ids=None,
                          cohort=None,
                          unnest=False,
                          remove_prefix=False,
                          key='cds_haplotypes'):
    """
    Extract all unique sample IDs from a haplotype dictionary.
    
    Args:
        haplotypes: Dictionary containing haplotype information
        key: The key in the haplotype dictionary containing sample information (default: 'cds_haplotypes')
        max_tx_ids: The maximum number of transcript IDs to process
        cohort: The cohort to filter samples by
        unnest: Whether to unnest the sample IDs
        remove_prefix: Whether to remove the prefix from the sample IDs
        key: The key in the haplotype dictionary containing sample information (default: 'cds_haplotypes')

    Returns:
        Set of unique sample IDs
    """
    if haplotypes is None:
        haplotypes = get_haplotypes(cache_only=True,
                                    use_protein_ids=key=="protein_haplotypes",
                                     max_tx_ids=max_tx_ids)
    if tx_ids is not None:
        tx_ids = utils.intersect(tx_ids, 
                                list(haplotypes.keys()))
    
    all_sample_ids = {}
    for tx_id in tqdm(haplotypes.keys(), 
                      desc="Extracting sample IDs",
                      leave=False):
        sample_ids = set()
        if key in haplotypes[tx_id]:
            for haplotype in haplotypes[tx_id][key]:
                if 'samples' in haplotype:
                    # Add all sample IDs from this haplotype
                    sample_ids_tmp = haplotype['samples'].keys()
                    if cohort is not None:
                        sample_ids_tmp = [x for x in sample_ids_tmp if x.startswith(cohort)]
                    if remove_prefix:
                        sample_ids.update(
                            {sample_id.split(":")[-1] 
                             for sample_id in sample_ids_tmp})
                    else:
                        sample_ids.update(haplotype['samples'].keys())
            all_sample_ids[tx_id] = sample_ids

    if unnest:
        return list(set.union(*all_sample_ids.values()))
    else:
        return all_sample_ids


def haplotypes_to_samples(haplotypes=None, 
                          tx_ids=None, 
                          max_tx_ids=None,
                          samples=None, 
                          return_seqs=True,
                          cohort=None, 
                          key='protein_haplotypes',
                          as_df=False,
                          add_sample_metadata=False,
                          add_ref=True, 
                          duplicate_ref=False,
                          verbose=False):
    """
    Process transcript haplotypes and organize sequences by sample.
    
    Args:
        tx_ids: List of transcript IDs to process
        haplotypes: Dictionary containing haplotype information
        ds: Dataset containing sample information
        cohort: Cohort identifier string to prepend to sample IDs
        verbose: Whether to print detailed processing information
        as_df: Whether to return a DataFrame instead of a dictionary
        add_sample_metadata: Whether to add sample metadata (from 1000 Genomes Project) to the DataFrame.
            Only works if as_df is True.
        duplicate_ref: Whether to duplicate the REF haplotype to simulate a diploid genome.
            Only works if as_df is True.
    Returns:
        Dictionary mapping transcript IDs to sample sequences
    """
    if haplotypes is None:
        haplotypes = get_haplotypes(cache_only=True,
                                    tx_ids=tx_ids,
                                    use_protein_ids=key=="protein_haplotypes",
                                     max_tx_ids=max_tx_ids)
    if tx_ids is None:
        tx_ids = list(haplotypes.keys())

    if key=="protein_haplotypes" and tx_ids[0].startswith("ENSP"):
        txid_map = get_txid_map(haplotypes=haplotypes, invert=True)
        tx_ids = [txid_map[tx_id] for tx_id in tx_ids]
        
    if samples is None:
        samples = get_haplotype_samples(haplotypes=haplotypes,  
                                        max_tx_ids=max_tx_ids,
                                        cohort=cohort,
                                        unnest=True, 
                                        # 'GGVP:HG02757' and '1000GENOMES:phase_3:HG02757' are both possible sample IDs
                                        remove_prefix=False,
                                        key=key)

    if verbose:
        print(f"Number of samples: {len(samples)}")
        print(f"Number of transcripts to process: {len(tx_ids)}")
    
    tx_sample_seqs = {}
    # Iterate over all transcript IDs
    for tx_id in tqdm(tx_ids, 
                      desc=f"Processing sample to haplotype {'sequence' if return_seqs else 'name'} maps", 
                      leave=False):
        sample_seqs = {sample: [] for sample in samples}
        
        # Iterate over all haplotype indices for this transcript
        for hap_idx in range(len(haplotypes[tx_id][key])):
            
            if 'samples' not in haplotypes[tx_id][key][hap_idx]:
                if verbose>1:
                    print(f"'samples' misssing from haplotypes for '{tx_id}'")
                continue
            
            sample_map = haplotypes[tx_id][key][hap_idx]['samples']
            # Split the keys in sample_map by ":" and only keep the last item
            # e.g. "1000Genomes:phase3:HG03235" --> "HG03235"
            # if cohort is None:
            #     sample_map = {k.split(":")[-1]: v for k, v in sample_map.items()}
            seq = haplotypes[tx_id][key][hap_idx]['seq']
            hap_name = haplotypes[tx_id][key][hap_idx]['name']
            
            if verbose>1:
                print(f"Haplotype index {hap_idx}, sequence length: {len(seq)}")
            
            for sample in samples:
                sample_id = sample
                if cohort is not None:
                    sample_id = cohort+":"+sample
                if sample_id in sample_map.keys():

                    # Get the number of haplotypes with this sequence for this sample
                    sample_count = sample_map[sample_id]

                    # Return sequences
                    if return_seqs:
                        sample_seqs[sample] += [seq]*sample_count
                    # Return haplotype names 
                    else:
                        sample_seqs[sample] += [hap_name]*sample_count
                        
                    # Reduce verbosity to avoid excessive output
                    if verbose>1:
                        if hap_idx == 0 and tx_id == tx_ids[0]:
                            print(f"  Sample {sample}: added {sample_count} sequences from haplotype {hap_idx}")
                elif hap_idx == 0 and tx_id == tx_ids[0]:  # Only print this message once per sample for the first transcript
                    if verbose>1:
                        print(f"  Sample {sample}: not found in sample_map")
        
        # Update the tx_sample_seqs dictionary
        tx_sample_seqs[tx_id] = sample_seqs
    
    # Return a DataFrame if requested
    if as_df:  

        # Convert haplotypes to DataFrame
        if verbose:
            print("Converting haplotypes to DataFrame")
        df = pd.concat(
            {k: pd.Series(v) for k, v in tx_sample_seqs.items()},
            names=["ENST_haplosaurus", "cohort_sample"]
        ).reset_index(name="haplotype")
        
        # Parse cohort and sample information
        df['cohort'] = df['cohort_sample'].str.split(":").str[0]
        df.loc[df['cohort'].str.startswith("HGDP"), "cohort"] = "HGDP"
        df['sample'] = df['cohort_sample'].str.split(":").str[-1]
        
        # Add ploid information
        df['ploid_count'] = df['haplotype'].str.len()
        df['ploid'] = df['ploid_count'].apply(lambda x: list(range(x)))
        df = df.explode(["haplotype", "ploid"], ignore_index=True)

        # Add sample metadata
        if add_sample_metadata:
            if verbose:
                print("Adding sample metadata to the DataFrame")
            sample_metadata = og.get_sample_metadata(harmonized=True)
            sample_metadata.drop_duplicates(subset=["sample"], inplace=True)
            if verbose:
                print("Samples before merging:", df['sample'].nunique())
            df = df.merge(sample_metadata, 
                          on=["sample"], 
                          how="left") 
            if verbose:
                print("Samples after merging:", df['sample'].nunique())

            # Add REF haplotypes to the dataframe
            if add_ref:
                if verbose:
                    print("Adding REF haplotypes to the DataFrame")
                haps_to_ref = df.loc[df['haplotype'].str.endswith(":REF")].copy()
                haps_to_ref.loc[:,["sample","population","population_name","superpopulation","superpopulation_name"]] = "REF"
                haps_to_ref.loc[:,["sex"]] = pd.Int64Dtype().na_value
                haps_to_ref.loc[:,["ploid_count"]] = 2
                haps_to_ref = haps_to_ref.drop_duplicates()

                # Simulate a diploid genome by duplicating the REF haplotype
                if duplicate_ref: 
                    df = pd.concat([haps_to_ref.assign(ploid=0), 
                                    haps_to_ref.copy().assign(ploid=1), 
                                    df.loc[df['sample']!="REF"].copy()
                                    ])
                else: 
                    df = pd.concat([haps_to_ref.assign(ploid=0), 
                                    df.loc[df['sample']!="REF"].copy()
                                    ])
                    
                if verbose:
                    print("Samples after adding REF haplotypes:", df['sample'].nunique())
        
        # Ensure there's not any duplicates
        df.drop_duplicates(subset=["sample","haplotype","ploid"],inplace=True)
                    
        if verbose:
            print(f"Final shape: {df.shape}")
            for col in ['haplotype','cohort','cohort_sample',
                        'sample','population',
                        'superpopulation','sex']:
                if verbose and col in df.columns:
                    print(f">> {col}(s): {df[col].nunique()}")

        return df
    else:
        return tx_sample_seqs



def count_haplotype_samples(tx_sample_seqs, 
                            verbose=False):
    """
    Count the number of sequences of each length for each transcript.
    
    Args:
        tx_seqs: Dictionary mapping transcript IDs to sample sequences
        verbose: Whether to print detailed per-transcript counts
        
    Returns:
        tuple: (tx_seq_counts, total_counts) where tx_seq_counts is a dictionary
               mapping transcript IDs to Counter objects of sequence lengths,
               and total_counts is a Counter of all sequence lengths
    """
    from collections import Counter
    # Count sequences per transcript
    tx_seq_counts = {}
    for tx_id, samples in tx_sample_seqs.items():
        # Count how many sequences of each length exist for this transcript
        length_counts = Counter([len(seqs) for seqs in samples.values()])
        tx_seq_counts[tx_id] = length_counts

    # Create a total tally across all transcripts
    total_counts = Counter()
    for counts in tx_seq_counts.values():
        total_counts.update(counts)
    
    print("Total sequence length counts across all transcripts:")
    print(total_counts)
    
    # Display per-transcript counts if verbose
    if verbose:
        print("\nPer-transcript sequence length counts:")
        for tx_id, counts in tx_seq_counts.items():
            print(f"{tx_id}: {dict(counts)}")
            
    return tx_seq_counts, total_counts



def dynamic_batching(hap_seqs, 
                     max_batch_size=1000000):
    
    """
    Convert dictionary to list of tuples (transcript_id, haplotype_id, sequence)
    [(id, seq) for id, seq in hap_seqs.items()]

    Batch sequences so that total amino acids in each batch doesn't exceed 1 million

    Args:
        hap_seqs: Dictionary mapping transcript IDs to haplotype sequences
        max_batch_size: Maximum number of amino acids in each batch

    Returns:
        List of batches, each containing a dictionary of transcript IDs to haplotype sequences

    Example:
        hap_seqs = {
            'ENST00000380152': {
                'haplotype_1': 'MSEQWENCE',
                'haplotype_2': 'MSEQWENCE'
            },
            'ENST00000380153': {
                'haplotype_1': 'MSEQWENCE',
                'haplotype_2': 'MSEQWENCE'
            }
        }
        dynamic_batching(hap_seqs)
    """
    batches = []
    current_batch = {}
    current_batch_size = 0
    
    for seq_id, seq in hap_seqs.items():
        seq = bp.preprocess_sequence(seq)
        seq_length = len(seq)
        
        # If adding this sequence would exceed the limit, start a new batch
        if current_batch_size + seq_length > max_batch_size and current_batch:
            batches.append(current_batch)
            current_batch = {}
            current_batch_size = 0
        
        # Add sequence to current batch
        current_batch[seq_id] = seq
        current_batch_size += seq_length
    
    # Add the last batch if it's not empty
    if current_batch:
        batches.append(current_batch)
    
    return batches

 
def haplosaurus_dataloader(tx_ids,
                           key='protein_haplotypes',
                           preprocess=True,
                           cache_only=True,
                           verbose=True
                          ):
    """
    Load haplotype data for specified transcript IDs into a DataFrame.
    
    This function provides a convenient way to retrieve haplotype data by chaining
    together multiple processing steps: fetching haplotypes, extracting sequences,
    and converting to a DataFrame format.
    
    Args:
        tx_ids (str or list): Transcript ID(s) to retrieve haplotypes for.
        key (str, optional): The haplotype key to use, typically 'protein_haplotypes'
            or 'nucleotide_haplotypes'. Defaults to 'protein_haplotypes'.
        preprocess (bool, optional): Whether to preprocess sequences in the DataFrame.
            Defaults to True.
        cache_only (bool, optional): Whether to only use cached data. Defaults to True.
        verbose (bool, optional): Whether to display progress information. Defaults to True.
    
    Returns:
        pandas.DataFrame: DataFrame containing haplotype information including sequences
            and associated metadata.
    """
    haplotypes = get_haplotypes(tx_ids=tx_ids, 
                                    leave=False,
                                    cache_only=cache_only,
                                    verbose=verbose)
    hap_df = haplotypes_to_df(haplotypes,
                                 preprocess=preprocess,
                                 key=key)
    return hap_df


def get_haplotype_diffs(haplotypes,
                        key='protein_haplotypes',
                        tx_ids=None,
                        skip_ref=True,
                        as_df=True,
                        verbose=True):
    """
    Extract diffs (variants present in the haplotype sequence) 
    from a haplotype dictionary and return a DataFrame.
    
    Args:
        haplotypes: Dictionary containing haplotype information
        key: The key in the haplotype dictionary containing the haplotype information
        tx_ids: List of transcript IDs to process
        skip_ref: Whether to skip reference haplotypes
        as_df: Whether to return a DataFrame
        verbose: Whether to print verbose output

    Returns:
        pandas.DataFrame: DataFrame containing haplotype diffs
    """

    # Create a list to store all rows
    rows = []
    if tx_ids is None:
        tx_ids = list(haplotypes.keys())
    

    for tx_id in tqdm(tx_ids, 
                    desc="Extracting haplotype diffs",
                    leave=False):
        for hap in haplotypes[tx_id][key]: 
            default_row_dict=  {'ENST': tx_id,
                                'haplotype': hap['name']} 
            if 'diffs' not in hap: 
                continue
            hap_diffs = hap['diffs']
            if len(hap_diffs) > 0:
                # For each diff in the list, create a separate row
                for diff in hap_diffs:
                    # Create a base dictionary with ENST and haplotype
                    row_dict = {
                        'ENST': tx_id,
                        'haplotype': hap['name']
                    }
                    row_dict.update(diff)
                    rows.append(row_dict)
            else: 
                if not skip_ref:
                    rows.append(default_row_dict)
                else:
                    continue

    if as_df:
        # Create dataframe directly from the list of dictionaries
        diff_df = pd.DataFrame.from_records(rows)

        if verbose:
            print(f"{len(diff_df)} diffs from {len(haplotypes)} {key} extracted")

    return diff_df


def add_haplotype_diffs(df,
                        haplotype_col='haplotype',
                        enst_col='ENST',
                        haplotypes=None,
                        hap_diffs=None,
                        key='protein_haplotypes',
                        prefix='diff_',
                        columns=['diff_polyphen_prediction','diff_sift_prediction'],
                        force=False,
                        verbose=True):
    """
    Add haplotype diffs to a DataFrame.
    
    Args:
        df: DataFrame to add haplotype diffs to
        haplotype_col: Column name containing haplotype information
        haplotypes: Dictionary containing haplotype information
        hap_diffs: DataFrame containing haplotype diffs
        key: The key in the haplotype dictionary containing the haplotype information
        prefix: Prefix for the haplotype diff columns
        force: Whether to force overwriting existing haplotype diff columns
        verbose: Whether to print verbose output

    Returns:
        DataFrame with haplotype diffs added
    """
    # Check for existing diff cols
    if any(df.columns.str.startswith(prefix)):
        if not force:
            return df
        else:
            if verbose:
                print(f"Overwriting existing haplotype diff column '{prefix}'")
            # Delete existing diff columns
            df = df.drop(columns=[c for c in df.columns if c.startswith(prefix) or c in ['diff']])
    
    if hap_diffs is None:
        if haplotypes is None:
            haplotypes = get_haplotypes(tx_ids=df[enst_col].unique(),
                                        key=key,
                                        verbose=verbose, 
                                        cache_only=True)
        hap_diffs = get_haplotype_diffs(haplotypes,
                                        key=key,
                                        verbose=verbose)
    # Make sure all columns start with the prefix
    hap_diffs.columns = [prefix+col if not col.startswith(prefix) and col not in ['ENST','haplotype','diff'] else col for col in hap_diffs.columns]
    
    # Pivot the hap_diffs DataFrame
    columns = ['diff_polyphen_prediction'] #'diff_polyphen_prediction'
    hap_diffs_pivot = hap_diffs.groupby(['haplotype',*columns])['diff'].count().reset_index().sort_values('diff', ascending=False).pivot(
        index='haplotype', 
        columns=columns, 
        values='diff'
        ).add_prefix(prefix).fillna(0)
    
    df = df.merge(hap_diffs_pivot,
                  left_on=haplotype_col,
                  right_on='haplotype',
                  how='left')
     # Fill NA for diff columns only
    df = df.fillna({col: 0 for col in df.columns if col.startswith(prefix)})

    return df
    

def split_haplosaurus_results(
    merged_json,
    save_dir=DIR_DICT["HGDP_haplotypes"],
    use_protein_ids: bool = False,
    return_haplotypes: bool = False,
    max_haplotypes: int = None,
    max_line_len: int = None,
    compress: bool = True,
    force: bool = False,
    verbose: bool = False
):
    """
    Split a merged Haplosaurus JSON file (one line per transcript) into individual JSON files per transcript.
    The expected input file is derived from running the [Haplosaurus tool within the VEP tool](https://github.com/Ensembl/ensembl-vep?tab=readme-ov-file#haplosaurus) 
    when using the `--json` flag.

    This function reads a gzipped JSON file where each line contains haplotype information for a single transcript.
    It saves each transcript's haplotype data as a separate gzipped JSON file in the specified directory.

    Args:
        merged_json (str or Path): Path to the gzipped merged JSON file containing haplotype data for multiple transcripts.
        save_dir (str or Path, optional): Directory to save the individual transcript JSON files. 
            Defaults to DIR_DICT["haplotypes_HGDP"].
        use_protein_ids (bool, optional): If True, use the "protein_haplotypes" key to extract transcript IDs.
            If False, use the "cds_haplotypes" key. Defaults to False.
        return_haplotypes (bool, optional): If True, return a dictionary mapping transcript IDs to their haplotype data.
            If False, return the save_dir path. Defaults to False.
        max_haplotypes (int, optional): Maximum number of haplotypes to save. Defaults to None.
        max_line_len (int, optional): Maximum length of a line in the merged JSON file. Defaults to None.
            This is to avoid loading very long lines into memory which can take a very long time.
        compress (bool, optional): Whether to compress the output files. Defaults to True.
        force (bool, optional): Whether to overwrite existing files. Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        dict or str:
            - If return_haplotypes is True: Returns a dictionary {tx_id: haplotype_data}.
            - If return_haplotypes is False: Returns the save_dir path.

    Notes:
        - The merged JSON file is expected to be gzipped and contain one JSON object per line.
        - Each output file is named as "<transcript_id>.json.gz".
        - The transcript ID is extracted from the haplotype name using the specified key.

    Example:
        >>> split_haplosaurus_results("all_haplotypes.json.gz", save_dir="output_dir", use_protein_ids=True)
        'output_dir'
    """
    import gzip
    import json

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True) 
    
    # Get set of existing output files for fast lookup
    if not force:
        existing_files = set(os.path.basename(f) for f in glob.glob(os.path.join(save_dir, "*.json*")))
    else:
        existing_files = set()

    # Select the key to use for extracting haplotype names
    key = "protein_haplotypes" if use_protein_ids else "cds_haplotypes"

    # Initialize the haplotypes dictionary if needed
    haplotypes = {} if return_haplotypes else None

    # Read and process the merged JSON file
    with gzip.open(merged_json, "rt", encoding="utf-8") as f:
        for line in tqdm(f, "Loading json"):
            line = line.strip()
            if line:
                if max_line_len is not None and len(line) > max_line_len:
                    if verbose:
                        print(f"Skipping line because it is too long: {len(line)} > {max_line_len}")
                    continue
                data = json.loads(line)
                if max_haplotypes is not None and len(data[key]) > max_haplotypes:
                    if verbose:
                        print(f"Skipping {tx_id} because it has more than {max_haplotypes} haplotypes")
                    continue
                # Extract transcript ID from the haplotype name
                tx_id = data[key][0]['name'].split(':')[0]
                out_fname = f"{tx_id}.json.gz"
                save_path = os.path.join(save_dir, out_fname)
                if (not force) and (out_fname in existing_files):
                    if verbose:
                        print(f"Skipping {tx_id} because it already exists")
                    continue

                # Save the data
                utils.save_json(obj=data,
                                save_path=save_path,
                                compress=compress,
                                verbose=verbose)
                # Store the haplotypes in the dictionary if requested
                if return_haplotypes:
                    haplotypes[tx_id] = data

    # Return the haplotypes dictionary if requested
    if return_haplotypes:
        return haplotypes
    else:
        return save_dir

def merge_haplotype_datasets(haplotype_datasets,
                             tx_id_method="union",
                             use_deepcopy: bool = True,
                             verbose: bool = True):
    """
    Merge a list of haplotype datasets (dicts keyed by tx_id) into a single merged dict.
    For each tx_id, merges total_population_counts, protein_haplotypes, and cds_haplotypes.
    Sets total_haplotype_count to None for merged entries.

    Args:
        haplotype_datasets (dict): A dict of haplotype datasets (dicts keyed by tx_id) to merge. 
            Each top-level key is a string identifier for the dataset (e.g. "1KG", "HGDP").
        use_deepcopy (bool): Whether to use deepcopy to merge the datasets. Defaults to False.
        verbose (bool): Whether to print verbose output. Defaults to True.

    Returns:
        dict: A merged haplotype dataset (dict keyed by tx_id).


    Example:
        import src.haplosaurus as hs
        haplotypes_1kg = hs.get_haplotypes(cache = hs.DIR_DICT["haplotypes"],
                                    tx_ids = tx_ids_all,
                                    cache_only = True)
        haplotypes_hgdp = hs.get_haplotypes(cache = hs.DIR_DICT["HGDP_haplotypes"],
                                    tx_ids = tx_ids_all,
                                    cache_only = True)
        haplotypes_merged = merge_haplotype_datasets({"1KG":haplotypes_1kg, "HGDP":haplotypes_hgdp})
    """  
    import copy
    import numpy as np

    # Remove any items with None values from each dataset using list comprehensions
    haplotype_datasets = {
        ds_key:ds  for ds_key, ds in haplotype_datasets.items() if ds is not None
    }

    # If there is only one dataset, return it
    if len(haplotype_datasets) == 1:
        return list(haplotype_datasets.values())[0]

    # Use deepcopy or copy depending on the use_deepcopy flag
    if use_deepcopy:
        if verbose:
            print("Using deepcopy")
        merged = copy.deepcopy(list(haplotype_datasets.values())[0])
    else:
        if verbose:
            print("Using copy")
        merged = list(haplotype_datasets.values())[0].copy()
    
    if tx_id_method == "union":
        tx_ids = set(merged.keys())
        for ds in haplotype_datasets.values():
            tx_ids.update(set(ds.keys()))
        tx_ids = list(tx_ids)
        if verbose:
            print(f"Merging {len(tx_ids)} tx_ids using union")
    elif tx_id_method == "intersection":
        tx_ids = set(merged.keys())
        for ds in haplotype_datasets.values():
            tx_ids.intersection_update(set(ds.keys()))
        tx_ids = list(tx_ids)
        if len(tx_ids)<len(merged):
            # subset merged to tx_ids
            merged = {tx_id:merged[tx_id] for tx_id in tx_ids}
        if verbose:
            print(f"Merging {len(tx_ids)} tx_ids using intersection")
    else:
        raise ValueError(f"Invalid tx_id_method: {tx_id_method}")
    
    # Helper functions
    def _check_all(freq_dict,
                    dataset_id):
        non_all_keys = [k for k in freq_dict if k != "_all" and not k.endswith(":ALL")]
        # Rename if only ALL columns are present
        if len(non_all_keys) == 0:
            all_keys = [k for k in freq_dict if k == "_all" or k.endswith(":ALL")]
            if len(all_keys)>0:
                freq_dict[dataset_id] = freq_dict[all_keys[0]] 
        return freq_dict
    
    def _update_all(freq_dict,
                    agg_func = np.mean,
                    type_func = float): 
        all_keys = [k for k in freq_dict if k != "_all" and not k.endswith(":ALL")]
        if all_keys:
            freq_dict["_all"] = type_func(agg_func([freq_dict[k] for k in all_keys]))
        return freq_dict

    # Iterate over the rest of the datasets
    for i, (dataset_id, ds) in tqdm(enumerate(haplotype_datasets.items()), 
                    desc="Merging haplotype datasets"):
        
        # Skip the first dataset (used for 'merged' variable)
        if i == 0:
            continue

        # Iterate over the haplotypes in the current dataset
        for tx_id, hap_tx_other in tqdm(ds.items(), 
                                        desc="Merging haplotypes", 
                                        leave=False):
            if tx_id not in tx_ids:
                if verbose:
                    print(f"Skipping {tx_id} because it is not in the tx_ids list")
                continue
            
            # If tx_id is not present, just copy it in 
            if tx_id not in merged and tx_id in tx_ids: 
                merged[tx_id] = copy.deepcopy(hap_tx_other) 
                continue

            hap_tx = merged[tx_id]

            #### total_population_counts ####
            hap_tx['total_population_counts'].update(hap_tx_other['total_population_counts'])

            #### protein_haplotypes ####
            phap_ids_other = set(get_haplotype_names(hap_tx_other, key='protein_haplotypes'))
            phap_ids_merged = set(get_haplotype_names(hap_tx, key='protein_haplotypes'))
            phap_ids_new = phap_ids_other - phap_ids_merged
            phap_ids_shared = phap_ids_merged & phap_ids_other

            # Simply append new haplotypes 
            if phap_ids_new: 
                hap_tx['protein_haplotypes'] += [x for x in hap_tx_other['protein_haplotypes'] if x['name'] in phap_ids_new]

            # For existing haplotypes, the procedure is more complex
            ## dict_keys(['frequency', 'other_hexes', 'type', 'count', 'has_indel', 'diffs', 'samples', 
            ## 'contributing_variants', 'flags', 'population_counts', 
            ## 'name', 'seq', 'hex', 'population_frequencies', 'aligned_sequences'])
            # Can skip: other_hexes, type, count, has_indel, diffs, contributing_variants, flags, name, seq, hex, aligned_sequences
            if phap_ids_shared:
                phap_other = {x['name']:x for x in hap_tx_other['protein_haplotypes'] if x['name'] in phap_ids_shared}
                phap_merged = {x['name']:x for x in hap_tx['protein_haplotypes']}

                # Merge frequency
                for phap_id in phap_ids_shared:  
                    phap_merged[phap_id]['frequency'] = float(np.mean([phap_merged[phap_id]['frequency'], phap_other[phap_id]['frequency']])) 

                    # Merge samples
                    phap_merged[phap_id]['samples'].update(phap_other[phap_id]['samples'])

                    # Merge population_counts
                    phap_other[phap_id]['population_counts'] = _check_all(phap_other[phap_id]['population_counts'], dataset_id)
                    phap_merged[phap_id]['population_counts'].update(phap_other[phap_id]['population_counts'])
                    phap_merged[phap_id]['population_counts'] = _update_all(phap_merged[phap_id]['population_counts'], 
                                                                            agg_func = np.sum, 
                                                                            type_func = int)

                    # Merge population_frequencies 
                    phap_merged[phap_id]['population_frequencies'].update(phap_other[phap_id]['population_frequencies']) 
                    phap_merged[phap_id]['population_frequencies'] = _update_all(phap_merged[phap_id]['population_frequencies'], 
                                                                                agg_func = np.mean, 
                                                                                type_func = float)
                # Update the merged haplotypes
                hap_tx['protein_haplotypes'] = list(phap_merged.values())

            #### cds_haplotypes ####
            chap_ids_other = set(get_haplotype_names(hap_tx_other, key='cds_haplotypes'))
            chap_ids_merged = set(get_haplotype_names(hap_tx, key='cds_haplotypes'))
            chap_ids_new = chap_ids_other - chap_ids_merged
            chap_ids_shared = chap_ids_merged & chap_ids_other

            # Simply append new haplotypes 
            if chap_ids_new:
                hap_tx['cds_haplotypes'] += [x for x in hap_tx_other['cds_haplotypes'] if x['name'] in chap_ids_new]

            # For existing haplotypes, the procedure is more complex
            if chap_ids_shared:
                chap_other = {x['name']:x for x in hap_tx_other['cds_haplotypes'] if x['name'] in chap_ids_shared}
                chap_merged = {x['name']:x for x in hap_tx['cds_haplotypes']}

                # Merge frequency
                for chap_id in chap_ids_shared:  
                    chap_merged[chap_id]['frequency'] = float(np.mean([chap_merged[chap_id]['frequency'], chap_other[chap_id]['frequency']])) 

                    # Merge samples
                    chap_merged[chap_id]['samples'].update(chap_other[chap_id]['samples'])

                    # Merge population_counts
                    chap_merged[chap_id]['population_counts'] = _check_all(chap_merged[chap_id]['population_counts'], dataset_id)
                    chap_merged[chap_id]['population_counts'].update(chap_other[chap_id]['population_counts'])
                    chap_merged[chap_id]['population_counts'] = _update_all(chap_merged[chap_id]['population_counts'], 
                                                                            agg_func = np.sum, 
                                                                            type_func = int)

                    # Merge population_frequencies
                    chap_merged[chap_id]['population_frequencies'].update(chap_other[chap_id]['population_frequencies'])
                    chap_merged[chap_id]['population_frequencies'] = _update_all(chap_merged[chap_id]['population_frequencies'], 
                                                                                agg_func = np.mean, 
                                                                                type_func = float)

                # Update the merged haplotypes
                hap_tx['cds_haplotypes'] = list(chap_merged.values())

            # Set total_haplotype_count to None (as in original code)
            hap_tx['total_haplotype_count'] = None

    if verbose:
        haplotype_names = get_haplotype_names(merged, key='protein_haplotypes', as_df=True)
        print(f"Merged {haplotype_names['haplotype'].nunique()} haplotypes across {len(merged)} tx_ids.")
    return merged
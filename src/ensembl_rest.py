import os
from typing import Dict, List, Optional, Union, Tuple, Set
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import ensembl_rest

import src.utils as utils
import src.config as config
import pooch

cache_dir = pooch.os_cache('ensembl_rest')

DIR_DICT = {
    "ensembl_rest": os.path.join(cache_dir,""),
    "vep": os.path.join(cache_dir,"vep",""),
    "haplotypes": os.path.join(cache_dir,"haplotypes",""),
    "variants": os.path.join(cache_dir,"variants",""),
    "variation": os.path.join(cache_dir,"variation",""),
    "xref_external": os.path.join(cache_dir,"xref_external",""),
}

def _params_to_filename(params: Optional[Dict],
                        suffix: str = '.json.gz') -> str:
    """
    Convert a dictionary of parameters to a filename suffix.

    Args:
        params (Optional[Dict]): A dictionary of parameters.
        suffix (str): The suffix for the filename.

    Returns:
        str: The filename suffix.
    """
    if params is None:
        return 'file' + suffix
    return ";".join([f"{k}:{v}" for k,v in sorted(params.items())]) + suffix



def xref_external(ids: List[str], 
                  client=None, 
                  species='homo_sapiens', 
                  cache=DIR_DICT["xref_external"],
                  force: bool = False,
                  verbose: bool = True): 
    """
    Get the Ensembl IDs for a list of IDs.
    For further details, see:
        https://ensemblrest.readthedocs.io/en/latest/#ensembl_rest.EnsemblClient.xref_external
        https://rest.ensembl.org/documentation/info/xref_external

    Args:
        ids (List[str]): The IDs to get the Ensembl IDs for.
        client (ensembl_rest.EnsemblClient, optional): The Ensembl client.
        species (str, optional): The species.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary mapping the IDs to their Ensembl IDs.
    """
    client = get_ensembl_client(client=client)
    ids = utils.process_ids(ids)

    if cache is not None:
        checksum = utils.ids_to_checksum(ids)
        save_path = os.path.join(cache, f'{checksum}.json.gz')
        map_dict = utils.load_json(save_path,
                                    force=force,
                                    verbose=verbose)
        if map_dict is not None:
            return map_dict

    map_dict = {}
    for id in tqdm(ids, desc="Getting Ensembl IDs"):
        try:
            res = client.xref_external(species=species, symbol=id)
            if len(res) > 0:
                map_dict[id] = {x['type']:x['id'] for x in res}
        except Exception as e:
            print(e)
    if cache is not None: 
        utils.save_json(map_dict, save_path)
    return map_dict
    
def get_ensembl_client(client: Optional[ensembl_rest.EnsemblClient] = None,
                       force: bool = False) -> ensembl_rest.EnsemblClient:
    """Get an Ensembl REST API client.

    Creates a new Ensembl REST API client if one is not provided or if force=True.
    For more information, see:
        https://ensemblrest.readthedocs.io/en/latest/#ensembl_rest.EnsemblClient
        https://rest.ensembl.org/

    Args:
        client (EnsemblClient, optional): Existing Ensembl client to use. Defaults to None.
        force (bool, optional): Whether to force creation of new client even if one is provided. 
            Defaults to False.

    Returns:
        EnsemblClient: An Ensembl REST API client instance, either the provided one or a new one.

    Example:
        >>> client = get_ensembl_client()
        >>> # Use existing client
        >>> same_client = get_ensembl_client(client=client)
        >>> # Force new client
        >>> new_client = get_ensembl_client(client=client, force=True)
    """
    if client is None or force:
        client = ensembl_rest.EnsemblClient()
    return client

def lookup_post(ids,
                batch_size: int = 500, 
                params: dict = {},
                client=None):
    """
    Map Ensembl IDs to other Ensembl ID synonyms using the Ensembl REST API.
    For more information, see:
        https://ensemblrest.readthedocs.io/en/latest/#ensembl_rest.EnsemblClient.lookup_post
        https://rest.ensembl.org/documentation/info/lookup_post
    """
    client = get_ensembl_client(client=client)
    batches = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
    map_dict = {}
    all_params = {}
    for batch in tqdm(batches):
        all_params = {'ids':batch}
        if params is not None and len(params) > 0:
            all_params.update(params)
        res = client.lookup_post(params=all_params)
        if list(res.values())[0] is not None:
            map_dict.update(res) 
    return map_dict

def _map_ids_lookup_post(df: pd.DataFrame, 
                         input_col: str,
                         client=None,
                         return_df: bool = False,
                         verbose: bool = True):
    """
    Map IDs to Ensembl IDs. 
    For example, map Ensembl Protein IDs (ENSPxxxx) to Ensembl Transcript IDs (ENSTxxxx).

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the IDs.
        col (str): The column to map.
        batch_size (int): The batch size for the mapping.

    Returns:
        dict: A dictionary mapping IDs to Ensembl IDs.
    """
    # Map Protein ID to Transcript ID
    res = lookup_post(ids=df[input_col].tolist(),
                            client=client)
    map_dict = {k:{v['object_type'].lower():v['Parent']} for k,v in res.items() if v['object_type'] in ['Gene','Transcript','Translation']}
    # Get all possible keys
    from itertools import chain
    all_keys = list(set(chain.from_iterable([list(v.keys()) for v in map_dict.values()])))

    if return_df:
        col_key = {'ENSG':'gene','ENSP':'translation','ENST':'transcript'}
        for col, key in col_key.items():
            if key in all_keys:
                map_k_v = {k:v[key] if key in v.keys() else {k:None} for k,v in map_dict.items()}
                if col in df.columns:
                    if verbose:
                        print(f"Warning: Overwriting column '{col}'")
                df[col] = df[input_col].map(map_k_v)
        return df
    else:
        return map_dict

def _map_ids_xref_external(df,
                          input_col,
                          force=False,
                          verbose=True):

    # Using Ensembl REST API
    map_dict = xref_external(ids=df[input_col], 
                                force=force, 
                                verbose=verbose)
    col_key = {'ENSG':'gene','ENSP':'translation','ENST':'transcript'}
    for col, key in col_key.items():
        map_k_v = {k:v[key] for k,v in map_dict.items()}
        if col in df.columns:
            if verbose:
                print(f"Warning: Overwriting column '{col}'")
        df[col] = df[input_col].map(map_k_v)
    return df

def map_ids(df,
            input_col,
            force=False,
            method=['xref_external','lookup_post'],
            verbose=True):
    """
    Map IDs to Ensembl IDs.
    """
    method = utils.one_only(method)
    if method == 'xref_external':
        return _map_ids_xref_external(df=df,
                                      input_col=input_col,
                                      force=force, 
                                      verbose=verbose)
    elif method == 'lookup_post':
        return _map_ids_lookup_post(df=df, 
                                    input_col=input_col, 
                                    force=force, 
                                    return_df=True,
                                    verbose=verbose)
    else:
        raise ValueError(f"Invalid method: {method}")




def transcript_haplotypes_get(ids: Optional[List[str]] = None,
                              cache: Path = Path(DIR_DICT["haplotypes"]),
                              species: str = "homo_sapiens",
                              params: dict = config.PARAMS_HAPLOTYPES,
                              client: Optional[ensembl_rest.EnsemblClient] = None,
                              force: bool = False,
                              cache_only: bool = False,
                              error: bool = False,
                              verbose: bool = True) -> dict:
    """Get haplotype information for transcript IDs from Ensembl REST API.
    For more information, see:
        Ensembl REST API: https://rest.ensembl.org/documentation/info/transcript_haplotypes_get
        ensembl-rest module: https://ensemblrest.readthedocs.io/en/latest/#ensembl_rest.EnsemblClient.transcript_haplotypes_get
        Haplosaurus: https://useast.ensembl.org/info/docs/tools/vep/haplo/index.html

    Args:
        tx_ids (list, optional): List of transcript IDs to get haplotypes for. If None, will use all cached haplotypes. Defaults to None.
        save_dir (Path, optional): Directory to cache haplotype data. Defaults to DIR_DICT["haplotypes"].
        species (str, optional): Species name for Ensembl API. Defaults to "homo_sapiens".
        params (dict, optional): Parameters for Ensembl API request. Defaults to PARAMS_HAPLOTYPES.
        client (EnsemblClient, optional): Ensembl REST API client. If None, will create new client. Defaults to None.
        force (bool, optional): Whether to force API request even if cached data exists. Defaults to False.
        cache_only (bool, optional): Whether to only return cached data without making API requests. Defaults to False.
        error (bool, optional): Whether to raise exceptions on API errors. Defaults to False.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.

    Returns:
        dict: Dictionary mapping transcript IDs to their haplotype information.
            Each value contains protein haplotype sequences and metadata from Ensembl.

    Raises:
        Exception: If error=True and API request fails for a transcript.
    """
    client = get_ensembl_client(client=client)
    haplotypes = {}

    # If cache only, don't search for new tx_ids  
    if cache_only:
        force = False

    # Get haplotypes
    for tx_id in tqdm(ids,
                      desc="Getting haplotypes"):
        save_path = os.path.join(cache,f"{tx_id}.json.gz")
        hap_tx_id = utils.load_json(save_path,
                                    force=force,
                                    verbose=verbose>1)
        if hap_tx_id is not None:
            haplotypes[tx_id] = hap_tx_id
            continue
        if cache_only:
            continue
        else:
            try:
                haplotypes[tx_id] = client.transcript_haplotypes_get(
                    id=tx_id,
                    species=species,
                    params=params
                    )
                utils.save_json(obj=haplotypes[tx_id],
                                 save_path=save_path,
                                 verbose=verbose>1)
            except Exception as e:
                if error:
                    raise e
                else:
                    if verbose:
                        print(f"Error getting haplotypes for {tx_id}: {e}")
                    continue
    return haplotypes

def get_vep(ids: Union[str, List[str]],
            species: str = 'homo_sapiens',
            params: Dict = config.PARAMS_VEP,
            save_dir: Path = Path(DIR_DICT["vep"]),
            client: Optional[ensembl_rest.EnsemblClient] = None,
            desc: str = "Getting variant info",
            leave: bool = True,
            force: bool = False,
            verbose: bool = True,
            cache_only: bool = False) -> Dict[str, List[Dict]]:
    """Get Variant Effect Predictor (VEP) results from Ensembl REST API.

    Retrieves VEP annotations for a list of variant IDs, with caching to avoid repeated API calls.

    Args:
        ids (str or list): Variant identifier(s) to look up
        species (str, optional): Species name for Ensembl API. Defaults to 'homo_sapiens'.
        params (dict, optional): Parameters for VEP API request. Defaults to PARAMS_VEP.
        save_dir (Path, optional): Directory to cache VEP results. Defaults to DIR_DICT["vep"].
        client (EnsemblClient, optional): Ensembl REST API client. If None, will create new client.
        desc (str, optional): Progress bar description. Defaults to "Getting variant info".
        leave (bool, optional): Whether to leave progress bar. Defaults to True.
        force (bool, optional): Whether to force API request even if cached. Defaults to False.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.
        cache_only (bool, optional): Whether to only return cached results. Defaults to False.

    Returns:
        dict: Dictionary mapping variant IDs to their VEP results from Ensembl.
            Each value contains the variant's predicted effects and consequences.

    Example:
        >>> vep_results = get_vep('rs699')
        >>> print(vep_results['rs699'][0]['most_severe_consequence'])
        'missense_variant'
    """

    if cache_only:
        force = False

    client = get_ensembl_client(client=client)
    variant_vep = {}
    ids = utils.process_ids(ids)

    for id in tqdm(ids,
                   desc=desc,
                   leave=leave):
        # Check if variant info is already cached
        save_path = f"{save_dir}/{id}.json.gz"
        variant_vep_id = utils.load_json(save_path, 
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
        utils.save_json(obj={id:variant_vep_id},
                        save_path=save_path,
                        verbose=verbose>1)
                  
        variant_vep[id] = variant_vep_id
        
    return variant_vep


def filter_vep(variant_vep: Dict[str, List[Dict]],
               consequence_terms: Optional[List[str]] = None, 
               exact: bool = False,
               desc: str = "Filtering variants VEP",
               leave: bool = False,
               verbose: bool = True):
    """
    Filter VEP results.

    Args:
        variant_vep (Dict[str, List[Dict]]): The VEP results.
        consequence_terms (Optional[List[str]]): The consequence terms to filter on.
        exact (bool): Whether to filter on exact matches.
        desc (str): The description of the filter.
    """
    if consequence_terms is None:
        return variant_vep
    
    consequence_terms = utils.as_list(consequence_terms)
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

def get_variation(variant_id: str,
                  save_dir: Optional[Path] = Path(DIR_DICT["variation"]),
                  species: str = 'homo_sapiens',
                  params: Dict = config.PARAMS_VARIATION,
                  verbose: bool = True,
                  client: Optional[ensembl_rest.EnsemblClient] = None,
                  force: bool = False,
                  error: bool = True):
    """
    Uses a variant identifier (e.g. rsID) to return the variation features
      including optional genotype, phenotype and population data
    See the following for more information:
        https://rest.ensembl.org/documentation/info/variation_id

    Args:
        variant_id (str): The variant identifier.
        save_dir (Optional[Path]): The directory to save the variation.
    """
    if save_dir is not None:
        filename = _params_to_filename(params)
        save_path = os.path.join(save_dir,variant_id,filename)
        variation = utils.load_json(save_path, 
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
        utils.save_json(obj=variation,
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




# https://mutalyzer.nl/api/

import requests
import json
import os
import pooch
import time
from tqdm import tqdm
import pandas as pd

# Local imports
import src.utils as utils


def backtranslate_hgvsp_ids(hgvsp_ids,
                            remove_parentheses=True,
                            sleep_time=0.1,
                            add_prefix=None,
                            verbose=False):
    """
    Back translate HGVSp IDs to genomic coordinates using Mutalyzer API
    
    Parameters
    ----------
    hgvsp_ids : list or array-like
        List of HGVSp IDs to back translate
    remove_parentheses : bool, default=True
        Whether to remove parentheses from HGVSc notation
    sleep_time : float, default=0.1
        Time to sleep between API calls (rate limiting)
    verbose : bool, default=False
        Whether to print progress information
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: HGVSp, HGVSc, position, REF, ALT
    """
    
    # Back translate each variant
    backtranslated_variants = []
    for hgvsp in tqdm(hgvsp_ids, 
                    desc="Backtranslating variants"): 
        result = query_mutalyzer(hgvsp, 
                                  endpoint="back_translate")
        if result:
            backtranslated_variants.append({
                'HGVSp': hgvsp,
                'HGVSc': result
            })
        time.sleep(sleep_time)  # Rate limiting to be respectful to the API
    
    # Convert to DataFrame
    backtranslated_df = pd.DataFrame(backtranslated_variants).explode("HGVSc")
    
    if remove_parentheses:
        backtranslated_df['HGVSc'] = backtranslated_df['HGVSc'].str.replace(r'[()]', '', regex=True)
    
    # Parse the start pos, ref and alt from the HGVSc column
    backtranslated_df['c.position'] = backtranslated_df['HGVSc'].str.extract(r'c\.(\d+)').astype(int)
    backtranslated_df['c.REF'] = backtranslated_df['HGVSc'].str.extract(r'c\.\d+([A-Z]+)>')[0]
    backtranslated_df['c.ALT'] = backtranslated_df['HGVSc'].str.extract(r'>([A-Z]+)')[0]
    
    if add_prefix:
        backtranslated_df.columns = [add_prefix + col for col in backtranslated_df.columns]
    
    if verbose:
        print(f"Successfully backtranslated {len(backtranslated_df)} variants")
    
    return backtranslated_df

def query_mutalyzer(id, 
                    endpoint,
                     base_url="https://mutalyzer.nl/api/",
                     force=False
                     ):
    """
    Query Mutalyzer API
    
    Parameters
    ----------
    id : str
        ID to query
    base_url : str, default="https://mutalyzer.nl/api/"
        Base URL of the Mutalyzer API
    endpoint : str, default="back_translate"
        Endpoint of the Mutalyzer API
    force : bool, default=False
        Whether to force a new query even if the result is already cached
        
    Returns
    -------
    dict or None
        JSON response from Mutalyzer API containing genomic coordinates,
        or None if request fails
    """  
    cache = pooch.os_cache("mutalyzer_"+endpoint)
    
    # Check if result is already cached
    os.makedirs(cache, exist_ok=True)
    cache_path = os.path.join(cache, f"{id}.json")
    if os.path.exists(cache_path) and not force:
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If cache file is corrupted, remove it and continue
            cache_path.unlink(missing_ok=True)
    
    # If not cached, make API request
    url = f"{base_url}/{endpoint}/{id}"    
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        
        # Cache the result
        try:
            with open(cache_path, 'w') as f:
                json.dump(result, f)
        except IOError as e:
            print(f"Warning: Could not cache result for {id}: {e}")
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error for {id}: {e}")
        return None
    

def normalize_hgvs_ids(ids,
                       base_url="https://mutalyzer.nl/api/",
                       endpoint="normalize"):
    """
    Normalize HGVS IDs using Mutalyzer API
    
    Parameters
    ----------
    ids : str or list
        HGVS ID(s) to normalize
    base_url : str, default="https://mutalyzer.nl/api/"
        Base URL of the Mutalyzer API
    endpoint : str, default="normalize"
        Endpoint of the Mutalyzer API
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing normalized HGVS descriptions and protein information
    """
    ids = utils.process_ids(ids)
    dfs = []
    for id in tqdm(ids, desc="Normalizing HGVS IDs"):
        result = query_mutalyzer(id, 
                                 endpoint=endpoint, 
                                 base_url=base_url) 
        if result is None:
            continue

        df = pd.DataFrame() 
        if result['equivalent_descriptions']: 

            df_eqs = []
            for key, value in result['equivalent_descriptions'].items():
                df_eq =  pd.DataFrame(value)
                df_eq["reference"] = df_eq["reference"].apply(lambda x: x['selector']['id']).astype(str)
                df_eq = df_eq.add_prefix(key + ".")
                df_eqs.append(df_eq)
            if len(df_eqs) > 0:
                df_eqs = pd.concat(df_eqs)  
            df = pd.concat([df, df_eqs], axis=1) 
             
        if 'protein' in result:   
            df_prot = pd.DataFrame(result['protein'], index=[0])
            df_prot.columns = ['p.' + col for col in df_prot.columns]
            df = pd.concat([df, df_prot], axis=1)
        if 'rna' in result: 
            df["r.description"] = result['rna']['description']
        
        # Add input and normalized description
        df.insert(0, "input_description", id)
        df.insert(1, "normalized_description", result['normalized_description'])
        dfs.append(df)
    # Convert dict columns to strings before dropping duplicates
    df = pd.concat(dfs)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    
    return df.drop_duplicates()



# import hgvs.parser
# import hgvs.dataproviders.uta
# import hgvs.assemblymapper

# # Initialize a data provider (e.g., UTA)
# hdp = hgvs.dataproviders.uta.connect()

# # Initialize an HGVS parser
# hp = hgvs.parser.Parser()

# # Parse an HGVS string
# # NC_000017.11, NM_007294.4, ENST00000357654
# variant_c = hp.parse_hgvs_variant("NM_007294.4:c.3083G>A")

# # Project the genomic variant to a transcript (example)
# # This requires a relevant transcript and potentially other data
# am = hgvs.assemblymapper.AssemblyMapper(hdp, assembly_name='GRCh38')
# variant_g = am.c_to_g(variant_c)
# variant_g
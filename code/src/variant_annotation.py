def get_mv_db():
    import myvariant
    mv = myvariant.MyVariantInfo()
    return mv

def find_field(search_string, 
               mv=None):
    """
    Find all fields in mv that contain the search_string
    https://docs.myvariant.info/en/latest/doc/data.html#available-fields
    """
    if mv is None:
        mv = get_mv_db()
    fields = mv.get_fields()
    return [k for k in fields if search_string in k]




def embl_get_variants(uniprot_id,
                      as_df=True):
    """
    Retrieve variants for a given transcript using the EBI Proteins API.

    Args:
        transcript_id (str): The transcript ID to query variants for.

    Returns:
        list: A list of variants associated with the transcript.
    """
    # 'https://www.ebi.ac.uk/proteins/api/variation?&accession=O43657'
    import requests, sys
    base_url = "https://www.ebi.ac.uk/proteins/api" 
    requestURL = f"{base_url}/variation?offset=0&size=100&accession={uniprot_id}"
    r = requests.get(requestURL, headers={ "Accept" : "application/json"})
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    responseBody = r.json()
    if as_df:
        import pandas as pd
        df1 = pd.DataFrame({k: [v] for k, v in responseBody[0].items() if k != 'features'})
        df2 = pd.DataFrame(responseBody[0]['features'])
        df1 = pd.concat([df1] *  len(df2), ignore_index=True)
        df = pd.concat([df1, df2], axis=1)
        return df
    return responseBody[0]

import sys
sys.path.append("code")
from src.utils import as_list

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

def get_clinvar_db(url="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"):
    import pysam
    return pysam.VariantFile(url)

def get_clinvar_db_headers(cv=None):
    if cv is None:
        cv = get_clinvar_db()
    return list(cv.header.info)

def filter_variants(recs, 
                    filter=None, 
                    regions=None,
                    verbose=True):
    recs_filtered = []
    if regions is not None:
        regions = as_list(regions)
        for region in regions:
            chrom = region.split(":")[0]
            start = int(region.split(":")[1].split("-")[0])
            end = int(region.split(":")[1].split("-")[1])
            recs_filtered += [rec for rec in recs if (rec.contig == "chr"+str(chrom).replace("chr", "") or rec.contig == chrom.replace("chr", "")) and rec.start >= start and rec.stop <= end]
        recs = recs_filtered
    if filter is not None:
        for rec in recs:
            for k,v in filter.items():
                if k in rec.info:
                    info = rec.info[k]
                    if type(info) == str:
                        info = (info,)
                    info = tuple([x.lower().replace(' ','_') for x in info])
                    if type(v) == str:
                        v = (v,) # Ensure v is a tuple
                    v = tuple([x.lower().replace(' ','_') for x in v])
                    if set(info).intersection(set(v)):
                        recs_filtered.append(rec)
    if verbose:
        print(f">> {len(recs_filtered)} variants remain after filtering")
    return recs_filtered

def get_clinvar_variants(db,
                         transcript_ids,
                         filters=None, # ={'CLNSIG': ('Pathogenic',)},
                         regions=None,
                         coding_only=True,
                         cv=None,
                         save_path=None,
                         force=False,
                         verbose=True):
    """
    Get clinvar variants for a given transcript
    See here for clinvar field descriptions: https://www.ncbi.nlm.nih.gov/clinvar/docs/clinsig/
    
    practice guideline (4 stars): The classification from the practice guideline record is used as the classification on the VCV and RCV records, no matter what other submitters may have reported.
    Reviewed by expert panel (3 stars): The classification from the expert panel record is used as the classification on the VCV and RCV records, no matter what other submitters may have reported.
    criteria provided, single submitter (1 star): The classification from all submitted records with this review status is used to calculate the aggregate classification on the VCV and RCV records.
    For example, the classification on a single SCV record with review status "criteria provided, single submitter" supersedes classifications on multiple SCV records with lower review statuses.
    no assertion criteria provided (0 stars): The classification from all submitted records with this review status is used to calculate the aggregate classification on the VCV and RCV records, if there is no record with higher precedence.
    """
    from tqdm.auto import tqdm
    import os
    import pickle

    if save_path is not None:
        if os.path.exists(save_path) and not force:
            print(f"Skipping {save_path} because it already exists")
            with open(save_path, 'rb') as f:
                return pickle.load(f)

    if cv is None:
        cv = get_clinvar_db()
    recs_all = []
    transcript_ids = as_list(transcript_ids)
    variant_counts = {}
    for transcript_id in tqdm(transcript_ids, desc="Fetching clinvar variants"): 
        if verbose:
            print(transcript_id)
        try:
            # Fetch all variants in the transcript
            tx = db.transcript_by_id(transcript_id)
            recs  = [x for x in cv.fetch(tx.contig, tx.start, tx.end)]
        except:
            if verbose:
                print(f"No variants found for {transcript_id}")
            continue
        if coding_only:
            if tx.biotype != 'protein_coding':
                if verbose:
                    print(f"Transcript {transcript_id} is not protein coding")
                continue
            regions = [f"chr{tx.contig}:{range[0]}-{range[1]}" for range in tx.coding_sequence_position_ranges]
        if verbose:
            print(f"> Found {len(recs)} clinvar variants for {transcript_id}")
        if regions:
            if verbose:
                print(f"> Filtering by regions.")
            recs = filter_variants(recs=recs, 
                                   regions=regions,
                                   verbose=verbose)
        if filters:
            if verbose:
                print(f"> Filtering by filters.")
            for k,v in filters.items():
                recs = filter_variants(recs=recs, 
                                       filter={k:v}, 
                                       verbose=verbose)
        variant_counts[transcript_id] = len(recs)
        recs_all += recs
    if save_path is not None:
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump((recs_all, variant_counts), f)
    return recs_all, variant_counts


def get_variant_chrom(recs):
    return [x.contig for x in recs]
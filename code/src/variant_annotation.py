import sys
sys.path.append("code")
from src.utils import as_list, save_pickle, load_pickle, save_vcf, sort_variants

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

def get_clinvar_db_info_descriptions(cv=None):
    if cv is None:
        cv = get_clinvar_db()
    return {k:v.description for k,v in dict(cv.header.info.items()).items()}

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
                         verbose=False):
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
    import pysam
    import os
    vcf_in = None
    if save_path is not None:
        if os.path.exists(save_path) and not force:
            print(f"Using existing file: {save_path}")
            vcf_in = pysam.VariantFile(save_path, 
                                       index_filename=save_path+'.tbi')
            header = vcf_in.header
            save_path = None # Don't overwrite existing file
    if vcf_in is None:
        print("Creating new clinvar VCF")
        if cv is None:
            cv = get_clinvar_db()
        vcf_in = cv
        header = cv.header
    # Add transcript_id to the header
    if 'transcript_id' not in vcf_in.header.info:
        vcf_in.header.info.add(
               "transcript_id", 
               number='1', 
               type='String', 
               description="Transcript ID associated with the variant"
           ) 
    
    recs_all = []
    transcript_ids = as_list(transcript_ids)
    variant_counts = {} 
    transcripts_to_variants = {}

    for transcript_id in tqdm(transcript_ids, desc="Fetching clinvar variants"): 
        if verbose:
            print(transcript_id)
        try:
            # Fetch all variants in the transcript
            tx = db.transcript_by_id(transcript_id)
            recs  = [x for x in vcf_in.fetch(tx.contig, tx.start, tx.end)]
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
        # Add transcript_id to the variant
        for rec in recs:
            rec.info['transcript_id'] = transcript_id
        # Update variant counts
        variant_counts[transcript_id] = len(recs)
        transcripts_to_variants[transcript_id] = [rec.id for rec in recs]
        # Add to all variants
        recs_all += recs
    
    # Sort variants so they can be indexed
    recs_all = sort_variants(recs_all)
    # Write the filtered variants to the output VCF
    save_vcf(recs_all, save_path, header) 
    if verbose:
        print(f"Found {len(recs_all)} clinvar variants")
    # Return the results
    return recs_all, variant_counts, transcripts_to_variants

def get_clinvar_variants_pathogenic(db,
                                    filters = {'CLNSIG': ('Pathogenic',),
                                               'CLNREVSTAT': ('practice_guideline', # 4-star
                                                              'reviewed_by_expert_panel','expert_panel' # 3-star
                                                             ),
                                                },
                                    mc_descendants=None,
                                    min_variants_per_transcript=1,
                                    max_variants_per_transcript=1,
                                    **kwargs):
   if mc_descendants is not None:
       descendants = get_sequence_ontology_descendants(mc_descendants, as_str=True)
       filters['MC'] = tuple(descendants)
   recs, variant_counts, transcripts_to_variants = get_clinvar_variants(
        filters = filters,
        db=db, 
        **kwargs
    )
   transcripts_selected, recs_selected = select_variants(
    recs, 
    variant_counts,
    min_variants_per_transcript=min_variants_per_transcript,
    max_variants_per_transcript=max_variants_per_transcript
    )
   return {
       'recs': recs,
       'recs_selected': recs_selected,
       'transcripts_selected': transcripts_selected,
       'transcripts_to_variants': transcripts_to_variants,
       'variant_counts': variant_counts
       }

def get_clinvar_variants_benign(db,
                                    filters = {'CLNSIG': ('Benign',),
                                               'CLNREVSTAT': ('practice_guideline', # 4-star
                                                              'reviewed_by_expert_panel','expert_panel' # 3-star
                                                             )
                                                },
                                    min_variants_per_transcript=1,
                                    max_variants_per_transcript=1,
                                    **kwargs):
   return get_clinvar_variants_pathogenic(db=db,
                                    filters = filters,
                                    min_variants_per_transcript=min_variants_per_transcript,
                                    max_variants_per_transcript=max_variants_per_transcript,
                                    **kwargs)


def get_variant_chrom(recs):
    return [x.contig for x in recs]

def select_variants(recs, 
                    variant_counts,
                    min_variants_per_transcript=1,
                    max_variants_per_transcript=None,
                    group_by_transcript=False,
                    verbose=True):
    """
    Select variants for a given transcript
    """
    from tqdm.auto import tqdm
    transcripts_selected = {k:v for k,v in variant_counts.items() if v > min_variants_per_transcript} 
    if verbose:
        print(f"Selected {len(transcripts_selected)} transcripts.")
    # Create dict mapping transcript_id to first variant found for that transcript
    recs_selected = {}
    for rec in tqdm(recs, desc="Selecting variants"):
        transcript_id = rec.info['transcript_id'] 
        if transcript_id in transcripts_selected: 
            if transcript_id not in recs_selected:
                recs_selected[transcript_id] = []
            elif max_variants_per_transcript is not None and len(recs_selected[transcript_id]) >= max_variants_per_transcript:
                continue
            else:
                recs_selected[transcript_id] += [rec]
    # Convert dict values to list
    if group_by_transcript is False:
        recs_selected = [rec for recs in recs_selected.values() for rec in recs]
        if verbose:
            print(f"Selected {len(recs_selected)} variants.")
    else:
        if verbose:
            n_variants = sum([len(x) for x in recs_selected.values()])
            print(f"Selected {len(recs_selected)} transcripts with {n_variants} variants.")
    return transcripts_selected, recs_selected


def get_sequence_ontology(**kwargs):
    # Import SO ontology
    import owlready2
    so_url = "http://purl.obolibrary.org/obo/so.owl"
    return owlready2.get_ontology(so_url, **kwargs).load()

def get_sequence_ontology_descendants(ancestor_label,
                                      so=None,
                                      include_self=True,
                                      as_str=False,
                                      verbose=True):
    if so is None:
        so = get_sequence_ontology()
    # get all descendant terms of 'coding_sequence_variant'
    ancestor = so.search_one(label=ancestor_label)
    descendants = ancestor.descendants(include_self=include_self)
    if verbose:
        print(f"Found {len(descendants)} descendants of '{ancestor_label}'")
    if as_str:
        return [f"{x.name.replace('_', ':')}|{x.label[0]}" for x in descendants]
    return descendants

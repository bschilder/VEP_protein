def import_gtf_db(gtf_file="GRCh38/gencode.v47.annotation.gtf.gz", 
                  dbfn=None, 
                  force=False):
    import gffutils
    import os
    if dbfn is None:
        dbfn = gtf_file.replace(".gtf.gz", ".db")
    if os.path.exists(dbfn) and not force:
        print("Using existing db.")
        db = gffutils.FeatureDB(dbfn, keep_order=True)
    else:
        print("Creating db.")
        db = gffutils.create_db(
            gtf_file,
            dbfn=gtf_file,
            merge_strategy="create_unique",
            disable_infer_transcripts=True,
            disable_infer_genes=True,
            verbose=True,
            force=force,
        )
    return db

def import_feature_metadata(db,
                            feature_type='transcript'):
    from tqdm.auto import tqdm
    feature_meta = []
    total = db.count_features_of_type(feature_type)
    for feature in tqdm(db.features_of_type(feature_type), desc=f"Processing {feature_type} metadata", total=total):
        attr = {k:v[0] for k,v in feature.attributes.items()} 
        feature_meta.append({
            'id': feature.id,
            'chrom': feature.chrom,
            'start': feature.start,
            'end': feature.end,
            'strand': feature.strand,
            'length': feature.end - feature.start,
            **attr
        })
    print("Converting to DataFrame.")
    feature_meta = pd.DataFrame(feature_meta)
    print(feature_meta.shape[0], "features found.")
    return feature_meta



def get_all_ids(db, featuretype, verbose=True):
    if verbose:
        print(f"Getting all {featuretype} ids.")
    return [x.id for x in db.features_of_type(featuretype)] 

def get_exons(db,
              transcript_ids=None,
              transcript_types=['protein_coding']):
    from tqdm.auto import tqdm
    exons = {}
    all_transcripts = get_all_ids(db, featuretype='transcript')
    # Filter transcript_ids if specified
    if transcript_ids is None:
        transcript_ids = all_transcripts
    else:
        transcript_ids = intersect(transcript_ids, all_transcripts)
    # Iterate over transcripts
    for transcript_id in tqdm(transcript_ids, desc="Extracting exons"):
        transcript = db.get(transcript_id)
        if transcript_types is not None:
            if transcript.attributes['transcript_type'] not in transcript_types:
                continue
        transcript_id = transcript.id
        exons[transcript_id] = []
        for exon in db.children(transcript_id, featuretype='exon'):
            attr = {k:v[0] for k,v in exon.attributes.items()} 
            exons[transcript_id].append({
            'id': exon.id,
            'chrom': exon.chrom,
            'start': exon.start,
            'end': exon.end,
            'strand': exon.strand,
            'length': exon.end - exon.start,
            **attr
        }) 
    return exons
# exons = get_exons(db, transcript_ids=knownCanonical[4].tolist())

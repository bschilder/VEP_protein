def get_vep(rec,
            allele=None,
            contig=None,
            start=None,
            stop=None,
            assembly="GRCh38",
            fields="all",
            species="homo_sapiens",
            **kwargs):
    
    import ensembl_rest as ensembl_rest
    # Get variant annotation from ensembl rest
    # Docs: https://rest.ensembl.org/documentation/info/vep_region_get
    if allele is None:
        allele = rec.alleles[1]
    if contig is None:
        contig = rec.contig
    if start is None:
        start = rec.pos
    if stop is None:
        stop = rec.pos+1
    region = f"{contig}:{start}-{stop}/{allele}"
    recs_vep = ensembl_rest.vep_region_get(
        region=region,  # Wrap region in object
        assembly=assembly,
        fields=fields, 
        species=species,
        **kwargs
    )
    return recs_vep
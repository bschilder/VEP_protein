from biomart import BiomartServer
import pandas as pd
import pooch

def _results_to_df(results,
                   verbose=True):
    if verbose:
        print('Converting results to dataframe')
    # Convert response text to dataframe
    txt = results.text
    df = pd.DataFrame([x.split('\t') for x in txt.split('\n') if x])
    if len(df) > 0:  # Check if dataframe has content
        # First row contains headers
        id_map = pd.DataFrame(df.values[1:], columns=df.iloc[0])
    return id_map


def get_ensembl_mappings(ensembl_ids, 
                         dataset='hsapiens_gene_ensembl', 
                         attributes=['ensembl_gene_id', 'hgnc_symbol'],
                         verbose=True):
    """Maps Ensembl IDs to other attributes using biomart.
    
    Args:
        ensembl_ids (list): List of Ensembl IDs to map.
        dataset (str): Ensembl dataset to use (default: 'hsapiens_gene_ensembl').
        attributes (list): List of attributes to retrieve (default: ['ensembl_gene_id', 'hgnc_symbol']).
    
    Returns:
        dict: Dictionary mapping Ensembl IDs to corresponding attributes.

    Example:
        ensembl_ids_to_map = ['ENSG00000139618', 'ENSG00000141510', 'ENSG00000175899']
        mapped_ids = get_ensembl_mappings(ensembl_ids_to_map)
        print(mapped_ids)
    """
    server = BiomartServer('http://useast.ensembl.org/biomart')
    mart = server.datasets[dataset]
    
    results = mart.search({
        'filters': {'ensembl_gene_id': ensembl_ids},
        'attributes': attributes
    }, header=1)
    
    return _results_to_df(results)


def get_id_map(dataset='hsapiens_gene_ensembl',
               filters={'with_refseq_peptide': 'only'},
                attributes=['ensembl_gene_id', 
                            'ensembl_gene_id_version',
                                'ensembl_transcript_id', 
                                'ensembl_transcript_id_version', 
                                'ensembl_peptide_id', 
                                'ensembl_peptide_id_version', 
                                'refseq_peptide', 
                                'refseq_peptide_predicted'],
                cache_dir=pooch.os_cache('biomart'),
                force=False,
                verbose=True):
    import os
    save_path = os.path.join(cache_dir, 'id_map.csv.gz')
    if os.path.exists(save_path) and not force:
        if verbose:
            print(f"Loading from cache ==> {save_path}")
        df = pd.read_csv(save_path, sep='\t')
        return df
    if verbose:
        print("Querying biomart")
    server = BiomartServer('http://useast.ensembl.org/biomart')
    mart = server.datasets[dataset]
    results = mart.search({
        'filters': filters,
        'attributes': attributes
    }, header=1)

    df = _results_to_df(results, verbose=verbose)
    
    # Cache results 
    if cache_dir is not None:
        if verbose:
            print(f"Caching ==> {save_path}")
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(save_path, index=False, sep='\t')

    return df

def map_ids(df, 
            id_map=None,
            on_left='RefSeq peptide ID',
            on_right='RefSeq peptide ID',
            how='left',
            **kwargs):
    if id_map is None:
        id_map = get_id_map()
    if len((set(df.columns) & set(id_map.columns)) - set([on_left, on_right])) > 0:
        raise ValueError('Columns in df and id_map overlap')
    return df.merge(id_map, 
                    left_on=on_left, 
                    right_on=on_right, 
                    how=how,
                    **kwargs)
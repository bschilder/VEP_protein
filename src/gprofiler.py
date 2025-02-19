from src.utils import as_list

def get_id_map(ids, 
               target_namespace='ENSP',
               unique_only=True,
               drop_na=['incoming','converted'],
               rename_converted=False,
               keep_cols=None,
               as_dict=False,
               verbose=True,
               **kwargs):
    """
    Get a mapping of IDs to a target namespace.
    See https://biit.cs.ut.ee/gprofiler/convert for more information.

    Args:
        ids: list of IDs to map
        target_namespace: namespace to map to
        as_dict: return a dictionary instead of a DataFrame. 
        When 1, the dictionary keys are the incoming IDs and the values are the converted IDs. 
        When -1, the dictionary keys are the converted IDs and the values are the incoming IDs.
        When 0 or False, return a DataFrame.
        **kwargs: additional arguments to pass to gprofiler.convert

    Returns:
        pandas DataFrame with the mapping

    Example:
        >>> map_ids(['ENSP00000456328', 'ENSP00000456329'], target_namespace='ENST')
    """
    import pandas as pd
    from gprofiler import GProfiler
    ids = as_list(ids)

    query = [x for x in ids if pd.notna(x)]
    if unique_only:
        query = list(set(query))
    if verbose:
        print(f'Running gprofiler.convert with {len(query)} queries.')
    gp = GProfiler(return_dataframe=True)
    id_map = gp.convert(organism='hsapiens',
                query=query,
                target_namespace=target_namespace,
                **kwargs)
    if drop_na is not None:
        id_map = id_map.dropna(subset=drop_na)
    if as_dict is not False:
        if as_dict == -1:
            id_dict = dict(zip(id_map['converted'], id_map['incoming']))
        elif as_dict == 1:
            id_dict = dict(zip(id_map['incoming'], id_map['converted']))
        return id_dict
    if keep_cols is not None:
        id_map = id_map[keep_cols].drop_duplicates()
    if rename_converted:
        id_map.rename(columns={'converted': target_namespace}, inplace=True)
    return id_map

def map_ids(df, 
            id_map=None,
            target_namespace='ENSP',
            on_left='protein',
            on_right='incoming',
            keep_cols=['incoming','converted','name'],
            how='left',
            **kwargs):
    if id_map is None:
        id_map = get_id_map(ids=df[on_left].unique().tolist(), 
                            target_namespace=target_namespace,
                            keep_cols=keep_cols,
                            rename_converted=True) 
        id_map.rename(columns={on_right: on_left,
                               'name':'HGNC'}, 
                      inplace=True)  
    overlap_cols = list(set(df.columns) & set(id_map.columns) - set([on_left, on_right]))
    if len(overlap_cols) > 0:
        print('Warning: Dropping columns in df and id_map overlap:',",".join(overlap_cols))
        df = df.drop(columns=overlap_cols)
    return df.merge(id_map, 
                    on=on_left, 
                    how=how,
                    **kwargs)
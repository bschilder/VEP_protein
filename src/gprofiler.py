
import os
import pandas as pd
import pooch
from gprofiler import GProfiler

import src.utils as utils


def get_id_map(ids, 
               target_namespace='ENSP',
               organism='hsapiens',
               unique_only=True,
               drop_na=['incoming','converted'],
               rename_converted=False,
               keep_cols=None,
               rows_per_id=None,
               as_dict=False,
               cache=pooch.os_cache('gprofiler'),
               force=False,
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
    ids = utils.as_list(ids)

    # Get the ID mapping 
    query = [x for x in ids if pd.notna(x)]
    if unique_only:
        query = list(set(query))
    if verbose:
        print(f'Running gprofiler.convert with {len(query)} queries.')

    # Create save path
    if cache is not None: 
        save_path = os.path.join(cache, 
                                 f'idmap_{utils.as_checksum("".join(query)+"_"+target_namespace)}.csv.gz')
        if os.path.exists(save_path) and not force:
            if verbose:
                print(f"Loading ID mapping from: '{save_path}'")
            id_map = pd.read_csv(save_path)
            cache = None
        else:
            id_map = None
    else:
        id_map = None

    # Get the ID mapping
    if id_map is None:
        gp = GProfiler(return_dataframe=True)
        id_map = gp.convert(organism=organism,
                            query=query,
                            target_namespace=target_namespace,
                            **kwargs)
    # Save the ID mapping
    if cache is not None:
        os.makedirs(cache, exist_ok=True)
        if verbose:
            print(f"Saving ID mapping to: '{save_path}'")
        id_map.to_csv(save_path, index=False) 

    # Filter and process df
    if drop_na is not None:
        if verbose:
            print(f"Dropping rows with NA in {drop_na}.")
        # Replace 'None' with None
        for col in drop_na:
            id_map.loc[id_map[col]=='None', col] = pd.NA
        # Drop rows with NA in drop_na
        id_map = id_map.dropna(subset=drop_na)
    if keep_cols is not None:
        if verbose:
            print(f"Dropping all but {len(keep_cols)} columns.")
        id_map = id_map[keep_cols].drop_duplicates()
    if rows_per_id is not None:
        if verbose:
            print(f"Keeping only the first {rows_per_id} rows per incoming ID.")
        id_map = id_map.groupby('incoming').head(rows_per_id)
    
    # Return a dictionary
    if as_dict is not False:
        if as_dict == -1:
            id_dict = dict(zip(id_map['converted'], id_map['incoming']))
        elif as_dict == 1:
            id_dict = dict(zip(id_map['incoming'], id_map['converted']))
        return id_dict
    
    # Rename cols
    if rename_converted:
        id_map.rename(columns={'converted': target_namespace}, inplace=True)
    
    # Return
    return id_map

def map_ids(df, 
            id_map=None,
            target_namespace='ENSP',
            rows_per_id=None,
            on_left='protein',
            on_right='incoming',
            keep_cols=['incoming','converted','name'],
            how='left',
            force=False,
            cache=pooch.os_cache('gprofiler'),
            verbose=False,
            **kwargs):
    """
    Map IDs to a target namespace.

    Args:
        df: pandas DataFrame with IDs to map
        id_map: pandas DataFrame with ID mapping
        target_namespace: namespace to map to
        on_left: column name to map from
        on_right: column name to map to
        keep_cols: columns to keep in the ID mapping
        how: how to merge the DataFrame
        **kwargs: additional arguments passed to pandas.merge

    Returns:
        pandas DataFrame with the mapped IDs
    
    Example:
        >>> map_ids(df, target_namespace='ENSP')
    """
    assert on_left in df.columns, f"Column '{on_left}' not found in DataFrame"

    existing_cols = utils.intersect(df.columns, [target_namespace,'incoming','converted','name'])
    if len(existing_cols) > 0:
        if verbose:
            print('Warning: Dropping columns in df and id_map overlap:',",".join(existing_cols))        
        df = df.drop(columns=existing_cols)

    if id_map is None:
        # Get the ID mapping
        id_map = get_id_map(ids=df[on_left].unique().tolist(), 
                            target_namespace=target_namespace,
                            keep_cols=keep_cols,
                            rename_converted=True,
                            rows_per_id=rows_per_id,
                            force=force,
                            cache=cache,
                            verbose=verbose) 
        id_map.rename(columns={on_right: on_left,
                               'name':'HGNC'}, 
                      inplace=True)  
    
    assert target_namespace in id_map.columns, f"Column '{target_namespace}' not found in ID mapping"
    
    # Save the ID mapping
    if verbose:
        print(f"Mapping IDs to: {target_namespace}")
    overlap_cols = list(set(df.columns) & set(id_map.columns) - set([on_left, on_right]))
    if len(overlap_cols) > 0:
        if verbose:
            print('Warning: Dropping columns in df and id_map overlap:',",".join(overlap_cols))
        df = df.drop(columns=overlap_cols)
    
    # Merge the ID mapping
    return df.merge(id_map, 
                    on=on_left, 
                    how=how,
                    **kwargs)


def map_and_filter(df1, 
                   df2, 
                   input_col1=None,
                   input_col2=None,
                   target_namespace='ENSP',
                   rows_per_id=(1,1), 
                   verbose=True): 
    """
    Map and filter two DataFrames.
    """
    # Convert rows_per_id to tuple of length 2
    if len(rows_per_id) == 1:
        rows_per_id = (rows_per_id, rows_per_id)

    # Convert lists to DataFrames
    if isinstance(df1, list):
        if input_col1 is None:
            input_col1 = 'id'
        df1 = utils.list_to_df(df1, cols=[input_col1])
    if isinstance(df2, list):
        if input_col2 is None:
            input_col2 = 'id'
        df2 = utils.list_to_df(df2, cols=[input_col2])

    assert input_col1 in df1.columns, f"Column '{input_col1}' not found in df1"
    assert input_col2 in df2.columns, f"Column '{input_col2}' not found in df2"
    
    # Map the IDs
    if verbose:
        print(f"Mapping df1 IDs to: {target_namespace}")
    id_map1 = map_ids(df=df1,
                      on_left=input_col1,
                      target_namespace=target_namespace,
                      verbose=verbose)
    if verbose:
        print(f"Mapping df2 IDs to: {target_namespace}")
    id_map2 = map_ids(df=df2,
                     on_left=input_col2,
                     target_namespace=target_namespace,
                     verbose=verbose)
    
    shared_ids = utils.intersect(id_map1[target_namespace].unique(),
                                 id_map2[target_namespace].unique())
    id_map1 = id_map1[id_map1[target_namespace].isin(shared_ids)]
    id_map2 = id_map2[id_map2[target_namespace].isin(shared_ids)]
    
    # Get the first occurence of each experiment_id (to avoid artifacts of ID mapping)
    if rows_per_id is not None:
        if rows_per_id[0] is not None:
            print("df1: Keeping only the first",rows_per_id[0],"row(s) per ID.")
            id_map1 = id_map1.groupby([input_col1]).head(rows_per_id[0])
        if rows_per_id[1] is not None:
            print("df2: Keeping only the first",rows_per_id[1],"row(s) per ID.")
            id_map2 = id_map2.groupby([input_col2]).head(rows_per_id[1])

    return id_map1, id_map2
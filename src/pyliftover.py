from pyliftover import LiftOver


def liftover(df, 
            from_build='hg19', 
            to_build='hg38', 
            chrom_col='chrom', 
            position_cols  =['start', 'end'],
            force=False):
    """
    Convert genomic coordinates between different genome builds using pyliftover.
    
    Args:
        df (pd.DataFrame): DataFrame containing genomic coordinates
        from_build (str): Source genome build (default: 'hg19')
        to_build (str): Target genome build (default: 'hg38') 
        chrom_col (str): Chromosome column name (default: 'chrom')
        position_cols (list): Position column names (default: ['start', 'end'])
        force (bool): Whether to overwrite existing columns (default: False)
        
    Returns:
        pd.DataFrame: DataFrame with converted coordinates

    Example:
        df = pd.DataFrame({'chrom': ['chr17'], 'start': [41196312], 'end': [41277381]})
        df = liftover(df)
        df
    """ 
    # Initialize the liftover converter
    lo = LiftOver(from_build, to_build) 
    
    for col in position_cols:
        # Store original positions 
        new_pos_col = f'{col}_{to_build}'
        if new_pos_col in df.columns and not force:
            continue 
        new_positions = []
        for i,row in df.iterrows():
            chrom = row[chrom_col]
            pos = row[col]
            # Convert each position
            converted = lo.convert_coordinate("chr"+str(chrom).replace("chr",""), pos) 
            # print(converted)
            if converted:
                new_positions.append(converted[0][1])
            else:
                new_positions.append(pos)
        df[new_pos_col] = new_positions 
    return df


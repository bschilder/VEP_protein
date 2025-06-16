from pyliftover import LiftOver


def liftover(df, 
            from_build='hg19', 
            to_build='hg38', 
            chrom_col='chrom', 
            position_cols  =['start', 'end']):
    """
    Convert genomic coordinates between different genome builds using pyliftover.
    
    Args:
        df (pd.DataFrame): DataFrame containing genomic coordinates
        from_build (str): Source genome build (default: 'hg19')
        to_build (str): Target genome build (default: 'hg38') 
        chrom_col (str): Chromosome column name (default: 'chr17')
        pos_col (str): Position column name (default: 'position')
        
    Returns:
        pd.DataFrame: DataFrame with converted coordinates

    Example:
        df = pd.DataFrame({'chrom': ['chr17'], 'start': [41196312], 'end': [41277381]})
        df = liftover(df)
        df
    """ 
    # Initialize the liftover converter
    lo = LiftOver(from_build, to_build)
    
    # Store original positions
    for col in position_cols:
        df[f'{col}_{from_build}'] = df[col]
    
    # Convert each position
    new_positions = []
    for col in position_cols:
        for pos in df[f'{col}_{from_build}']:
            converted = lo.convert_coordinate(chrom_col, pos)
        if converted:
            new_positions.append(converted[0][1])
        else:
            new_positions.append(pos)
            
    # Update positions in dataframe
    for col in position_cols:
        df[col] = new_positions[col]
    
    return df


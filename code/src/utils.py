def is_pd(x):
    """
    Check if the input object is a pandas DataFrame.

    Parameters:
    x (object): The object to be checked.

    Returns:
    bool: True if the object is a pandas DataFrame, False otherwise.
    """
    import pandas as pd
    return isinstance(x, pd.DataFrame)

def as_list(x,
            type_func=None):
    """
    Convert a string to a list when possible.

    Args:
        x: The input value to be converted to a list.
        type_func: (optional) A function to apply to each element of the list.

    Returns:
        A list containing the converted value(s).
    """
    if is_pd(x):
        return [x] 
    if x == None:
        return x
    if type(x) != list:
        x = [x]
    if type_func != None:
        x = [type_func(y) for y in x]
    return x

def intersect(x,y):
    return list(set(x) & set(y))

def list_vcf(dir='/grid/koo/home/schilder/projects/GenomeEncoder/data/1KG/vcf_formatted/*.vcf.gz'):
    # List all VCF files in the 1KG directory
    import glob
    vcf_files = glob.glob(dir)
    print(len(vcf_files),"VCF files found.")
    return vcf_files


def get_canonical_transcripts():
    import pandas as pd
    knownCanonical = pd.read_csv("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/knownCanonical.txt.gz", 
                                 sep="\t", 
                                 header=None)
    # Extract the TranscriptId from the knownCanonical file
    knownCanonical['TranscriptId'] = knownCanonical[4].str.split('.').str[0]
    return knownCanonical

def add_codon_buffer(seq, codon_buffer='N'):
    if codon_buffer is not None:
        seq += codon_buffer * (3 - len(seq) % 3)
    return seq
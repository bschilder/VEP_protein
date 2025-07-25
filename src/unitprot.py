# https://biopython.org/docs/1.75/api/Bio.SeqIO.UniprotIO.html?highlight=uniprot#module-Bio.SeqIO.UniprotIO


from Bio import SeqIO
from io import StringIO
import requests
import pandas as pd

def import_uniprot(protein_id, 
                   parser="uniprot-xml",
                   verbose=False):
    """
    Import protein data from UniProt database for a given protein ID.
    
    Args:
        protein_id (str): UniProt protein identifier
        verbose (bool, optional): If True, prints record details. Defaults to False.
        
    Returns:
        Bio.SeqIO.Record: Parsed UniProt XML records containing protein data
        
    Example:
        >>> records = import_uniprot("P12345", verbose=True)
        >>> for record in records:
        ...     print(record.id)
    """
    response = requests.get(f"https://rest.uniprot.org/uniprotkb/{protein_id}.xml")
    if response.status_code == 200:
        records = SeqIO.parse(StringIO(response.text), parser)
        if verbose:
            for record in records:
                print(record.id)
                print(record.name)
                print(record.seq)
                print(record.description)
                # ... other attributes like feature 
        return records
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")

def get_seqs(protein_id):
    """
    Retrieve protein sequences from UniProt for a given protein ID.
    NOTE: You must reimport the records to get the sequences. If you try accessing them after the records have already been called by another function, they will be empty.
    
    Args:
        protein_id (str): UniProt protein identifier
        
    Returns:
        list: List of Bio.Seq objects containing protein sequences
        
    Example:
        >>> seqs = get_seqs("P12345")
        >>> print(len(seqs))  # Number of sequences retrieved
    """
    records = import_uniprot(protein_id)
    seqs = []
    for record in records:
        seqs.append(record.seq)
    return seqs

def get_features(protein_id):
    """
    Retrieve and process protein features from UniProt for a given protein ID.
    NOTE: You must reimport the records to get the features. If you try accessing them after the records have already been called by another function, they will be empty.
    
    Args:
        protein_id (str): UniProt protein identifier
        
    Returns:
        pd.DataFrame: DataFrame containing protein features with columns:
            - id: Feature identifier
            - type: Feature type
            - start: Start position
            - end: End position
            - length: Feature length (end - start)
            - Additional columns from feature qualifiers
            
    Example:
        >>> features = get_features("P12345")
        >>> print(features.columns)
    """
    records = import_uniprot(protein_id)
    features_df = []
    for record in records:
        for feature in record.features:
            fdict = {}
            fdict["id"] = feature.id
            fdict["type"] = feature.type
            fdict["start"] = feature.location.start
            fdict["end"] = feature.location.end 
            fdict.update(feature.qualifiers)
            features_df.append(fdict)

    features_df = pd.DataFrame(features_df)
    features_df["length"] = features_df["end"] - features_df["start"]

    return features_df


def get_dbxrefs(protein_id):
    """
    Retrieve database cross-references from UniProt for a given protein ID.
    
    Args:
        protein_id (str): UniProt protein identifier
        
    Returns:
        pd.DataFrame: DataFrame containing database cross-references with columns:
            - dbxref: Original cross-reference string
            - db: Database name
            - id: Database identifier
            
    Example:
        >>> dbxrefs = get_dbxrefs("P12345")
        >>> print(dbxrefs.columns)
    """
    records = import_uniprot(protein_id)
    dbxrefs = []
    for record in records:       
        dbxrefs.append(list(record.dbxrefs))
    dbxrefs = pd.DataFrame(dbxrefs, index=["dbxref"]).T
    # Split the dbxref column into two columns at the first occurrence of ":"
    dbxrefs[['db', 'id']] = dbxrefs['dbxref'].str.split(':', n=1, expand=True)
    return dbxrefs 
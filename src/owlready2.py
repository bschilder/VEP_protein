import pooch
import owlready2 as owl
from tqdm import tqdm
import src.utils as utils
import pandas as pd

def process_id(id: str):
    """
    Processes an ID to remove the MONDO: prefix and replace : with _
    """
    return f"*{id.replace(":", "_")}"

def get_onto(onto_url: str,
             known_hash: str,
             verbose: bool = False):
    """
    Loads an ontology from a file.
    """

    onto_path = pooch.retrieve(onto_url, 
                                known_hash=known_hash, 
                                progressbar=verbose)

    # Load the ontology
    onto = owl.get_ontology(onto_path).load()
    return onto


def get_onto_mondo(verbose: bool = False):
    """
    Loads the MONDO ontology.
    """
    return get_onto("http://purl.obolibrary.org/obo/mondo.owl", 
                    known_hash="faf39917bca366b5b7ee014d499e5004abfd67e2e136c32d535052a39e882d0b", 
                    verbose=verbose)

def get_onto_icdo(verbose: bool = False):
    """
    Loads the ICD-10 ontology.
    """
    return get_onto("https://raw.githubusercontent.com/icdo/ICDO/master/src/ontology/icdo.owl", 
                    known_hash="1e20c40d97bbeae1531a063632f948c4c2617eb72e097904a5a861e8bf67bae4", 
                    verbose=verbose)

def get_icd10_codes(url="https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2020/icd10cm_codes_2020.txt"):
    """
    Loads the ICD-10 codes.
    """
    df = pd.read_csv(url, sep="\t", header=None)[0].str.split(' ', n=1, expand=True)
    df.columns = ["code", "label"]
    df["code"] = df["code"].str.strip()
    df["label"] = df["label"].str.strip()
    return df

def get_id_map(onto,  
               first_only: bool = True,
                verbose: bool = True):
    """
    Maps IDs to names.
    """ 
    
    # Get all entities and their labels at once
    all_entities = list(onto.classes())
    id_map = {entity.name.replace("_", ":"): entity.label.first() if first_only else entity.label for entity in all_entities}
    return id_map

def map_ids_to_labels(ids,
                      onto,
                      verbose: bool = False):
    """
    Maps IDs to labels.
    """
    id_map = get_id_map(onto,
                        first_only=True,
                        verbose=verbose)
    return [id_map[id] for id in ids if id in id_map]


def get_ancestors(onto,
                  id: str,
                  lvl: int = None,
                  prefix: str = None,
                  return_ids: bool = False,
                  verbose: bool = False):
    """
    Gets the ancestors of an ID.
    """
    # Recursion
    if isinstance(id, list):
        if verbose:
            print(f"Getting ancestors for {id}")
        return [get_ancestors(onto, i, lvl, prefix, return_ids, verbose) for i in id]
    else:
        if id is None:
            return None
        if verbose:
            print(f"Getting ancestors for {id}")
        onto_res = onto.search_one(iri = process_id(id))
        if onto_res is None:
            return None
                                   
        anc = list(onto_res.ancestors())
        anc.reverse()


        if prefix is not None:
                anc = [a for a in anc if prefix+"_" in a.name]

        if return_ids:
            anc = [a.name.replace("_", ":") for a in anc]
            
        
        if lvl is not None:
            anc = anc[lvl]


        if verbose:
            print(f"Ancestors of {id}: {anc}")

        return anc

def get_mrca(onto,
              id1, 
              id2,
              verbose: bool = False):
    """
    Finds the Most Recent Common Ancestor using owlready2.
    Args:
        onto: Ontology object
        id1: First ID
        id2: Second ID
    Returns:
        Most Recent Common Ancestor

    Example:
        mrca = find_mrca_owlready2(onto, ids[0], ids[-1])
    """
    class1 = onto.search_one(iri = process_id(id1))
    class2 = onto.search_one(iri = process_id(id2))

    if not class1 or not class2:
        return None # Handle cases where classes are not found

    ancestors1 = set(class1.ancestors())
    ancestors2 = set(class2.ancestors())

    common_ancestors = ancestors1.intersection(ancestors2)

    if not common_ancestors:
        return None # Handle no common ancestors

    # Find the most recent (lowest) common ancestor
    mrca = sorted(common_ancestors, key=lambda x: len(list(x.ancestors())), reverse=True)[0]

    return mrca


def get_mrca_counts(onto,
                    ids,
                    verbose: bool = True):
    """
    Finds the Most Recent Common Ancestor using owlready2.
    """
    from tqdm import tqdm
    # Create a dictionary to store MRCA counts
    mrca_counts = {}

    # Iterate through all pairs of IDs
    for i in tqdm(range(len(ids)), 
                  desc="Computing MRCA counts", 
                  disable=not verbose, 
                  leave=False):
        for j in range(i+1, len(ids)):
            id1 = ids[i]
            id2 = ids[j]
            
            # Find MRCA for this pair
            mrca = get_mrca(onto, id1, id2)
            
            if mrca:
                # Get the MONDO ID from the IRI
                mrca_id = mrca.iri.split('/')[-1].replace('_', ':')
                
                # Update count
                mrca_counts[mrca_id] = mrca_counts.get(mrca_id, 0) + 1

    # Sort MRCAs by frequency
    sorted_mrcas = sorted(mrca_counts.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_mrcas



import sys
sys.path.append("code")
from src.utils import as_list, one_only

def get_ontology(url, **kwargs):
    # Import SO ontology
    import owlready2
    return owlready2.get_ontology(url, **kwargs).load()

def get_sequence_ontology(**kwargs):
    return get_ontology("http://purl.obolibrary.org/obo/so.owl", **kwargs)

def default_ontology(**kwargs):
    return get_sequence_ontology(**kwargs)

def _multi_handler(lst, multi, sep="."):
    # lst = as_list(lst)
    if multi == "join":
        return sep.join(lst)
    elif multi == "first":
        return lst[0]
    elif multi == "all":
        return lst

def _return_as(x, return_as, multi=["join","first","all"][0]):  
    import owlready2
    multi = one_only(multi)
    if not isinstance(x, owlready2.entity.ThingClass):
        raise ValueError(f"Entity must be an owlready2.entity.ThingClass, got {type(x)}")
    if return_as == 'entity':
        return x
    elif return_as == 'label':
        return _multi_handler(x.label, multi)
    elif return_as == 'id':
        return _multi_handler(x.id, multi)
    elif return_as == 'id|label':
        return _multi_handler(x.id, multi)+"|"+_multi_handler(x.label, multi)
    else:
        raise ValueError(f"Invalid return_as: {return_as}")

def _get_kin(label_or_id, 
            ont, 
            kin_type=["descendants","ancestors"],
            include_self=True,
            return_as=['entity','label','id','id|label'],
            verbose=True):
    from functools import partial
    kin_type = one_only(kin_type) 
    return_as = one_only(return_as)
    # recursion
    if isinstance(label_or_id, list) and len(label_or_id)==1:
        label_or_id = label_or_id[0]
    if isinstance(label_or_id, list):
        return {l: _get_kin(l,
                            ont=ont,
                            include_self=include_self,
                            return_as=return_as,
                            verbose=verbose)
                for l in label_or_id}
    # get all descendant terms of 'coding_sequence_variant'
    if is_label_or_id(label_or_id, ont) == 'label':
        entity = ont.search_one(label=label_or_id)
    else:
        entity = ont.search_one(id=label_or_id)
    # if no ancestor is found, return just the ancestor label
    if entity is None:
        if verbose:
            print(f"No entity found for '{label_or_id}'") 
        if include_self:
            return label_or_id
        else:
            return _return_as(entity, return_as)
    # get descendants or ancestors
    if kin_type == "descendants":
        kin = entity.descendants(include_self=include_self)
    elif kin_type == "ancestors":
        kin = entity.ancestors(include_self=include_self) 
    if verbose:
        print(f"Found {len(kin)} {kin_type} of '{label_or_id}'")
    return list(map(partial(_return_as, return_as=return_as), kin))

def get_descendants(label_or_id,
                    ont=None,
                    include_self=True,
                    return_as=['entity','label','id','id|label'],
                    verbose=True): 
    if ont is None:
        ont = default_ontology()
    return _get_kin(label_or_id,
                    ont=ont,
                    kin_type="descendants",
                    include_self=include_self,
                    return_as=return_as,
                    verbose=verbose)

def get_ancestors(label_or_id,
                  ont=None,
                  include_self=True,
                  return_as=[None,'label','id','id|label'],
                  verbose=True):
    if ont is None:
        ont = default_ontology()
    return _get_kin(label_or_id,
                    ont=ont,
                    kin_type="ancestors",
                    include_self=include_self,
                    return_as=return_as,
                    verbose=verbose)

def get_labels(ont, unnest=True):
    labels = [x.label for x in ont.classes()]
    if unnest:
        from itertools import chain
        labels = list(chain.from_iterable(labels))
    return labels

def get_ids(ont, unnest=True):
    ids = [x.id for x in ont.classes()]
    if unnest:
        from itertools import chain
        ids = list(chain.from_iterable(ids))
    return ids

def is_label_or_id(label_or_id, ont=None):
    if ont is None:
        ont = default_ontology()
    so_labels = get_labels(ont)
    so_ids = get_ids(ont)
    if label_or_id in so_ids:
        return 'id'
    if label_or_id in so_labels:
        return 'label'
    return None
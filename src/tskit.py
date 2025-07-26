import tskit
import tszip
# !conda install -c conda-forge -y tskit tszip

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from tqdm import tqdm
from collections import defaultdict 
from typing import List, Dict, Any, Optional, Iterator, Union
import re

import src.utils as utils

def list_trees(dir: str, suffix: str = ".trees.tsz"):
    """
    List all tree files in a directory with a given suffix.

    Parameters
    ----------
    dir : str
        Directory to search for tree files.
    suffix : str, optional
        File suffix to match (default: ".trees.tsz").

    Returns
    -------
    list of str
        List of file paths matching the suffix in the directory.
    """
    files =  glob.glob(os.path.expanduser(os.path.join(dir, f"*{suffix}")))
    print(f"Found {len(files)} tree file(s) in {dir}")
    return files

def load_tree(tree_file):
    """
    Load a single tree sequence from a compressed file.

    Parameters
    ----------
    tree_file : str
        Path to the tree file.

    Returns
    -------
    tskit.TreeSequence
        Loaded tree sequence object.
    """
    ts = tszip.load(tree_file)

    ## Add new attributes to the TreeSequence object
    ts.file = tree_file
    ts.name = _filename_to_name(tree_file)
    ts.chromosome = _filename_to_chromosome(tree_file)

    return ts

def load_trees(tree_files):
    """
    Load multiple tree sequences from a list of files.

    Parameters
    ----------
    tree_files : list of str
        List of tree file paths.

    Returns
    -------
    list of tskit.TreeSequence
        List of loaded tree sequence objects.
    """
    return [load_tree(tree_file) for tree_file in tree_files]

def get_individuals_metadata(ts, metadata_filters={}):
    """
    Get the metadata for all individuals in a tree sequence.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence to analyze.
    metadata_filters : dict, optional
    """
    # Get the metadata for all individuals
    ind_metadata = {i.id: json.loads(i.metadata) for i in ts.individuals()}

    # Filter the individuals by the metadata filters
    if metadata_filters is not None:
        ind_metadata = {k: v for k, v in ind_metadata.items()
                        if all(
                            (v.get(fk, None) in fval if isinstance(fval, list) else v.get(fk, None) == fval)
                            for fk, fval in metadata_filters.items()
                        )
                        }
    return ind_metadata

def get_superpop_to_nodes(ts, metadata_field="super_population"):
    """
    Build a mapping from superpopulation (or other population metadata field) to sample node IDs.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence to analyze.
    metadata_field : str, optional
        The population metadata field to use as the group key (default: "super_population").
        Other options include:
        - "name": the population name
        - "region": the region of the population

    Returns
    -------
    dict
        Dictionary mapping superpopulation names to lists of sample node IDs.
    """ 
    pop_metadata = [json.loads(pop.metadata) for pop in ts.populations()]
    superpop_to_nodes = defaultdict(list)
    for ind in ts.individuals(): 
        for node_id in ind.nodes:
            node = ts.node(node_id)
            pop_idx = node.population
            superpop = pop_metadata[pop_idx].get(metadata_field, None)
            if superpop is not None:
                superpop_to_nodes[superpop].append(node_id)
    return superpop_to_nodes

def filter_populations(ts, 
                       populations, 
                       metadata_field="name",
                       return_sample_idx=False):
    """
    Returns a list of population IDs from the tree sequence whose metadata field matches any value in the given populations list.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence containing the populations.
    populations : list
        A list of values to match against the specified metadata field.
    metadata_field : str, optional
        The metadata field to use for filtering populations (default: "name").
    return_sample_idx : bool, optional
        If True, return the sample indices instead of the population IDs.

    Returns
    -------
    list
        List of population IDs whose metadata field matches any value in `populations`.
    """
    print(f"Filtering populations")
    import json
    pop_idx = [p.id for p in ts.populations() if json.loads(p.metadata).get(metadata_field, None) in populations]
    if return_sample_idx:
        return np.unique(np.concatenate([ts.samples(population=idx) for idx in pop_idx]))
    else:
        return pop_idx

def get_group_distance(ts, 
                        metadata_field="super_population",
                        func="divergence",
                        show_plot=True,
                        populations=None,
                        figsize=(8, 6),
                        plot_title=None,
                        dist_kwargs={},
                        plot_kwargs={}):
    """
    Compute and visualize genetic relatedness between groups (e.g., superpopulations) in a tree sequence.

    This function computes a pairwise genetic distance matrix (mean divergence) between groups,
    performs hierarchical clustering, and optionally plots a dendrogram.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence to analyze.
    metadata_field : str, optional
        The population metadata field to use as the group key (default: "super_population").
    func: str, optional
        The function to use to compute the distance between groups.
        - "divergence": compute mean pairwise divergence between all pairs of nodes from the two superpops
        - "relatedness": compute genetic relatedness between all pairs of nodes from the two superpops
        - "Fst": compute Fst between all pairs of nodes from the two superpops
    show_plot : bool, optional
        Whether to display a dendrogram plot (default: True).
    figsize : tuple, optional
        The size of the figure for the dendrogram plot (default: (8, 4)).
    plot_title : str, optional
        The title of the dendrogram plot (default: None).
    kwargs : dict, optional
        Additional keyword arguments to pass to the dendrogram plot.

    Returns
    -------
    dict
        Dictionary containing:
            - 'dist_matrix': numpy.ndarray, pairwise distance matrix
            - 'condensed_dist': numpy.ndarray, condensed distance matrix for clustering
            - 'Z': linkage matrix from hierarchical clustering
            - 'superpops': list of group names
            - 'superpop_samples': dict mapping group names to sample node IDs
            - 'superpop_to_nodes': dict mapping group names to all node IDs
            - 'pop_metadata': list of population metadata dicts
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform


    # Filter tree sequence to only include the specified populations
    if populations is not None:
        samples_idx = filter_populations(ts, 
                                         populations=populations, 
                                         metadata_field="name", 
                                         return_sample_idx=True)
        ts = ts.simplify(samples_idx, 
                         filter_sites=True, 
                         filter_populations=True, 
                         filter_nodes=True,
                         filter_individuals=True)

    # Get population metadata for all populations in the tree sequence
    pop_metadata = [json.loads(pop.metadata) for pop in ts.populations()]

    # Build mapping from superpop name to list of sample node IDs
    superpop_to_nodes = get_superpop_to_nodes(ts, metadata_field=metadata_field)

    # Only keep superpops with at least 2 samples
    superpop_samples = {k: v for k, v in superpop_to_nodes.items() if len(v) > 1}

    # Compute pairwise genetic distances (mean divergence) between superpopulations
    superpops = list(superpop_samples.keys()) 
    sample_sets = [v for v in superpop_to_nodes.values()]
    indexes = [(i, j) for i in range(len(sample_sets)) for j in range(i + 1, len(sample_sets))]
   
    # Compute genetic distance
    print(f"Computing {func}...")
    if func == "divergence":
        # "Divergence" here refers to the mean genetic distance between all pairs of nodes from two groups.
        # It quantifies the average number of differences (e.g., mutations) between samples in different groups.
        # The following computes the mean pairwise divergence between all pairs of nodes from the two superpopulations.
        dist = ts.divergence(sample_sets=sample_sets, indexes=indexes, **dist_kwargs)
    elif func == "relatedness":
        dist = ts.genetic_relatedness(sample_sets=sample_sets, indexes=indexes, **dist_kwargs)
        # Invert the distances so that higher values mean more relatedness (if appropriate).
        # Here, we invert by subtracting from the maximum, but avoid division by zero.
        dist = np.max(dist) - dist
    elif func == "Fst":
        # Fst (fixation index) is a measure of population differentiation due to genetic structure.
        # It quantifies the proportion of genetic variance that can be explained by population differences.
        # Here, we compute pairwise Fst between all pairs of sample sets (superpopulations).
        dist = ts.Fst(sample_sets=sample_sets, indexes=indexes, **dist_kwargs)
        # Normalize so that all distances are non-negative (shift if necessary)
        min_dist = np.min(dist)
        if min_dist < 0:
            dist = dist - min_dist
    else:
        raise ValueError(f"Invalid function: {func}")

    # Xdist = np.zeros((len(superpops), len(superpops)))
    # for idx, (i, j) in enumerate(indexes):
    #     Xdist[i, j] = dist[idx]
    #     Xdist[j, i] = dist[idx] 

    # Perform hierarchical clustering 
    Xdist = pd.DataFrame(
        squareform(dist),
        index=superpops,
        columns=superpops
    )
    
    print("Computing linkage matrix...")
    Z = linkage(dist, method='ward')

    # Plot dendrogram 
    import matplotlib
    orig_backend = matplotlib.get_backend()
    plt.ioff()  # Turn interactive mode off to suppress showing
    fig = plt.figure(figsize=figsize)
    # The order of labels for dendrogram should match the order of superpops (rows/cols of Xdist)
    # The dendrogram function will automatically order the leaves according to the clustering
    dendrogram(Z, labels=superpops, leaf_rotation=90, **plot_kwargs)
    if plot_title is None:
        plt.title(f"Dendrogram of {metadata_field} ({func})")
    else:
        plt.title(plot_title)
    plt.ylabel(f"Genetic distance ({func})")
    plt.xlabel(f"{metadata_field}")
    plt.tight_layout()
    if show_plot:
        plt.show()
    plt.ion()  # Restore interactive mode

    return {
        'dist': dist,
        'Xdist': Xdist,
        'Z': Z,
        'superpops': superpops,
        'superpop_samples': superpop_samples,
        'superpop_to_nodes': superpop_to_nodes,
        'pop_metadata': pop_metadata,
        'fig': fig
    }

def plot_superpop_dendrogram(
    X, 
    og_meta=None, 
    palette=None, 
    groupby_col="Population Description", 
    sample_col="sample",
    site_cols=None, 
    figsize=(8, 10), 
    title="Hierarchical clustering dendrogram of superpopulations (by group centroid)",
    ylabel="Ward distance (by group centroid)"
):
    """
    Compute and plot a dendrogram of superpopulation centroids in high-dimensional space.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with samples as rows and features as columns.
    og_meta : pd.DataFrame
        Metadata DataFrame with at least columns for sample IDs and group labels.
    palette : dict, optional
        Mapping from group label to color hex code.
    groupby_col : str, optional
        Column in og_meta to group by (default: "Population Description").
    sample_col : str, optional
        Column in X to use as sample IDs.
    site_cols : list, optional
        List of columns in X to use as features. If None, uses all columns.
    figsize : tuple, optional
        Figure size for the plot.
    title : str, optional
        Title for the dendrogram plot.
    ylabel : str, optional
        Y-axis label for the dendrogram plot.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram

    # Load metadata if not provided
    if og_meta is None:
        import src.onekg as og
        og_meta = og.get_sample_metadata()
    if site_cols is None:
        site_cols = X.columns.tolist()
    if palette is None: 
        palette = utils.get_superpop_palette()

    # Compute centroids
    group_centroids = (
        X.reset_index()
         .merge(og_meta, on=sample_col, how="left")
         .groupby(groupby_col)[site_cols]
         .mean()
    )

    # Hierarchical clustering
    Z_group = linkage(group_centroids.values, method='ward')
    group_labels = group_centroids.index.tolist()
    group_colors = [palette.get(sp, "#333333") for sp in group_labels]

    # Plot dendrogram
    fig, ax = plt.subplots(figsize=figsize)
    dendro = dendrogram(
        Z_group,
        labels=group_labels,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=None
    )

    # Color tick labels
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center', va='top', fontsize=10)
    xticks = ax.get_xticklabels()
    for tick, sp in zip(xticks, dendro['ivl']):
        tick.set_color(palette.get(sp, "#333333"))

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    return {'fig':fig, 'ax':ax, 
            'dendro':dendro, 
            'group_centroids':group_centroids, 
            'group_labels':group_labels, 
            'group_colors':group_colors,
            'Z_group':Z_group}



#####################################
##### Tree sequence collections #####
#####################################

def _map_trees(self, func, *args, **kwargs):
    """
    Apply a function to each tree sequence and return results.
    """
    return [func(ts, *args, **kwargs) for _, ts in self.items()]

def _map_trees_with_names(
    self,
    func,
    *args,
    name_by_chrom=True,
    show_progress=False,
    verbose=True,
    **kwargs
):
    """
    Applies a given function to each tree sequence in the collection and returns the results
    as a dictionary keyed by either chromosome or original name.

    Parameters
    ----------
    self : object
        The collection object containing tree sequences, expected to have an .items() method.
    func : callable
        The function to apply to each tree sequence. Should accept a tskit.TreeSequence as its first argument.
    *args :
        Additional positional arguments to pass to `func`.
    name_by_chrom : bool, optional
        If True (default), use the chromosome name (parsed from the filename) as the key in the result dictionary.
        If False, use the original name from the collection.
    show_progress : bool, optional
        If True, display a progress bar using tqdm (default: True).
    verbose : bool, optional
        If True, print additional information during processing (default: False).
    **kwargs :
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    dict
        A dictionary mapping chromosome names (or original names) to the result of `func` applied to each tree sequence.

    Notes
    -----
    - The function expects the collection to have an .items() method yielding (name, tree_sequence) pairs.
    - If `name_by_chrom` is True, the key is determined by `_filename_to_chromosome(name)`.
    - If `show_progress` is True, a tqdm progress bar is shown.
    """
    result = {}
    items = list(self.items())
    iterator = tqdm(items, desc="Processing trees") if show_progress else items
    for name, ts in iterator:
        if verbose:
            print(f"Processing: {name}")
        key = _filename_to_chromosome(name) if name_by_chrom else name
        result[key] = func(ts, *args, **kwargs)
    return result

def _filter_trees(self, predicate):
    """
    Filter tree sequences based on a predicate function.
    """
    filtered_names = []
    filtered_seqs = []
    for name, ts in self.items():
        if predicate(ts):
            filtered_names.append(name)
            filtered_seqs.append(ts)
    return filtered_names, filtered_seqs

def _get_summary_stats(self):
    """
    Get summary statistics for all tree sequences in the collection.
    """ 
    stats = []
    for name, ts in self.items():
        stats.append({
            'name': name,
            'chromosome': _filename_to_chromosome(name),
            'num_samples': ts.num_samples,
            'num_nodes': ts.num_nodes,
            'num_edges': ts.num_edges,
            'num_sites': ts.num_sites,
            'num_mutations': ts.num_mutations,
            'num_populations': ts.num_populations,
            'sequence_length': ts.sequence_length,
            'num_trees': ts.num_trees
        })
    return pd.DataFrame(stats)

def _iterate_all_trees(self):
    """
    Iterate over all individual trees across all tree sequences.
    """
    for name, ts in self.items():
        for tree_idx, tree in enumerate(ts.trees()):
            yield name, tree_idx, tree

def _iterate_all_sites(self):
    """
    Iterate over all sites across all tree sequences.
    """
    for name, ts in self.items():
        for site_idx, site in enumerate(ts.sites()):
            yield name, site_idx, site

def _iterate_all_mutations(self):
    """
    Iterate over all mutations across all tree sequences.
    """
    for name, ts in self.items():
        for mut_idx, mutation in enumerate(ts.mutations()):
            yield name, mut_idx, mutation

def _filename_to_name(filename: str):
    """
    Extract name from filename.
    """
    return os.path.basename(filename).replace('.trees.tsz', '')

def _filename_to_chromosome(filename: str):
    """
    Extract chromosome name from filename.
    """
    return re.search(r'(chr\d+)', filename).group(1) if re.search(r'(chr\d+)', filename) else None

class TreeSequenceCollection:
    """
    A collection class for multiple tskit TreeSequence objects that provides
    easy iteration and batch processing capabilities.
    """
    def __init__(self, tree_sequences: list, 
                 names: Optional[list] = None,
                 metadata: Optional[dict] = None):
        self.tree_sequences = tree_sequences
        self.names = names if names is not None else [f"tree_{i}" for i in range(len(tree_sequences))]
        self.chromosomes = [_filename_to_chromosome(n) for n in self.names]
        self.metadata = metadata or {}
        if len(self.tree_sequences) != len(self.names):
            raise ValueError("Number of tree sequences must match number of names")

    def __len__(self):
        return len(self.tree_sequences)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx in self.names:
                return self.tree_sequences[self.names.index(idx)]
            elif idx in self.chromosomes:
                return self.tree_sequences[self.chromosomes.index(idx)]
            else:
                raise KeyError(f"Tree sequence '{idx}' not found")
        else:
            return self.tree_sequences[idx]

    def __iter__(self):
        return iter(self.tree_sequences)

    def items(self):
        return zip(self.names, self.tree_sequences)

    def map_trees(self, func, *args, **kwargs):
        return _map_trees(self, func, *args, **kwargs)

    def map_trees_with_names(self, func, *args, **kwargs):
        return _map_trees_with_names(self, func, *args, **kwargs)

    def filter_trees(self, predicate):
        filtered_names, filtered_seqs = _filter_trees(self, predicate)
        return TreeSequenceCollection(filtered_seqs, filtered_names, self.metadata)

    def get_summary_stats(self):
        return _get_summary_stats(self)

    def iterate_all_trees(self):
        return _iterate_all_trees(self)

    def iterate_all_sites(self):
        return _iterate_all_sites(self)

    def iterate_all_mutations(self):
        return _iterate_all_mutations(self)

class LazyTreeSequenceCollection:
    """
    A lazy-loading collection of tree sequences that only loads files when accessed.
    """
    def __init__(self, tree_files: list, 
                 names: Optional[list] = None,
                 metadata: Optional[dict] = None):
        import re
        self.tree_files = tree_files
        self.names = names if names is not None else [_filename_to_name(f) for f in tree_files]
        self.chromosomes = [_filename_to_chromosome(n) for n in self.names]
        self.metadata = metadata or {}
        self._loaded_sequences = {}
        if len(self.tree_files) != len(self.names):
            raise ValueError("Number of tree files must match number of names")

    def __len__(self):
        return len(self.tree_files)

    def __getitem__(self, idx):
        # If idx is a string, it can be a name or a chromosome
        if isinstance(idx, str):
            if idx in self.names:
                file_idx = self.names.index(idx)
            elif idx in self.chromosomes:
                file_idx = self.chromosomes.index(idx)
            else:
                raise KeyError(f"Tree sequence '{idx}' not found")
        else:
            file_idx = idx

        # Load the tree sequence if not already loaded
        if file_idx not in self._loaded_sequences:
            self._loaded_sequences[file_idx] = load_tree(self.tree_files[file_idx])
        
        return self._loaded_sequences[file_idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _ensure_preloaded(self, show_progress: bool = True):
        # Preload all tree sequences if not already loaded
        if len(self._loaded_sequences) < len(self.tree_files):
            self.preload(show_progress=show_progress)

    @property
    def tree_sequences(self):
        self._ensure_preloaded()
        # Return the list of loaded tree sequences in the correct order
        return [self._loaded_sequences[i] for i in range(len(self.tree_files))]

    def items(self):
        self._ensure_preloaded()
        return zip(self.names, self.tree_sequences)

    def map_trees(self, func, *args, **kwargs):
        self._ensure_preloaded()
        return _map_trees(self, func, *args, **kwargs)

    def map_trees_with_names(self, func, *args, **kwargs):
        self._ensure_preloaded()
        return _map_trees_with_names(self, func, *args, **kwargs)

    def filter_trees(self, predicate):
        self._ensure_preloaded()
        filtered_names, filtered_seqs = _filter_trees(self, predicate)
        return TreeSequenceCollection(filtered_seqs, filtered_names, self.metadata)

    def get_summary_stats(self):
        self._ensure_preloaded()
        return _get_summary_stats(self)

    def iterate_all_trees(self):
        self._ensure_preloaded()
        return _iterate_all_trees(self)

    def iterate_all_sites(self):
        self._ensure_preloaded()
        return _iterate_all_sites(self)

    def iterate_all_mutations(self):
        self._ensure_preloaded()
        return _iterate_all_mutations(self)

    def preload(self, idx: list = None, show_progress: bool = True):
        if idx is None:
            indices = list(range(len(self)))
        else:
            indices = [
                self.names.index(i) if isinstance(i, str) and i in self.names
                else self.chromosomes.index(i) if isinstance(i, str) and i in self.chromosomes
                else i
                for i in idx
            ]
        if show_progress:
            tree_sequences = [self[i] for i in tqdm(indices, desc="Preloading tree sequences")]
        else:
            tree_sequences = [self[i] for i in indices]
        # names = [self.names[i] for i in indices] 
        # return TreeSequenceCollection(tree_sequences, names, self.metadata)

    def clear_cache(self):
        self._loaded_sequences.clear()

    def __repr__(self):
        info_lines = [
            f"<LazyTreeSequenceCollection:",
            f"  {len(self)} tree files" + (f"\n  >> {', '.join(self.names[:3])}" + ("..." if len(self.names) > 3 else "") if self.names else ""),
            (
                f"  {len(self.chromosomes) if hasattr(self, 'chromosomes') else 'N/A'} chromosomes"
                + (
                    f"\n  >> {', '.join(self.chromosomes)}"
                    if hasattr(self, 'chromosomes') and self.chromosomes else ""
                )
            ),
            f"  {len(self._loaded_sequences)} loaded",
            f"  metadata keys: {list(self.metadata.keys()) if self.metadata else 'None'}",
            f">"
        ]
        info_lines = [line for line in info_lines if line.strip() != ""]
        return "\n".join(info_lines)

    def __str__(self):
        lines = [
            f"LazyTreeSequenceCollection with {len(self)} tree files" + (f"\n  >> {', '.join(self.names[:5])}" + ("..." if len(self.names) > 5 else "") if self.names else ""),
            (
                f"Number of chromosomes: {len(self.chromosomes) if hasattr(self, 'chromosomes') else 'N/A'}"
                + (
                    f"\n  >> {', '.join(self.chromosomes)}"
                    if hasattr(self, 'chromosomes') and self.chromosomes else ""
                )
            ),
            f"Loaded: {len(self._loaded_sequences)} / {len(self)}",
            f"Metadata keys: {list(self.metadata.keys()) if self.metadata else 'None'}"
        ]
        return "\n".join(lines)

    def filter(self, predicate):
        filtered_names, filtered_seqs = _filter_trees(self, predicate)
        filtered_files = []
        for name in filtered_names:
            idx = self.names.index(name)
            filtered_files.append(self.tree_files[idx])
        return LazyTreeSequenceCollection(filtered_files, filtered_names, self.metadata)

     
def create_tree_collection_from_files(tree_files: list, 
                                     names: Optional[list] = None,
                                     metadata: Optional[dict] = None,
                                     show_progress: bool = True,
                                     lazy: bool = False):
    if names is None:
        names = [_filename_to_name(f) for f in tree_files]
    if lazy:
        return LazyTreeSequenceCollection(tree_files, names, metadata)
    else:
        if show_progress:
            tree_sequences = [load_tree(f) for f in tqdm(tree_files, desc="Loading tree sequences")]
        else:
            tree_sequences = [load_tree(f) for f in tree_files]
        return TreeSequenceCollection(tree_sequences, names, metadata)

def create_tree_collection_from_directory(dir: str,
                                         suffix: str = ".trees.tsz",
                                         names: Optional[list] = None,
                                         metadata: Optional[dict] = None,
                                         show_progress: bool = True,
                                         lazy: bool = False) -> Union[TreeSequenceCollection, LazyTreeSequenceCollection]:
    """
    Create a TreeSequenceCollection from all tree files in a directory.
    
    Parameters
    ----------
    dir : str
        Directory containing tree files
    suffix : str, optional
        File suffix to match (default: ".trees.tsz")
    names : List[str], optional
        Names for each tree sequence (default: None, will use filenames)
    metadata : Dict[str, Any], optional
        Additional metadata for the collection
    show_progress : bool, optional
        Whether to show a progress bar during loading
    lazy : bool, optional
        If True, create a lazy-loading collection (default: False)
    Returns
    -------
    TreeSequenceCollection
        Collection containing all loaded tree sequences
    """
    tree_files = list_trees(dir, suffix)
    return create_tree_collection_from_files(tree_files, names, metadata, show_progress, lazy)
 

def create_merged_collection(tree_collection: TreeSequenceCollection,
                             method: str = "concatenate",
                             chunk_size: Optional[int] = None) -> tskit.TreeSequence:
    """
    Create a single merged tree sequence from a collection.

    Parameters
    ----------
    tree_collection : TreeSequenceCollection
        Collection of tree sequences to merge
    method : str, optional
        Method for merging (see merge_tree_sequences)
    chunk_size : int, optional
        If provided, merge in chunks of this size to manage memory

    Returns
    -------
    tskit.TreeSequence
        Merged tree sequence
    """ 
    def _union_with_mapping(ts1, ts2, check_shared_equality=False):
        # tskit.TreeSequence.union(ts, other, node_mapping, check_shared_equality=False)
        # We need to provide a node_mapping array of length ts2.num_nodes
        
        node_mapping = np.full(ts2.num_nodes, tskit.NULL, dtype=np.int32)
        return ts1.union(ts2, node_mapping, check_shared_equality=check_shared_equality)

    def _merge_tree_sequences(tree_sequences: List[tskit.TreeSequence], method: str = "concatenate", show_progress: bool = False) -> tskit.TreeSequence:
        if not tree_sequences:
            raise ValueError("No tree sequences provided")
        if len(tree_sequences) == 1:
            return tree_sequences[0]
        if method == "concatenate":
            merged = tree_sequences[0]
            iterator = tree_sequences[1:]
            if show_progress:
                iterator = tqdm(iterator, desc="Merging tree sequences")
            for ts in iterator:
                merged = _union_with_mapping(merged, ts, check_shared_equality=False)
            return merged
        elif method == "union":
            merged = tree_sequences[0]
            iterator = tree_sequences[1:]
            if show_progress:
                iterator = tqdm(iterator, desc="Merging tree sequences")
            for ts in iterator:
                merged = _union_with_mapping(merged, ts, check_shared_equality=True)
            return merged
        else:
            raise ValueError(f"Unknown merge method: {method}")

    sequences = list(tree_collection)
    if chunk_size is None or len(sequences) <= chunk_size:
        return _merge_tree_sequences(sequences, method, show_progress=True)

    # Merge in chunks
    merged = sequences[0]
    num_chunks = ((len(sequences) - 1) // chunk_size)
    for i in tqdm(range(chunk_size, len(sequences), chunk_size), desc="Merging chunks"):
        chunk = sequences[i:i + chunk_size]
        chunk_merged = _merge_tree_sequences(chunk, method, show_progress=False)
        merged = _merge_tree_sequences([merged, chunk_merged], method, show_progress=False)

    return merged

def batch_process_trees(tree_collection: TreeSequenceCollection,
                       func: callable,
                       batch_size: int = 10,
                       *args, **kwargs) -> List[Any]:
    """
    Process tree sequences in batches to manage memory usage.
    
    Parameters
    ----------
    tree_collection : TreeSequenceCollection
        Collection of tree sequences to process
    func : callable
        Function to apply to each batch
    batch_size : int, optional
        Number of tree sequences to process in each batch
    *args, **kwargs
        Additional arguments to pass to func
        
    Returns
    -------
    List[Any]
        Results from processing all batches
    """
    results = []
    sequences = list(tree_collection)
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        batch_result = func(batch, *args, **kwargs)
        results.append(batch_result)
    
    return results

def get_population_variant_matrix(ts: tskit.TreeSequence,
                                  metadata_field: str = "name"):
    """
    Get the population x variant matrix for a tree sequence.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence to compute the population x variant matrix for
    metadata_field : str, optional
        The field in the population metadata to use for the population name
        (default: "name")

    Returns
    -------
    np.ndarray
        The population x variant matrix
    """
    # Efficiently compute the population x variant matrix without repeated simplification
    import json

    # Map population names to their sample indices in the tree sequence
    pop_samples_dict = {}
    for p in ts.populations():
        pop_name = json.loads(p.metadata)[metadata_field]
        pop_samples_dict[pop_name] = ts.samples(population=p.id)

    # Get the full genotype matrix once (shape: variants x samples)
    print("Loading genotype matrix")
    G = ts.genotype_matrix()  # shape: (num_sites, num_samples)

    # For each population, compute the mean genotype per variant across its samples
    pop_names = list(pop_samples_dict.keys())
    pxv = np.stack([
        G[:, pop_samples_dict[pop_name]].mean(axis=1)
        for pop_name in tqdm(pop_names, desc="Computing population variant matrices")
    ])
    print(pxv.shape)
    
    return pxv
 
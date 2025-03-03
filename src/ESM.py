import torch
import esm2
import os
import glob
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm

import src.utils as utils 
import src.haplosaurus as hs 
import src.config as config
import src.proteingym as pg
import src.gprofiler as gp
import src.biopython as bp


 
def list_models(prefix='esm',
                recommended_models=False,
                return_list=False):
    """Lists available ESM models from esm2.pretrained.
    See here for descriptions of each model and recommendations for which ones to use:
    https://github.com/facebookresearch/esm?tab=readme-ov-file#main-models-you-should-use-
    
    Args:
        prefix (str, optional): Filter models starting with this prefix. Defaults to 'esm'.
        recommended_models (bool, optional): If True, only returns models recommended by the ESM team. Defaults to False.
        return_list (bool, optional): If True, returns list of model names. If False, prints them. Defaults to False.
        
    Returns:
        list: List of model names if return_list=True, otherwise None
    """
    models = [model_name for model_name in dir(esm2.pretrained) if model_name.startswith(prefix)]
    # Filter by recommended models
    if recommended_models:
        recc_models = [
            "esm2_t36_3B_UR50D", 
            'esm2_t48_15B_UR50D',
            'esmfold_v1',
            'esm_msa1b_t12_100M_UR50S',
            'esm1v_t33_650M_UR90S_1',
            'esm1v_t33_650M_UR90S_5',
            'esm_if1_gvp4_t16_142M_UR50'
        ]
        models = utils.intersect(models, recc_models)
    # Return list of models
    if return_list:
        return models
    else:
        print(f"Available models:")
        for model in models:
            print(f"- {model}")

def list_scoring_strategies(model: str = None,
                            options: list = ["wt-marginals", "masked-marginals", "pseudo-ppl"]):
    models = list_models(return_list=True)
    scoring_strategies = {} 
    for m in models:
        if model is None or m == model:
            scoring_strategies[m] = options
    return scoring_strategies

def get_torch_data_i(transcript_id, 
                     results, 
                     alphabet,
                     suffix='',
                     to_stop=True):
    proteoform_counter = 1
    # print(transcript_id)
    if 'aa_seqs' not in results:
        print(f"'aa_seqs' key not found in results for {transcript_id}")
        return
    if not results['aa_seqs']:
        print(f"'aa_seqs' is empty for {transcript_id}")
        return
    data = []
    for sample, (seq1, seq2) in tqdm(results['aa_seqs'].items(), 
                                     desc=f"{transcript_id}: Processing samples.",
                                     leave=False):
        sample_suffix = f"{sample}{suffix}"
        samples.append(sample_suffix) 
        # Process both phases
        for phase, seq in [("phase1", seq1), ("phase2", seq2)]:
            # only use the substring up to the first stop codon
            if seq is None:
                continue
            stop_codon_count = seq.count('*')
            if to_stop is True:
                if '*' in seq:
                    seq = seq[:seq.find('*')]
            proteoform_id = utils.create_proteoform_id(transcript_id, seq, alphabet)
            if seq not in seq_to_proteoform.keys():
                if transcript_id not in transcript_to_proteoform:
                    transcript_to_proteoform[transcript_id] = []
                seq_to_proteoform[seq] = proteoform_id
                data.append((proteoform_id, seq))
                proteoform_to_stopcodons[proteoform_id] = stop_codon_count
                transcript_to_proteoform[transcript_id].append(proteoform_id)
                proteoform_counter += 1
            sample_to_proteoform[(transcript_id,sample_suffix,phase)] = proteoform_id
    return data
    # return data, seq_to_proteoform, transcript_to_proteoform, sample_to_proteoform, proteoform_to_stopcodons, samples

def get_torch_data_transcripts(results_all=None,
                               save_dir=None,
                               transcript_ids=None,
                               max_transcripts=None): 
    # Get transcript IDs
    if results_all is not None:
        print(f"Loading data from results_all")
        transcript_ids_x = results_all.keys()
    elif save_dir is not None:
        if isinstance(save_dir, str):
            print(f"Loading data from {save_dir}")
            files = glob.glob(f"{save_dir}/*.pkl")
            transcript_ids_x = [os.path.basename(f).replace('.pkl','') for f in files]
        elif isinstance(save_dir, dict):
            transcript_ids_x = set()
            for sample_group,save_dir_group in save_dir.items():
                files = glob.glob(f"{save_dir_group}/*.pkl")
                transcript_ids_x.update([os.path.basename(f).replace('.pkl','') for f in files])
            transcript_ids_x = list(transcript_ids_x)
    # Filter transcript IDs
    if transcript_ids is not None:
        transcript_ids_final = utils.intersect(transcript_ids_x, transcript_ids)
    else:
        transcript_ids_final = transcript_ids_x
    if max_transcripts is not None:
        transcript_ids_final = transcript_ids_final[:max_transcripts]
    if len(transcript_ids_final) == 0:
        raise ValueError(f"No transcripts found in {save_dir}")
    return transcript_ids_final

def get_torch_data(results_all=None,
                   alphabet=None,
                   transcript_ids=None,
                   max_transcripts=None,
                   save_dir = "1KG/sequence_dict_all",
                   to_stop=True,
                   verbose=True
                   ):
    # Prepare Data for ESM2 Model
    global seq_to_proteoform, transcript_to_proteoform, sample_to_proteoform, proteoform_to_stopcodons, samples
    seq_to_proteoform = {} # Maps transcript sequence to variant ID
    transcript_to_proteoform = {} # Maps transcript to variant IDs
    sample_to_proteoform = {} # Maps transcript,sample,phase to variant ID
    proteoform_to_stopcodons = {}
    samples = []
    batches = {} 
    transcript_ids_final = get_torch_data_transcripts(
        results_all=results_all,
        save_dir=save_dir,
        transcript_ids=transcript_ids,
        max_transcripts=max_transcripts
    )
    # Iterate over transcripts
    for transcript_id in tqdm(transcript_ids_final,
                              desc=f"Processing transcripts"):
    
        batches[transcript_id] = []
        if results_all is not None:
            results_tx = results_all[transcript_id]
        elif save_dir is not None:
            if isinstance(save_dir, str): 
                f = f"{save_dir}/{transcript_id}.pkl"
                results_tx = utils.load_pickle(f, verbose=verbose>1) 
                if results_tx is not None:
                     batches[transcript_id] += get_torch_data_i(
                        transcript_id=transcript_id, 
                        results=results_tx, 
                        alphabet=alphabet, 
                        to_stop=to_stop
                        )
            elif isinstance(save_dir, dict):
                for sample_group,save_dir_group in save_dir.items():
                    f = f"{save_dir_group}/{transcript_id}.pkl"
                    results_tx = utils.load_pickle(f, verbose=verbose>1)
                    if results_tx is not None:
                        batches[transcript_id] += get_torch_data_i(
                            transcript_id=transcript_id, 
                            results=results_tx, 
                            alphabet=alphabet, 
                            suffix="_"+sample_group,
                            to_stop=to_stop
                            )
    # Return vars
    return seq_to_proteoform, transcript_to_proteoform, sample_to_proteoform, proteoform_to_stopcodons, samples, batches


def get_embeddings(batches,
                   seed=42,
                   model=None,
                   alphabet=None,
                   repr_layers=[33],
                   force=False, 
                   max_transcripts=None,
                   verbose=False,
                   error=True,
                   save_dir="1KG/embeddings/esm2_t33_650M_UR50D",
                   tx_suffix_dict=None,
                   desc=f"Embedding transcript proteoforms",
                   return_paths_only=False,
                   save_hdf5=True,
                   deterministic=True,
                   **kwargs):  
    
    # Set seeds
    utils.set_seeds_torch(seed, deterministic)
    
    if verbose:
        print(f"Random seed set to {seed}")
    
    if model is None:
        if verbose:
            print(f"Using model: esm2_t33_650M_UR50D")
        model, alphabet = esm2.pretrained.esm2_t33_650M_UR50D()
    if alphabet is None:
        alphabet = model.alphabet
    batch_converter = alphabet.get_batch_converter()
    # Init vars
    results = {}
    save_paths = {}
    failed_tx_ids = [] 
    
    if max_transcripts!=None:
        if verbose:
            print(f"Limiting to {max_transcripts} transcripts")
        batches = {k: v for k, v in list(batches.items())[:max_transcripts]}

    for tx_id, data in tqdm(batches.items(), 
                                    desc=desc): 
        if tx_suffix_dict is not None and tx_id in tx_suffix_dict:
            save_path_suffix = tx_suffix_dict[tx_id]
        else:
            save_path_suffix = ''
        if save_hdf5:
            ext = '.h5'
        else:
            ext = '.pkl'
        save_path = os.path.abspath(f"{save_dir}/{tx_id}{save_path_suffix}{ext}")
        try:
            # Load existing embeddings
            if os.path.exists(save_path) and force is False and return_paths_only is True:
                save_paths[tx_id] = save_path
                continue
            if save_hdf5:
                results_tx = load_esm_res_hdf5(save_path, 
                                     force=force, 
                                     verbose=verbose>1)
            else:
                results_tx = utils.load_pickle(save_path, 
                                               force=force, 
                                               verbose=verbose>1)
            if results_tx is not None: 
                save_paths[tx_id] = save_path
                results[tx_id] = results_tx
                continue
            else:
                # Prepare batches for model
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                # Generate embeddings
                with torch.no_grad():
                    start_time = time.time()
                    embeddings = model(batch_tokens, 
                                       repr_layers=repr_layers, 
                                       return_contacts=False,
                                       **kwargs)
                    embed_time = time.time() - start_time
                    if return_paths_only is False:
                        results[tx_id] = {
                            'embeddings':embeddings,
                            'batch_labels':batch_labels, 
                            'batch_strs':batch_strs,
                            'batch_tokens':batch_tokens,
                            'embed_time':embed_time
                        }
                    else:
                        results[tx_id] = save_path
                # Save 
                if save_hdf5:
                    save_esm_res_hdf5(results[tx_id],
                                    save_path,
                                    force=force,
                                    verbose=verbose>1)
                else:
                    utils.save_pickle(obj=results[tx_id],
                                save_path=save_path,
                                verbose=verbose>1) 
                # Only add to save_paths if the file was created
                save_paths[tx_id] = save_path
        except Exception as e: 
            # print(data)
            if error:
                raise e
            else:
                if verbose:
                    print(f"Failed to embed transcript {tx_id}: {e}")   
                failed_tx_ids.append(tx_id)
                continue

    return {'results':results, 
            'save_paths':save_paths, 
            'failed_tx_ids':failed_tx_ids}

def get_representations(save_paths=None,
                        tx_ids=None,
                        save_dir=None, 
                        repr_layers=[33],
                        batches=None,
                        esm_results=None,
                        suffix=None,
                        verbose=True):
    # Prepare variables
    if isinstance(repr_layers, list):
        repr_layers = repr_layers[0]
    if save_paths is None and save_dir is not None:
        import glob
        save_paths = glob.glob(f"{save_dir}/*.pkl", recursive=True)
    if isinstance(save_paths, dict):
        save_paths = save_paths.values()
    else:
        save_paths = save_paths
    if verbose: 
        print(f"Found {len(save_paths)} ESM-2 embeddingfiles")
    # Get sequence representations function
    def _get_representations_i(results, 
                                data,
                                repr_layers,
                                suffix_final=''):
        batch_labels = results["batch_labels"]
        return [(batch_labels[i]+suffix_final, results["embeddings"]["representations"][repr_layers][i, 1 : len(seq) + 1].mean(0, keepdim=False)) for i,(_,seq) in enumerate(data)]
    # Iterate over save paths
    tx_ids_final = set()
    seq_reps = []
    for f in tqdm(save_paths, 
                  desc="Extracting sequence representations"):
        fname_split = os.path.basename(f).replace('.pkl','').split('_')
        tx_id = fname_split[0] 
        if tx_ids is not None:
            if tx_id not in tx_ids:
                continue
        tx_ids_final.add(tx_id)
        if suffix is None:
            suffix_final = ''
        else:
            if len(fname_split) == 2:
                suffix_final = ":".join(['_'+suffix,fname_split[1]])
            else:
                suffix_final = suffix
        
        if esm_results is None:
            results = utils.load_pickle(f, verbose=verbose>1)
        else:
            if tx_id in esm_results:
                results = esm_results[tx_id]
            else:
                results = esm_results
        seq_reps += _get_representations_i(
            results=results, 
            data=batches[tx_id],
            repr_layers=repr_layers,
            suffix_final=suffix_final
            )
        if verbose:
            print(">",tx_id,":",len(seq_reps),'sequence representations extracted.')
    if verbose:
        print(f"TOTAL: {len(seq_reps)} sequence representations extracted across {len(tx_ids_final)} transcript(s).")
    return seq_reps

def get_representation_matrix(seq_reps,
                              save_path=None,
                              force=True):
    # restructure sequence_representations to be a 2D array
    def _get_representation_matrix(seq_reps):
        return torch.stack([x[1] for x in seq_reps])
    
    if save_path is not None:
        if not os.path.exists(save_path) or force:
            print("Saving sequence representations")
            X = _get_representation_matrix(seq_reps)
            torch.save(X, save_path)
        else:
            print("Loading existing sequence representations") 
            X = torch.load(save_path)
    else:
        X = _get_representation_matrix(seq_reps)
    print(X.shape)
    return X

def _get_representation_variances_i(seq_reps, group,
                                   tx_id_sep=":"):
    transcript_variances = {}
    transcript_variance_means = {}
    tx_ids = list(set([x[0].split(tx_id_sep)[0] for x in seq_reps]))
    # Compute the variance of each transcript embedding acrosss variants
    for tx_id in tqdm(tx_ids, 
                      desc=f"Computing transcript variances for {group}"):
        variant_embeddings = [x[1] for x in seq_reps if x[0].split(tx_id_sep)[0] == tx_id]
        if len(variant_embeddings) == 1:
            # Set variance to 0 when there is only 1 isoform
            transcript_variances[tx_id] = torch.zeros(1,variant_embeddings[0].shape[0])
            transcript_variance_means[tx_id] = 0.0
        else:
            transcript_variances[tx_id] = torch.stack([x for i,x in enumerate(variant_embeddings)]).var(dim=0)
            transcript_variance_means[tx_id] = transcript_variances[tx_id].mean()
    return transcript_variances, transcript_variance_means

def get_representation_variances(seq_reps, 
                                 tx_id_sep=":",
                                 groups=['WT', 'Pathogenic','Benign'],
                                 prefix="_"):
    
    variances_by_tx = {}
    tx_ids = list(set([x[0].split(tx_id_sep)[0] for x in seq_reps]))
    
    for tx_id in tx_ids:
        variances_by_tx[tx_id] = {}
        
    if groups is not None:
        for group in groups:
            seq_reps_group = [x for x in seq_reps if prefix+group in x[0]]
            variances, means = _get_representation_variances_i(seq_reps_group, group, tx_id_sep)
            for tx_id in variances:
                variances_by_tx[tx_id][group] = (variances[tx_id], means[tx_id])
        return variances_by_tx
    else:
        variances, means = _get_representation_variances_i(seq_reps, "all", tx_id_sep)
        for tx_id in variances:
            variances_by_tx[tx_id]["all"] = (variances[tx_id], means[tx_id])
        return variances_by_tx
     

def get_reduction_df(seq_reps, 
                    method=["UMAP","TSVD"][0],
                    haplotypes=None,
                    add_freqs=True,
                    add_tx_id=True,
                    embedding=None,
                    nan_indices=None,
                    tx_id_sep=":",
                    split="_",
                    verbose=True):
        
    if embedding is None or nan_indices is None:
        X = get_representation_matrix(seq_reps)
        if method == "UMAP":
            model, embedding, nan_indices = utils.run_umap(X)
        elif method == "TSVD":
            model, embedding, nan_indices = utils.run_tsvd(X)
    df = pd.DataFrame(
        embedding, 
        columns=[method+" "+str(i+1) for i in range(embedding.shape[1])]
        )
    # Add label and label_base columns
    df['label'] = [x[0] for i,x in enumerate(seq_reps) if not nan_indices[i]]
    df['label_base'] = df['label'].str.split(split).str[0] 
    # Count the number of variants in the label_base (the number of ">" or "<" in the label_base)
    df['edits'] = utils.count_variants(df['label_base'], 
                               tx_id_sep=tx_id_sep)
    # Add group column
    df['group'] = get_haplotype_group(df, col="label")
    # Add protein_id column
    df['protein_id'] = [x.split(tx_id_sep)[0] for x in df['label']]
    if haplotypes is not None and add_tx_id:
        df = hs.add_txid(df, haplotypes, verbose=verbose)
        # Add haplotype frequencies
    if haplotypes is not None and add_freqs:
        df = hs.add_haplotype_freqs(df, haplotypes, verbose=verbose)
    return df

def _get_color_map(df, color_col, color_palette): 
    
    palette = sns.color_palette(color_palette, 
                                n_colors=len(df[color_col].unique())) 
    color_map = dict(sorted(zip(df[color_col].unique(), palette)))
    return color_map     

def _infer_xy_cols(df, x, y, 
                   search_strings=["UMAP","TSVD"]):
    # Find columns that match the search strings followed by a number
    # Find which search string has at least 2 matching columns
    search_strings = utils.as_list(search_strings)
    search_string = None
    for s in search_strings:
        matching_cols = [col for col in df.columns if s.lower() in col.lower()]
        if len(matching_cols) >= 2:
            search_string = s
            break
    if search_string is None:
        raise ValueError(f"Could not find at least 2 columns matching any of {search_strings}")
    # Find the columns that match the search string followed by a number
    import re
    # find columns that match the pattern "UMAP (some number)"
    if x is None:
        x = [col for col in df.columns if re.match(r"(?i)"+search_string+"\s*\d+", col)][0]
    if y is None:
        y = [col for col in df.columns if re.match(r"(?i)"+search_string+"\s*\d+", col)][1]
    return x, y

def _plot_umap_interactive(df,
                            x,
                            y,
                            opacity,
                            facet_col,
                            col_wrap,
                            color,
                            color_palette,
                            color_col,
                            size,
                            sizes,
                            sharex,
                            sharey,
                            highlight_label,
                            highlight_color,
                            highlight_size,
                            highlight_linewidth,
                            highlight_marker,
                            title,
                            add_density,
                            **kwargs):
        # Set size max
        if sizes is not None:
            size_max = max(sizes)
        else:
            size_max = None
        # Set color map
        if color_col is not None:
            color = None
            color_map = _get_color_map(df, color_col, color_palette)
            df['color'] = df[color_col].map(color_map)
        # Plot
        fig = px.scatter(df, 
                         x=x, 
                         y=y,
                         facet_col=facet_col,
                         facet_col_wrap=col_wrap,
                         hover_data=['label'],
                         size=size, 
                         size_max=size_max,  
                         opacity=opacity,
                         title=title,
                         color='color' if color_col is not None else color,
                         **kwargs)
        # Add px.density_contour underneath the points if add_density is True
        if add_density:
            fig.add_trace(px.density_contour(df, 
                                             x=x, 
                                             y=y,
                                             facet_col=facet_col,
                                             facet_col_wrap=col_wrap,
                                             **kwargs).data[0]
                                             )
        if highlight_label is not None:
            # Add red circles around highlighted points
            highlight_df = df[df['label'].str.contains(highlight_label)]
            fig.add_trace(px.scatter(highlight_df,
                                   x=x,
                                   y=y,
                                   facet_col=facet_col,
                                   facet_col_wrap=col_wrap).update_traces(
                                       mode='markers',
                                       marker=dict(symbol='circle-open',
                                                 size=highlight_size,
                                                 color=highlight_color,
                                                 line=dict(width=highlight_linewidth,
                                                           color=highlight_color))).data[0])
        if sharex is False:
            fig.update_xaxes(matches=None)
        if sharey is False:
            fig.update_yaxes(matches=None)
        return fig

def _plot_umap_static(df,
                    x,
                    y,
                    opacity,
                    facet_col,
                    col_wrap,
                    color,
                    color_palette,
                    color_col,
                    size,
                    sizes,
                    sharex,
                    sharey,
                    highlight_label,
                    highlight_color,
                    highlight_size,
                    highlight_linewidth,
                    highlight_marker,
                    title,
                    add_edges,
                    add_density,
                    figsize,
                    show,
                    style_col,
                    **kwargs):
    
    # Set color map
    if color_col is not None:
        color= None
        color_map = _get_color_map(df, color_col, color_palette)
    
    if add_density:
        legend_bg = 'black'
        legend_text = 'white'
    else:
        legend_bg = 'white'
        legend_text = 'black' 
    if highlight_color is not None:
        highlight_color = legend_text
    # Plot
    marker_map = utils.get_marker_map()
    fig, ax = plt.subplots(figsize=figsize)

    # First plot the density contours
    # Create a faceted plot by transcript
    if facet_col is not None:
        g = sns.FacetGrid(df, 
                            col=facet_col,    
                            col_wrap=col_wrap,
                            sharex=sharex,
                            sharey=sharey)
        # Add KDE plot to each facet if add_density is True
        if add_density:
            g.map_dataframe(sns.kdeplot, 
                            x=x, y=y,
                        fill=True,
                        cmap='viridis',
                        **kwargs)
        # Add scatter points to each facet
        g.map_dataframe(plt.scatter,
                    x=x, y=y, 
                    alpha=opacity,
                    s=1,
                    color=color)
        
        # Set background color for each subplot if showing density
        for ax in g.axes.flat:
            if add_density:
                ax.set_facecolor('#2F0154')
            if highlight_label is not None:
                # Add red circles around highlighted points for each facet
                highlight_df = df[df['label'].str.contains(highlight_label)]
                highlight_df_facet = highlight_df[highlight_df[facet_col] == ax.get_title().split(' = ')[1]]
                ax.scatter(highlight_df_facet[x], highlight_df_facet[y],
                            facecolors='none', 
                            edgecolors=highlight_color, 
                            s=highlight_size, 
                            linewidth=highlight_linewidth, 
                            marker=highlight_marker)
            if add_edges:
                draw_haplotype_trio_edges(df,
                                        x=x,
                                        y=y,
                                        palettes=config.PALETTES,
                                        connect_pb=False,
                                        ax=ax)
    else:
        if add_density:
            plt.gca().set_facecolor('#2F0154') # Set background to slightly darker than darkest viridis color
            edge_zorder=1
            g = sns.kdeplot(data=df, x=x, y=y, 
                        fill=True, 
                        cmap='viridis',
                        **kwargs
                        ) 
            plt.scatter(data=df,
                        x=x, y=y, 
                        alpha=opacity, 
                        s=df[size] if isinstance(size,str) else size,
                        sizes=sizes,
                        color=df[color_col].map(color_map)
                        )
            
        else:
            edge_zorder=-1
            g = sns.scatterplot(data=df,
                        x=x, y=y, 
                        alpha=opacity,
                        size=size, 
                        sizes=sizes,
                        hue=color_col,
                        style=style_col,
                        markers=marker_map,
                        palette=color_palette,
                        **kwargs
                        )
        
        if highlight_label is not None:
            # Add red circles around highlighted points
            highlight_df = df[df['label'].str.contains(highlight_label)]
            plt.scatter(highlight_df[x], highlight_df[y],
                        facecolors='none', 
                        edgecolors=highlight_color, 
                        s=highlight_df[size] if isinstance(size,str) else size,
                        sizes=sizes,
                        linewidth=highlight_linewidth, 
                        marker=highlight_marker)
        if add_edges:
            draw_haplotype_trio_edges(df,
                                    x=x,
                                    y=y,
                                    palettes=config.PALETTES,
                                    connect_pb=False,
                                    ax=ax,
                                    zorder=edge_zorder
                                    ) 
        # Update legend markers to circles if the legend title matches color_col or size
        if not add_density:
            for handle in g.legend_.legendHandles:
                if not hasattr(handle, 'get_marker') and not hasattr(handle, '_legmarker'):  # Skip non-marker legends
                    continue
                legend_title = g.legend_.get_title().get_text()
                if hasattr(handle, '_legmarker'):  # Size legend
                    if legend_title == size:
                        handle._legmarker.set_marker('o')
                elif legend_title == color_col:
                    handle.set_marker('o')
        # Set plot title
        if title is not None:
            plt.title(title) 
        # Show plot
        if show:
            plt.show()

def plot_umap(df,
              x=None,
              y=None,
              interact=False,
              opacity=0.5,
              facet_col=None,
              col_wrap=3,
              color='white',
              color_palette='tab10',
              color_col=None,
              size=None, 
              sizes=None,
              sharex=True,
              sharey=True,
              highlight_label=None,
              highlight_color='white',
              highlight_size=100,
              highlight_linewidth=2,
              highlight_marker='D',
              title=None,
              add_density=True,
              show=True,
              add_edges=True,
              figsize=(12, 12),
              style_col=None,
              **kwargs):  
    # Set size 
    if isinstance(size, list) or isinstance(size, str):
        if size not in df.columns:
            print(f"Size column {size} not found in embedding_df")
            size = None 
    # Set x and y if not provided
    x, y = _infer_xy_cols(df, x, y)
    # Plot interactively
    if interact is True:
       fig = _plot_umap_interactive(
           df=df,
           x=x,
           y=y,
           opacity=opacity,
           facet_col=facet_col,
           col_wrap=col_wrap,
           color=color,
           color_palette=color_palette,
           color_col=color_col,
           size=size, 
           sizes=sizes,
           sharex=sharex,
           sharey=sharey,
           highlight_label=highlight_label,
           highlight_color=highlight_color,
           highlight_size=highlight_size,
           highlight_linewidth=highlight_linewidth,
           highlight_marker=highlight_marker,
           title=title,
           add_density=add_density,
           add_edges=add_edges,
           show=show,
           figsize=figsize,
           style_col=style_col,
           **kwargs
       )
       return fig
    # Plot statically
    else:
        _plot_umap_static(
             df=df,
           x=x,
           y=y,
           opacity=opacity,
           facet_col=facet_col,
           col_wrap=col_wrap,
           color=color,
           color_palette=color_palette,
           color_col=color_col,
           size=size, 
           sizes=sizes,
           sharex=sharex,
           sharey=sharey,
           highlight_label=highlight_label,
           highlight_color=highlight_color,
           highlight_size=highlight_size,
           highlight_linewidth=highlight_linewidth,
           highlight_marker=highlight_marker,
           title=title,
           add_edges=add_edges,
           add_density=add_density,
           show=show,
           figsize=figsize,
           style_col=style_col,
           **kwargs
        )
    

def draw_haplotype_trio_edges(df, 
                             x='UMAP 1',
                             y='UMAP 2',
                             palettes=config.PALETTES, 
                             connect_pb=True,
                             alpha=0.2,
                             linestyle='--',
                             zorder=-1,
                             ax=None):
    """Draw connecting lines between points with same haplotype in a UMAP plot.
    
    Args:
        df: DataFrame containing UMAP coordinates and labels
        x: Name of x-axis column (default: 'UMAP 1') 
        y: Name of y-axis column (default: 'UMAP 2')
        palettes: Dict mapping group names to color palettes
        connect_pb: Whether to connect Pathogenic-Benign pairs
        alpha: Transparency of lines
        linestyle: Style of connecting lines
        zorder: Z-order of lines (negative to plot under points)
        
    Example:
        # Plot UMAP points and add connecting lines between related haplotypes
        fig = plot_umap(df, 
                       x='UMAP 1', 
                       y='UMAP 2',
                       color_col='group')
        draw_haplotype_trio_edges(df,
                                x='UMAP 1',
                                y='UMAP 2', 
                                alpha=0.3,
                                connect_pb=False)
    """
    if ax is None:
        ax = plt.gca()
    
    # Get unique haplotypes from label_base
    unique_haplotypes = df['label_base'].unique()

    # Create color palette with same number of colors as haplotypes
    colors_p = plt.cm.get_cmap(palettes['Pathogenic'])(np.linspace(0, 1, len(unique_haplotypes)))
    colors_b = plt.cm.get_cmap(palettes['Benign'])(np.linspace(0, 1, len(unique_haplotypes)))
    
    for hap, color_p, color_b in zip(unique_haplotypes, colors_p, colors_b):
        # Get points for this haplotype
        points = df[df['label_base'] == hap]
        
        # Draw lines between points if we have all 3 groups
        if len(points) == 3:
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    # Skip connections between Pathogenic and Benign if connect_pb=False
                    if not connect_pb and ((points.iloc[i]['group'].startswith('Pathogenic') and points.iloc[j]['group'].startswith('Benign')) or
                                           (points.iloc[i]['group'].startswith('Benign') and points.iloc[j]['group'].startswith('Pathogenic'))
                                           ):
                        continue
                        
                    # Determine line color based on variant types being connected
                    if points.iloc[i]['group'].startswith('Pathogenic') or points.iloc[j]['group'].startswith('Pathogenic'):
                        line_color = color_p
                    elif points.iloc[i]['group'].startswith('Benign') or points.iloc[j]['group'].startswith('Benign'):
                        line_color = color_b
                    else:
                        line_color = 'gray'
                        
                    ax.plot([points.iloc[i][x], points.iloc[j][x]], 
                            [points.iloc[i][y], points.iloc[j][y]], 
                            color=line_color, 
                            alpha=alpha, 
                            linestyle=linestyle, 
                            zorder=zorder)

def get_haplotype_group(df,
                        col="label",
                        split="_",
                        default="WT"):
    if col=='index':
        return pd.Series([x[1] if len(x) > 1 else default for x in df.index.str.split(split)])
    else:
        return df[col].apply(lambda x: x.split(split)[1] if split in x else default)

def list_embeddings(dir, 
                    suffix='',
                    ext="pkl",
                    recursive=False,
                    as_dict=True):
    files = glob.glob(f"{dir}/*{suffix}.{ext}", recursive=recursive)
    if as_dict:
        return {os.path.basename(f).replace(f".{ext}",""):f for f in files}
    else:
        return files
    
def check_batch_info(batches,
                      split="_",
                      as_df=True,
                      verbose=True):
    # Infer ngroups 
    batch_groups = {k:len(set([x[0].split(split)[1] for x in batches[k]])) for k in batches.keys()}
    # check batch sizes
    batch_sizes = {k:len(v) for k,v in batches.items()}
    # get batch size per group
    batch_sizes_per_group = {k:v//batch_groups[k] for k,v in batch_sizes.items()}
    # check that all batch sizes are divible by the number of groups
    batch_remainders = {k:v%batch_groups[k] for k,v in batch_sizes.items()}
    # get the number of unique haplotypes per trancript
    haplotype_counts = {k:len(set([x[0][:x[0].find(split)] for x in batches[k]])) for k in batches.keys()}
    if as_df:
        batch_info =  pd.concat([
            pd.DataFrame(batch_sizes, index=['size']).T,
            pd.DataFrame(batch_groups, index=['groups']).T,
            pd.DataFrame(batch_sizes_per_group, index=['size_per_group']).T,
            pd.DataFrame(batch_remainders, index=['remainder']).T,
            pd.DataFrame(haplotype_counts, index=['haplotype_count']).T
        ], axis=1)
        if verbose:
            print(batch_info)
        return batch_info
    else:
        if verbose:
            print(batch_sizes)
            print(batch_remainders)
        return {'sizes':batch_sizes, 
                'groups':batch_groups,
                'sizes_per_group':batch_sizes_per_group,
                'remainders':batch_remainders,
                'haplotypes':haplotype_counts}
  

def save_esm_res_hdf5(esm_res, 
                      file_path, 
                      compression="gzip",
                      compression_opts=4,
                      force=False,
                      verbose=True):
    """
    Save the esm_res dictionary to an HDF5 file with compression.

    Args:
        esm_res (dict): The esm_res dictionary to save.
        file_path (str): The path to the HDF5 file.
        force (bool): Whether to overwrite existing file.
    """
    import h5py

    if os.path.exists(file_path) and not force:
        raise ValueError(f"File {file_path} already exists. Set force=True to overwrite.")
    if not file_path.endswith(".h5"):
        raise ValueError("File must have a .h5 extension")
    
    if verbose:
        print(f"Saving ESM results to {file_path}")
    # Create the file
    with h5py.File(file_path, 'w') as f:
        # Handle embeddings
        if verbose:
            print("> Saving: 'embeddings'")
        embeddings_group = f.create_group('embeddings')
        
        # Check if 'representations' key exists
        representations = esm_res['embeddings'].get('representations', {})
        if not representations:
            raise KeyError("Key 'representations' not found in 'esm_res['embeddings']'.")
        
        for layer, tensor in representations.items():
            # Ensure tensor is a PyTorch tensor
            if isinstance(tensor, torch.Tensor):
                embeddings_group.create_dataset(
                    f'rep_layer_{layer}',
                    data=tensor.cpu().numpy(),
                    compression=compression,
                    compression_opts=compression_opts
                )
            else:
                print(f"Skipping layer {layer}: Not a torch.Tensor")
        
        # Save logits if present
        if 'logits' in esm_res['embeddings']:
            if isinstance(esm_res['embeddings']['logits'], torch.Tensor):
                if verbose:
                    print("> Saving: 'logits'")
                embeddings_group.create_dataset(
                    'logits',
                    data=esm_res['embeddings']['logits'].cpu().numpy(),
                    compression=compression,
                    compression_opts=compression_opts
                )
            else:
                print("Skipping 'logits': Not a torch.Tensor")
        
        # Save batch_tokens
        if verbose:
            print("> Saving: 'batch_tokens'")
        f.create_dataset(
            'batch_tokens',
            data=esm_res['batch_tokens'].cpu().numpy(),
            compression=compression,
            compression_opts=compression_opts
        )
        
        # Save batch_labels as variable-length strings
        dt = h5py.string_dtype(encoding='utf-8')
        if verbose:
            print("> Saving: 'batch_labels'")
        f.create_dataset(
            'batch_labels',
            data=np.array(esm_res['batch_labels'], dtype=object),
            dtype=dt,
            compression=compression,
            compression_opts=compression_opts
        )
        
        # Save batch_strs as variable-length strings
        if verbose:
            print("> Saving: 'batch_strs'")
        f.create_dataset(
            'batch_strs',
            data=np.array(esm_res['batch_strs'], dtype=object),
            dtype=dt,
            compression=compression,
            compression_opts=compression_opts
        )

def load_esm_res_hdf5(file_path, verbose=True):
    """
    Load the esm_res dictionary from an HDF5 file.

    Args:
        file_path (str): The path to the HDF5 file.
        verbose (bool): Whether to print verbose messages.

    Returns:
        dict: The loaded esm_res dictionary.
    """
    import h5py

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    if not file_path.endswith(".h5"):
        raise ValueError("File must have a .h5 extension")

    if verbose:
        print(f"Loading ESM results from {file_path}")

    esm_res = {}

    with h5py.File(file_path, 'r') as f:
        # Load embeddings
        if 'embeddings' in f:
            if verbose:
                print("> Loading: 'embeddings'")
            embeddings_group = f['embeddings']
            esm_res['embeddings'] = {}

            # Load representations
            if 'rep_layer_0' in embeddings_group or any(key.startswith('rep_layer_') for key in embeddings_group):
                representations = {}
                for key in embeddings_group:
                    if key.startswith('rep_layer_'):
                        layer_num = key.replace('rep_layer_', '')
                        tensor = torch.tensor(embeddings_group[key][()])
                        representations[layer_num] = tensor
                        if verbose:
                            print(f"  Loaded representation from layer {layer_num}")
                esm_res['embeddings']['representations'] = representations
            else:
                if verbose:
                    print("  No 'representations' found in 'embeddings'.")

            # Load logits if present
            if 'logits' in embeddings_group:
                esm_res['embeddings']['logits'] = torch.tensor(embeddings_group['logits'][()])
                if verbose:
                    print("  Loaded 'logits'")
            else:
                if verbose:
                    print("  No 'logits' found in 'embeddings'.")
        else:
            if verbose:
                print("No 'embeddings' group found in the HDF5 file.")

        # Load batch_tokens
        if 'batch_tokens' in f:
            if verbose:
                print("> Loading: 'batch_tokens'")
            esm_res['batch_tokens'] = torch.tensor(f['batch_tokens'][()])
        else:
            if verbose:
                print("No 'batch_tokens' dataset found in the HDF5 file.")

        # Load batch_labels
        if 'batch_labels' in f:
            if verbose:
                print("> Loading: 'batch_labels'")
            batch_labels = f['batch_labels'][()]
            # Decode bytes to strings if necessary
            if isinstance(batch_labels[0], bytes):
                batch_labels = [label.decode('utf-8') for label in batch_labels]
            esm_res['batch_labels'] = batch_labels
        else:
            if verbose:
                print("No 'batch_labels' dataset found in the HDF5 file.")

        # Load batch_strs
        if 'batch_strs' in f:
            if verbose:
                print("> Loading: 'batch_strs'")
            batch_strs = f['batch_strs'][()]
            # Decode bytes to strings if necessary
            if isinstance(batch_strs[0], bytes):
                batch_strs = [s.decode('utf-8') for s in batch_strs]
            esm_res['batch_strs'] = batch_strs
        else:
            if verbose:
                print("No 'batch_strs' dataset found in the HDF5 file.")

    return esm_res

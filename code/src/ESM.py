try:
    import sys
    sys.path.append("code")
    from src.utils import create_proteoform_id, load_pickle, save_pickle, intersect, run_umap
except:
    print("Could not import utils")


def get_torch_data_i(transcript_id, 
                     results, 
                     alphabet,
                     suffix='',
                     to_stop=True):
    from tqdm.auto import tqdm
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
            proteoform_id = create_proteoform_id(transcript_id, seq, alphabet)
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
    import os
    import glob 
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
        transcript_ids_final = intersect(transcript_ids_x, transcript_ids)
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
    from tqdm.auto import tqdm  
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
                results_tx = load_pickle(f, verbose=verbose>1) 
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
                    results_tx = load_pickle(f, verbose=verbose>1)
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
                   model=None,
                   alphabet=None,
                   repr_layers=[33],
                   force=False, 
                   max_transcripts=None,
                   verbose=False,
                   error=True,
                   save_dir="1KG/embeddings/esm2_t33_650M_UR50D",
                   desc=f"Embedding transcript proteoforms",
                   **kwargs): 
                   
    import os
    from tqdm.auto import tqdm
    import torch 
    if model is None:
        if verbose:
            print(f"Using model: esm2_t33_650M_UR50D")
        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    if alphabet is None:
        alphabet = model.alphabet
    batch_converter = alphabet.get_batch_converter()
    # Init vars
    results = {}
    save_paths = {}
    failed_transcripts = [] 
    
    if max_transcripts!=None:
        if verbose:
            print(f"Limiting to {max_transcripts} transcripts")
        batches = {k: v for k, v in list(batches.items())[:max_transcripts]}

    for transcript_id, data in tqdm(batches.items(), 
                                    desc=desc): 
        save_path = os.path.abspath(f"{save_dir}/{transcript_id}.pkl")
        try:
            # Load existing embeddings
            results_tx = load_pickle(save_path, 
                                     force=force, 
                                     verbose=verbose>1)
            if results_tx is not None: 
                save_paths[transcript_id] = save_path
                results[transcript_id] = results_tx
                continue
            else:
                # Prepare batches for model
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                # Generate embeddings
                with torch.no_grad():
                    embeddings = model(batch_tokens, 
                                    repr_layers=repr_layers, 
                                    return_contacts=False,
                                    **kwargs)
                    results[transcript_id] = {
                        'embeddings':embeddings,
                        'batch_labels':batch_labels, 
                        'batch_strs':batch_strs,
                        'batch_tokens':batch_tokens
                    }
                # Save 
                save_pickle(obj=results[transcript_id],
                            save_path=save_path,
                            verbose=verbose>1) 
                # Only add to save_paths if the file was created
                save_paths[transcript_id] = save_path
        except Exception as e: 
            # print(data)
            if error:
                raise e
            else:
                if verbose:
                    print(f"Failed to embed transcript {transcript_id}: {e}")   
                failed_transcripts.append(transcript_id)
                continue

    return results, save_paths, failed_transcripts

def get_representations(save_paths=None,
                        save_dir=None, 
                        repr_layers=[33],
                        batches=None,
                        esm_results=None):
    from tqdm.auto import tqdm 
    import pickle
    import os

    if isinstance(repr_layers, list):
        repr_layers = repr_layers[0]
    if save_paths is None and save_dir is not None:
        import glob
        save_paths = glob.glob(f"{save_dir}/*.pkl")
    if isinstance(save_paths, dict):
        save_paths = save_paths.values()
    else:
        save_paths = save_paths
    print(f"Found {len(save_paths)} ESM2 files")
    
    def get_sequence_representations(results, 
                                     data,
                                     repr_layers):
        batch_labels = results["batch_labels"]
        return [(batch_labels[i], results["embeddings"]["representations"][repr_layers][i, 1 : len(seq) + 1].mean(0, keepdim=False)) for i,(_,seq) in enumerate(data)]
    
    seq_reps = []
    for f in tqdm(save_paths, desc="Extracting sequence representations"):
        transcript_id = os.path.basename(f).replace('.pkl','')
        if esm_results is None:
            with open(f,'rb') as handle:
                results = pickle.load(handle) 
        else:
            results = esm_results[transcript_id]
        seq_reps += get_sequence_representations(results, 
                                                 batches[transcript_id],
                                                 repr_layers)
    return seq_reps

def get_representation_matrix(seq_reps,
                              save_path = f"1KG/embeddings/esm2_t33_650M_UR50D/sequence_representation_matrix.pt",
                              force = True):
    import torch
    import os
    # restructure sequence_representations to be a 2D array
    if not os.path.exists(save_path) or force is True:
        print("Restructuring sequence representations into 2D array")
        X = torch.stack([x[1] for x in seq_reps])
        # Save X
        torch.save(X, save_path)
    else:
        print("Loading existing sequence representations")
        X = torch.load(save_path)
    print(X.shape)
    return X

def get_representation_variances(seq_reps, 
                                 tx_id_sep=":"):
    import torch
    from tqdm.auto import tqdm
    transcript_variances = {}
    transcript_variance_means = {}
    transcript_ids = list(set([x[0].split(tx_id_sep)[0] for x in seq_reps]))
    # Compute the variance of each transcript embedding acrosss variants
    for transcript_id in tqdm(transcript_ids, desc="Computing transcript variances"):
        # print(transcript_id)
        variant_embeddings = [x[1] for x in seq_reps if x[0].split(tx_id_sep)[0] == transcript_id]
        if len(variant_embeddings) == 1:
            # Set variance to 0 when there is only 1 isoform
            transcript_variances[transcript_id] = torch.zeros(1,variant_embeddings[0].shape[0])
            transcript_variance_means[transcript_id] = 0.0
        else:
            transcript_variances[transcript_id] = torch.stack([x for i,x in enumerate(variant_embeddings)]).var(dim=0)
            transcript_variance_means[transcript_id] = transcript_variances[transcript_id].mean()
        # print(f"> Mean variance: { transcript_variance_means[transcript_id]}")
    return transcript_variances, transcript_variance_means

def get_umap_df(seq_reps,
                embedding=None,
                nan_indices=None,
                tx_id_sep=":"):
    import pandas as pd
    if embedding is None or nan_indices is None:
        X = get_representation_matrix(seq_reps)
        reducer, embedding, nan_indices = run_umap(X)
    embedding_df = pd.DataFrame(embedding, columns=['UMAP 1', 'UMAP 2'])
    embedding_df['label'] = [x[0] for i,x in enumerate(seq_reps) if not nan_indices[i]]
    embedding_df['transcript_id'] = [x.split(tx_id_sep)[0] for x in embedding_df['label']]
    embedding_df['transcript'] = [x.split(tx_id_sep)[0].split('.')[0] for x in embedding_df['label']]
    return embedding_df

def plot_umap(embedding_df,
              interact=False,
              opacity=0.1,
              facet_col=None,
              col_wrap=3,
              color='white',
              color_palette='tab10',
              color_col=None,
              size=None,
              size_max=None,  
              sizes=None,
              sharex=True,
              sharey=True,
              highlight_label=None,
              highlight_color='white',
              highlight_size=100,
              highlight_linewidth=2,
              highlight_marker='D',
              title=None,
              **kwargs): 
    import seaborn as sns
    if color_col is not None:
        palette = sns.color_palette(color_palette, n_colors=len(embedding_df[color_col].unique()))
        color= None
        color_map = dict(sorted(zip(embedding_df[color_col].unique(), palette)))

    if interact is True:
        import plotly.express as px
        fig = px.scatter(embedding_df, 
                         x='UMAP 1', 
                         y='UMAP 2',
                         facet_col=facet_col,
                         facet_col_wrap=col_wrap,
                         hover_data=['label'],
                         size=size, 
                         size_max=size_max,  
                         opacity=opacity,
                         title=title,  
                         **kwargs)
        # Add px.density_contour underneath the points
        fig.add_trace(px.density_contour(embedding_df, 
                                         x='UMAP 1', 
                                         y='UMAP 2',
                                         facet_col=facet_col,
                                         facet_col_wrap=col_wrap,
                                         **kwargs).data[0]
                                         )
        if highlight_label is not None:
            # Add red circles around highlighted points
            highlight_df = embedding_df[embedding_df['label'].str.contains(highlight_label)]
            fig.add_trace(px.scatter(highlight_df,
                                   x='UMAP 1',
                                   y='UMAP 2',
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
    else:
        from matplotlib import pyplot as plt 
        import seaborn as sns
        plt.figure(figsize=(12, 12))  # Increase figure size
        # First plot the density contours
        # First plot the density contours with dark background matching lowest density
        # Create a faceted plot by transcript
        if facet_col is not None:
            g = sns.FacetGrid(embedding_df, 
                              col=facet_col,    
                              col_wrap=col_wrap,
                              sharex=sharex,
                              sharey=sharey)
            # Add KDE plot to each facet
            g.map_dataframe(sns.kdeplot, 
                            x='UMAP 1', y='UMAP 2',
                        fill=True,
                        cmap='viridis',
                        **kwargs)
            # Add scatter points to each facet
            g.map_dataframe(plt.scatter,
                        x='UMAP 1', y='UMAP 2', 
                        alpha=opacity,
                        s=1,
                        color=color)
            
            # Set background color for each subplot
            for ax in g.axes.flat:
                ax.set_facecolor('#2F0154')
                if highlight_label is not None:
                    # Add red circles around highlighted points for each facet
                    highlight_df = embedding_df[embedding_df['label'].str.contains(highlight_label)]
                    highlight_df_facet = highlight_df[highlight_df[facet_col] == ax.get_title().split(' = ')[1]]
                    ax.scatter(highlight_df_facet['UMAP 1'], highlight_df_facet['UMAP 2'],
                             facecolors='none', 
                             edgecolors=highlight_color, 
                             s=highlight_size, 
                             linewidth=highlight_linewidth, 
                             marker=highlight_marker)
        else:
            plt.gca().set_facecolor('#2F0154') # Set background to slightly darker than darkest viridis color
            sns.kdeplot(data=embedding_df, x='UMAP 1', y='UMAP 2', 
                        fill=True, 
                        cmap='viridis',
                        **kwargs
                        ) 
            # Then overlay scatter points with some transparency
            if color_col is not None:
                scatter = plt.scatter(data=embedding_df,
                            x='UMAP 1', y='UMAP 2', 
                            alpha=opacity, # Make points semi-transparent
                            s=size, # Small point size
                            sizes=sizes,
                            color=embedding_df[color_col].map(color_map))
                # Add legend mapping labels to colors
                legend_elements = [plt.scatter([], [], c=color, label=label) 
                                 for label, color in color_map.items()]
                
                # Add highlight label to legend if specified
                if highlight_label is not None:
                    legend_elements.append(plt.scatter([], [], 
                                                    facecolors='none',
                                                    edgecolors=highlight_color,
                                                    s=50,
                                                    linewidth=highlight_linewidth,
                                                    marker=highlight_marker,
                                                    label=highlight_label))
         
                plt.legend(handles=legend_elements,
                           title=color_col, 
                           facecolor='black', edgecolor='black', 
                           labelcolor='white', title_fontsize=10).get_title().set_color('white')
            else:
                plt.scatter(data=embedding_df,
                            x='UMAP 1', y='UMAP 2', 
                            alpha=opacity, # Make points semi-transparent
                            s=size, # Small point size
                            sizes=sizes,
                            color=color) # White points
            
            if highlight_label is not None:
                # Add red circles around highlighted points
                highlight_df = embedding_df[embedding_df['label'].str.contains(highlight_label)]
                plt.scatter(highlight_df['UMAP 1'], highlight_df['UMAP 2'],
                          facecolors='none', 
                          edgecolors=highlight_color, 
                          s=highlight_size, 
                          linewidth=highlight_linewidth, 
                          marker=highlight_marker)
            
            if title is not None:
                plt.title(title) 
        plt.show()

def get_patient_tensor(seq_reps,
                       sample_to_proteoform,
                       drop_nan=True):
    import torch
    from tqdm.auto import tqdm

    # Create empty 3D tensor
    samples = set([x[1] for x in sample_to_proteoform.keys()])
    transcripts = set([x[0] for x in sample_to_proteoform.keys()])
    phases = ['phase1', 'phase2']
    tensor = torch.zeros((len(samples), 
                          len(transcripts), 
                          len(phases),
                          seq_reps[0][1].shape[0]))

    seq_reps_keys = [x[0] for x in seq_reps]
    proteoform_ids = set()
    for (transcript, sample, phase), proteoform_id in tqdm(sample_to_proteoform.items(),
                                                            desc="Populating patient tensor"):
        # get indices of sample, transcript, phase
        sample_idx = list(samples).index(sample)
        transcript_idx = list(transcripts).index(transcript)
        phase_idx = phases.index(phase)
        # seq_rep 
        if proteoform_id not in seq_reps_keys:
            continue

        proteoform_ids.add(proteoform_id)
        seq_rep_idx = seq_reps_keys.index(proteoform_id)
        seq_rep = seq_reps[seq_rep_idx][1]
        if drop_nan and torch.isnan(seq_rep).any():
            continue
        tensor[sample_idx, transcript_idx, phase_idx, :] = seq_rep
    # Drop samples with no proteoforms
    if drop_nan is True:
        tensor = tensor[~torch.all(torch.all(tensor == 0, dim=-1))][0]
    return {'tensor':tensor,
            'samples':samples,
            'transcripts':transcripts,
            'phases':phases}

def run_tensor_factorization(patient_tensor,
                              n_components={'samples':10,
                                            'transcripts':10,
                                            'phases':10,
                                            'features':10},
                              n_epochs=500,
                              learning_rate=0.01):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm.auto import tqdm

    # Get tensor dimensions
    n_samples, n_transcripts, n_phases, n_features = patient_tensor.shape

    # Initialize factor matrices
    A = nn.Parameter(torch.randn(n_samples, n_components['samples'])) # Sample factors
    B = nn.Parameter(torch.randn(n_transcripts, n_components['transcripts'])) # Transcript factors  
    C = nn.Parameter(torch.randn(n_phases, n_components['phases'])) # Phase factors
    D = nn.Parameter(torch.randn(n_features, n_components['features'])) # Feature factors

    # Define optimizer
    optimizer = optim.Adam([A, B, C, D], lr=learning_rate)

    # Training loop
    pbar = tqdm(range(n_epochs), desc="Epochs completed")
    for epoch in pbar:
        # Forward pass - reconstruct tensor
        pred = torch.einsum('ac,bc,dc,ec->abde', A, B, C, D)
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(pred, patient_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Store learned factors
    factors = {
        'samples': A.detach(),
        'transcripts': B.detach(), 
        'phases': C.detach(),
        'features': D.detach()
    } 
    return factors

def get_transcript_contributions(factors, 
                                 patient_tensor, 
                                 l2_norm=True):
    import torch
    import pandas as pd
    A = factors['samples']
    B = factors['transcripts']
    C = factors['phases']
    D = factors['features']
    transcript_contributions = torch.einsum('ac,bc,dc,ec->ab', A, B, C, D)
    tx_contrib_df = pd.DataFrame(transcript_contributions.numpy(), 
                            index=patient_tensor['samples'], 
                            columns=patient_tensor['transcripts'])
    if l2_norm is True:
        l2_contrib = torch.norm(transcript_contributions, dim=0)
        # rescale to 0-1
        l2_contrib = l2_contrib / torch.max(l2_contrib)
        l2_contrib_df = pd.DataFrame(l2_contrib.numpy(), 
             index=patient_tensor['transcripts'],
             columns=["contribution"]).sort_values(by="contribution", ascending=False)   
    else:
        l2_contrib_df = None
    return tx_contrib_df, l2_contrib_df

def plot_tensor_factorization(factors, 
                              keys=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if keys is None:
        keys = factors.keys()
    for factor_name in keys:
        factor_matrix = factors[factor_name]
        plt.figure(figsize=(10, 6))
        sns.heatmap(factor_matrix.T, annot=True, cmap='viridis', fmt='.2f')
        plt.title(f'Learned {factor_name} matrix')
        plt.show()
import sys
sys.path.append("code")
from src.utils import *

def get_torch_data(results_all=None,
                   alphabet=None,
                   transcript_ids=None,
                   save_dir = "1KG/sequence_dict_all",
                   ):
    # Prepare Data for ESM2 Model
    global seq_to_proteoform, transcript_to_proteoform, sample_to_proteoform, proteoform_to_stopcodons, samples, batches
    seq_to_proteoform = {} # Maps transcript sequence to variant ID
    transcript_to_proteoform = {} # Maps transcript to variant IDs
    sample_to_proteoform = {} # Maps transcript,sample,phase to variant ID
    proteoform_to_stopcodons = {}
    samples = []
    batches = {}
    from tqdm.auto import tqdm
    def get_torch_data_i(transcript_id, results):
        data = []
        proteoform_counter = 1
        # print(transcript_id)
        for sample, (seq1, seq2) in tqdm(results['aa_seqs'].items(), 
                                         desc=f"{transcript_id}: Processing samples.", leave=False):
            samples.append(sample) 
            # Process both phases
            for phase, seq in [("phase1", seq1), ("phase2", seq2)]:
                # only use the substring up to the first stop codon
                if seq is not None:
                    seq = seq[:seq.find('*')]
                if seq not in seq_to_proteoform.keys() and seq is not None:
                    if transcript_id not in transcript_to_proteoform:
                        transcript_to_proteoform[transcript_id] = []
                    proteoform_id = create_proteoform_id(transcript_id, seq, alphabet)
                    seq_to_proteoform[seq] = proteoform_id
                    data.append((proteoform_id, seq))
                    proteoform_to_stopcodons[proteoform_id] = seq.count('*')
                    transcript_to_proteoform[transcript_id].append(proteoform_id)
                    proteoform_counter += 1
                sample_to_proteoform[(transcript_id,sample,phase)] = proteoform_id
        batches[transcript_id] = data

    # When using results_all
    import os
    import glob
    import pickle
    # Get transcript IDs
    if results_all is not None:
        print(f"Loading data from results_all")
        transcript_ids_x = results_all.keys()
    elif save_dir is not None:
        print(f"Loading data from {save_dir}")
        files = glob.glob(f"{save_dir}/*.pkl")
        transcript_ids_x = [os.path.basename(f).replace('.pkl','') for f in files]
    # Filter transcript IDs
    if transcript_ids is not None:
        transcript_ids_final = intersect(transcript_ids_x, transcript_ids)
    else:
        transcript_ids_final = transcript_ids_x
    # Iterate over transcripts
    for transcript_id in tqdm(transcript_ids_final,
                              desc=f"Processing transcripts"):
        if results_all is not None:
            results = results_all[transcript_id]
        elif save_dir is not None:
            results = pickle.load(open(f"{save_dir}/{transcript_id}.pkl",'rb'))
        get_torch_data_i(transcript_id, results)
    # Return vars
    return seq_to_proteoform, transcript_to_proteoform, sample_to_proteoform, proteoform_to_stopcodons, samples, batches

def get_embeddings(batches,
                   model=None,
                   alphabet=None,
                   repr_layers=[33],
                   force = False, 
                   max_transcripts = None,
                   verbose = False,
                   save_dir="1KG/embeddings/esm2_t33_650M_UR50D"):
    import os
    import pickle
    from tqdm.auto import tqdm
    import torch

    os.makedirs(save_dir, exist_ok=True)

    if model is None:
        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    if alphabet is None:
        alphabet = model.alphabet
    batch_converter = alphabet.get_batch_converter()
    # Init vars
    results = {}
    save_paths = {}
    failed_transcripts = []
    i = 1
    
    if max_transcripts!=None:
        batches = {k: v for k, v in list(batches.items())[:max_transcripts]}

    for transcript_id, data in tqdm(batches.items(), 
                                    desc=f"Embedding transcript proteoforms"): 
        save_path = os.path.abspath(f"{save_dir}/{transcript_id}.pkl")
        try:
            if os.path.exists(save_path) and force is not True:
                if verbose:
                    print(f"Loading existing file: {save_path}")
                with open(save_path,'rb') as handle:
                    results[transcript_id] = pickle.load(handle) 
            else:
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                # Generate Embeddings
                with torch.no_grad():
                    embeddings = model(batch_tokens, 
                                       repr_layers=repr_layers, 
                                       return_contacts=False)
                    results[transcript_id] = {
                        'embeddings':embeddings,
                        'batch_labels':batch_labels, 
                        'batch_strs':batch_strs,
                        'batch_tokens':batch_tokens
                    }
                # Save 
                with open(save_path,'wb') as handle:
                    pickle.dump(results[transcript_id],
                                handle)
                i += 1
            # Only add to save_paths if the file was created
            save_paths[transcript_id] = save_path
        except Exception as e: 
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

def get_representation_variances(seq_reps):
    import torch
    from tqdm.auto import tqdm
    transcript_variances = {}
    transcript_variance_means = {}
    transcript_ids = list(set([x[0].split('_')[0] for x in seq_reps]))
    # Compute the variance of each transcript embedding acrosss variants
    for transcript_id in tqdm(transcript_ids, desc="Computing transcript variances"):
        # print(transcript_id)
        variant_embeddings = [x[1] for x in seq_reps if x[0].split('_')[0] == transcript_id]
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
                nan_indices=None):
    import pandas as pd
    if embedding is None or nan_indices is None:
        X = get_representation_matrix(seq_reps)
        reducer, embedding, nan_indices = run_umap(X)
    embedding_df = pd.DataFrame(embedding, columns=['UMAP 1', 'UMAP 2'])
    embedding_df['label'] = [x[0] for i,x in enumerate(seq_reps) if not nan_indices[i]]
    embedding_df['transcript_id'] = [x.split('_')[0] for x in embedding_df['label']]
    embedding_df['transcript'] = [x.split('_')[0].split('.')[0] for x in embedding_df['label']]
    return embedding_df

def plot_umap(embedding_df,
              interact=False,
              opacity=0.1,
              facet_col=None,
              col_wrap=3,
              sharex=True,
              sharey=True,
              **kwargs): 
    if interact is True:
        import plotly.express as px
        fig = px.scatter(embedding_df, 
                         x='UMAP 1', 
                         y='UMAP 2',
                         facet_col=facet_col,
                         facet_col_wrap=col_wrap,
                         hover_data=['label'], 
                         size_max=10, 
                         opacity=opacity,  
                         **kwargs)
        # Add px.density_contour underneath the points
        fig.add_trace(px.density_contour(embedding_df, 
                                         x='UMAP 1', 
                                         y='UMAP 2',
                                         facet_col=facet_col,
                                         facet_col_wrap=col_wrap,
                                         **kwargs).data[0]
                                         )
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
                        color='white')
            
            # Set background color for each subplot
            for ax in g.axes.flat:
                ax.set_facecolor('#2F0154')
        else:
            plt.gca().set_facecolor('#2F0154') # Set background to slightly darker than darkest viridis color
            sns.kdeplot(data=embedding_df, x='UMAP 1', y='UMAP 2', 
                        fill=True, 
                        cmap='viridis',
                        **kwargs
                        ) 
            # Then overlay scatter points with some transparency
            plt.scatter(data=embedding_df,
                        x='UMAP 1', y='UMAP 2',
                        alpha=opacity, # Make points semi-transparent
                        s=1, # Small point size
                        color='white') # White points
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
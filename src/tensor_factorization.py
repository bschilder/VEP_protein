import torch
from tqdm.auto import tqdm


def get_patient_tensor(seq_reps,
                       sample_to_proteoform,
                       drop_nan=True):
    
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
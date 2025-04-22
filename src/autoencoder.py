import torch
from tqdm.auto import tqdm
import src.utils as utils


def make_autoencoder(input_dim, 
                     hidden_dims=[512, 256, 128, 64], 
                     latent_dim=32,
                     n_heads=4,
                     transformer_layers=2):
    """
    Create an autoencoder model with transformer layers to learn relationships
    between transcripts and phases for each patient.
    
    Parameters:
    -----------
    input_dim : int
        Dimension of the input features
    hidden_dims : list
        List of hidden dimensions for the encoder (decoder will be symmetric)
    latent_dim : int
        Dimension of the latent space
    n_heads : int
        Number of attention heads in transformer layers
    transformer_layers : int
        Number of transformer encoder layers
        
    Returns:
    --------
    model : torch.nn.Module
        Autoencoder model with transformer layers
    """
    import torch.nn as nn
    
    class PatientAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dims, latent_dim, n_heads, transformer_layers):
            super(PatientAutoencoder, self).__init__()
            
            # Feature embedding layers
            self.feature_embedding = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU()
            )
            
            # Transformer encoder to learn relationships between transcripts and phases
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dims[0],
                nhead=n_heads,
                dim_feedforward=hidden_dims[0] * 4,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers
            )
            
            # Build encoder layers after transformer
            encoder_layers = []
            prev_dim = hidden_dims[0]
            for dim in hidden_dims[1:]:
                encoder_layers.append(nn.Linear(prev_dim, dim))
                encoder_layers.append(nn.ReLU())
                prev_dim = dim
            encoder_layers.append(nn.Linear(prev_dim, latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)
            
            # Build decoder layers (symmetric to encoder)
            decoder_layers = []
            prev_dim = latent_dim
            for dim in reversed(hidden_dims[1:]):
                decoder_layers.append(nn.Linear(prev_dim, dim))
                decoder_layers.append(nn.ReLU())
                prev_dim = dim
            decoder_layers.append(nn.Linear(prev_dim, hidden_dims[0]))
            decoder_layers.append(nn.ReLU())
            self.decoder = nn.Sequential(*decoder_layers)
            
            # Final projection back to original feature space
            self.output_projection = nn.Linear(hidden_dims[0], input_dim)
            
            # Store dimensions for reshaping
            self.hidden_dims = hidden_dims
            self.latent_dim = latent_dim
        
        def forward(self, x, mask=None):
            # x shape: [batch_size, n_transcripts * n_phases, n_features]
            batch_size, seq_len, _ = x.shape
            
            # Embed features
            embedded = self.feature_embedding(x)  # [batch_size, seq_len, hidden_dim]
            
            # Apply transformer to learn relationships between transcripts and phases
            if mask is not None:
                transformed = self.transformer_encoder(embedded, src_key_padding_mask=mask)
            else:
                transformed = self.transformer_encoder(embedded)
            
            # Pool across sequence dimension (mean pooling)
            pooled = transformed.mean(dim=1)  # [batch_size, hidden_dim]
            
            # Encode to latent space
            encoded = self.encoder(pooled)  # [batch_size, latent_dim]
            
            # Decode from latent space
            decoded = self.decoder(encoded)  # [batch_size, hidden_dim]
            
            # Expand back to sequence length
            expanded = decoded.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
            
            # Project back to original feature space
            output = self.output_projection(expanded)  # [batch_size, seq_len, n_features]
            
            return output
        
        def encode(self, x, mask=None):
            batch_size, seq_len, _ = x.shape
            
            # Embed features
            embedded = self.feature_embedding(x)
            
            # Apply transformer
            if mask is not None:
                transformed = self.transformer_encoder(embedded, src_key_padding_mask=mask)
            else:
                transformed = self.transformer_encoder(embedded)
            
            # Pool across sequence dimension
            pooled = transformed.mean(dim=1)
            
            # Encode to latent space
            encoded = self.encoder(pooled)
            
            return encoded
    
    return PatientAutoencoder(input_dim, hidden_dims, latent_dim, n_heads, transformer_layers)

def train_autoencoder(patient_tensor, 
                      model,
                      batch_size=8,
                      learning_rate=1e-3,
                      num_epochs=100,
                      device=None,
                      save_path=None,
                      verbose=True):
    """
    Train a deep autoencoder to compress the patient tensor.
    
    Parameters:
    -----------
    patient_tensor : torch.Tensor
        The patient tensor with shape [n_samples, n_transcripts, n_phases, n_features]
    model : torch.nn.Module
        A self-supervised model
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate for optimizer
    num_epochs : int
        Number of epochs to train
    device : str
        Device to use for training ('cuda' or 'cpu')
    save_path : str
        Path to save the trained model
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    model : torch.nn.Module
        Trained autoencoder model
    history : dict
        Training history
    """
    
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get dimensions
    n_samples, n_transcripts, n_phases, n_features = patient_tensor.shape
    
    # Reshape tensor for transformer-based autoencoder
    # Combine transcripts and phases into a single sequence dimension
    X = patient_tensor.reshape(n_samples, n_transcripts * n_phases, n_features)
    
    # Create masks for padding (where all features are zero)
    padding_mask = ~torch.any(X != 0, dim=2)  # True where all features are zero
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, padding_mask)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True)
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=learning_rate)
    
    # Training loop
    history = {'loss': []}
    pbar = tqdm(range(num_epochs), 
                desc="Epochs completed") if verbose else range(num_epochs)
    
    for epoch in pbar:
        epoch_loss = 0
        for batch in dataloader:
            # Get batch
            x, mask = batch
            x = x.to(device)
            mask = mask.to(device)
            
            # Forward pass
            outputs = model(x, mask)
            
            # Calculate loss only on non-padded elements
            loss = 0
            for i in range(x.size(0)):  # For each sample in batch
                # Get non-padded positions
                valid_pos = ~mask[i]
                if valid_pos.sum() > 0:  # If there are valid positions
                    # Calculate MSE only on valid positions
                    sample_loss = criterion(outputs[i, valid_pos], x[i, valid_pos])
                    loss += sample_loss
            
            loss = loss / x.size(0)  # Average loss across batch
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x.size(0)
        
        # Calculate average loss for the epoch
        epoch_loss /= n_samples
        history['loss'].append(epoch_loss)
        
        if verbose and isinstance(pbar, tqdm):
            pbar.set_postfix({'loss': f'{epoch_loss:.6f}'})
    
    # Save model if path is provided
    if save_path is not None:
        utils.save_torch({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'input_dim': n_features,
            'history': history,
            'n_transcripts': n_transcripts,
            'n_phases': n_phases
        }, save_path)
    
    return model, history

def encode_patient_tensor(model, 
                          patient_tensor, 
                          device=None, 
                          batch_size=32,
                          save_path=None):
    """
    Encode the patient tensor using a trained autoencoder in batches.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained autoencoder model
    patient_tensor : torch.Tensor
        The patient tensor with shape [n_samples, n_transcripts, n_phases, n_features]
    device : str
        Device to use for encoding ('cuda' or 'cpu')
    batch_size : int
        Size of batches for processing
    save_path : str
        Path to save the encoded tensor
        
    Returns:
    --------
    encoded_tensor : torch.Tensor
        Encoded patient tensor with shape [n_samples, latent_dim]
    """
    import torch
    from tqdm.auto import tqdm
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get dimensions
    n_samples, n_transcripts, n_phases, n_features = patient_tensor.shape
    
    # Reshape tensor for transformer-based autoencoder
    X = patient_tensor.reshape(n_samples, n_transcripts * n_phases, n_features)
    
    # Create masks for padding (where all features are zero)
    padding_mask = ~torch.any(X != 0, dim=2)  # True where all features are zero
    
    # Prepare model
    model.eval()
    model = model.to(device)
    
    # Process in batches
    encoded_batches = []
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size),
                      desc="Encoding patient tensor",
                      total=n_samples // batch_size,
                      leave=False):
            # Get batch
            batch_end = min(i + batch_size, n_samples)
            X_batch = X[i:batch_end].to(device)
            mask_batch = padding_mask[i:batch_end].to(device)
            
            # Encode batch
            encoded_batch = model.encode(X_batch, mask_batch).cpu()
            encoded_batches.append(encoded_batch)
    
    # Concatenate all batches
    encoded_tensor = torch.cat(encoded_batches, dim=0)

    if save_path is not None:
        utils.save_torch(encoded_tensor, save_path) 
    
    return encoded_tensor

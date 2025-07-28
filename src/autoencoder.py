import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class XDataset(Dataset):
    def __init__(self, X, contrastive=False, noise_std=0.1):
        # Store as numpy for NaN handling, but keep original shape
        self.X = np.asarray(X)
        self.contrastive = contrastive
        self.noise_std = noise_std

    def __len__(self):
        return self.X.shape[0]

    def _get_tensor_and_mask(self, idx):
        x_np = self.X[idx]
        mask = ~np.isnan(x_np)
        # Replace NaNs with zero for input, mask will be used in loss
        x = torch.tensor(np.nan_to_num(x_np, nan=0.0), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        return x, mask

    def __getitem__(self, idx):
        if self.contrastive:
            x, mask = self._get_tensor_and_mask(idx)
            # Add noise only to observed values
            noise1 = torch.randn_like(x) * self.noise_std * mask
            noise2 = torch.randn_like(x) * self.noise_std * mask
            x1 = x + noise1
            x2 = x + noise2
            # Return masks for both views and targets
            return x1, x2, x, x, mask, mask  # (view1, view2, original1, original2, mask1, mask2)
        else:
            x, mask = self._get_tensor_and_mask(idx)
            return x, x, mask  # input, target, mask

# Use a simple autoencoder (not UNet) for tabular data
class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim, 
                 embedding_dim=2,
                 hidden_dims=[128, 64, 32]):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.Linear(hidden_dims[2], embedding_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[2]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Linear(hidden_dims[0], input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

def masked_mse_loss(pred, target, mask):
    # mask: 1 for observed, 0 for missing
    # Only compute loss on observed values
    diff = (pred - target) * mask
    mse = (diff ** 2).sum() / (mask.sum() + 1e-8)
    return mse

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for contrastive learning.
    z1, z2: (batch_size, embedding_dim)
    """
    batch_size = z1.size(0)
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)  # (2*batch_size, embedding_dim)
    similarity_matrix = torch.matmul(representations, representations.T)  # (2*batch_size, 2*batch_size)
    # Remove self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    similarity_matrix = similarity_matrix / temperature
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

    # For each anchor, the positive is at index: (i + batch_size) % (2*batch_size)
    positive_indices = (torch.arange(2 * batch_size, device=z1.device) + batch_size) % (2 * batch_size)
    logits = similarity_matrix
    loss = nn.CrossEntropyLoss()(logits, positive_indices)
    return loss

def prepare_data(X, normalize=True):
    import pandas as pd
    # Optionally scale the data to avoid degenerate (zero) embeddings
    # Fit scaler only on observed values
    if isinstance(X, pd.DataFrame):
        X_np = np.asarray(X.values)
    else:
        X_np = np.asarray(X)

    mask = ~np.isnan(X_np)
    if normalize:
        scaler = StandardScaler()
        # Flatten to 1D for scaler, then reshape
        X_flat = X_np[mask].reshape(-1, 1)
        scaler.fit(X_flat)
        # Transform, but keep NaNs
        X_scaled = X_np.copy()
        X_scaled[mask] = scaler.transform(X_np[mask].reshape(-1, 1)).flatten()
    else:
        scaler = None
        X_scaled = X_np.copy()
    return X_scaled, mask, scaler

def train_autoencoder(X, 
                      n_epochs=100, 
                      shuffle=True,
                      embedding_dim=2,
                      hidden_dims=[128, 64, 32],
                      batch_size=32, 
                      lr=1e-3, 
                      weight_decay=1e-5,
                      use_contrastive=False,
                      contrastive_weight=1.0,
                      temperature=0.5,
                      noise_std=0.1, 
                      seed=42,
                      normalize=True):
    """
    If use_contrastive is True, use a combination of reconstruction and contrastive loss.
    Handles NaNs in X by masking them during loss computation.
    Set normalize=False to keep original scales (no per-column normalization).
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    X_scaled, mask, scaler = prepare_data(X, normalize=normalize)

    # Prepare data
    dataset = XDataset(X_scaled, contrastive=use_contrastive, noise_std=noise_std)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle)

    # Model, optimizer, loss
    input_dim = X.shape[1]
    autoencoder = TabularAutoencoder(input_dim=input_dim, 
                                     embedding_dim=embedding_dim, 
                                     hidden_dims=hidden_dims)
    
    autoencoder = autoencoder.to(device)
    optimizer = optim.Adam(autoencoder.parameters(), 
                           lr=lr, 
                           weight_decay=weight_decay)

    # Training loop 
    for epoch in range(n_epochs):
        autoencoder.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            if use_contrastive:
                x1, x2, y1, y2, mask1, mask2 = batch
                x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
                mask1, mask2 = mask1.to(device), mask2.to(device)
                out1, z1 = autoencoder(x1)
                out2, z2 = autoencoder(x2)
                # Reconstruction loss for both views, only on observed values
                recon_loss = (masked_mse_loss(out1, y1, mask1) + masked_mse_loss(out2, y2, mask2)) / 2
                # Contrastive loss
                contrastive_loss = nt_xent_loss(z1, z2, temperature=temperature)
                loss = recon_loss + contrastive_weight * contrastive_loss
                total_loss += loss.item() * x1.size(0)
            else:
                xb, yb, maskb = batch
                xb, yb, maskb = xb.to(device), yb.to(device), maskb.to(device)
                out, _ = autoencoder(xb)
                loss = masked_mse_loss(out, yb, maskb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
                continue  # skip the rest, already stepped
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(dataset)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    # Get embeddings
    autoencoder.eval()
    with torch.no_grad():
        # Prepare X_tensor with NaNs replaced by zero
        X_tensor = torch.tensor(np.nan_to_num(X_scaled, nan=0.0), dtype=torch.float32).to(device)
        _, embeddings = autoencoder(X_tensor)
        embeddings = embeddings.cpu().numpy()

    # Compute reconstruction error
    reconstruction_error = compute_reconstruction_error(autoencoder, 
                                                        X_scaled, 
                                                        mask=mask)
    print(f"Reconstruction error: {reconstruction_error:.4f}")

    return {    "embeddings": embeddings, 
                "X_tensor": X_tensor, 
                "X_scaled": X_scaled,
                "model": autoencoder,
                "scaler": scaler,
                "mask": mask,
                "reconstruction_error": reconstruction_error}

def compute_reconstruction_error(autoencoder, X, mask=None, device='cpu'):
    """
    Compute the mean squared reconstruction error for X.
    If mask is provided, only compute error on observed values.
    X: numpy array or torch tensor, shape (n_samples, n_features)
    mask: numpy array or torch tensor, same shape as X, 1 for observed, 0 for missing

    Example:
        >>> # Assume autoencoder is a trained model, X_test is a numpy array
        >>> # Optionally, mask_test is a numpy array with 1 for observed, 0 for missing
        >>> error = compute_reconstruction_error(autoencoder, X_test)
        >>> print("Reconstruction error (all values):", error)
        >>> # With mask
        >>> error_masked = compute_reconstruction_error(autoencoder, X_test, mask=mask_test)
        >>> print("Reconstruction error (observed only):", error_masked)
    """
    # Ensure model is on the correct device
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    with torch.no_grad():
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(np.nan_to_num(X, nan=0.0), dtype=torch.float32)
        else:
            X_tensor = X.detach().clone()
        X_tensor = X_tensor.to(device)
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask_tensor = torch.tensor(mask, dtype=torch.float32)
            else:
                mask_tensor = mask.detach().clone()
            mask_tensor = mask_tensor.to(device)
        out, _ = autoencoder(X_tensor)
        if mask is not None:
            mse = ((out - X_tensor) ** 2) * mask_tensor
            error = mse.sum() / mask_tensor.sum()
        else:
            mse = (out - X_tensor) ** 2
            error = mse.mean()
    return error.item()

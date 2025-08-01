import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class XDataset(Dataset):
    """
    PyTorch Dataset for tabular data with optional support for contrastive learning and missing values.

    Args:
        X (np.ndarray or pd.DataFrame): Input data of shape (n_samples, n_features). Can contain NaNs.
        contrastive (bool): If True, returns two noisy views of each sample for contrastive learning.
        noise_std (float): Standard deviation of Gaussian noise added to observed values for contrastive views.

    Behavior:
        - For each sample, missing values (NaN) are replaced with zero for input, and a mask is generated (1 for observed, 0 for missing).
        - If contrastive=True, returns two noisy views (x1, x2), both original targets (x, x), and both masks.
        - If contrastive=False, returns (x, x, mask) for standard autoencoder training.
    """
    def __init__(self, X, contrastive=False, noise_std=0.1):
        # Store as numpy for NaN handling, but keep original shape
        self.X = np.asarray(X)
        self.contrastive = contrastive
        self.noise_std = noise_std

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return self.X.shape[0]

    def _get_tensor_and_mask(self, idx):
        """
        Internal helper to get a single sample and its mask.

        Args:
            idx (int): Index of the sample.

        Returns:
            x (torch.Tensor): Input vector with NaNs replaced by zero, shape (n_features,).
            mask (torch.Tensor): Mask vector, 1 for observed, 0 for missing, shape (n_features,).
        """
        x_np = self.X[idx]
        mask = ~np.isnan(x_np)
        # Replace NaNs with zero for input, mask will be used in loss
        x = torch.tensor(np.nan_to_num(x_np, nan=0.0), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        return x, mask

    def __getitem__(self, idx):
        """
        Get a sample for training.

        Args:
            idx (int): Index of the sample.

        Returns:
            If contrastive:
                x1 (Tensor): First noisy view of input, shape (n_features,).
                x2 (Tensor): Second noisy view of input, shape (n_features,).
                x (Tensor): Original input (target for both views), shape (n_features,).
                x (Tensor): Original input (target for both views), shape (n_features,).
                mask (Tensor): Mask for first view, shape (n_features,).
                mask (Tensor): Mask for second view, shape (n_features,).
            If not contrastive:
                x (Tensor): Input vector, shape (n_features,).
                x (Tensor): Target vector (same as input), shape (n_features,).
                mask (Tensor): Mask vector, shape (n_features,).
        """
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

class TabularAutoencoder(nn.Module):
    """
    Simple feedforward autoencoder for tabular data.

    Args:
        input_dim (int): Number of input features.
        embedding_dim (int): Size of the latent embedding (default: 2).
        hidden_dims (list of int): List of hidden layer sizes (default: [128, 64, 32]).

    Architecture:
        - Encoder: MLP with hidden layers and batch normalization, outputs embedding of size embedding_dim.
        - Decoder: MLP with hidden layers and batch normalization, reconstructs input from embedding.

    Forward:
        x (Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            out (Tensor): Reconstructed input, shape (batch_size, input_dim).
            z (Tensor): Latent embedding, shape (batch_size, embedding_dim).
    """
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
        """
        Forward pass through the autoencoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            out (Tensor): Reconstructed input, shape (batch_size, input_dim).
            z (Tensor): Latent embedding, shape (batch_size, embedding_dim).
        """
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

def masked_mse_loss(pred, target, mask):
    """
    Compute mean squared error (MSE) loss between pred and target, only over observed (non-missing) values.

    Args:
        pred (Tensor): Predicted output, shape (..., n_features).
        target (Tensor): Target output, shape (..., n_features).
        mask (Tensor): Mask tensor, 1 for observed, 0 for missing, shape (..., n_features).

    Returns:
        mse (Tensor): Scalar MSE loss over observed values.
    """
    # mask: 1 for observed, 0 for missing
    # Only compute loss on observed values
    diff = (pred - target) * mask
    mse = (diff ** 2).sum() / (mask.sum() + 1e-8)
    return mse

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Compute the Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for contrastive learning.

    This loss is used in self-supervised contrastive learning (e.g., SimCLR).
    For a batch of paired embeddings (z1, z2), encourages each pair to be close in embedding space,
    while pushing apart all other samples in the batch.

    Args:
        z1 (Tensor): First set of embeddings, shape (batch_size, embedding_dim).
        z2 (Tensor): Second set of embeddings, shape (batch_size, embedding_dim).
        temperature (float): Temperature parameter for scaling similarities.

    Returns:
        loss (Tensor): Scalar NT-Xent loss.
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

def pairwise_distance_matrix(x, metric='euclidean'):
    """
    Compute the pairwise distance matrix for a batch of vectors.

    Args:
        x (Tensor): Input tensor of shape (batch_size, dim).
        metric (str): Distance metric to use. One of:
            - 'euclidean': Euclidean (L2) distance.
            - 'cosine': Cosine distance (1 - cosine similarity).

    Returns:
        dist (Tensor): Pairwise distance matrix of shape (batch_size, batch_size).

    Raises:
        ValueError: If an unsupported metric is specified.

    Notes:
        - For Euclidean, uses efficient broadcasting to compute all pairwise distances.
        - For cosine, returns 1 - cosine similarity.
    """
    if metric == 'euclidean':
        # x: (N, D)
        # dist(i, j) = sqrt(sum_k (x[i,k] - x[j,k])^2)
        # Efficient computation using broadcasting:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        x_square = (x ** 2).sum(dim=1, keepdim=True)  # (N, 1)
        dist2 = x_square + x_square.t() - 2 * torch.matmul(x, x.t())
        dist2 = torch.clamp(dist2, min=0.0)
        dist = torch.sqrt(dist2 + 1e-8)
        return dist
    elif metric == 'cosine':
        # Cosine distance: 1 - cosine similarity
        x_norm = nn.functional.normalize(x, dim=1)
        sim = torch.matmul(x_norm, x_norm.t())
        dist = 1 - sim
        return dist
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def pairwise_distance_loss(x_high, x_low, mask=None, metric='euclidean'):
    """
    Loss between pairwise distance matrices of high-dim (input) and low-dim (embedding) representations.
    x_high: (batch_size, input_dim)
    x_low: (batch_size, embedding_dim)
    mask: (batch_size, input_dim) or None. If provided, only use rows with all observed values.
    metric: 'euclidean' or 'cosine'
    Returns: scalar loss
    """
    # If mask is provided, only use rows with all observed values
    if mask is not None:
        # mask: (batch_size, input_dim)
        # Only keep rows where all features are observed (mask==1)
        row_mask = (mask.sum(dim=1) == mask.size(1))
        if row_mask.sum() < 2:
            # Not enough fully observed rows, skip loss
            return torch.tensor(0.0, device=x_high.device, dtype=x_high.dtype)
        x_high = x_high[row_mask]
        x_low = x_low[row_mask]
    if x_high.size(0) < 2:
        return torch.tensor(0.0, device=x_high.device, dtype=x_high.dtype)
    D_high = pairwise_distance_matrix(x_high, metric=metric)
    D_low = pairwise_distance_matrix(x_low, metric=metric)
    # Normalize distances to mean 1 to avoid scale issues
    D_high = D_high / (D_high.mean() + 1e-8)
    D_low = D_low / (D_low.mean() + 1e-8)
    # Use MSE between distance matrices (excluding diagonal)
    mask_offdiag = ~torch.eye(D_high.size(0), dtype=torch.bool, device=x_high.device)
    loss = ((D_high[mask_offdiag] - D_low[mask_offdiag]) ** 2).mean()
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

def train_autoencoder(
    X, 
    n_epochs=100, 
    shuffle=True,
    embedding_dim=2,
    hidden_dims=[128, 64, 32],
    batch_size=32, 
    lr=1e-3, 
    weight_decay=1e-5,
    use_contrastive=False,
    contrastive_loss_weight=1.0,
    pairwise_loss_weight=1.0,
    reconstruction_loss_weight=1.0,
    temperature=0.5,
    noise_std=0.1, 
    seed=42,
    normalize=True,
    device=None,
    metric='euclidean'
):
    """
    Train a tabular autoencoder (optionally with contrastive and pairwise distance losses).

    Args:
        X (np.ndarray or pd.DataFrame): Input data of shape (n_samples, n_features). Can contain NaNs.
        n_epochs (int): Number of training epochs.
        shuffle (bool): Whether to shuffle the dataset each epoch.
        embedding_dim (int): Dimension of the embedding (bottleneck) layer.
        hidden_dims (list of int): List of hidden layer sizes for the encoder/decoder.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) for the optimizer.
        use_contrastive (bool): If True, use a combination of reconstruction and contrastive loss.
        contrastive_loss_weight (float): Weight for the contrastive loss (if use_contrastive=True).
        pairwise_loss_weight (float): Weight for the pairwise distance loss between input and embedding spaces.
        reconstruction_loss_weight (float): Weight for the high-dimensional (input space) reconstruction loss.
        temperature (float): Temperature parameter for contrastive loss.
        noise_std (float): Standard deviation of noise for contrastive augmentation.
        seed (int): Random seed for reproducibility.
        normalize (bool): If True, normalize each feature (column) to zero mean and unit variance.

    Returns:
        dict: Dictionary containing:
            - "embeddings": The learned low-dimensional representations (np.ndarray).
            - "X_tensor": The input data as a torch tensor (with NaNs replaced by zero).
            - "X_scaled": The (optionally normalized) input data as a numpy array.
            - "model": The trained autoencoder model.
            - "scaler": The fitted scaler (if normalization was used), else None.
            - "mask": Boolean mask of observed values (True for observed, False for NaN).
            - "reconstruction_error": Mean squared error on observed values.

    Notes:
        - Handles NaNs in X by masking them during loss computation.
        - Set normalize=False to keep original scales (no per-column normalization).
        - If use_contrastive is True, uses a combination of reconstruction, contrastive, and pairwise distance losses.
        - The returned "embeddings" are the output of the encoder for the full dataset.
    """
    from tqdm import trange, tqdm

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

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

    # Training loop with tqdm showing average loss per epoch (works in Jupyter)
    from tqdm.notebook import trange, tqdm

    epoch_bar = trange(n_epochs, desc="Epochs", leave=True)
    for epoch in epoch_bar:
        autoencoder.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Batches", leave=False):
            optimizer.zero_grad()
            if use_contrastive:
                x1, x2, y1, y2, mask1, mask2 = batch
                x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
                mask1, mask2 = mask1.to(device), mask2.to(device)
                out1, z1 = autoencoder(x1)
                out2, z2 = autoencoder(x2)
                # Only compute losses if their weights are nonzero
                loss = 0.0
                if reconstruction_loss_weight != 0:
                    recon_loss = (masked_mse_loss(out1, y1, mask1) + masked_mse_loss(out2, y2, mask2)) / 2
                    loss = loss + reconstruction_loss_weight * recon_loss
                if contrastive_loss_weight != 0:
                    contrastive_loss = nt_xent_loss(z1, z2, temperature=temperature)
                    loss = loss + contrastive_loss_weight * contrastive_loss
                if pairwise_loss_weight != 0:
                    pairwise_loss1 = pairwise_distance_loss(x1, z1, mask1, metric=metric)
                    pairwise_loss2 = pairwise_distance_loss(x2, z2, mask2, metric=metric)
                    pairwise_loss = (pairwise_loss1 + pairwise_loss2) / 2
                    loss = loss + pairwise_loss_weight * pairwise_loss
                total_loss += loss.item() * x1.size(0)
            else:
                xb, yb, maskb = batch
                xb, yb, maskb = xb.to(device), yb.to(device), maskb.to(device)
                out, z = autoencoder(xb)
                loss = 0.0
                if reconstruction_loss_weight != 0:
                    recon_loss = masked_mse_loss(out, yb, maskb)
                    loss = loss + reconstruction_loss_weight * recon_loss
                if pairwise_loss_weight != 0:
                    pairwise_loss = pairwise_distance_loss(xb, z, maskb)
                    loss = loss + pairwise_loss_weight * pairwise_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
                continue  # skip the rest, already stepped
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(dataset)
        epoch_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
 
        # if (epoch+1) % 10 == 0 or epoch == 0:
        #     print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

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
    Compute the mean squared reconstruction error for a trained autoencoder.

    Args:
        autoencoder (nn.Module): Trained autoencoder model.
        X (np.ndarray or torch.Tensor): Input data of shape (n_samples, n_features).
            Can be a numpy array (with possible NaNs) or a torch tensor.
        mask (np.ndarray or torch.Tensor, optional): Boolean or float mask of same shape as X.
            1/True for observed values, 0/False for missing. If None, computes error on all values.
        device (str or torch.device): Device to run the computation on.

    Returns:
        float: Mean squared reconstruction error (averaged over observed values if mask is provided).

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
 


class UNet1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_bn=True):
        super(UNet1DBlock, self).__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(out_channels) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm1d(out_channels) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)

class UNet1DAutoencoder(nn.Module):
    """
    A UNet-based Autoencoder for 1D data (e.g., signals, sequences).
    Input shape: (batch, channels, length)
    """
    def __init__(self, in_channels=1, base_channels=32, latent_dim=2, seq_len=128, embedding_size=2):
        super(UNet1DAutoencoder, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = UNet1DBlock(base_channels, base_channels*2, down=True)
        self.enc3 = UNet1DBlock(base_channels*2, base_channels*4, down=True)
        self.enc4 = UNet1DBlock(base_channels*4, base_channels*8, down=True)
        # Bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv1d(base_channels*8, base_channels*8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Embedding projection to 2D (or embedding_size)
        self.embedding_proj = nn.Conv1d(base_channels*8, embedding_size, kernel_size=1)
        # Decoder
        self.dec4 = UNet1DBlock(embedding_size, base_channels*4, down=False)
        self.dec3 = UNet1DBlock(base_channels*8, base_channels*2, down=False)  # skip connection
        self.dec2 = UNet1DBlock(base_channels*4, base_channels, down=False)    # skip connection
        self.dec1 = nn.Sequential(
            nn.Conv1d(base_channels*2, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        # Bottleneck
        b = self.bottleneck_conv(e4)
        embedding = self.embedding_proj(b)  # shape: (batch, embedding_size, L_bottleneck)
        # Decoder with skip connections
        d4 = self.dec4(embedding)
        d4_cat = torch.cat([d4, e3], dim=1)
        d3 = self.dec3(d4_cat)
        d3_cat = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3_cat)
        d2_cat = torch.cat([d2, e1], dim=1)
        out = self.dec1(d2_cat)
        # For 2D embedding, flatten spatial dimension (mean over L_bottleneck)
        embedding_flat = embedding.mean(dim=2)  # shape: (batch, embedding_size)
        return out, embedding_flat

class UNet1DAutoencoderWithBottleneck(nn.Module):
    """
    A UNet-based Autoencoder for 1D data (e.g., signals) with configurable bottleneck size and embedding size.
    """
    def __init__(self, in_channels=1, base_channels=32, bottleneck_channels=None, latent_dim=2, seq_len=128, embedding_size=2):
        super().__init__()
        if bottleneck_channels is None:
            bottleneck_channels = base_channels * 8
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = UNet1DBlock(base_channels, base_channels*2, down=True)
        self.enc3 = UNet1DBlock(base_channels*2, base_channels*4, down=True)
        self.enc4 = UNet1DBlock(base_channels*4, bottleneck_channels, down=True)
        self.bottleneck_conv = nn.Sequential(
            nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.embedding_proj = nn.Conv1d(bottleneck_channels, embedding_size, kernel_size=1)
        self.dec4 = UNet1DBlock(embedding_size, base_channels*4, down=False)
        self.dec3 = UNet1DBlock(base_channels*8, base_channels*2, down=False)  # skip connection
        self.dec2 = UNet1DBlock(base_channels*4, base_channels, down=False)    # skip connection
        self.dec1 = nn.Sequential(
            nn.Conv1d(base_channels*2, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck_conv(e4)
        embedding = self.embedding_proj(b)  # shape: (batch, embedding_size, L_bottleneck)
        d4 = self.dec4(embedding)
        d4_cat = torch.cat([d4, e3], dim=1)
        d3 = self.dec3(d4_cat)
        d3_cat = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3_cat)
        d2_cat = torch.cat([d2, e1], dim=1)
        out = self.dec1(d2_cat)
        embedding_flat = embedding.mean(dim=2)  # shape: (batch, embedding_size)
        return out, embedding_flat

def train_unet_1d(
        X, 
        n_epochs=100, 
        shuffle=True,
        batch_size=32, 
        lr=1e-3, 
        weight_decay=1e-5,
        seed=42,
        normalize=True,
        seq_len=128,
        in_channels=1,
        base_channels=32,
        latent_dim=2,
        bottleneck_channels=None,
        embedding_size=2,
        device=None
    ):
    """
    Train a UNet-based autoencoder for 1D data (e.g., signals, sequences).

    Args:
        X (np.ndarray): Input data of shape (n_samples, channels, length) or (n_samples, length).
        n_epochs (int): Number of training epochs.
        shuffle (bool): Whether to shuffle the dataset each epoch.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) for the optimizer.
        seed (int): Random seed for reproducibility.
        normalize (bool): If True, normalize each sequence to [0, 1].
        seq_len (int): Length of the 1D sequence.
        in_channels (int): Number of input channels.
        base_channels (int): Base number of channels for UNet.
        latent_dim (int): Latent dimension (not used directly, for compatibility).
        bottleneck_channels (int or None): Number of channels in the bottleneck layer. If None, defaults to base_channels*8.
        embedding_size (int): Size of the embedding vector from the bottleneck (default 2).
        device (str or torch.device): Device to use.

    Returns:
        dict: Dictionary containing:
            - "reconstructions": The reconstructed sequences (np.ndarray).
            - "X_tensor": The input data as a torch tensor.
            - "embeddings": The bottleneck embeddings (np.ndarray, shape [n_samples, embedding_size]).
            - "model": The trained UNet1DAutoencoder model.
            - "reconstruction_error": Mean squared error on the dataset.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import trange, tqdm

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Prepare data
    X_np = np.asarray(X)
    if X_np.ndim == 2:
        # (n_samples, L) -> (n_samples, 1, L)
        X_np = X_np[:, None, :]
    elif X_np.ndim == 3:
        # (n_samples, C, L)
        pass
    else:
        raise ValueError("Input X must have shape (n_samples, L) or (n_samples, C, L)")

    if normalize:
        X_np = X_np.astype(np.float32)
        X_min = X_np.min()
        X_max = X_np.max()
        if X_max > X_min:
            X_np = (X_np - X_min) / (X_max - X_min)
        else:
            X_np = np.zeros_like(X_np, dtype=np.float32)

    X_tensor = torch.from_numpy(X_np).float().to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False
    ) 

    model = UNet1DAutoencoderWithBottleneck(
        in_channels=in_channels,
        base_channels=base_channels,
        bottleneck_channels=bottleneck_channels,
        latent_dim=latent_dim,
        seq_len=seq_len,
        embedding_size=embedding_size
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model.train()
    for epoch in trange(n_epochs, desc="Epochs"):
        epoch_loss = 0.0
        for (batch_x,) in tqdm(data_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        # Optionally print progress
        # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss / len(X_tensor):.6f}")
    
    # Get embeddings
    model.eval()
    with torch.no_grad():
        # Prepare X_tensor with NaNs replaced by zero
        X_tensor = torch.tensor(np.nan_to_num(X_np, nan=0.0), dtype=torch.float32).to(device)
        # Forward pass to get embeddings
        output = model(X_tensor)
        if isinstance(output, tuple) and len(output) == 2:
            _, embeddings = output
        else:
            embeddings = output
        embeddings = embeddings.cpu().numpy()

    # Evaluation
    model.eval()
    with torch.no_grad():
        reconstructions, _ = model(X_tensor)
        reconstructions_np = reconstructions.cpu().numpy()
        mse = ((reconstructions_np - X_np) ** 2).mean()

    return {
        "reconstructions": reconstructions_np,
        "X_tensor": X_tensor,
        "embeddings": embeddings,
        "model": model,
        "reconstruction_error": mse
    }


class TabularVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for tabular data.

    Args:
        input_dim (int): Number of input features.
        embedding_dim (int): Dimensionality of the latent space (default: 2).
        hidden_dims (list of int): Sizes of hidden layers in the encoder/decoder (default: [128, 64, 32]).

    Architecture:
        - Encoder: MLP with hidden layers, outputs mean and log-variance for latent distribution.
        - Decoder: MLP reconstructs input from latent variable.

    Methods:
        encode(x): Returns mean and log-variance of latent distribution for input x.
        reparameterize(mu, logvar): Samples latent variable z using the reparameterization trick.
        forward(x): Returns reconstruction, mean, log-variance, and latent variable for input x.
    """
    def __init__(self, input_dim, embedding_dim=2, hidden_dims=[128, 64, 32]):
        super().__init__()
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[2])
        )
        self.fc_mu = nn.Linear(hidden_dims[2], embedding_dim)
        self.fc_logvar = nn.Linear(hidden_dims[2], embedding_dim)
        # Decoder network
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

    def encode(self, x):
        """
        Encode input x into mean and log-variance of the latent distribution.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            mu (Tensor): Mean of latent distribution.
            logvar (Tensor): Log-variance of latent distribution.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Sample from latent distribution using the reparameterization trick.

        Args:
            mu (Tensor): Mean of latent distribution.
            logvar (Tensor): Log-variance of latent distribution.

        Returns:
            z (Tensor): Sampled latent variable.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            recon (Tensor): Reconstructed input.
            mu (Tensor): Mean of latent distribution.
            logvar (Tensor): Log-variance of latent distribution.
            z (Tensor): Sampled latent variable.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Compute the VAE loss as the sum of reconstruction loss and (weighted) KL divergence.

    Args:
        recon_x (Tensor): Reconstructed input.
        x (Tensor): Original input.
        mu (Tensor): Mean of latent distribution.
        logvar (Tensor): Log-variance of latent distribution.
        beta (float): Weight for KL divergence term (default: 1.0).

    Returns:
        total_loss (Tensor): Total VAE loss.
        recon_loss (Tensor): Reconstruction loss (MSE).
        kl_div (Tensor): KL divergence loss.

    Notes:
        - The beta parameter controls the trade-off between reconstruction accuracy and the regularization of the latent space.
        - Increasing beta puts more emphasis on the KL divergence term, encouraging the latent embeddings to be closer to a standard normal distribution (more "regularized" and disentangled, but possibly less faithful reconstructions).
        - Decreasing beta allows the model to focus more on reconstruction, which can lead to less regularized (and potentially more entangled) latent embeddings.
        - In summary, higher beta values produce more "compressed" and regularized embeddings, while lower beta values allow embeddings to capture more information about the input at the cost of less regularization.
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # KL divergence between approximate posterior and prior
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div, recon_loss, kl_div

def train_vae(
    X,
    n_epochs=100,
    embedding_dim=2,
    hidden_dims=[128, 64, 32],
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-5,
    beta=1.0,
    seed=42,
    normalize=True,
    device=None
):
    """
    Train a Variational Autoencoder (VAE) for tabular data.

    Args:
        X (np.ndarray or pd.DataFrame): Input data of shape (n_samples, n_features). Can contain NaNs.
        n_epochs (int): Number of training epochs.
        embedding_dim (int): Dimensionality of the latent space.
        hidden_dims (list of int): List of hidden layer sizes for encoder/decoder.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for optimizer.
        weight_decay (float): Weight decay (L2 regularization) for optimizer.
        beta (float): Weight for KL divergence term in the loss.
        seed (int): Random seed for reproducibility.
        normalize (bool): If True, normalize each feature to zero mean and unit variance.
        device (str or torch.device or None): Device to use ("cuda", "cpu", or None for auto).

    Returns:
        dict: Dictionary containing:
            - "reconstructions": Reconstructed data as a numpy array.
            - "X_tensor": Input data as a torch tensor (with NaNs replaced by zero).
            - "embeddings": Latent mean representations (numpy array).
            - "model": Trained VAE model.
            - "reconstruction_error": Mean squared error on observed values.
            - "scaler": Fitted scaler (if normalization was used), else None.
            - "mask": Boolean mask of observed values (True for observed, False for NaN).

    Notes:
        - Handles NaNs in X by masking them during loss computation and replacing with zero for input.
        - The returned "embeddings" are the mean of the latent distribution for the full dataset.
        - If normalize=True, each feature is scaled to zero mean and unit variance.
    """
    from tqdm import trange, tqdm
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Prepare data (normalize and mask NaNs)
    if isinstance(X, pd.DataFrame):
        X_np = np.asarray(X.values)
    else:
        X_np = np.asarray(X)
    mask = ~np.isnan(X_np)
    if normalize:
        scaler = StandardScaler()
        X_flat = X_np[mask].reshape(-1, 1)
        scaler.fit(X_flat)
        X_scaled = X_np.copy()
        X_scaled[mask] = scaler.transform(X_np[mask].reshape(-1, 1)).flatten()
    else:
        scaler = None
        X_scaled = X_np.copy()

    # Replace NaNs with zero for input
    X_tensor = torch.tensor(np.nan_to_num(X_scaled, nan=0.0), dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_tensor.shape[1]
    model = TabularVAE(input_dim=input_dim, embedding_dim=embedding_dim, hidden_dims=hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    model.train()
    for epoch in trange(n_epochs, desc="VAE Epochs"):
        epoch_loss = 0.0
        for (batch_x,) in tqdm(data_loader, desc=f"VAE Epoch {epoch+1}/{n_epochs}", leave=False):
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar, _ = model(batch_x)
            loss, recon_loss, kl_div = vae_loss_function(recon, batch_x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        # Optionally print progress
        # print(f"VAE Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss / len(X_tensor):.6f}")

    # Get embeddings and reconstructions
    model.eval()
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        recon, mu, logvar, z = model(X_tensor)
        embeddings = mu.cpu().numpy()
        reconstructions_np = recon.cpu().numpy()
        # Compute MSE on observed values only
        mse = ((reconstructions_np - X_np)[mask] ** 2).mean()

    return {
        "reconstructions": reconstructions_np,
        "X_tensor": X_tensor,
        "embeddings": embeddings,
        "model": model,
        "reconstruction_error": mse,
        "scaler": scaler,
        "mask": mask
    }

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest
from tqdm import tqdm

import src.vep_analysis as va
import src.utils as utils

def wtvariants_to_clusters_xgboost(dr_df, 
                                   target = "cluster",
                                   random_state=42,
                                   haplotype_col="haplotype",
                                   wt_variant_split="[,]|[|]",
                                   device="cuda"):
    # Random Forest and XGBoost are both ensemble machine learning algorithms based on decision trees,
    #  but they differ in how they build and combine those trees.

    # Random Forest:
    # - Builds many decision trees independently in parallel (bagging).
    # - Each tree is trained on a random subset of the data and features.
    # - The final prediction is made by averaging (regression) or majority vote (classification) across all trees.
    # - Tends to reduce variance and helps prevent overfitting.

    # XGBoost:
    # - Builds trees sequentially, where each new tree tries to correct the errors of the previous trees (boosting).
    # - Uses gradient boosting, optimizing a loss function using gradient descent.
    # - Incorporates regularization to reduce overfitting and can handle missing data natively.
    # - Generally achieves higher accuracy and is faster due to optimizations, but is more complex and sensitive to hyperparameters.

    # In summary: Random Forest uses parallel, independent trees (bagging), while XGBoost uses sequential, dependent trees (boosting) with advanced optimizations.
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # Prepare features and target
    # One-hot encode wt_variant (multi-hot, as haplotypes can have multiple variants)
    X_wt = get_wt_variant_matrix(dr_df,
                                 haplotype_col=haplotype_col,
                                 wt_variant_split=wt_variant_split)

    # Target: cluster
    y = dr_df[target]

    # Remove any rows with cluster label '-1' (DBSCAN noise)
    mask_valid = y != "-1"
    X_wt = X_wt.loc[mask_valid]
    y = y.loc[mask_valid]

    # Convert cluster labels to integer type for classification
    y_int = y.astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_wt, y_int, 
                                                        test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y_int)

    # Train XGBoost classifier with GPU support using new device parameter
    model = xgb.XGBClassifier( 
        eval_metric="mlogloss",
        random_state=random_state,
        tree_method="hist",         # Use standard histogram method
        device=device               # Set device to CUDA for GPU acceleration
    )
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Feature importance: which wt_variants are most predictive of cluster
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "wt_variant": X_wt.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("Top predictive wt_variants for cluster assignment:")
    print(feature_importance_df.head(10))

    return feature_importance_df, model



# Define a simple feedforward neural network 
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)
     
def get_wt_variant_matrix(
    vep_df, 
    haplotype_col="haplotype", 
    wt_variant_split="[,]|[|]",
    as_df=False
):
    """
    Generate a binary matrix indicating the presence of wild-type (wt) variants for each haplotype.

    Parameters
    ----------
    vep_df : pd.DataFrame
        DataFrame containing at least a column with haplotype identifiers.
    haplotype_col : str, default="haplotype"
        Name of the column in `vep_df` containing haplotype identifiers.
    wt_variant_split : str, default="[,]|[|]"
        Regular expression pattern to split the wt_variant string.
    as_df : bool, default=False
        If True, return the exploded DataFrame with haplotype and wt_variant columns.
        If False, return a binary matrix (DataFrame) with haplotypes as rows and wt_variants as columns.

    Returns
    -------
    pd.DataFrame
        If as_df is True: DataFrame with columns [haplotype_col, "wt_variant", "dummy"].
        If as_df is False: Binary matrix (DataFrame) with haplotypes as index and wt_variants as columns.
    """
    wt_variants = vep_df[[haplotype_col]].drop_duplicates()

    # Remove any parts after the substring "md5:" in the haplotype column
    wt_variants[haplotype_col] = wt_variants[haplotype_col].str.replace(r"[|]md5:.*$", "", regex=True)
    

    # Extract the wt_variant part from the haplotype string and split by the provided pattern
    wt_variants["wt_variant"] = (
        wt_variants[haplotype_col]
        .str.split(":", n=1)
        .str[1]
        .str.split(wt_variant_split)
    )
    wt_variants = wt_variants.explode("wt_variant")
    wt_variants["dummy"] = 1
    if as_df:
        return wt_variants
    else:
        X_wt = wt_variants.pivot_table(
            index=haplotype_col,
            columns="wt_variant",
            values="dummy",
            aggfunc="sum",
            fill_value=0,
        )
        X_wt = (X_wt > 0).astype(int)
        print('wt_matrix shape: ',X_wt.shape)
        return X_wt
    

def get_vep_matrix(vep_df,
                   haplotype_col = "haplotype",
                   site_col = "site",
                   target = "VEP",
                   pivot_table_kwargs = {}):
    X_vep = vep_df.pivot_table(index=haplotype_col, 
                             columns=site_col, 
                             values=target,
                             **pivot_table_kwargs) 
    print('vep_matrix shape: ',X_vep.shape)
    return X_vep
        
def wtvariants_to_vep_neural_network(vep_df,
                                     target = "VEP",  
                                     haplotype_col = "haplotype",
                                     site_col = "site",
                                     wt_variant_split = "[,]|[|]",
                                     n_epochs = 100,
                                     random_state=42, 
                                     model=SimpleNN,
                                     lr=1e-3,
                                     device="cuda",
                                     attribution_method = "weights",
                                     shap_sample_size = 100,
                                     pivot_table_kwargs = {}): 
    
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    y_vep = get_vep_matrix(vep_df,
                           haplotype_col=haplotype_col,
                           site_col=site_col,
                           target=target,
                           **pivot_table_kwargs)

    # Option to learn which (site, wt_variant) interactions are important for determining clusters or VEP values
    # Set target_mode = "value" for VEP value regression, "cluster" for cluster classification
    X_wt = get_wt_variant_matrix(vep_df, 
                                 haplotype_col=haplotype_col, 
                                 wt_variant_split=wt_variant_split)

    # --- Handle NaNs without imputation: remove any rows with NaNs in X or y --- 
    # Find rows with any NaNs in X_wt or y_vep
    nan_mask = (~X_wt.isna().any(axis=1)) & (~y_vep.isna().any(axis=1))
    X_wt_clean = X_wt.loc[nan_mask]
    y_vep_clean = y_vep.loc[nan_mask]

    # 4. Train a simple neural network to predict continuous VEP values for each site 
    # Convert to torch tensors
    X_tensor = torch.tensor(X_wt_clean.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_vep_clean.values, dtype=torch.float32) 

    input_dim = X_tensor.shape[1]
    output_dim = y_tensor.shape[1]
    model = model(input_dim, output_dim)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop 
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    # Now get the most important interactions between input features and output sites
    # We'll use the absolute value of the weights in the first and last layer as a proxy for interaction strength
    if attribution_method == "shap":
        # Use SHAP values to estimate feature importance for each output site
        import shap

        # SHAP expects a function that returns numpy arrays
        def model_predict(x_numpy):
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32, device=device)
            with torch.no_grad():
                return model(x_tensor).cpu().numpy()

        # Use a small sample for background (SHAP can be slow)
        if shap_sample_size is None:
            background = X_wt_clean.values
        else:
            background = X_wt_clean.sample(n=min(shap_sample_size, len(X_wt_clean)), random_state=42).values

        # KernelExplainer works for general models, but is slow; DeepExplainer is faster for simple NNs
        try:
            print("Using SHAP DeepExplainer")
            explainer = shap.DeepExplainer(model, torch.tensor(background, dtype=torch.float32, device=device))
            shap_values = explainer.shap_values(torch.tensor(X_wt_clean.values, dtype=torch.float32, device=device))
            # shap_values is a list of arrays, one per output; shape: (n_samples, n_features)
            # We'll average absolute SHAP values across samples for each (input, output) pair
            if isinstance(shap_values, list):
                # Multi-output: list of arrays, one per output
                interaction_matrix = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values], axis=1)  # (n_features, n_outputs)
            else:
                # Single output: shape (n_samples, n_features)
                interaction_matrix = np.abs(shap_values).mean(axis=0)[:, None]
        except Exception as e:
            print("SHAP DeepExplainer failed, falling back to KernelExplainer. Error:", e)
            explainer = shap.KernelExplainer(model_predict, background)
            shap_values = explainer.shap_values(X_wt_clean.values, nsamples=100)
            if isinstance(shap_values, list):
                interaction_matrix = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values], axis=1)
            else:
                interaction_matrix = np.abs(shap_values).mean(axis=0)[:, None]
        # interaction_matrix shape: (input_dim, output_dim)
    elif attribution_method == "fasttreeshap":
        # Use FastTreeSHAP for feature importance (requires tree-based model)
        # You must have fasttreeshap installed: pip install fasttreeshap
        print("Using FastTreeSHAP")
        try: 
            import fasttreeshap 
            # Fit a tree-based model (e.g., XGBoost) for feature importance
            import xgboost as xgb
            # For multi-output regression, fit one model per output
            X_np = X_wt_clean.values
            Y_np = y_vep_clean.values
            n_outputs = Y_np.shape[1]
            interaction_matrix = np.zeros((X_np.shape[1], n_outputs))
            for j in range(n_outputs):
                dtrain = xgb.DMatrix(X_np, label=Y_np[:, j])
                params = {"objective": "reg:squarederror", "verbosity": 0}
                bst = xgb.train(params, dtrain, num_boost_round=50)
                # FastTreeSHAP expects a fitted tree model and data
                explainer = fasttreeshap.TreeExplainer(bst)
                shap_values = explainer.shap_values(X_np)
                # Average absolute SHAP values for each feature
                interaction_matrix[:, j] = np.abs(shap_values).mean(axis=0)
            # interaction_matrix shape: (input_dim, output_dim)
        except ImportError as e:
            raise ImportError("fasttreeshap or xgboost is not installed. Please install with 'pip install fasttreeshap xgboost'.") from e
    
    elif attribution_method == "weights":
        with torch.no_grad():
            # Estimate input-output (feature-site) interaction strengths using the neural network's weights.
            # 
            # This approach works as follows:
            # - In a simple feedforward neural network with one hidden layer (or in this case, a 3-layer network),
            #   the influence of an input feature on an output can be approximated by considering the paths through the network.
            # - Each path from input i to output j passes through all hidden units k.
            # - The contribution of input i to output j via hidden unit k is the product of the weight from input i to hidden k (W1[k, i])
            #   and the weight from hidden k to output j (W2[j, k]).
            # - Summing over all hidden units gives the total (linearized) influence of input i on output j.
            # - Taking the absolute value of each weight before multiplying (i.e., |W1[k, i]| * |W2[j, k]|) ensures we capture the magnitude
            #   of the effect, regardless of sign, and is a common heuristic for feature importance in neural networks.
            # - The resulting interaction_matrix[i, j] = sum_k |W1[k, i]| * |W2[j, k]| gives a measure of how strongly input i can affect output j.
            #
            # This method is a simplification and ignores nonlinearities (activations), but is often used for a quick estimate of feature importance
            # or input-output saliency in fully connected networks.
            #
            # Implementation:
            first_layer_weights = model.net[0].weight.cpu().numpy()  # shape: (hidden_dim, input_dim)
            last_layer_weights = model.net[2].weight.cpu().numpy()   # shape: (output_dim, hidden_dim)
            # Compute the interaction matrix as described above
            interaction_matrix = abs(first_layer_weights).T @ abs(last_layer_weights.T)
            # interaction_matrix shape: (input_dim, output_dim)

    input_features = X_wt_clean.columns
    output_sites = y_vep_clean.columns
    interaction_df = pd.DataFrame(interaction_matrix, 
                                index=input_features, 
                                columns=output_sites)
    interaction_df = interaction_df.stack().reset_index()
    interaction_df.columns = ['wt_variant', 'site', 'interaction_strength']

    # Add info about how many haplotypes each wt_variant is present in
    wt_variant_counts = X_wt_clean.sum(axis=0).to_dict()
    interaction_df['n_haplotypes'] = interaction_df['wt_variant'].map(wt_variant_counts)
    # Compute a score that gives equal weight to interaction strength and number of haplotypes
    # Here, we use the geometric mean to balance both factors equally
    interaction_df["interaction_strength_weighted"] = np.sqrt(
        interaction_df["interaction_strength"] * interaction_df["n_haplotypes"]
    )
    interaction_df = interaction_df.sort_values('interaction_strength', ascending=False)

    return interaction_df, model


def wtvariants_to_vep_linear_model(
        vep_df,
        target="VEP",
        haplotype_col="haplotype",
        site_col="site",
        wt_variant_split="[,]|[|]",
        model_type="ridge",  # "ridge" or "lasso"
        alpha=1.0,
        random_state=42, 
        pivot_table_kwargs={},
        add_positions=True,
    ):
        """
        Fit a Ridge or Lasso regression model to predict VEP values from wt_variant features,
        and compute input-output (wt_variant-site) interaction strengths.

        Parameters
        ----------
        vep_df : pd.DataFrame
            DataFrame containing variant effect predictor (VEP) data.
        target : str, default="VEP"
            Name of the column in vep_df containing the target VEP values.
        haplotype_col : str, default="haplotype"
            Name of the column in vep_df identifying haplotypes.
        site_col : str, default="site"
            Name of the column in vep_df identifying sites.
        wt_variant_split : str, default="[,]|[|]"
            Regular expression used to split wt_variant strings.
        model_type : {"ridge", "lasso"}, default="ridge"
            Type of linear model to fit. "ridge" for Ridge regression, "lasso" for Lasso regression.
        alpha : float, default=1.0
            Regularization strength; must be a positive float. Larger values specify stronger regularization.
            For Ridge regression, this corresponds to the L2 penalty term, and for Lasso regression, to the L1 penalty term.
        random_state : int, default=42
            Random seed for reproducibility.
        pivot_table_kwargs : dict, default={}
            Additional keyword arguments to pass to the pivot table function.
        add_positions : bool, default=True
            If True, add positions to the interaction_df.

        Returns
        -------
        interaction_df : pd.DataFrame
            DataFrame with columns ['wt_variant', 'site', 'interaction_strength', 'n_haplotypes', 'interaction_strength_weighted']
        model : fitted sklearn model
        """
        from sklearn.linear_model import Ridge, Lasso

        np.random.seed(random_state)

        # Prepare target matrix (haplotype x site: VEP)
        y_vep = get_vep_matrix(
            vep_df,
            haplotype_col=haplotype_col,
            site_col=site_col,
            target=target,
            **pivot_table_kwargs
        )

        # Prepare input matrix (haplotype x wt_variant: binary matrix)
        X_wt = get_wt_variant_matrix(
            vep_df,
            haplotype_col=haplotype_col,
            wt_variant_split=wt_variant_split
        )

        # Remove any rows with NaNs in X or y and report how many rows were dropped
        nan_mask = (~X_wt.isna().any(axis=1)) & (~y_vep.isna().any(axis=1))
        n_dropped = (~nan_mask).sum()
        if n_dropped > 0:
            print(f"Dropped {n_dropped} haplotypes due to NaNs in input or target matrices.")
        X_wt_clean = X_wt.loc[nan_mask]
        y_vep_clean = y_vep.loc[nan_mask]

        # Fit model
        if model_type == "ridge":
            model = Ridge(alpha=alpha, random_state=random_state)
        elif model_type == "lasso":
            model = Lasso(alpha=alpha, random_state=random_state)
        else:
            raise ValueError("model_type must be 'ridge' or 'lasso'")

        # Ridge/Lasso in sklearn supports multi-target regression (haplotype x site)
        model.fit(X_wt_clean.values, y_vep_clean.values)

        # Attribution: use absolute value of coefficients as interaction strength
        # model.coef_ shape: (n_sites, n_wt_variants)
        # We want (n_wt_variants, n_sites)
        coef_matrix = np.abs(model.coef_.T)  # shape: (n_wt_variants, n_sites)

        input_features = X_wt_clean.columns
        output_sites = y_vep_clean.columns
        interaction_df = pd.DataFrame(
            coef_matrix,
            index=input_features,
            columns=output_sites
        )
        interaction_df = interaction_df.stack().reset_index()
        interaction_df.columns = ['wt_variant', 'site', 'interaction_strength']

        # Add info about how many haplotypes each wt_variant is present in
        wt_variant_counts = X_wt_clean.sum(axis=0).to_dict()
        interaction_df['n_haplotypes'] = interaction_df['wt_variant'].map(wt_variant_counts)
        # Compute a score that gives equal weight to interaction strength and number of haplotypes
        interaction_df["interaction_strength_weighted"] = np.sqrt(
            interaction_df["interaction_strength"] * interaction_df["n_haplotypes"]
        )
        interaction_df = interaction_df.sort_values('interaction_strength', ascending=False)
        interaction_df["clinical_variant"] = interaction_df["site"].str.split(":").str[1]

        if add_positions: 
                interaction_df=  utils.variants_to_positions(interaction_df, 
                                                            variant_col="wt_variant", 
                                                            position_col="wt_position",
                                                            ref_col="wt_REF",
                                                            alt_col="wt_ALT")
                interaction_df = utils.variants_to_positions(interaction_df, 
                                                            variant_col="clinical_variant", 
                                                            position_col="clinical_position",
                                                            ref_col="clinical_REF",
                                                            alt_col="clinical_ALT")
        return interaction_df, model


def wtvariants_to_vep_autoencoder(model,
                                  vep_df,
                                  haplotype_col="haplotype",
                                  site_col="site",
                                  target="VEP",
                                  wt_variant_split="[,]|[|]",
                                  random_state=42,
                                  pivot_table_kwargs={}): 

    # Get device from model parameters
    device = next(model.parameters()).device


    np.random.seed(random_state)

  

    # Prepare input matrix (haplotype x wt_variant: binary matrix)
    X_vep = get_vep_matrix(
        vep_df,
        haplotype_col=haplotype_col,
        site_col=site_col,
        target=target,
        **pivot_table_kwargs
    )
 
    # DataFrame to store results
    wt_site_importance = []

    # Compute baseline reconstruction (no masking)
    X_tensor_full = torch.tensor(X_vep.values, dtype=torch.float32, device=device)
    with torch.no_grad():
        # The autoencoder returns a tuple (recon, embedding), so we need to unpack it
        recon_full = model(X_tensor_full)[0].cpu().numpy()

    for wt in X_vep.columns:
        haplotypes_with_wt_variant = X_vep.index[X_vep[wt] == 1]

        X_masked = X_vep.copy()
        X_masked.loc[haplotypes_with_wt_variant.tolist(), :] = 0  # mask out this wt_variant for all haplotypes

        # Prepare input for autoencoder
        X_tensor = torch.tensor(X_masked.values, dtype=torch.float32, device=device)

        # Compute reconstruction with this wt masked
        with torch.no_grad():
            recon_masked = model(X_tensor)[0].cpu().numpy()

        # For each site (column), compute the mean absolute difference in reconstruction
        for i, site in enumerate(X_vep.columns):
            # Only consider haplotypes where this wt_variant was present (i.e., those that were masked)
            idx = [X_vep.index.get_loc(h) for h in haplotypes_with_wt_variant]
            if len(idx) == 0:
                continue
            diff = abs(recon_full[idx, i] - recon_masked[idx, i]).mean()
            wt_site_importance.append({
                "wt_variant": wt,
                "site": site,
                # Mean absolute difference in reconstruction
                "interaction_strength": diff
            })

    # Convert to DataFrame
    interaction_df = pd.DataFrame(wt_site_importance).sort_values("interaction_strength", ascending=False)

    # Show top results
    print(interaction_df.head(10))




def check_multi_dms_overlap(multi_dms_df, vep_files):
    """
    Check overlap between multi-mutant DMS data and VEP variant effect predictions.

    For each DMS experiment with multiple mutations (e.g., double mutants), this function:
      - Maps VEP files to RefSeq protein IDs.
      - Loads DMS data and extracts double mutants.
      - Compares DMS mutant alleles and positions to VEP-predicted variants.
      - Reports the number of overlapping variants by allele and by position.

    Args:
        multi_dms_df (pd.DataFrame): DataFrame of DMS experiments with multiple mutations,
            as returned by `get_multi_dms()`.
        vep_files (pd.DataFrame): DataFrame of VEP variant effect prediction files,
            must include a "protein" column.

    Returns:
        pd.DataFrame: DataFrame summarizing overlap counts for each DMS experiment.
            Columns include:
                - gene: Gene symbol
                - source_file: Path to DMS data file
                - wt_vs_mut1: Overlap between VEP wildtype variant and DMS mut1 (allele)
                - wt_vs_mut2: Overlap between VEP wildtype variant and DMS mut2 (allele)
                - site_vs_mut1: Overlap between VEP site variant and DMS mut1 (allele)
                - site_vs_mut2: Overlap between VEP site variant and DMS mut2 (allele)
                - wtpos_vs_mut1pos: Overlap between VEP wildtype position and DMS mut1 position
                - wtpos_vs_mut2pos: Overlap between VEP wildtype position and DMS mut2 position
                - sitepos_vs_mut1pos: Overlap between VEP site position and DMS mut1 position
                - sitepos_vs_mut2pos: Overlap between VEP site position and DMS mut2 position

    Example:
        >>> import src.vep_analysis as va
        >>> import src.proteingym as pg

        >>> multi_dms_df = pg.get_multi_dms(keys=["DMS_ProteinGym_substitutions.zip"])
        >>> vvep_files = va.list_vep_files(as_df=True)
        >>> overlap_df = pg.check_multi_dms_overlap(multi_dms_df, vep_files)
        >>> print(overlap_df.head())
    """
    import src.proteingym as pg
    import src.gprofiler as gp
 

    # Map RefSeq protein IDs to gene names
    vep_files_map = gp.map_ids(
        vep_files,
        on_left="protein",
        target_namespace="REFSEQ_PEPTIDE"
    )

    # Convert wt_variant from "1092V>L" to "V1092L" format
    def wt_variant_to_hgvs(wt_variant):
        import re
        # Match: number(s) + letter > letter, e.g. 1092V>L
        m = re.match(r"(\d+)([A-Z])>([A-Z])", str(wt_variant))
        if m:
            pos, ref, alt = m.groups()
            return f"{ref}{pos}{alt}"
        else:
            return wt_variant

    overlap_df = []
    for i, row in tqdm(multi_dms_df.iterrows(),
                       total=len(multi_dms_df),
                       desc="Iterating over DMS file"):
        ### DMS DATA ####
        dms = pd.read_csv(row["source_file"])
        dms = dms.loc[dms["mutant"].str.contains(":")]
        # Split the 'mutant' column into two new columns: 'mut1' and 'mut2'
        # Assumes double mutants are separated by ":"
        dms[["mut1", "mut2"]] = dms["mutant"].str.split(":", n=1, expand=True)
        dms = utils.variants_to_positions(df=dms, variant_col="mut1", position_col="mut1_pos", ref_col="mut1_ref", alt_col="mut1_alt")
        dms = utils.variants_to_positions(df=dms, variant_col="mut2", position_col="mut2_pos", ref_col="mut2_ref", alt_col="mut2_alt")

        #### VEP DATA ####
        vep_files_map_sub = vep_files_map.loc[vep_files_map["HGNC"] == row["gene"]]
        # model_location = "esm1_t34_670M_UR50D"
        # vep_files_map_sub = vep_files_map_sub.loc[vep_files_map_sub["model_location"]==model_location]

        if vep_files_map_sub.empty:
            print("No rows in vep_files found.")
            continue

        vep_df = va.merge_vep(
            add_metadata=True,
            scoring_strategy=["masked-marginals"],
            vep_files=vep_files_map_sub
        )
        print("haplotypes:", vep_df["haplotype"].nunique())
        vep_df['site'] = vep_df['protein'] + ":" + vep_df['mutant']

        interaction_df, model = wtvariants_to_vep_linear_model(vep_df, model_type="ridge")

        interaction_df["wt_hgvs"] = interaction_df["wt_variant"].apply(wt_variant_to_hgvs)
        interaction_df["site_hgvs"] = interaction_df["site"].str.split(":").str[1]
        interaction_df = utils.variants_to_positions(df=interaction_df, variant_col="wt_hgvs", position_col="wt_pos", ref_col="wt_ref", alt_col="wt_alt")
        interaction_df = utils.variants_to_positions(df=interaction_df, variant_col="site_hgvs", position_col="site_pos", ref_col="site_ref", alt_col="site_alt")

        overlap = {
            "gene": row["gene"],
            "source_file": row["source_file"],
            "wt_vs_mut1": len(utils.intersect(interaction_df["wt_hgvs"], dms["mut1"])),
            "wt_vs_mut2": len(utils.intersect(interaction_df["wt_hgvs"], dms["mut2"])),
            "site_vs_mut1": len(utils.intersect(interaction_df["site_hgvs"], dms["mut1"])),
            "site_vs_mut2": len(utils.intersect(interaction_df["site_hgvs"], dms["mut2"])),
            # Positional overlap
            "wtpos_vs_mut1pos": len(utils.intersect(interaction_df["wt_pos"], dms["mut1_pos"])),
            "wtpos_vs_mut2pos": len(utils.intersect(interaction_df["wt_pos"], dms["mut2_pos"])),
            "sitepos_vs_mut1pos": len(utils.intersect(interaction_df["site_pos"], dms["mut1_pos"])),
            "sitepos_vs_mut2pos": len(utils.intersect(interaction_df["site_pos"], dms["mut2_pos"]))
        }
        overlap_df.append(overlap)

        # Optionally print details for debugging
        interaction_dms = pd.concat([
            interaction_df.merge(dms, left_on=["wt_hgvs", "site_hgvs"], right_on=["mut1", "mut2"], how="inner"),
            interaction_df.merge(dms, left_on=["wt_hgvs", "site_hgvs"], right_on=["mut2", "mut1"], how="inner"),
        ])
        print("interaction_dms (by hgvs):", interaction_dms.shape)

        # Only check using position, not allele
        interaction_dms_pos = pd.concat([
            interaction_df.merge(
                dms,
                left_on=["wt_pos", "site_pos"],
                right_on=["mut1_pos", "mut2_pos"],
                how="inner"
            ),
            interaction_df.merge(
                dms,
                left_on=["wt_pos", "site_pos"],
                right_on=["mut2_pos", "mut1_pos"],
                how="inner"
            ),
        ])
        print("interaction_dms (by position):", interaction_dms_pos.shape)

    overlap_df = pd.DataFrame(overlap_df)
    return overlap_df

 

def run_contact_enrichment(contact_map_binary,
                            Xoutliers_sig, 
                            Xoutliers_all=None,
                            use_Xoutliers_all_non_na_baseline=False,
                            exclude_diagonals=True,
                            verbose=True):
    """
    Analyze overlap between non-NA Xoutliers_sig and contact map.

    Parameters
    ----------
    contact_map_binary : np.ndarray
        Binary contact map (1 = contact, 0 = no contact).
    Xoutliers_sig : np.ndarray or DataFrame, optional
        Matrix of significant outlier values (e.g., z-scores). 
    Xoutliers_all : np.ndarray or DataFrame, optional
        Matrix of all outlier values (e.g., z-scores).
    use_Xoutliers_all_non_na_baseline : bool, default False
        If True, use the number of non-NA values in Xs as the baseline for the binomial test.
        If False, use the total number of positions in the contact map.
    exclude_diagonals : bool, default True
        If True, exclude diagonal elements from the contact map.
    verbose : bool, default True
        If True, print verbose output.
    """
    
    Xoutliers_sig = Xoutliers_sig.copy()
    # Get binary contact map (1 = contact, 0 = no contact) 
    contact_mask = (contact_map_binary == 1)
    if exclude_diagonals:
        contact_mask = contact_mask & (np.eye(contact_mask.shape[0]) == 0)
    num_contacts = np.sum(contact_mask)
    total_positions = contact_mask.size

    # Get mask of non-NA values in Xoutliers
    if hasattr(Xoutliers_sig, "values"):
        Xoutliers_sig = Xoutliers_sig.values 

    # Get mask of non-NA values in Xoutliers_sig
    non_na_mask = ~np.isnan(Xoutliers_sig)
    if exclude_diagonals:
        non_na_mask = non_na_mask & (np.eye(non_na_mask.shape[0]) == 0)
    num_non_na = np.sum(non_na_mask)

    # Overlap: non-NA Xoutliers that are also contacts
    non_na_and_contact = non_na_mask & contact_mask
    num_non_na_and_contact = np.sum(non_na_and_contact)

    if num_non_na > 0:
        # What proportion of the contacts are also non-NA Xoutliers?
        # prop_non_na_overlap_contacts = num_non_na_and_contact / num_contacts
        # print(f"Proportion of non-NA Xoutliers that overlap with contacts: {prop_non_na_overlap_contacts:.3f} ({num_non_na_and_contact}/{num_non_na})")

        # What proportion of the non-NA Xoutliers_sig are also contacts?
        prop_non_na_overlap_contacts = num_non_na_and_contact / num_non_na
        if verbose:
            print(f"Proportion of non-NA Xoutliers that overlap with contacts: {prop_non_na_overlap_contacts:.3f} ({num_non_na_and_contact}/{num_non_na})")
    else:
        if verbose:
            print("No non-NA values in Xoutliers.")

    if verbose:
        print(f"Total number of non-NA Xoutliers: {num_non_na}")
        print(f"Total number of contacts: {num_contacts}")

    # Determine expected probability
    if use_Xoutliers_all_non_na_baseline and Xoutliers_all is not None:
        Xoutliers_all = Xoutliers_all.copy()
        if hasattr(Xoutliers_all, "values"):
            Xoutliers_all = Xoutliers_all.values 

        num_non_na_all_and_contact = np.sum((~np.isnan(Xoutliers_all)) & contact_mask)
        if verbose:
            print(f"Using Xoutliers_all non-NA baseline: {num_non_na_all_and_contact} positions with contacts")

        # What proportion of the total positions are also non-NA Xoutliers?
        # expected_prob = num_non_na_all_and_contact / total_positions
        # print(f"Baseline computation: {baseline_total} positions with contacts / {total_positions} total positions = {expected_prob:.3e}")

        # What proportion of the non-NA Xoutliers_all are also contacts?
        expected_prob = num_non_na_all_and_contact / np.sum(~np.isnan(Xoutliers_all))
        if verbose:
            print(f"Proportion of non-NA Xoutliers_all that overlap with contacts: {expected_prob:.3f} ({num_non_na_all_and_contact}/{num_non_na})")
 
    else:
         # What proportion of the total positions are also contacts?
        expected_prob = num_contacts / total_positions
        if verbose:
            print(f"Using total positions baseline: {num_contacts} positions with contacts / {total_positions} total positions = {expected_prob:.3e}")

   
    binom_test = binomtest(k=num_non_na_and_contact, # The number of successes.
                           n=num_non_na, # The number of trials.
                           p=expected_prob, # The hypothesized probability of success.
                           alternative='two-sided' 
                           ) 

    if verbose:
        print(f"Expected overlap by chance: {expected_prob:.3e}")
        print(f"Binomial test p-value: {binom_test.pvalue:.3e}")

    # Enrichment statistic: fold enrichment of observed overlap vs. expected
    # enrichment is >1, depletion is <1
    enrichment = prop_non_na_overlap_contacts / expected_prob if expected_prob > 0 else np.nan
    if verbose:
        print(f"Fold enrichment of contact overlap among non-NA Xoutliers: {enrichment:.2f}x")

    return {"observed_prob": prop_non_na_overlap_contacts,
            "expected_prob": expected_prob,
            "binom_test": binom_test,
            "enrichment": enrichment}

def compute_enrichment_vs_threshold(
    outlier_df, 
    contact_map_binary, 
    full_length, 
    num_thresholds=50, 
    linear_sampling=True,
    run_contact_enrichment_func=None,
    utils_module=None,
    max_percentage=0.95,
    interaction_col="z_score_abs", 
    x_id_col="wt_variant",
    y_id_col="clinical_variant",
    x_pos_col="wt_position",
    y_pos_col="clinical_position",
):
    """
    Compute enrichment statistics for a range of z_score_abs thresholds.

    Parameters
    ----------
    outlier_df : pd.DataFrame
        DataFrame containing outlier data with z_score column.
    contact_map_binary : np.ndarray
        Binary contact map.
    full_length : int
        Full length of the protein (for fill_coordinates).
    num_thresholds : int, optional
        Number of thresholds to sample, by default 50.
    linear_sampling : bool, optional
        If True, use linear sampling of thresholds; else geometric, by default True.
    run_contact_enrichment_func : callable, optional
        Function to compute enrichment, by default uses run_contact_enrichment.
    utils_module : module, optional
        Module containing fill_coordinates, by default uses utils.
    z_score_col : str, optional
        Name of the z-score column, by default "z_score".

    Returns
    -------
    list of dict
        List of enrichment results for each threshold.
    """
    if run_contact_enrichment_func is None:
        run_contact_enrichment_func = run_contact_enrichment
    if utils_module is None:
        import utils
        utils_module = utils

    df = outlier_df.copy()
    min_val = df[interaction_col].min()
    max_val = df[interaction_col].max() * max_percentage

    if linear_sampling: 
        thresholds = np.linspace(min_val, max_val, num=num_thresholds)
    else:
        # Use a geometric progression for denser sampling at the beginning
        if min_val <= 0:
            shift = abs(min_val) + 1e-6
            min_val_shifted = min_val + shift
            max_val_shifted = max_val + shift
            thresholds = np.geomspace(min_val_shifted, max_val_shifted, num=num_thresholds) - shift
        else:
            thresholds = np.geomspace(min_val, max_val, num=num_thresholds)

    results = []
    for threshold in tqdm(thresholds): 
        df_thresh = df[df[interaction_col] > threshold]
        Xoutliers_all = utils_module.fill_coordinates(
            df_thresh, 
            x_id_col=x_id_col,
            y_id_col=y_id_col,
            x_pos_col=x_pos_col,
            y_pos_col=y_pos_col,
            value_col=interaction_col,
            full_length=full_length
        )   

        # Compute enrichment by setting the expected probability to:
        # the proportion of residue-residue pairs that are contacts.
        res_interaction = run_contact_enrichment_func(
            contact_map_binary,  
            Xoutliers_sig=Xoutliers_all,
            use_Xoutliers_all_non_na_baseline=False,
            verbose=False
        )
        res_interaction["interaction_threshold"] = threshold
        res_interaction["n_interactions"] = df_thresh.shape[0]
        results.append(res_interaction)
    return pd.DataFrame(results)


def plot_enrichment_vs_interactions(results, log_x_axis=False, show=True, ax=None):
    """
    Plot enrichment and number of interactions vs interaction threshold.

    Parameters
    ----------
    results : pd.DataFrame or list of dict
        DataFrame or list of dicts with columns:
            - interaction_threshold
            - enrichment
            - n_interactions
    log_x_axis : bool, default True
        Whether to use log scale for x-axis.
    show : bool, default True
        Whether to call plt.show().
    ax : matplotlib.axes.Axes or None
        Optionally provide an axis to plot on. If None, a new figure is created.

    Returns
    -------
    (fig, ax1, ax2)
        The matplotlib Figure and Axes objects.
    """

    if not isinstance(results, pd.DataFrame):
        results_df = pd.DataFrame(results)
    else:
        results_df = results

    if ax is None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax
        fig = ax1.figure

    # Plot enrichment (left y-axis)
    color = "tab:blue"
    sns.lineplot(
        data=results_df, 
        x="interaction_threshold", y="enrichment", 
        marker="o", ax=ax1, color=color
    )
    ax1.set_ylabel("Contact Enrichment", color=color)
    ax1.set_xlabel("Interaction Threshold")
    ax1.tick_params(axis='y', labelcolor=color)

    if log_x_axis:
        ax1.set_xscale("log")

    # Add text labels for min and max enrichment next to the first and last dot
    first_idx = 0
    last_idx = len(results_df) - 1
    first_lab_coords = results_df.iloc[first_idx]["interaction_threshold"], results_df.iloc[first_idx]["enrichment"]
    last_lab_coords = results_df.iloc[last_idx]["interaction_threshold"], results_df.iloc[last_idx]["enrichment"]
    offset = results_df["interaction_threshold"].max() * 0.01

    ax1.text(first_lab_coords[0]+offset, first_lab_coords[1], f"{first_lab_coords[1]:.0f}x", va='top', ha='left', fontsize=10, color=color)
    ax1.text(last_lab_coords[0]-offset, last_lab_coords[1], f"{last_lab_coords[1]:.0f}x", va='center', ha='right', fontsize=10, color=color)

    # Plot number of interactions (right y-axis)
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    sns.lineplot(
        data=results_df,
        x="interaction_threshold", y="n_interactions",
        marker="s", ax=ax2, color=color2
    )
    ax2.set_ylabel("Number of interactions", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    if log_x_axis:
        ax2.set_xscale("log")

    # Optionally, add text labels for min and max n_interactions
    # Uncomment if desired
    # first_nint_coords = results_df.iloc[first_idx]["interaction_threshold"], results_df.iloc[first_idx]["n_interactions"]
    # last_nint_coords = results_df.iloc[last_idx]["interaction_threshold"], results_df.iloc[last_idx]["n_interactions"]
    # ax2.text(first_nint_coords[0]+0.01, first_nint_coords[1], f"{int(first_nint_coords[1])}", va='center', ha='left', fontsize=10, color=color2)
    # ax2.text(last_nint_coords[0], last_nint_coords[1]-0.1, f"{int(last_nint_coords[1])}", va='top', ha='left', fontsize=10, color=color2)

    if show:
        plt.show()

    return fig, ax1, ax2
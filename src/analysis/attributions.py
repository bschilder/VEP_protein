from re import T
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest
from tqdm import tqdm

import src.utils as utils
import src.vep_analysis as va
import src.haplosaurus as hs
import src.onekg as og
import src.colabfold.colabfold as cf
import src.analysis.matrices as mc

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
        model_kwargs={},
        pivot_table_kwargs={},
        add_positions=True,
        standardize_variants=True,
        test_epistasis=False,
        epistasis_alpha=None,
        epistasis_pvalue_threshold=0.05,
    ):
    """
    Fit a Ridge or Lasso regression model to predict VEP values from wt_variant features,
    and compute input-output (wt_variant-site) interaction strengths, including directionality.

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
    alpha : float, default=1.0 (same default as sklearn)
        Regularization strength; must be a positive float. Larger values specify stronger regularization.
        For Ridge regression, this corresponds to the L2 penalty term, and for Lasso regression, to the L1 penalty term.
    random_state : int, default=42
        Random seed for reproducibility.
    pivot_table_kwargs : dict, default={}
        Additional keyword arguments to pass to the pivot table function.
    add_positions : bool, default=True
        If True, add positions to the interaction_df.
    standardize_variants : bool, default=True
        If True, standardize the variant names first.
    test_epistasis : bool, default=False
        If True, test whether each (wt_variant, clinical_variant) interaction is truly epistatic
        (non-additive) rather than simply additive. This fits separate models for each pair and
        compares additive vs interaction models using statistical tests.
    epistasis_alpha : float, optional
        Regularization strength for epistasis testing models. If None, uses the same alpha as the
        main model. Only used when test_epistasis=True.
    epistasis_pvalue_threshold : float, default=0.05
        P-value threshold for determining epistatic vs additive interactions. Only used when
        test_epistasis=True.
    
    Returns
    -------
    interaction_df : pd.DataFrame
        DataFrame with columns ['wt_variant', 'site', 'interaction_strength', 'interaction_strength_signed', 'n_haplotypes', 'interaction_strength_weighted']
        If test_epistasis=True, also includes columns: 'is_epistatic', 'epistasis_pvalue', 'epistasis_fstat', 
        'additive_r2', 'interaction_r2', 'delta_r2', 'additive_mse', 'interaction_mse', 'epistasis_coefficient'
    model : fitted sklearn model
    X_wt_clean : pd.DataFrame
        Cleaned input matrix (haplotype x wt_variant: binary matrix)
    y_vep_clean : pd.DataFrame
        Cleaned target matrix (haplotype x site: VEP)
    coef_matrix_signed : pd.DataFrame
        Signed interaction strength matrix (wt_variant x site)
    coef_matrix_abs : pd.DataFrame
        Absolute interaction strength matrix (wt_variant x site)
    metrics : pd.DataFrame
        Metrics for the model
    epistasis_results : dict, optional
        If test_epistasis=True, contains summary statistics about epistatic vs additive interactions.
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
    # Standardize column names of y_vep using standardize_variant
    if standardize_variants:
        y_vep.columns = [utils.standardize_variant(col) for col in y_vep.columns]
    

    # Prepare input matrix (haplotype x wt_variant: binary matrix)
    X_wt = get_wt_variant_matrix(
        vep_df,
        haplotype_col=haplotype_col,
        wt_variant_split=wt_variant_split
    )
    # Standardize column names of X_wt using standardize_variant
    if standardize_variants:
        X_wt.columns = [utils.standardize_variant(col) for col in X_wt.columns]
    

    # Remove any rows with NaNs in X or y and report how many rows were dropped
    nan_mask = (~X_wt.isna().any(axis=1)) & (~y_vep.isna().any(axis=1))
    n_dropped = (~nan_mask).sum()
    if n_dropped > 0:
        print(f"Dropped {n_dropped} haplotypes due to NaNs in input or target matrices.")
    X_wt_clean = X_wt.loc[nan_mask]
    y_vep_clean = y_vep.loc[nan_mask]

    # Fit model
    if model_type == "ridge":
        model = Ridge(alpha=alpha, random_state=random_state, **model_kwargs)
    elif model_type == "lasso":
        model = Lasso(alpha=alpha, random_state=random_state, **model_kwargs)
    else:
        raise ValueError("model_type must be 'ridge' or 'lasso'")

    # Ridge/Lasso in sklearn supports multi-target regression (haplotype x site)
    model.fit(X_wt_clean.values, y_vep_clean.values)

    # Attribution: use both signed and absolute value of coefficients as interaction strength
    # model.coef_ shape: (n_sites, n_wt_variants)
    # We want (n_wt_variants, n_sites)
    coef_matrix_signed = model.coef_.T  # shape: (n_wt_variants, n_sites)
    coef_matrix_abs = np.abs(coef_matrix_signed)

    input_features = X_wt_clean.columns
    output_sites = y_vep_clean.columns
    interaction_df = pd.DataFrame(
        coef_matrix_abs,
        index=input_features,
        columns=output_sites
    )
    interaction_df = interaction_df.stack().reset_index()
    interaction_df.columns = ['wt_variant', 'site', 'interaction_strength']

    # Add signed interaction strength
    interaction_df_signed = pd.DataFrame(
        coef_matrix_signed,
        index=input_features,
        columns=output_sites
    ).stack().reset_index(drop=True)

    interaction_df["interaction_strength_signed"] = interaction_df_signed
    interaction_df["outlier_type"] = interaction_df["interaction_strength_signed"].apply(
        lambda x: "more pathogenic" if x < 0 else ("neutral" if x == 0 else "more benign")
    )

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
        interaction_df = utils.variants_to_positions(interaction_df, 
                                                    variant_col="wt_variant", 
                                                    position_col="wt_position",
                                                    ref_col="wt_REF",
                                                    alt_col="wt_ALT")
        interaction_df = utils.variants_to_positions(interaction_df, 
                                                    variant_col="clinical_variant", 
                                                    position_col="clinical_position",
                                                    ref_col="clinical_REF",
                                                    alt_col="clinical_ALT")
        
    # Turn coef_matrix_signed and coef_matrix_abs into DataFrames with correct row/col names
    coef_matrix_signed_df = pd.DataFrame(
        coef_matrix_signed,
        index=input_features,
        columns=output_sites
    )
    coef_matrix_abs_df = pd.DataFrame(
        coef_matrix_abs,
        index=input_features,
        columns=output_sites
    )

    # REport
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

    # Predict values for the training set 
    y_pred = model.predict(X_wt_clean)

    # Calculate metrics
    metrics = {
        # Note: In regression, R2 and explained variance score can be different if predictions or targets are not centered or have non-constant residuals. In typical use with linear models and centered data they are often equal, but in general:
        #   R2 = 1 - (sum((y_true - y_pred)^2) / sum((y_true - y_true.mean())^2))
        #   Explained variance = 1 - (Var(y_true - y_pred) / Var(y_true))
        # They are equal when errors have zero mean and are homoscedastic, but can differ otherwise.
        "R2": r2_score(y_vep_clean, y_pred),
        "Explained Variance": explained_variance_score(y_vep_clean, y_pred),
        "MSE": mean_squared_error(y_vep_clean, y_pred),
        # mean_squared_error does not support 'squared' argument in your version.
        # Instead, compute RMSE by taking the square root of MSE.
        "RMSE": mean_squared_error(y_vep_clean, y_pred) ** 0.5,
        "MAE": mean_absolute_error(y_vep_clean, y_pred)
    }

    metrics_df = pd.DataFrame([metrics])

    # Epistasis testing: test whether interactions are truly epistatic vs additive
    epistasis_results = None
    if test_epistasis:
        from scipy.stats import f as f_distribution
        from sklearn.metrics import r2_score, mean_squared_error
        
        if epistasis_alpha is None:
            epistasis_alpha = alpha
        
        print("Testing epistasis for each (wt_variant, clinical_variant) pair...")
        
        # Initialize columns for epistasis results
        interaction_df['is_epistatic'] = False
        interaction_df['epistasis_pvalue'] = np.nan
        interaction_df['epistasis_fstat'] = np.nan
        interaction_df['additive_r2'] = np.nan
        interaction_df['interaction_r2'] = np.nan
        interaction_df['delta_r2'] = np.nan
        interaction_df['additive_mse'] = np.nan
        interaction_df['interaction_mse'] = np.nan
        interaction_df['epistasis_coefficient'] = np.nan
        
        # For epistasis testing, we test whether the joint effect of (WT variant, clinical variant) is
        # additive (sum of individual effects) or epistatic (non-additive, more or less than sum).
        #
        # For a given (WT variant i, clinical variant j) pair:
        # - Estimate individual effect of WT variant i: mean VEP difference (WT=1 vs WT=0) across all clinical variants
        # - Estimate individual effect of clinical variant j: mean VEP for this clinical variant vs baseline (haplotypes without WT i)
        # - Compare: Is the joint effect (from linear model coefficient β_ij) = sum of individual effects (additive)?
        #
        # Additive model: VEP = β₀ + β₁×WT + β₂×Clinical (main effects only, no interaction)
        # Interaction model: VEP = β₀ + β₁×WT + β₂×Clinical + β₃×(WT × Clinical) (allows interaction)
        print(f"Testing epistasis between WT variants and clinical variants for {len(interaction_df)} pairs")
        print(f"Note: Testing if joint effects are additive (sum of individual effects) or epistatic (non-additive)")
        
        # Pre-compute individual effects for efficiency
        # Effect of each WT variant: mean difference across all clinical variants
        wt_individual_effects = {}
        for wt_var in X_wt_clean.columns:
            wt_indicator = X_wt_clean[wt_var].values.astype(float)
            # Mean VEP difference: haplotypes with WT vs without WT, averaged across all clinical variants
            wt_mask = ~np.isnan(wt_indicator)
            if wt_mask.sum() > 0:
                vep_with_wt = y_vep_clean.loc[wt_mask & (wt_indicator > 0.5), :].mean(axis=0).mean()
                vep_without_wt = y_vep_clean.loc[wt_mask & (wt_indicator < 0.5), :].mean(axis=0).mean()
                wt_individual_effects[wt_var] = vep_with_wt - vep_without_wt
            else:
                wt_individual_effects[wt_var] = 0.0
        
        # Effect of each clinical variant: mean VEP for that variant (baseline is mean across all variants)
        clinical_individual_effects = {}
        vep_baseline = y_vep_clean.mean(axis=0).mean()  # Overall mean VEP
        for site in y_vep_clean.columns:
            vep_site_mean = y_vep_clean[site].mean()
            clinical_individual_effects[site] = vep_site_mean - vep_baseline
        
        # Test each (wt_variant, site) pair
        n_tested = 0
        n_epistatic = 0
        n_skipped_missing = 0
        n_skipped_insufficient_samples = 0
        n_skipped_no_variation = 0
        n_skipped_insufficient_combinations = 0
        
        for idx, row in tqdm(interaction_df.iterrows(), total=len(interaction_df), desc="Epistasis testing"):
            wt_var = row['wt_variant']
            site = row['site']
            
            # Skip if we don't have enough data
            if wt_var not in X_wt_clean.columns or site not in y_vep_clean.columns:
                n_skipped_missing += 1
                continue
            
            # Get binary indicator for WT variant and VEP values for this clinical variant
            wt_indicator = X_wt_clean[wt_var].values.astype(float)
            y_values = y_vep_clean[site].values
            
            # Remove rows with NaN
            valid_mask = ~(np.isnan(wt_indicator) | np.isnan(y_values))
            
            if valid_mask.sum() < 10:  # Need at least 10 samples
                n_skipped_insufficient_samples += 1
                continue
            
            wt_clean = wt_indicator[valid_mask]
            y_clean = y_values[valid_mask]
            
            # Check if we have sufficient variation in WT variant (not all 0s or all 1s)
            wt_std = wt_clean.std()
            if wt_std < 1e-10:
                n_skipped_no_variation += 1
                continue
            
            # Check if we have both WT=0 and WT=1 cases
            combinations = set(wt_clean.astype(int))
            if len(combinations) < 2:
                n_skipped_insufficient_combinations += 1
                continue
            
            n_tested += 1
            
            # Fit additive model: VEP = β₀ + β₁×WT
            # This assumes the effect is purely from the WT variant (additive with clinical variant)
            # The clinical variant effect is constant (absorbed into intercept)
            X_additive = np.column_stack([np.ones(len(wt_clean)), wt_clean])
            
            if model_type == "ridge":
                model_additive = Ridge(alpha=epistasis_alpha, random_state=random_state, **model_kwargs)
            else:
                model_additive = Lasso(alpha=epistasis_alpha, random_state=random_state, **model_kwargs)
            
            model_additive.fit(X_additive, y_clean)
            y_pred_additive = model_additive.predict(X_additive)
            
            # Get the joint effect from the main linear model (coefficient β_ij)
            # This is the effect when both WT variant and clinical variant are present
            joint_effect = row['interaction_strength_signed']  # Signed coefficient from main model
            
            # Expected additive effect: sum of individual effects
            wt_effect = wt_individual_effects.get(wt_var, 0.0)
            clinical_effect = clinical_individual_effects.get(site, 0.0)
            expected_additive_effect = wt_effect + clinical_effect
            
            # Test if the WT effect for this specific clinical variant deviates from the average WT effect
            # If the WT effect is significantly different for this clinical variant vs average, it's epistatic
            #
            # Additive model: VEP = β₀ + β₁×WT
            #   This gives us the observed WT effect (β₁) for this clinical variant
            #
            # Interaction model: VEP = β₀ + β₁×WT + β₂×(WT × clinical_variant_specific_term)
            #   This allows the WT effect to vary, indicating epistasis
            #
            # We'll compare the observed WT effect (from additive model) to the average WT effect
            # If significantly different, it's epistatic
            
            # The additive model gives us β₁_additive (observed WT effect for this clinical variant)
            # We compare this to expected_additive_effect
            # To test if deviation is significant, we use an interaction model that allows
            # the WT effect to deviate from the expected additive effect
            
            # Interaction model: VEP = β₀ + β₁×WT + β₂×(WT × deviation_from_expected)
            # where deviation_from_expected helps capture epistasis
            # If β₂ is significant, it indicates the WT effect deviates from additivity (epistasis)
            
            # Create a term that is 1 when WT=1 and represents the deviation
            # We'll use the difference between expected and a baseline
            avg_wt_effect = wt_individual_effects.get(wt_var, 0.0)
            deviation_from_baseline = expected_additive_effect - avg_wt_effect
            epistasis_term = wt_clean * deviation_from_baseline
            X_interaction = np.column_stack([np.ones(len(wt_clean)), wt_clean, epistasis_term])
            
            if model_type == "ridge":
                model_interaction = Ridge(alpha=epistasis_alpha, random_state=random_state, **model_kwargs)
            else:
                model_interaction = Lasso(alpha=epistasis_alpha, random_state=random_state, **model_kwargs)
            
            model_interaction.fit(X_interaction, y_clean)
            y_pred_interaction = model_interaction.predict(X_interaction)
            
            # Calculate metrics
            r2_additive = r2_score(y_clean, y_pred_additive)
            r2_interaction = r2_score(y_clean, y_pred_interaction)
            mse_additive = mean_squared_error(y_clean, y_pred_additive)
            mse_interaction = mean_squared_error(y_clean, y_pred_interaction)
            delta_r2 = r2_interaction - r2_additive
            
            # Get interaction coefficient (β₂) - the epistasis term coefficient
            # X_interaction has 3 columns: [ones, wt_clean, epistasis_term]
            # So coef_ has 3 elements: [β₀ (intercept), β₁ (WT), β₂ (epistasis)]
            epistasis_coef = model_interaction.coef_[2] if len(model_interaction.coef_) > 2 else 0.0
            
            # Calculate the deviation from additivity
            deviation_from_additive = joint_effect - expected_additive_effect
            
            # F-test for model comparison
            # F = ((RSS_additive - RSS_interaction) / (df_additive - df_interaction)) / (RSS_interaction / df_interaction)
            # where RSS = residual sum of squares = MSE * n
            n_samples = len(y_clean)
            df_additive = n_samples - 2  # 2 parameters: intercept, WT
            df_interaction = n_samples - 3  # 3 parameters: intercept, WT, interaction term
            
            rss_additive = mse_additive * n_samples
            rss_interaction = mse_interaction * n_samples
            
            # F-test for model comparison
            # Note: With regularized models (Ridge/Lasso), the degrees of freedom are approximate
            # but the F-test still provides a useful comparison between additive and interaction models
            if rss_interaction > 1e-10 and df_interaction > 0 and (df_additive - df_interaction) > 0:
                f_stat = ((rss_additive - rss_interaction) / (df_additive - df_interaction)) / (rss_interaction / df_interaction)
                # Ensure F-statistic is non-negative (should be, but check for numerical issues)
                f_stat = max(0, f_stat)
                # P-value from F-distribution
                try:
                    pvalue = 1 - f_distribution.cdf(f_stat, df_additive - df_interaction, df_interaction)
                except:
                    pvalue = np.nan
            else:
                f_stat = np.nan
                pvalue = np.nan
            
            # Store results
            interaction_df.at[idx, 'is_epistatic'] = pvalue < epistasis_pvalue_threshold if not np.isnan(pvalue) else False
            interaction_df.at[idx, 'epistasis_pvalue'] = pvalue
            interaction_df.at[idx, 'epistasis_fstat'] = f_stat
            interaction_df.at[idx, 'additive_r2'] = r2_additive
            interaction_df.at[idx, 'interaction_r2'] = r2_interaction
            interaction_df.at[idx, 'delta_r2'] = delta_r2
            interaction_df.at[idx, 'additive_mse'] = mse_additive
            interaction_df.at[idx, 'interaction_mse'] = mse_interaction
            interaction_df.at[idx, 'epistasis_coefficient'] = epistasis_coef
            interaction_df.at[idx, 'joint_effect'] = joint_effect
            interaction_df.at[idx, 'expected_additive_effect'] = expected_additive_effect
            interaction_df.at[idx, 'deviation_from_additive'] = deviation_from_additive
            interaction_df.at[idx, 'wt_individual_effect'] = wt_effect
            interaction_df.at[idx, 'clinical_individual_effect'] = clinical_effect
            
            if interaction_df.at[idx, 'is_epistatic']:
                n_epistatic += 1
        
        print(f"\nEpistasis testing summary:")
        print(f"  Tested: {n_tested} (WT variant, clinical variant) pairs")
        print(f"  Found epistatic (non-additive joint effects): {n_epistatic} (p < {epistasis_pvalue_threshold})")
        print(f"  Skipped - missing columns: {n_skipped_missing}")
        print(f"  Skipped - insufficient samples (<10): {n_skipped_insufficient_samples}")
        print(f"  Skipped - no variation: {n_skipped_no_variation}")
        print(f"  Skipped - insufficient combinations (<2): {n_skipped_insufficient_combinations}")
        
        # Create summary statistics
        epistasis_results = {
            'n_tested': n_tested,
            'n_epistatic': n_epistatic,
            'n_additive': n_tested - n_epistatic,
            'epistasis_rate': n_epistatic / n_tested if n_tested > 0 else 0.0,
            'mean_delta_r2_epistatic': interaction_df[interaction_df['is_epistatic']]['delta_r2'].mean() if n_epistatic > 0 else np.nan,
            'mean_delta_r2_additive': interaction_df[~interaction_df['is_epistatic']]['delta_r2'].mean() if (n_tested - n_epistatic) > 0 else np.nan,
            'mean_epistasis_coefficient': interaction_df[interaction_df['is_epistatic']]['epistasis_coefficient'].mean() if n_epistatic > 0 else np.nan,
        }

    return_dict = {"interaction_df": interaction_df, "model": model, 
            "X_wt_clean": X_wt_clean, "y_vep_clean": y_vep_clean,
            "coef_matrix_signed": coef_matrix_signed_df, 
            "coef_matrix_abs": coef_matrix_abs_df,
            "metrics": metrics_df}
    
    if test_epistasis and epistasis_results is not None:
        return_dict["epistasis_results"] = epistasis_results
    
    return return_dict


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
    full_length=None, 
    num_thresholds=50, 
    linear_sampling=True,
    run_contact_enrichment_func=None,
    max_percentage=0.95,
    interaction_col="z_score_abs", 
    x_id_col="wt_variant",
    y_id_col="clinical_variant",
    x_pos_col="wt_position",
    y_pos_col="clinical_position",
    min_val=None,
    max_val=None,
    thresholds=None,
):
    """
    Compute enrichment statistics for a range of z_score_abs thresholds.

    Parameters
    ----------
    outlier_df : pd.DataFrame
        DataFrame containing outlier data with z_score column.
    contact_map_binary : np.ndarray
        Binary contact map.
    full_length : int, optional
        Full length of the protein (for fill_coordinates). If None, will be inferred from contact_map_binary.shape[0].
    num_thresholds : int, optional
        Number of thresholds to sample, by default 50.
    linear_sampling : bool, optional
        If True, use linear sampling of thresholds; else geometric, by default True.
    run_contact_enrichment_func : callable, optional
        Function to compute enrichment, by default uses run_contact_enrichment.
    z_score_col : str, optional
        Name of the z-score column, by default "z_score".
    thresholds : list or np.ndarray, optional
        List of thresholds to use. If provided, overrides automatic threshold computation.

    Returns
    -------
    list of dict
        List of enrichment results for each threshold.
    """
    if run_contact_enrichment_func is None:
        run_contact_enrichment_func = run_contact_enrichment
    
    if full_length is None:
        full_length = contact_map_binary.shape[0]

    df = outlier_df.copy()

    if thresholds is not None:
        thresholds = np.array(thresholds)
    else:
        min_val = df[interaction_col].min() if min_val is None else min_val
        max_val = df[interaction_col].max() * max_percentage if max_val is None else max_val

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
        Xoutliers_all = mc.fill_coordinates(
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


def plot_enrichment_vs_interactions(
    results, 
    log_x_axis=False, 
    log_y1_axis=False,
    log_y2_axis=False,
    show=True, 
    ax=None, 
    y1_label_freq=0.1,
    y2_label_freq=0.2,
    y1_label_kwargs={"va": "bottom", 
                     "ha": "right", 
                     "color": "black",
                     "clip_on": True},
    y2_label_kwargs={"va": "bottom", 
                     "ha": "left", 
                     "color": "grey", 
                     "clip_on": True},
    colors = ["black", "grey"],
    y1_first_label=True,
    y1_last_label=True,
    y2_first_label=True,
    y2_last_label=True,
    arrow_label="Stronger Joint Effects",
    y1_axis_label="● Contact Enrichment",
    y2_axis_label="■ Variant Pairs",
    x_axis_label="Absolute Joint Effect Threshold",
    title="WT-Clinical Variant Joint Effects vs. Contact Enrichment",
    figsize=(9, 6),
    add_arrow=True,
    arrow_title_padding=0.15,
    arrow_label_padding=0.02,
    arrow_label_fontsize=None,
    arrow_label_bold=True,
    arrow_right_of_xlabel=False,
):
    """
    Plot enrichment and number of interactions versus interaction threshold.

    This function creates a dual-axis line plot showing how contact enrichment and the number of interactions
    change as a function of the interaction threshold. The left y-axis displays the enrichment, while the right
    y-axis displays the number of interactions. Optionally, labels can be added to the first/last points or at
    a specified frequency along each line.

    Parameters
    ----------
    results : pd.DataFrame or list of dict
        DataFrame or list of dicts with columns:
            - 'interaction_threshold': The threshold value for interaction strength.
            - 'enrichment': The contact enrichment at each threshold.
            - 'n_interactions': The number of interactions above each threshold.
    log_x_axis : bool, default False
        Whether to use a logarithmic scale for the x-axis (interaction threshold).
    log_y1_axis : bool, default False
        Whether to use a logarithmic scale for the left y-axis (enrichment).
    log_y2_axis : bool, default False
        Whether to use a logarithmic scale for the right y-axis (number of interactions).
    show : bool, default True
        Whether to display the plot with plt.show().
    ax : matplotlib.axes.Axes or None, optional
        Axis to plot on. If None, a new figure and axis are created.
    y1_label_freq : float, default 0.0
        Frequency for labeling points on the enrichment (left y-axis) line.
        If 1, every point is labeled; if 0.5, every other point; if 0, no labels.
    y2_label_freq : float, default 0.0
        Frequency for labeling points on the number of interactions (right y-axis) line.
        Same convention as y1_label_freq.
    y1_label_kwargs : dict, optional
        Keyword arguments for enrichment text labels.
    y2_label_kwargs : dict, optional
        Keyword arguments for number of interactions text labels.
    colors : list of str, default ["black", "grey"]
        Colors for the enrichment and number of interactions lines, respectively.
    y1_first_label : bool, default False
        Whether to label the first point on the enrichment line.
    y1_last_label : bool, default False
        Whether to label the last point on the enrichment line.
    y2_first_label : bool, default False
        Whether to label the first point on the number of interactions line.
    y2_last_label : bool, default False
        Whether to label the last point on the number of interactions line.
    y1_axis_label : str, default "Contact Enrichment"
        Label for the left y-axis.
    y2_axis_label : str, default "Number of Interactions"
        Label for the right y-axis.
    x_axis_label : str, default "Interaction Threshold"
        Label for the x-axis.
    title : str, optional
        Title for the plot.
    figsize : tuple, default (7, 5)
        Size of the figure.
    add_arrow : bool, default False
        Whether to add a rightward arrow below the x-axis label labeled "stronger interactions".
    arrow_title_padding : float, default 0.05
        Padding between the x-axis title and the arrow (in axes fraction coordinates).
    arrow_label_padding : float, default 0.02
        Padding between the arrow and the arrow label below it (in axes fraction coordinates).
    arrow_label_fontsize : int or None, default None
        Font size for the arrow label. If None, uses "medium".
    arrow_label_bold : bool, default True
        Whether to make the arrow label bold.
    arrow_right_of_xlabel : bool, default False
        If True, positions the arrow to the right of the x-axis label on the same y-plane.
        If False, positions the arrow below the x-axis label (default behavior).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax1 : matplotlib.axes.Axes
        The left y-axis (enrichment) Axes object.
    ax2 : matplotlib.axes.Axes
        The right y-axis (number of interactions) Axes object.

    Notes
    -----
    - The function uses seaborn for line plotting and matplotlib for axis manipulation.
    - Labels can be added to the first/last points or at a regular interval along each line.
    - The function supports both DataFrame and list-of-dict input for results.
    - New: log_y1_axis and log_y2_axis control log scaling of the left and right y-axes.
    """
    import numpy as np
    
    def format_k_notation(value):
        """Format a number as 'k' notation (e.g., 5000 -> '5k')."""
        if value >= 1000:
            return f"{value/1000:.0f}k"
        else:
            return f"{value:.0f}"

    if not isinstance(results, pd.DataFrame):
        results_df = pd.DataFrame(results)
    else:
        results_df = results

    if ax is None:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        ax1 = ax
        fig = ax1.figure

    if title:
        ax1.set_title(title)

    # Plot enrichment (left y-axis)
    color = colors[0]
    sns.lineplot(
        data=results_df, 
        x="interaction_threshold", y="enrichment", 
        marker="o", ax=ax1, color=color
    )
    # Decrease marker border thickness and set zorder lower than text labels
    if ax1.get_lines():
        line = ax1.get_lines()[-1]
        line.set_markeredgewidth(0.5)
        line.set_zorder(1)
    ax1.set_ylabel(y1_axis_label, color=color)
    ax1.set_xlabel(x_axis_label)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add arrow below x-axis label if requested
    if add_arrow:
        xlabel_pos = ax1.get_xlabel()
        if xlabel_pos:
            if arrow_right_of_xlabel:
                # Position arrow to the right of xlabel, on the same y-plane
                arrow_y = 0.0  # Same y-level as xlabel
                # Position arrow starting just to the right of center, extending rightward
                arrow_start_x = 0.65
                arrow_end_x = 0.80
                # Position label centered above the arrow
                label_x = (arrow_start_x + arrow_end_x) / 2  # Center of arrow
                label_y = arrow_y + arrow_label_padding  # Position label above arrow
                if arrow_label_fontsize is None:
                    label_fontsize = "medium"
                else:
                    label_fontsize = arrow_label_fontsize
                annotate_kwargs = {
                    'xy': (label_x, label_y),
                    'xycoords': 'axes fraction',
                    'ha': 'center',
                    'va': 'bottom',
                    'fontsize': label_fontsize,
                    'color': 'black',
                    'zorder': 1000
                }
                if arrow_label_bold:
                    annotate_kwargs['weight'] = 'bold'
                ax1.annotate(
                    arrow_label,
                    **annotate_kwargs
                )
                ax1.annotate(
                    "",
                    xy=(arrow_end_x, arrow_y),
                    xytext=(arrow_start_x, arrow_y),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='black', lw=2.0),
                    ha='center',
                    va='center'
                )
            else:
                # Default: position arrow below x-axis label
                arrow_y = -arrow_title_padding
                label_y = arrow_y - arrow_label_padding
                if arrow_label_fontsize is None:
                    label_fontsize = "medium"
                else:
                    label_fontsize = arrow_label_fontsize
                annotate_kwargs = {
                    'xy': (0.5, label_y),
                    'xycoords': 'axes fraction',
                    'ha': 'center',
                    'va': 'top',
                    'fontsize': label_fontsize,
                    'color': 'black',
                    'zorder': 1000
                }
                if arrow_label_bold:
                    annotate_kwargs['weight'] = 'bold'
                ax1.annotate(
                    arrow_label,
                    **annotate_kwargs
                )
                ax1.annotate(
                    "",
                    xy=(0.7, arrow_y),
                    xytext=(0.3, arrow_y),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='black', lw=2.0),
                    ha='center',
                    va='top'
                )

    if log_x_axis:
        ax1.set_xscale("log") 
    if log_y1_axis:
        ax1.set_yscale("log")

    def add_min_max_labels(ax, 
                           results_df, 
                           x_col, 
                           y_col, 
                           first_label=True, 
                           last_label=True, 
                           label_fmt="{:.0f}", 
                           label_suffix="", 
                           label_kwargs=None, 
                           offset=0.0,
                           idx_offset=1,
                           y_offset=0.02,
                           log_y_axis=False):
        """
        Add text labels for minimum and maximum y values next to the first and last data points.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to add text to.
        results_df : pd.DataFrame
            DataFrame containing the data.
        x_col : str
            Name of the x column.
        y_col : str
            Name of the y column.
        first_label : bool
            Whether to label the first point.
        last_label : bool
            Whether to label the last point.
        label_fmt : str
            Format string for the label value.
        label_suffix : str
            Suffix to append to the label value.
        label_kwargs : dict or None
            Additional kwargs for ax.text.
        offset : float
            Y-offset for the label (in data units).
        idx_offset : int, default 1
            Index offset for first/last label (to avoid edge effects).
        y_offset : float, default 0.02
            Fraction of y-range to offset label vertically.
        log_y_axis : bool, default False
            Whether the y-axis is log-scaled (adjust y-offset accordingly).
        """
        if label_kwargs is None:
            label_kwargs = {}
        first_idx = idx_offset
        last_idx = len(results_df) - idx_offset
        first_lab_coords = results_df.iloc[first_idx][x_col], results_df.iloc[first_idx][y_col]
        last_lab_coords = results_df.iloc[last_idx][x_col], results_df.iloc[last_idx][y_col]

        y_vals = results_df[y_col].values
        if log_y_axis:
            # Avoid log(0) by filtering out non-positive values
            y_vals_pos = y_vals[y_vals > 0]
            if len(y_vals_pos) == 0:
                y_range = 1.0
            else:
                y_range = np.log10(y_vals_pos).max() - np.log10(y_vals_pos).min()
        else:
            y_range = y_vals.max() - y_vals.min()
        if log_y_axis:
            # Offset in log space, then exponentiate back
            def offset_y(y):
                if y > 0:
                    return y * (10 ** (y_offset * y_range))  # multiplicative offset
                else:
                    return y + 0.1  # fallback for non-positive
        else:
            def offset_y(y):
                return y + (y_range * y_offset if y_range > 0 else 0.1) + offset

        # Handle both format strings and callable formatters
        if callable(label_fmt):
            format_value = lambda y: label_fmt(y)
        else:
            format_value = lambda y: label_fmt.format(y)
        
        if first_label:
            x, y = first_lab_coords
            y_disp = offset_y(y)
            # Ensure zorder is set and can't be overridden by label_kwargs
            text_kwargs = {**label_kwargs, 'zorder': 1000}
            ax.text(
                x,
                y_disp,
                f"{format_value(y)}{label_suffix}",
                **text_kwargs
            )
        if last_label:
            x, y = last_lab_coords
            y_disp = offset_y(y)
            # Ensure zorder is set and can't be overridden by label_kwargs
            text_kwargs = {**label_kwargs, 'zorder': 1000}
            ax.text(
                x,
                y_disp,
                f"{format_value(y)}{label_suffix}",
                **text_kwargs
            )

    # Add min/max labels for enrichment (left y-axis)
    add_min_max_labels(
        ax1, results_df,
        x_col="interaction_threshold",
        y_col="enrichment",
        first_label=y1_first_label,
        last_label=y1_last_label,
        label_fmt="{:.0f}",
        label_suffix="x",
        label_kwargs=y1_label_kwargs,
        log_y_axis=log_y1_axis
    )
    # Add min/max labels for number of interactions (left y-axis, but can be used for right)
    # Use custom formatting function for "k" notation
    def format_n_interactions_label(y):
        return format_k_notation(y)
    add_min_max_labels(
        ax1, results_df,
        x_col="interaction_threshold",
        y_col="n_interactions",
        first_label=y2_first_label,
        last_label=y2_last_label,
        label_fmt=format_n_interactions_label,
        label_suffix="", 
        label_kwargs=y2_label_kwargs,
        log_y_axis=log_y2_axis
    ) 

    # Plot number of interactions (right y-axis)
    ax2 = ax1.twinx()
    color2 = colors[1]
    sns.lineplot(
        data=results_df,
        x="interaction_threshold", y="n_interactions",
        marker="s", ax=ax2, color=color2
    )
    # Decrease marker border thickness and set zorder lower than text labels
    if ax2.get_lines():
        line = ax2.get_lines()[-1]
        line.set_markeredgewidth(0.5)
        line.set_zorder(1)
    ax2.set_ylabel(y2_axis_label, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    if log_x_axis:
        ax2.set_xscale("log")
    if log_y2_axis:
        ax2.set_yscale("log")
    
    # Format y2 tick labels as "k" notation (e.g., 5000 -> 5k)
    from matplotlib.ticker import FuncFormatter
    def format_k(x, pos):
        if x >= 1000:
            return f"{x/1000:.0f}k"
        else:
            return f"{x:.0f}"
    ax2.yaxis.set_major_formatter(FuncFormatter(format_k))

    # Add text labels for enrichment at specified frequency (left y-axis)
    if y1_label_freq and y1_label_freq > 0:
        n_points = len(results_df)
        step = max(1, int(round(1/y1_label_freq)))
        y_vals = results_df["enrichment"].values
        if log_y1_axis:
            y_vals_pos = y_vals[y_vals > 0]
            if len(y_vals_pos) == 0:
                y_range = 1.0
            else:
                y_range = np.log10(y_vals_pos).max() - np.log10(y_vals_pos).min()
            def offset_y(y):
                if y > 0:
                    return y * (10 ** (0.015 * y_range))
                else:
                    return y + 0.1
        else:
            y_range = y_vals.max() - y_vals.min()
            def offset_y(y):
                return y + (y_range * 0.015 if y_range > 0 else 0.1)
        for i in range(0, n_points, step):
            row = results_df.iloc[i]
            x = row["interaction_threshold"]
            y = row["enrichment"]
            y_disp = offset_y(y)
            # Ensure zorder is set and can't be overridden by y1_label_kwargs
            text_kwargs = {**y1_label_kwargs, 'zorder': 1000}
            ax1.text(
                x, y_disp, f"{y:.1f}x", 
                **text_kwargs
            )

    # Add text labels for n_interactions at specified frequency (right y-axis)
    if y2_label_freq and y2_label_freq > 0:
        n_points = len(results_df)
        step = max(1, int(round(1/y2_label_freq)))
        y_vals = results_df["n_interactions"].values
        if log_y2_axis:
            y_vals_pos = y_vals[y_vals > 0]
            if len(y_vals_pos) == 0:
                y_range = 1.0
            else:
                y_range = np.log10(y_vals_pos).max() - np.log10(y_vals_pos).min()
            def offset_y(y):
                if y > 0:
                    return y * (10 ** (0.015 * y_range))
                else:
                    return y + 0.1
        else:
            y_range = y_vals.max() - y_vals.min()
            def offset_y(y):
                return y + (y_range * 0.015 if y_range > 0 else 0.1)
        for i in range(0, n_points, step):
            row = results_df.iloc[i]
            x = row["interaction_threshold"]
            y = row["n_interactions"]
            y_disp = offset_y(y)
            # Ensure zorder is set and can't be overridden by y2_label_kwargs
            text_kwargs = {**y2_label_kwargs, 'zorder': 1000}
            ax2.text(
                x, y_disp, format_k_notation(y), 
                **text_kwargs
            )

    # Add buffer to y-axis limits to prevent labels from being cut off
    # Calculate max y-value including label offsets for ax1 (enrichment)
    y1_vals = results_df["enrichment"].values
    if log_y1_axis:
        y1_vals_pos = y1_vals[y1_vals > 0]
        if len(y1_vals_pos) > 0:
            y1_max = y1_vals_pos.max()
            y1_range_log = np.log10(y1_vals_pos).max() - np.log10(y1_vals_pos).min()
            # Add buffer in log space (multiplicative)
            y1_upper = y1_max * (10 ** (0.1 * y1_range_log))  # 10% buffer in log space
        else:
            y1_upper = None
    else:
        y1_max = y1_vals.max()
        y1_range = y1_vals.max() - y1_vals.min()
        # Add buffer for labels (accounting for offset_y which adds ~1.5-2% of range)
        y1_upper = y1_max + (0.15 * y1_range if y1_range > 0 else y1_max * 0.15)
    
    # Calculate max y-value including label offsets for ax2 (n_interactions)
    y2_vals = results_df["n_interactions"].values
    if log_y2_axis:
        y2_vals_pos = y2_vals[y2_vals > 0]
        if len(y2_vals_pos) > 0:
            y2_max = y2_vals_pos.max()
            y2_range_log = np.log10(y2_vals_pos).max() - np.log10(y2_vals_pos).min()
            # Add buffer in log space (multiplicative)
            y2_upper = y2_max * (10 ** (0.1 * y2_range_log))  # 10% buffer in log space
        else:
            y2_upper = None
    else:
        y2_max = y2_vals.max()
        y2_range = y2_vals.max() - y2_vals.min()
        # Add buffer for labels (accounting for offset_y which adds ~1.5-2% of range)
        y2_upper = y2_max + (0.15 * y2_range if y2_range > 0 else y2_max * 0.15)
    
    # Set y-axis limits with buffer
    if y1_upper is not None:
        y1_lims = ax1.get_ylim()
        ax1.set_ylim([y1_lims[0], y1_upper])
    
    if y2_upper is not None:
        y2_lims = ax2.get_ylim()
        ax2.set_ylim([y2_lims[0], y2_upper])

    # Remove the top spines from both axes
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Ensure the layout is not squished
    fig.tight_layout()

    if show:
        plt.show()

    return {"fig": fig, "axes": [ax1, ax2], "data": results_df}



def add_wt_variants(vep_df,
                    haplotypes=None,
                    protein=None,
                    haps_df=None,
                    wt_prefix="wt_variant",
                    clinical_prefix="clinical_variant",
                    clinical_col="mutant",
                    add_positions=True):

    if wt_prefix in vep_df.columns:
        print(f"WT variants already added to {wt_prefix} column. Skipping.")
        return vep_df
    
    if protein is None:
        protein = vep_df['protein'].unique().tolist()[0]

    if haps_df is None:
        haps_df = hs.get_haplotype_names(haplotypes=haplotypes, 
                                        as_df=True)
        haps_df.loc[:,["variant"]] = haps_df['haplotype'].str.split(':').str[1].str.split(",")
        haps_df = haps_df.explode("variant").rename(columns={"variant": wt_prefix})

    # Filter VEP data for protein and merge with haplotype info
    vep_prot = vep_df[vep_df['protein'] == protein].merge(haps_df, on=["ENST","haplotype"], how="left")
    
    # Calculate variant-level statistics
    new_vep_mean_col = wt_prefix + "_VEP_mean"
    new_vep_std_col = wt_prefix + "_VEP_std"
    agg_stats = vep_prot.groupby(["protein",wt_prefix]).agg({"VEP":["mean","std"]}).reset_index()
    agg_stats.columns = ['protein', wt_prefix, new_vep_mean_col, new_vep_std_col]
    vep_prot = vep_prot.merge(agg_stats, on=["protein",wt_prefix], how="left").sort_values(new_vep_mean_col)

    if clinical_col in vep_prot.columns:
        vep_prot[clinical_prefix] = vep_prot[clinical_col]

    if add_positions:
        print(f"Adding positions to {wt_prefix} column")
        vep_prot = utils.variants_to_positions(vep_prot, 
                                                variant_col=wt_prefix,
                                                position_col=wt_prefix + "_position")
        if clinical_prefix in vep_prot.columns:
            print(f"Adding positions to {clinical_prefix} column")
            vep_prot = utils.variants_to_positions(vep_prot, 
                                                    variant_col=clinical_prefix,
                                                    position_col=clinical_prefix + "_position")
    
    return vep_prot


def get_clinical_vs_wt_variant_matrix(vep_prot,
                                      index="wt_variant",
                                      columns="clinical_variant", 
                                      value_col="VEP",
                                      aggfunc="mean",
                                      **kwargs):
    
    clinical_vs_wt = vep_prot.pivot_table(index=index, 
                                         columns=columns, 
                                         values=value_col, 
                                         aggfunc=aggfunc,
                                         **kwargs)
    
    return clinical_vs_wt
    
def get_wt_variant_vep_scores(vep_df,
                               haplotypes=None,
                               protein=None,
                               haps_df=None,
                               haps_to_samples=None):
    """Calculate VEP scores and frequencies for wild-type variants and generate a heatmap plot.
    
    Args:
        vep_df (pd.DataFrame): DataFrame containing VEP annotations
        haplotypes (list): List of haplotype identifiers
        protein (str, optional): Protein to analyze. If None, uses all proteins in vep_df
        haps_df (pd.DataFrame, optional): DataFrame mapping haplotypes to variants
        haps_to_samples (pd.DataFrame, optional): DataFrame mapping haplotypes to samples
        
    Returns:
        tuple: (var_vep_agg, var_freqs_agg, fig) containing:
            - vep_prot: DataFrame of VEP scores by protein and variant
            - var_vep_agg: DataFrame of aggregated VEP scores by variant
            - var_freqs_agg: DataFrame of variant frequencies
            - fig: Figure object with visualization
    """

    vep_df = vep_df.copy()
    
    if protein is None:
        protein = vep_df['protein'].unique().tolist()[0]

    if haps_df is None:
        haps_df = hs.get_haplotype_names(haplotypes=haplotypes, 
                                        as_df=True)
        haps_df.loc[:,["variant"]] = haps_df['haplotype'].str.split(':').str[1].str.split(",")
        haps_df = haps_df.explode("variant")

    if haps_to_samples is None:
        haps_to_samples = hs.haplotypes_to_samples(haplotypes=haplotypes, 
                                                    as_df=True,
                                                    add_sample_metadata=True,
                                                    return_seqs=False, 
                                                    add_ref=True)  
    
    # Filter VEP data for protein and merge with haplotype info
    vep_prot = vep_df[vep_df['protein'] == protein].merge(haps_df, on=["ENST","haplotype"], how="left")
    
    # Calculate variant-level statistics
    agg_stats = vep_prot.groupby(["protein","variant"]).agg({"VEP":["mean","std"]}).reset_index()
    agg_stats.columns = ['protein', 'variant', 'variant_VEP_mean', 'variant_VEP_std']
    vep_prot = vep_prot.merge(agg_stats, on=["protein","variant"], how="left").sort_values('variant_VEP_mean')
    
    # Calculate aggregated scores and frequencies
    var_vep_agg = _get_wt_variant_mean_vep_scores(vep_prot, haps_to_samples)
    var_freqs_agg = _get_wt_variant_mean_freqs(vep_prot, haplotypes)

    # Generate visualization
    fig, (ax1, ax2) = plot_wt_variant_vep_scores(var_freqs_agg, var_vep_agg)
        
    return vep_prot, var_vep_agg, var_freqs_agg, fig

def downsample_tick_labels(data, g, n_labels_x, n_labels_y):
    """
    Downsample tick labels for a clustermap visualization to improve readability.
    
    Args:
        data (pd.DataFrame): Input data used to create the clustermap
        g (sns.ClusterGrid): Seaborn clustermap object
        n_labels_x (int or bool): Number of x-axis labels to show. If False, no labels shown
        n_labels_y (int or bool): Number of y-axis labels to show. If False, no labels shown
    """
    # Get the reordered indices after clustering if dendrograms exist
    x_order = g.dendrogram_col.reordered_ind if hasattr(g, 'dendrogram_col') and g.dendrogram_col is not None else np.arange(len(data.columns))
    y_order = g.dendrogram_row.reordered_ind if hasattr(g, 'dendrogram_row') and g.dendrogram_row is not None else np.arange(len(data.index))

    # Calculate evenly spaced indices for labels
    x_indices = np.linspace(0, len(x_order)-1, n_labels_x, dtype=int)
    y_indices = np.linspace(0, len(y_order)-1, n_labels_y, dtype=int)

    # Set x-axis ticks and labels
    if n_labels_x is not False:
        g.ax_heatmap.set_xticks(range(len(data.columns)))
        # Use modulo to ensure even spacing of labels
        g.ax_heatmap.set_xticklabels([data.columns[x_order[i]] if i % (len(x_order)//n_labels_x) == 0 else '' for i in range(len(x_order))])

    # Set y-axis ticks and labels
    if n_labels_y is not False:
        g.ax_heatmap.set_yticks(range(len(data.index)))
        # Use modulo to ensure even spacing of labels
        g.ax_heatmap.set_yticklabels([data.index[y_order[i]] if i % (len(y_order)//n_labels_y) == 0 else '' for i in range(len(y_order))])

    return x_indices, y_indices, x_order, y_order


def sort_clinical_vs_wt_variant_matrix(vep_prot, clinical_vs_wt):
    
    print("Adding positions to WT variants")
    # Sort WT variants (rows) by residue position
    vep_prot=  utils.variants_to_positions(vep_prot, 
                                            variant_col='mutant',
                                            position_col='mutant_position')
    
    # Sort clinical variants (columns) by residue position
    print("Adding positions to clinical variants")
    vep_prot=  utils.variants_to_positions(vep_prot, 
                                            variant_col='variant',
                                            position_col='variant_position')
    
    # Sort rows by variant_position and columns by mutant_position
    print("Sorting clinical variants by WT variant positions")
    clinical_vs_wt = clinical_vs_wt.reindex(
        index=vep_prot.sort_values('variant_position')['variant'].unique(),
        columns=vep_prot.sort_values('mutant_position')['mutant'].unique()
    ) 
    return clinical_vs_wt, vep_prot


def pairwise_variant_sensitization_clustermap(vep_prot, 
                                            figsize=(20,15), 
                                            n_labels_x=50, 
                                            n_labels_y=50, 
                                            row_positions_col="mutant_position",
                                            col_positions_col="variant_position",
                                            normalize_procedure=["cols"],
                                            palette_positions="Blues",
                                            palette_heatmap="viridis",
                                            ref_linewidth=2,
                                            title_y=1.35,
                                            show_mean_vep_per_col=True,
                                            show_mean_vep_per_row=True,
                                            show_position_per_col=False,
                                            show_position_per_row=False,
                                            pow=1,
                                            **kwargs):
    
    vep_prot = vep_prot.copy()
    # Fill NaN values with 0
    # First create the pivot table
    clinical_vs_wt = vep_prot.pivot_table(index="wt_variant", 
                                         columns="clinical_variant", 
                                         values="VEP", 
                                         aggfunc="mean")

    clinical_vs_wt, vep_prot = sort_clinical_vs_wt_variant_matrix(vep_prot, clinical_vs_wt) 

    print("Creating color palettes")
    col_pos_cmap = utils.make_palette(vep_prot[col_positions_col].unique(), 
                                      palette=palette_positions )
    row_pos_cmap = utils.make_palette(vep_prot[row_positions_col].unique(), 
                                      palette=palette_positions ) 
    # Invert the scale
    print("Inverting scale")
    clinical_vs_wt = -clinical_vs_wt

    # Normalize the data
    clinical_vs_wt = utils.minmax_normalize(clinical_vs_wt, 
                                           procedure=normalize_procedure) 

    # Create column colors DataFrame with both position and mean VEP info
    print("Creating column colors")
    col_colors = vep_prot.groupby(["mutant", col_positions_col]).agg({"VEP": "mean"}).reset_index(level=col_positions_col)
    col_colors[col_positions_col] = col_colors[col_positions_col].map(col_pos_cmap) 

    mean_vep_per_col_cmap = utils.make_palette(col_colors["VEP"].unique(), palette="viridis")
    col_colors["VEP"] = col_colors["VEP"].map(mean_vep_per_col_cmap)
    print("Reindexing column colors")
    # Ensure unique column labels before reindexing
    col_colors = col_colors.loc[~col_colors.index.duplicated(keep='first')]
    col_colors = col_colors.reindex(clinical_vs_wt.columns)
    col_colors.rename(columns={"VEP": "mean VEP"}, inplace=True)
    
    # Create row colors DataFrame with both position and mean VEP info
    print("Creating row colors")
    row_colors = vep_prot.groupby(["variant",row_positions_col]).agg({"VEP": "mean"}).reset_index(level=row_positions_col)
    row_colors[row_positions_col] = row_colors[row_positions_col].map(row_pos_cmap) 
    
    mean_vep_per_row_cmap = utils.make_palette(row_colors["VEP"].unique(), palette="viridis")
    row_colors["VEP"] = row_colors["VEP"].map(mean_vep_per_row_cmap) 
    print("Reindexing row colors")
    row_colors = row_colors.loc[~row_colors.index.duplicated(keep='first')]
    row_colors = row_colors.reindex(clinical_vs_wt.index)
    row_colors.rename(columns={"VEP": "mean VEP"}, inplace=True)

    if not show_mean_vep_per_col:
        print("Dropping VEP column from column colors")
        col_colors = col_colors.drop(columns=['mean VEP'])
    if not show_mean_vep_per_row:
        print("Dropping VEP column from row colors")
        row_colors = row_colors.drop(columns=['mean VEP'])
    if not show_position_per_col:
        print("Dropping position column from column colors")
        col_colors = col_colors.drop(columns=[col_positions_col])
    if not show_position_per_row:
        print("Dropping position column from row colors")
        row_colors = row_colors.drop(columns=[row_positions_col])

    try:
        # Create clustermap with masked values
        # Ensure no duplicate labels in index and columns

        # Create a mask for the original NaN values
        print("Creating mask")
        mask = clinical_vs_wt.isna()
        # Fill NaN values with 0 to avoid errors
        clinical_vs_wt_clean = clinical_vs_wt.fillna(0)

        print("Creating clustermap")
        # Apply power transformation here so we can return the original data
        g = sns.clustermap(clinical_vs_wt_clean**pow,
                            figsize=figsize,
                            mask=mask,
                            col_colors=col_colors,
                            row_colors=row_colors,
                            cmap=palette_heatmap,
                            **kwargs)
    except Exception as e:
        print(f"Error creating clustermap: {str(e)}")
        return [col_colors, row_colors], clinical_vs_wt_clean

    # --- Robust tick label downsampling to avoid ZeroDivisionError ---
    def safe_downsample_tick_labels_middle(data, g, n_labels_x, n_labels_y):
        # Get the reordered indices after clustering if dendrograms exist
        x_order = g.dendrogram_col.reordered_ind if hasattr(g, 'dendrogram_col') and g.dendrogram_col is not None else np.arange(len(data.columns))
        y_order = g.dendrogram_row.reordered_ind if hasattr(g, 'dendrogram_row') and g.dendrogram_row is not None else np.arange(len(data.index))

        # X-axis: put ticks in the middle of each column
        n_cols = len(x_order)
        if not n_labels_x or n_labels_x <= 0:
            g.ax_heatmap.set_xticks([])
            g.ax_heatmap.set_xticklabels([])
            x_indices = []
        else:
            # Calculate which columns to label (downsample if needed)
            if n_labels_x >= n_cols:
                label_cols = np.arange(n_cols)
            else:
                label_cols = np.linspace(0, n_cols-1, n_labels_x, dtype=int)
            # Ticks at the center of each column: for imshow, columns are at integer positions, so center is at i+0.5
            xticks = [i + 0.5 for i in label_cols]
            xticklabels = [data.columns[x_order[i]] for i in label_cols]
            g.ax_heatmap.set_xticks(xticks)
            g.ax_heatmap.set_xticklabels(xticklabels, rotation=0)
            x_indices = label_cols

        # Y-axis: put ticks in the middle of each row
        n_rows = len(y_order)
        if not n_labels_y or n_labels_y <= 0:
            g.ax_heatmap.set_yticks([])
            g.ax_heatmap.set_yticklabels([])
            y_indices = []
        else:
            if n_labels_y >= n_rows:
                label_rows = np.arange(n_rows)
            else:
                label_rows = np.linspace(0, n_rows-1, n_labels_y, dtype=int)
            yticks = [i + 0.5 for i in label_rows]
            yticklabels = [data.index[y_order[i]] for i in label_rows]
            g.ax_heatmap.set_yticks(yticks)
            g.ax_heatmap.set_yticklabels(yticklabels, rotation=0)
            y_indices = label_rows

        return x_indices, y_indices, x_order, y_order

    x_indices, y_indices, x_order, y_order = safe_downsample_tick_labels_middle(clinical_vs_wt, g, n_labels_x, n_labels_y)

    # Add horizontal line for REF sample
    if ref_linewidth is not None:
        try:
            ref_idx = np.where(np.array([x for x in clinical_vs_wt.index[y_order]]) == 'REF')[0][0]
            g.ax_heatmap.axhline(y=ref_idx, color='red', linestyle='--', alpha=1, linewidth=ref_linewidth)
        except Exception as e:
            print(f"Could not add REF line: {e}")

    g.ax_heatmap.collections[0].colorbar.set_label("Mean VEP score")
    g.ax_heatmap.set_xlabel("Clinical variants")
    g.ax_heatmap.set_ylabel("WT variants")
    g.ax_heatmap.set_title(f"Pairwise Variant Sensitization Analysis\nProtein: {vep_prot['GENEINFO'].unique()[0].split(':')[0]} ({vep_prot['protein'].unique()[0]})\nInjected clinical variants (n={clinical_vs_wt.shape[1]} cols) x Natural WT variants (n={clinical_vs_wt.shape[0]} rows)",
                        ha="left", x=0.1, y=title_y)
    plt.show()
    return g, clinical_vs_wt, row_colors, col_colors

def sample_by_mutant_clustermap(vep_df, 
                                haps_to_samples,
                                protein=None,
                                figsize=(15,6),
                                palette_heatmap="viridis",
                                t=1.15,
                                criterion='inconsistent'):
    
    from scipy.cluster.hierarchy import fcluster

    # Set seed
    np.random.seed(42)

    ###### Data preparation ######
    # Get data for NP_009225.1 and merge with haplotypes
    protein_data = vep_df.copy()
    if protein is not None:
        protein_data = protein_data.loc[protein_data['protein'] == protein]
    protein_data = protein_data.merge(haps_to_samples, on=["haplotype"], how="left")
    print(protein_data.shape)

    gene_list = list(set([x.split(':')[0] for x in protein_data["GENEINFO"].unique()]))

    # Create pivot table for heatmap and transpose it
    heatmap_data = protein_data.pivot_table(
        index='sample',
        columns='mutant',
        values='VEP',
        aggfunc="mean"
    )
    # Min-max normalize the heatmap data
    heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    # Log transform the data first
    heatmap_data = np.log1p(heatmap_data)

    # Get super population info for each sample and create a numeric mapping
    pop_data = haps_to_samples[['sample', "superpopulation", "sex"]].drop_duplicates()
    pop_data = pop_data.set_index('sample')

    # Reindex pop_data to match the row order of heatmap_data
    pop_data = pop_data.reindex(heatmap_data.index)

    # Create numeric mapping for populations
    pop_mapping = {pop: i for i, pop in enumerate(pop_data["superpopulation"].unique())}
    pop_data['Superpop'] = pop_data["superpopulation"].map(pop_mapping)

   
    ###### Clustermap plotting ######
    # Create row colors dataframe with both Super Population and Gender
    sex_palette = {'male': 'lightblue', 'female': 'mistyrose'}
    row_colors = pd.DataFrame({
        'Superpop': pop_data["superpopulation"].map(utils.get_superpop_palette()),
        'Sex': pop_data['sex'].map(sex_palette)  # Map gender values to colors
    })

    # Create clustermap but don't display it
    g1 = sns.clustermap(heatmap_data,
                        figsize=figsize,
                    cmap=palette_heatmap,
                    cbar_kws={'label': 'VEP Score', 'location': 'left', 'pad': 0.025},
                    row_colors=row_colors,
                    yticklabels=False,
                    xticklabels=False)
    plt.close()  # Close the figure to prevent display

    # Get the column linkage
    col_linkage = g1.dendrogram_col.linkage

    # Use scipy's fcluster to get cluster assignments 
    col_clusters = fcluster(col_linkage, 
                            t=t, 
                            criterion=criterion)
    n_clusters = len(np.unique(col_clusters))
    print(f"Number of col clusters: {n_clusters}")

    # Create variant to cluster mapping
    variant_to_cluster = {}
    for variant_id, cluster_id in zip(heatmap_data.columns, col_clusters):
        variant_to_cluster[variant_id] = cluster_id

    # Create a color palette for the clusters
    cluster_colors = sns.color_palette('tab20', n_colors=n_clusters)
    col_colors = pd.Series([cluster_colors[i-1] for i in col_clusters], index=heatmap_data.columns)

    # Create final clustermap with both row and column colors
    g1 = sns.clustermap(heatmap_data,
                        figsize=figsize,
                    cmap=palette_heatmap,
                    cbar_kws={'label': 'VEP Score', 'location': 'left', 'pad': 0.025},
                    row_colors=row_colors,
                    col_colors=col_colors,
                    yticklabels=False,
                    xticklabels=False)

    # Add title centered on the heatmap
    g1.fig.suptitle(f'VEP Score Clusters\nProtein: {gene_list[0] if len(gene_list) == 1 else len(gene_list)} ({protein_data["protein"].unique()[0]})\n{heatmap_data.shape[1]} variants x {heatmap_data.shape[0]} samples',
                    x=0.5, y=1.02, ha='center')

    # Add x-axis label
    g1.ax_heatmap.set_xlabel('Clinical variant')

    # Add y-axis label and rotate it horizontally
    g1.ax_heatmap.set_ylabel('Sample', rotation=270, va="bottom")
    g1.ax_heatmap.yaxis.set_label_position('right')
    g1.ax_heatmap.yaxis.tick_right()

    # Find REF sample position
    if 'REF' in heatmap_data.index:
        ref_idx = np.where(heatmap_data.index == 'REF')[0][0]

        # Add vertical line for REF sample
        g1.ax_heatmap.axhline(y=ref_idx, color='red', linestyle='--', alpha=1, linewidth=5)

    # Add legends for row colors
    # Super Population legend
    palette = utils.get_superpop_palette()
    handles = [plt.Rectangle((0,0),1,1, facecolor=color) for color in palette.values()]
    labels = list(palette.keys())
    superpop_legend = plt.legend(handles, labels,
            title='Superpop', 
            bbox_to_anchor=(0.5, -0.15),
            loc='upper center',
            ncol=1)

    # Add the first legend to the plot
    plt.gca().add_artist(superpop_legend)

    # Gender legend
    sex_handles = [plt.Rectangle((0,0),1,1, facecolor=color) for color in sex_palette.values()]
    sex_labels = ['Male', 'Female']
    plt.legend(sex_handles, sex_labels,
            title='Sex',
            bbox_to_anchor=(0.5, -2),
            loc='upper center',
            ncol=1) 

    return g1, variant_to_cluster

def plot_vep_variance_zscore(vep_df):
    
    # Display results
    print("Distribution of z-scores for REF haplotypes:")
    print(vep_df.loc[vep_df['is_ref'], ['protein', 'mutant', 'VEP', 'mean', 'std', 'z_score','z_score_abs']].describe())


    # Sort clinsig by palette keys
    clinsig_order = list(utils.get_clinsig_palette().keys())
    vep_df['clinsig'] = pd.Categorical(vep_df['clinsig'], categories=clinsig_order, ordered=True)

    vep_df["z_score_log"] = np.log10(vep_df["z_score_abs"])
    # Plot distribution of z-scores
    plt.figure(figsize=(15, 4))
    sns.histplot(data=vep_df.loc[vep_df['is_ref']], 
                x='z_score',
                hue="clinsig",
                #  multiple="fill",
                palette=utils.get_clinsig_palette(),
                bins=1000)
    # plt.xlim(-5, 5)

    # Calculate percentages for each standard deviation
    ref_data = vep_df.loc[vep_df['is_ref'], 'z_score']
    ref_data_neg = ref_data[ref_data < 0]
    ref_data_pos = ref_data[ref_data > 0]

    for sd in [-2, -1]:
        percentage = ((ref_data_neg >= sd) & (ref_data_neg < 0)).mean() * 100
        print(percentage)
        plt.axvline(x=sd, color='black', linestyle='--', alpha=0.5)
        plt.text(sd-0.1, plt.ylim()[1]*0.95, f'{percentage:.1f}%', 
                horizontalalignment='center', verticalalignment='top',
                rotation=90)

    for sd in [1, 2]:
        percentage = ((ref_data_pos <= sd) & (ref_data_pos > 0)).mean() * 100
        plt.axvline(x=sd, color='black', linestyle='--', alpha=0.5)
        plt.text(sd+0.1, plt.ylim()[1]*0.95, f'{percentage:.1f}%', 
                horizontalalignment='center', verticalalignment='top',
                rotation=90)

    # Add breaks to x-axis
    plt.axvspan(-5, -4.5, alpha=0.2, color='gray')
    plt.axvspan(4.5, 5, alpha=0.2, color='gray')
    plt.text(-4.75, plt.ylim()[1]*0.5, '...', ha='center', va='center', fontsize=20)
    plt.text(4.75, plt.ylim()[1]*0.5, '...', ha='center', va='center', fontsize=20)

    plt.title('Distribution of REF haplotype z-scores')
    plt.xlabel('Z-score (standard deviations from mean)')
    plt.ylabel('Count')
    plt.show()

def add_rectangles(xy_pairs, 
                   X=None,
                   x_col="wt_variant",
                   y_col="clinical_variant",
                   xy_pairs_are_idx=False,
                   color_col="outlier_type",
                   shape_col=None,
                   shape_func=plt.Rectangle,
                   cmap=None,
                   palette=None,
                   height=1,
                   width=1,
                   linewidth=0.5,
                   angle=0,
                   rotation_point='xy',
                   fill=False,
                   **kwargs):
    """Add colored rectangles to highlight significant variant pairs in a plot.
    
    Parameters
    ----------
    X : pandas.DataFrame
        The main data matrix containing variant pairs
    xy_pairs : pandas.DataFrame
        DataFrame containing pairs of variants to highlight
    x_col : str, default="wt_variant"
        Column name in xy_pairs containing x-axis variant labels
    y_col : str, default="clinical_variant" 
        Column name in xy_pairs containing y-axis variant labels
    color_col : str, default="outlier_type"
        Column name in xy_pairs containing color categories
    cmap : dict, optional
        Custom color mapping dictionary
    palette : str, optional
        Seaborn color palette name to use if cmap not provided
        
    Returns
    -------
    None
        Adds rectangles to the current matplotlib axes
    """
    print("Adding rectangles")
    if palette is None:
        palette = "Set3"
    if cmap is None:
        cmap = utils.make_palette(xy_pairs[color_col].unique(), palette=palette)
        
    if not xy_pairs_are_idx:
        if X is None:
            raise ValueError("X must be provided if xy_pairs_are_idx is False")
    
    # Pre-compute shape function mapping if needed
    shape_funcs = None
    if shape_col is not None:
        unique_values = xy_pairs[shape_col].unique()
        if len(unique_values) == 2 and set(unique_values).issubset({0, 1, True, False}):
            shape_funcs = {1: plt.Rectangle, 0: plt.Circle}
    
    # Pre-compute index mappings if needed
    if not xy_pairs_are_idx:
        x_idx_map = {val: X.index.get_loc(val) for val in xy_pairs[x_col].unique()}
        y_idx_map = {val: X.columns.get_loc(val) for val in xy_pairs[y_col].unique()}
    
    # Pre-compute colors for all unique values
    color_map = {val: cmap[val] for val in xy_pairs[color_col].unique()}
    
    # Get current axes once
    ax = plt.gca()
    
    # Vectorized processing
    patches = []
    for _, row in xy_pairs.iterrows():
        # Get indices
        if xy_pairs_are_idx:
            row_idx, col_idx = row[x_col], row[y_col]
        else:
            row_idx, col_idx = x_idx_map[row[x_col]], y_idx_map[row[y_col]]
        
        # Get color and shape function
        color = color_map[row[color_col]]
        current_shape_func = shape_funcs[row[shape_col]] if shape_funcs is not None else shape_func
        
        # Create shape parameters based on shape type
        if current_shape_func == plt.Circle:
            shape_params = {
                'xy': (col_idx, row_idx),
                'radius': min(width, height) / 2,
                'fill': fill,
                'linewidth': linewidth,
                'edgecolor': color,
                'facecolor': color,
                **kwargs
            }
        else:
            # Rectangle-specific parameters
            shape_params = {
                'xy': (col_idx, row_idx),
                'width': width,
                'height': height,
                'angle': angle,
                'rotation_point': rotation_point,
                'fill': fill,
                'linewidth': linewidth,
                'edgecolor': color,
                'facecolor': color,
                **kwargs
            }
        
        # Create and store patch
        patches.append(current_shape_func(**shape_params))
    
    # Add all patches at once
    ax.add_collection(plt.matplotlib.collections.PatchCollection(patches, match_original=True))



def identify_outliers(X):
    import scipy.stats as stats 
    from statsmodels.stats.multitest import multipletests

    # Initialize lists to store results
    columns = []
    variants = []
    values = []
    outlier_types = []
    z_scores = []
    p_values = []  # New list to store p-values
    
    for col in X.columns:
        # Calculate mean and standard deviation
        mean = X[col].mean()
        std = X[col].std()
        
        # Identify values more than 2 standard deviations from mean
        outliers = X[col][abs(X[col] - mean) > 2 * std]
        
        if not outliers.empty:
            # Add high outliers
            high_outliers = outliers[outliers > mean]
            for idx, val in high_outliers.items():
                columns.append(col)
                variants.append(idx)
                values.append(val)
                outlier_types.append('high')
                z = (val - mean) / std
                z_scores.append(z)
                # Calculate two-tailed p-value from z-score
                p_values.append(2 * (1 - stats.norm.cdf(abs(z))))
            
            # Add low outliers
            low_outliers = outliers[outliers < mean]
            for idx, val in low_outliers.items():
                columns.append(col)
                variants.append(idx)
                values.append(val)
                outlier_types.append('low')
                z = (val - mean) / std
                z_scores.append(z)
                # Calculate two-tailed p-value from z-score
                p_values.append(2 * (1 - stats.norm.cdf(abs(z))))
    
    # Create DataFrame from results
    df =  pd.DataFrame({
        'clinical_variant': columns,
        'wt_variant': variants,
        'value': values,
        'outlier_type': outlier_types,
        'z_score': z_scores,
        'p_value': p_values  # Add p-values to DataFrame
    })

    # Get the p-values and apply FDR correction
    # rejected: Boolean array indicating which hypotheses were rejected after FDR correction
    # (True means the null hypothesis was rejected, i.e. the result is statistically significant)
    significant, p_adjusted, _, _ = multipletests(df['p_value'], method='fdr_bh')

    # Add adjusted p-values to DataFrame
    df['p_adjusted'] = p_adjusted
    df['significant'] = significant

    # Sort by adjusted p-value
    df = df.sort_values('p_adjusted')
    return df


def _get_wt_variant_mean_vep_scores(vep_prot,
                                    haps_to_samples):
    
    ## Mean per-variant VEP scores  
    var_vep_agg = vep_prot.copy().merge(haps_to_samples, on=["haplotype"], how="left").groupby(["superpopulation","variant"]).agg({"VEP":"mean"}).sort_values("VEP", ascending=False).reset_index()
    var_vep_agg = var_vep_agg.loc[var_vep_agg["superpopulation"] != "REF"].pivot(index="variant", columns="superpopulation", values="VEP")

    # var_vep_agg = var_vep_agg.fillna(0)
    # Normalize and invert the scale
    var_vep_agg = (var_vep_agg - var_vep_agg.min()) / (var_vep_agg.max() - var_vep_agg.min())
    var_vep_agg = 1 - var_vep_agg  # Invert the scale so higher values are better
    
    # Add ALL column after all normalization steps
    var_vep_agg['ALL'] = var_vep_agg.mean(axis=1, skipna=True)
    var_vep_agg = var_vep_agg[['ALL'] + [col for col in var_vep_agg.columns if col != 'ALL']]

    return var_vep_agg

def _get_wt_variant_mean_freqs(vep_prot,
                               haplotypes,
                               fillna=0):
    
    vep_prot = hs.add_haplotype_freqs(vep_prot, haplotypes=haplotypes)

    superpops = og.get_sample_metadata()["superpopulation"].dropna().unique().tolist()
    freq_cols = ["freq_1000GENOMES:phase_3:ALL"] + [f"freq_1000GENOMES:phase_3:{pop}" for pop in superpops]
    # Ensure columns are present in vep_prot
    freq_cols = [col for col in freq_cols if col in vep_prot.columns]

    # Take the mean across individuals for each haplotype
    var_freqs_agg = vep_prot.groupby(["haplotype","variant"]).agg(dict(zip(freq_cols, ["mean"]*len(freq_cols))))
    # Then take the sum across haplotypes
    var_freqs_agg = var_freqs_agg.groupby(["variant"]).agg(dict(zip(freq_cols, ["mean"]*len(freq_cols))))
    # Remove prefix before : from all column names
    var_freqs_agg.columns = [col.split(':')[-1] if ':' in col else col for col in var_freqs_agg.columns]

    # Divide by 2 to account for the fact that we have two haplotypes?
    # var_freqs_agg = var_freqs_agg/2 

    if fillna is not None:
        var_freqs_agg = var_freqs_agg.fillna(fillna)
    
    return var_freqs_agg

def plot_wt_variant_vep_scores(
    var_freqs_agg, 
    var_vep_agg, 
    highlight_cells=False, 
    figsize=(13, 16), 
    width_ratios=[6, 6]):
    """
    Plot two heatmaps side by side: variant frequencies and VEP scores.
    Optionally highlight cells with values >0 in both heatmaps at the same locations.

    Parameters
    ----------
    var_freqs_agg : pd.DataFrame
        DataFrame of variant frequencies (variants x superpopulations).
    vep_prot_pops : pd.DataFrame
        DataFrame of VEP scores (variants x superpopulations).
    highlight_cells : bool, optional
        If True, highlight cells with values >0 in both heatmaps at the same locations.
    figsize : tuple, optional
        Figure size.
    width_ratios : list, optional
        Width ratios for the subplots.
    """
    import matplotlib.colors as mcolors
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': width_ratios})

    # Cluster the y-axis labels
    Z = linkage(pdist(var_freqs_agg), method='ward')
    ordered_labels = var_freqs_agg.index[dendrogram(Z, no_plot=True)['leaves']]

    # Prepare mask for highlighting: only highlight cells that are >0 at the same coordinates in both plots 
    highlight_mask = None
    if highlight_cells:
        # Align both dataframes to the same index/columns order
        var_freqs_agg_aligned = var_freqs_agg.reindex(index=ordered_labels, columns=var_vep_agg.columns)
        var_vep_agg_aligned = var_vep_agg.reindex(index=ordered_labels, columns=var_vep_agg.columns)
        # Only highlight cells where both are >0
        highlight_mask = (
            (var_freqs_agg_aligned != 0) & (var_vep_agg_aligned != 0) &
            (~var_freqs_agg_aligned.isna()) & (~var_vep_agg_aligned.isna())
        )  

    # Plot frequency heatmap on left with clustered order, log color scale
    sns.heatmap(
        var_freqs_agg.reindex(ordered_labels),
        annot=True,
        fmt=".3f",
        cmap="viridis",
        linewidths=0.01,
        linecolor='grey',
        ax=ax1,
        yticklabels=False,
        cbar_kws={'location': 'left', 'pad': 0.04},
        norm=mcolors.LogNorm(
            vmin=var_freqs_agg[var_freqs_agg > 0].min().min(), 
            vmax=var_freqs_agg.max().max()
        )
    )
    ax1.set_title('Variant Frequencies')
    ax1.set_xlabel("Superpopulation")

    # Plot VEP scores on right using same order, log color scale
    sns.heatmap(
        var_vep_agg.reindex(ordered_labels),
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=0.01,
        linecolor='grey',
        ax=ax2,
        norm=mcolors.LogNorm(
            vmin=var_vep_agg[var_vep_agg > 0].min().min(),
            vmax=var_vep_agg.max().max()
        )
    )
    ax2.set_title('Variant VEP Scores')
    ax2.set_xlabel("Superpopulation")
    plt.ylabel(None)

    # Optionally overlay highlight rectangles for cells >0 in both plots at the same coordinates
    if highlight_cells and highlight_mask is not None:
        def highlight_rects(ax, mask_df):
            for i, idx in enumerate(mask_df.index):
                for j, col in enumerate(mask_df.columns):
                    if mask_df.iloc[i, j]:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2, clip_on=False))
        highlight_rects(ax1, highlight_mask)
        highlight_rects(ax2, highlight_mask)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_wt_variant_vep_scores_dotplot(var_vep_agg, var_freqs_agg, figsize=(6, 15)):
    # Create figure and axis
    plt.figure(figsize=figsize)

    # Melt the dataframes
    vep_melted = var_vep_agg.reset_index().melt(id_vars='variant', 
                                            var_name='population',
                                            value_name='vep_score')

    freq_melted = var_freqs_agg.reset_index().melt(id_vars='variant',
                                                var_name='population',
                                                value_name='frequency')

    # Merge the melted dataframes
    plot_data = vep_melted.merge(freq_melted, on=['variant', 'population'])

    # Create scatter plot using seaborn
    scatter = sns.scatterplot(data=plot_data,
                            x='population',
                            y='variant',
                            hue='vep_score',
                            size='frequency',
                            sizes=(0, 1000),  # Scale sizes for better visibility
                            palette='viridis',
                            alpha=0.7)

    # Get handles and labels for size legend
    handles, labels = scatter.get_legend_handles_labels()
    # Remove the hue legend
    scatter.legend_.remove()

    # Create custom size legend with meaningful labels
    size_legend = plt.legend(handles[1:], 
                            [f'{freq:.2f}' for freq in sorted(plot_data['frequency'].unique())],  # Sort frequencies
                            title='Frequency',
                            bbox_to_anchor=(1.05, 1),
                            loc='upper left',
                            markerscale=1.5)  # Make legend markers larger

    # Set labels and ticks
    plt.xticks(rotation=45, ha='right')

    # Set y-axis limits to remove extra whitespace
    plt.ylim(-0.75, len(plot_data['variant'].unique()) - 0.25)

    # Add title and adjust layout
    plt.title('Variant VEP Scores and Frequencies by Population')
    plt.tight_layout()

    # Show plot
    plt.show()

def plot_sensitization_maps(
    clinical_vs_wt, 
    vep_prot,  
    ref_linewidth=2, 
    z_score_threshold=4.5, 
    p_adjusted_threshold=0.001,
    highlight_cmap={"high": "red", "low": "blue"}
):
    """
    Plot pairwise variant sensitization maps with and without outlier highlighting.

    Parameters
    ----------
    clinical_vs_wt : pd.DataFrame
        Matrix of VEP scores (clinical variants x WT variants).
    vep_prot : pd.DataFrame
        DataFrame with protein/variant annotation.
    va : module
        Module with identify_outliers and add_rectangles functions.
    utils : module
        Module with variants_to_positions function.
    ref_linewidth : int, optional
        Line width for reference line, by default 2.
    z_score_threshold : float, optional
        Z-score threshold for outlier significance, by default 4.5.
    p_adjusted_threshold : float, optional
        Adjusted p-value threshold for outlier significance, by default 0.001.
    highlight_cmap : dict, optional
        Color map for outlier highlighting, by default {"high": "red", "low": "blue"}.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,10))

    # Get outlier analysis and display results
    outlier_df = identify_outliers(clinical_vs_wt)

    Xs = clinical_vs_wt**5

    # First heatmap - original
    sns.heatmap(Xs, 
                cmap='viridis', 
                cbar_kws={'label': 'VEP Score'}, 
                alpha=1, ax=ax1)
    ax1.set_xlabel("WT variant position")
    ax1.set_ylabel("Clinical variant position")
    ax1.set_title(
        f"Pairwise Variant Sensitization Map\nProtein: {vep_prot['GENEINFO'].unique()[0].split(':')[0]} "
        f"({vep_prot['protein'].unique()[0]})\nInjected clinical variants (n={clinical_vs_wt.shape[1]} cols) "
        f"x Natural WT variants (n={clinical_vs_wt.shape[0]} rows)"
    )
    ax1.tick_params(axis='x', rotation=90)
    if ref_linewidth is not None:
        ref_idx = np.where(np.array([x for x in clinical_vs_wt.index]) == 'REF')[0][0]
        print(ref_idx)
        ax1.axhline(y=ref_idx, color='red', linestyle='--', alpha=1, linewidth=ref_linewidth)

    # Add sequence positions to the outlier dataframe
    outlier_df = utils.variants_to_positions(
        outlier_df, 
        variant_col="wt_variant", 
        position_col="wt_position",
        ref_col="wt_REF",
        alt_col="wt_ALT"
    )
    outlier_df = utils.variants_to_positions(
        outlier_df, 
        variant_col="clinical_variant", 
        position_col="clinical_position",
        ref_col="clinical_REF",
        alt_col="clinical_ALT"
    )
    # Filter to most sig results
    outlier_sig = outlier_df[
        (outlier_df['significant']) &  # Must be significant after FDR correction
        (abs(outlier_df['z_score']) > z_score_threshold) &  # Must have very extreme z-scores
        (outlier_df['p_adjusted'] < p_adjusted_threshold)  # Must have very small adjusted p-values
    ].copy()

    if "clinsig" in outlier_sig.columns:
        outlier_sig['is_pathogenic'] = outlier_sig['clinsig'].isin(["path", "likely path"]) 

    # Remove outliers with the same sequence position
    outlier_sig = outlier_sig.loc[outlier_sig["wt_position"] != outlier_sig["clinical_position"]]

    # Remove large deletions
    outlier_sig = outlier_sig.loc[
        ~(outlier_sig["wt_variant"].str.contains("del{") | outlier_sig["wt_variant"].str.contains("del{"))
    ]
    outlier_sig = outlier_sig.merge(
        vep_prot[["mutant", "clinsig"]].drop_duplicates(), 
        left_on="clinical_variant", 
        right_on="mutant", how="left"
    )

    print(outlier_sig.shape)

    # Second heatmap - with highlighted outliers
    sns.heatmap(Xs, cmap='viridis', cbar_kws={'label': 'VEP Score'}, 
                alpha=0.3, ax=ax2)

    add_rectangles(
        X=Xs, 
        xy_pairs=outlier_sig, 
        x_col="wt_variant",
        y_col="clinical_variant",
        color_col="outlier_type",
        cmap=highlight_cmap
    )

    ax2.set_xlabel("WT variant position")
    ax2.set_ylabel("Clinical variant position")
    ax2.set_title(
        f"Pairwise Variant Sensitization Map Highlighted by Outliers\nProtein: {vep_prot['GENEINFO'].unique()[0].split(':')[0]} "
        f"({vep_prot['protein'].unique()[0]})\nInjected clinical variants (n={clinical_vs_wt.shape[1]} cols) "
        f"x Natural WT variants (n={clinical_vs_wt.shape[0]} rows)"
    )
    ax2.tick_params(axis='x', rotation=90)
    if ref_linewidth is not None:
        ref_idx = np.where(np.array([x for x in clinical_vs_wt.index]) == 'REF')[0][0]
        print(ref_idx)
        ax2.axhline(y=ref_idx, color='red', linestyle='--', alpha=1, linewidth=ref_linewidth) 

    plt.tight_layout() 


    return {'fig': fig, 'axes': (ax1, ax2), 
            'data': {'Xs': Xs,
                     'outlier_df': outlier_df, 
                     'outlier_sig': outlier_sig}}



def mask_by_percentile(matrix, percentile=50, invert=False):
    """
    Create a boolean mask for matrix values below a certain percentile.
    
    Parameters
    ----------
    matrix : np.ndarray or pd.DataFrame
        Matrix whose values to mask.
    percentile : float, default 50
        Percentile at which to mask (values below the percentile are masked).
    invert : bool, default False
        If True, invert the mask (mask values above the percentile).
    
    Returns
    -------
    mask : np.ndarray (dtype=bool)
        Boolean mask array.
    """
    m_arr = matrix.values if hasattr(matrix, "values") else matrix
    masking_threshold = np.percentile(m_arr[~np.isnan(m_arr)], q=percentile)
    mask = m_arr < masking_threshold
    if invert:
        mask = ~mask
    return mask

def plot_contact_and_sensitization_maps(
    contact_map_binned,
    outlier_sig,
    add_rect=True,
    figsize=None,
    cmap=("gnuplot2", "binary"),
    masking_percentile1=50,
    masking_percentile2=25, 
    title=None,
    x_label="Clinical Variant Position",
    y_label="WT Variant Position",
    invert_mask=False,
    height_width=20,
    heatmap_alpha=None,
    rectangles_alpha=0.9,
    cbar_label=r"3D Distance (Ångstroms)",
    rectangles_linewidth=0.5,
    rectangles_cmap={"high": "red", "low": "blue"},
    mask_color=None,
    show_plot=(True, True), 
    dpi=300,
    save_fig=False,
    save_path=None,
    pow1=1,
    pow2=1,
):
    """
    Plot contact map and overlay sensitization map with optional rectangles.

    Parameters
    ----------
    contact_map_binned : np.ndarray or pd.DataFrame
        Binned contact map.
    outlier_sig : pd.DataFrame
        DataFrame with outlier signal (for rectangles).
    add_rect : bool, default True
        Whether to add rectangles for outlier_sig.
    figsize : tuple, default (10, 10)
        Figure size.
    dpi : int, default 300
        DPI for figures.
    save_fig : bool, default False
        Whether to save the figure.
    save_path : str or None, default None
        Path to save the figure.
    cmap : str, default "gnuplot2"
        Colormap for contact map.
    height_width : int, default 20
        Rectangle width/height.
    heatmap_alpha : float or None, default None
        Alpha for heatmap.
    rectangles_alpha : float, default 0.9
        Alpha for rectangles.
    mask_color : str or None, default None
        Color for masked values.
    show_plot : tuple of bool, default (True, True)
        Tuple indicating which plots to show: (show_plot1, show_plot2)
    pow1 : float, default 1
        Power/exponent to apply to contact_map for plot 1.
    pow2 : float, default 1
        Power/exponent to apply to contact_map for plot 2.
    """

    outlier_sig = outlier_sig.copy()
    
    contact_map_unbinned = contact_map_binned

    if save_path is None:
        save_path = f"contact_map_{'sensitization_map' if add_rect else ''}.png"

    if heatmap_alpha is None:
        heatmap_alpha = 0.5 if add_rect else 1

    outlier_sig = outlier_sig.dropna(subset=["wt_position", "clinical_position","outlier_type"])

    # Get mask for plot 1
    mask = mask_by_percentile(contact_map_unbinned, percentile=masking_percentile1, invert=invert_mask)

    def add_plot_labels(g, 
                        bin_size,
                        n_bins,
                        xlabel="Clinical Variant Position", 
                        ylabel="WT Variant Position", 
                        title=None,
                        add_rect=False):
        if title is None:
            title = f"Contact Map {'x Variant Sensitization Map' if add_rect else ''}"
        if title is not None:
            plt.title(title)
        g.collections[0].colorbar.set_label(cbar_label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        mc.label_bins(bin_size=bin_size, n_bins=n_bins)

    def color_mask(g, mask_color = None):
        # Set the color for masked values 
        if mask_color is not None:
            g.collections[0].get_cmap().set_bad(mask_color)

    g1 = None
    g2 = None

    ##### PLOT 1 #####
    if show_plot[0]:
        fig1, ax1 = plt.subplots(figsize=figsize)
        # Apply pow1 if necessary
        matrix_plot1 = np.power(contact_map_unbinned, pow1) if pow1 != 1 else contact_map_unbinned
        g1 = sns.heatmap(matrix_plot1,
                        cmap=cmap[0],
                        mask=mask,
                        alpha=1) 
        add_plot_labels(g1, bin_size=1, 
                        n_bins=contact_map_unbinned.shape[0], 
                        add_rect=False, 
                        xlabel="Residue Position", 
                        ylabel="Residue Position",
                        title=title)
        color_mask(g1, mask_color=mask_color)
    else:
        fig1, ax1 = None, None

    ##### PLOT 2 #####
    if show_plot[1]:
        fig2, ax2 = plt.subplots(figsize=figsize)
        if masking_percentile2 is not None:
            mask_2 = mask_by_percentile(contact_map_unbinned, percentile=masking_percentile2, invert=invert_mask)
        else:
            mask_2 = None
        # Apply pow2 if necessary
        matrix_plot2 = np.power(contact_map_unbinned, pow2) if pow2 != 1 else contact_map_unbinned
        g2 = ax2.imshow(
            np.ma.masked_array(matrix_plot2, mask=mask_2) if mask_2 is not None else matrix_plot2,
            cmap=cmap[1],
            alpha=heatmap_alpha,
            interpolation='nearest'
        )
        # Add colorbar for the imshow plot
        cbar = plt.colorbar(g2, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

        ax2.set_xlabel(x_label)
        ax2.set_ylabel(y_label)
        ax2.set_title(title) 
        color_mask(g2, mask_color=mask_color)
        # Remove the border (spines) around the plot
        for spine in ax2.spines.values():
            spine.set_visible(False)
        
        # Add rectangles around high-confidence sensitization variants
        if add_rect:    
            add_rectangles(
                xy_pairs=outlier_sig,
                width=height_width,
                height=height_width,
                xy_pairs_are_idx=True,
                fill=False,
                alpha=rectangles_alpha,
                x_col="wt_position",
                y_col="clinical_position",
                color_col="outlier_type",
                cmap=rectangles_cmap,
                linewidth=rectangles_linewidth
            )
    else:
        fig2, ax2 = None, None

    if save_fig:
        plt.savefig(save_path, dpi=dpi)
    if any(show_plot):
        plt.show()

    return {'fig': {'subplot1': fig1, 'subplot2': fig2}, 
            'axes': {'subplot1': ax1, 'subplot2': ax2}, 
            'data': {'contact_map_unbinned': contact_map_unbinned, 
                     'outlier_sig': outlier_sig}}



def plot_clinsig_interaction_strength(
    ridge_df,
    annot_df=None,
    site_col="mutant",
    agg_func="mean",
    x="clinsig",    
    y="interaction_strength",
    palette=utils.get_clinsig_palette(),
    title="Marginal Effects per Clinical Variant",
    xlabel="Clinical Significance",
    ylabel="Marginal Effect Size",
    figsize=(5, 5),
    text_format="star",
    show_test_name=False,
    loc='inside',
    verbose=0,
    pvalue_format_string=" ({:.2g})",
    test='Mann-Whitney',
    annotator_kwargs={},
    xtick_rotation=0,
    bracket_linewidth=0.75,
    break_xtick_labels=False,
    hide_xtick_labels=False,
    show_legend=False,
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statannotations.Annotator import Annotator
    from itertools import combinations 
    import pandas as pd
    import numpy as np

    ridge_df = ridge_df.copy()
    annot_df = annot_df.copy()

    # Prepare bar_df
    if x in ridge_df.columns:
        bar_df = ridge_df.groupby(["clinical_variant", x])[y].agg(agg_func).reset_index()
    else:
        if annot_df is None:
            raise ValueError("annot_df is required when x is not in ridge_df.columns")
        annot_df[site_col] = annot_df[site_col].apply(utils.standardize_variant)
        ridge_df[y] = ridge_df[y].apply(utils.standardize_variant)
        bar_df = ridge_df.groupby("clinical_variant")[y].agg(agg_func).reset_index().merge(
            annot_df[[site_col, x]].drop_duplicates().rename(columns={site_col: "clinical_variant"})
        )

    # Standardize clinsig labels: replace underscores with spaces or newlines, expand "path" and "likely_path"
    # Use spaces instead of newlines if xtick_rotation is set to prevent line breaks
    # Unless break_xtick_labels is True, which allows breaks even when rotated
    separator = "\n" if (xtick_rotation == 0 or break_xtick_labels) else " "
    def clean_clinsig(clinsig):
        clinsig = clinsig.replace("_", separator)
        if clinsig == "path":
            return "pathogenic"
        elif clinsig == f"likely{separator}path":
            return f"likely{separator}pathogenic"
        return clinsig

    bar_df[x] = bar_df[x].astype(str).apply(clean_clinsig)

    # Get the canonical clinsig order and filter to those present in the data
    canonical_order = utils.get_clinsig_order()
    # Apply the same cleaning as above to the canonical order
    def clean_order_label(label):
        label = str(label).replace("_", separator)
        if label == "path":
            return "pathogenic"
        elif label == f"likely{separator}path":
            return f"likely{separator}pathogenic"
        return label
    cleaned_canonical_order = [clean_order_label(l) for l in canonical_order]
    # Only keep those present in the data, and ensure uniqueness
    present_clinsigs = list(bar_df[x].unique())
    clinsig_order = []
    seen = set()
    for l in cleaned_canonical_order:
        if l in present_clinsigs and l not in seen:
            clinsig_order.append(l)
            seen.add(l)

    # Sort bar_df by clinsig order
    bar_df[x] = pd.Categorical(bar_df[x], categories=clinsig_order, ordered=True)
    bar_df = bar_df.sort_values(x)

    # Remap palette keys to match cleaned clinsig labels
    palette_cleaned = {}
    for k, v in palette.items():
        k_clean = k.replace("_", separator)
        if k_clean == "path":
            k_clean = "pathogenic"
        elif k_clean == f"likely{separator}path":
            k_clean = f"likely{separator}pathogenic"
        palette_cleaned[k_clean] = v

    plt.figure(figsize=figsize)

    # Draw the boxplot, using the correct clinsig order
    ax = sns.boxplot(
        data=bar_df,
        x=x,
        y=y,
        hue=x,
        palette=palette_cleaned,
        showfliers=False,
        order=clinsig_order,
        hue_order=clinsig_order,
        linewidth=0.75  # Set linewidth for box borders
    )
    
    # Force boxplot border color to black with alpha=1
    # Use matplotlib's black color explicitly (RGB: 0, 0, 0) as RGBA tuple
    black_color = (0.0, 0.0, 0.0, 1.0)  # RGBA tuple for fully opaque black
    
    # Set edgecolor on all boxplot elements (artists are the boxes)
    for patch in ax.artists:
        patch.set_edgecolor(black_color)
        patch.set_linewidth(0.75)  # Increased linewidth for better visibility
        patch.set_alpha(1.0)  # Ensure patch itself is fully opaque
    
    # Also check patches (some seaborn versions use patches instead of artists)
    for patch in ax.patches:
        patch.set_edgecolor(black_color)
        patch.set_linewidth(0.75)  # Increased linewidth for better visibility
        patch.set_alpha(1.0)  # Ensure patch itself is fully opaque
    
    # Set all boxplot line elements (median, whiskers, caps) to black
    # At this point, all lines in ax.lines are boxplot elements (annotations are added later)
    for line in ax.lines:
        line.set_color('black')
        line.set_linewidth(0.75)
        line.set_alpha(1.0)

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Compute mean interaction_strength for each clinsig group
    means = bar_df.groupby(x, observed=True)[y].mean()
    # Sort clinsig groups by mean interaction_strength
    sorted_clinsig = means.sort_values().index.tolist()

    # Generate all pairwise combinations, sorted by distance between means (descending)
    pairwise = list(combinations(sorted_clinsig, 2))
    pairwise_sorted = sorted(pairwise, key=lambda pair: abs(means[pair[0]] - means[pair[1]]), reverse=True)

    # Add statistical annotations
    annotator = Annotator(
        ax,
        pairs=pairwise_sorted,
        data=bar_df,
        x=x,
        y=y,
        order=clinsig_order
    )
    # Set line_width for annotation brackets
    # Only set if not already specified in annotator_kwargs
    if 'line_width' not in annotator_kwargs:
        annotator_kwargs['line_width'] = bracket_linewidth
    
    annotator.configure(
        test=test,
        text_format=text_format,
        loc=loc,
        verbose=verbose,
        show_test_name=show_test_name,
        pvalue_format_string=pvalue_format_string,
        **annotator_kwargs
    )
    annotator.apply_and_annotate()
    
    # Also set linewidth of annotation bracket lines (fallback if line_width param doesn't work)
    # Only modify lines that are likely annotation brackets (horizontal lines at higher y positions)
    y_data_range = bar_df[y].max() - bar_df[y].min()
    y_max = bar_df[y].max()
    for line in ax.lines:
        if hasattr(line, 'get_linewidth') and hasattr(line, 'get_ydata'):
            current_lw = line.get_linewidth()
            ydata = line.get_ydata()
            # Check if this line is likely an annotation bracket (horizontal line above the data)
            if len(ydata) > 0 and current_lw > 0:
                y_mean = np.mean(ydata)
                y_std = np.std(ydata) if len(ydata) > 1 else 0
                # If line is significantly above the data range and relatively horizontal, it's likely an annotation bracket
                # Check: mean y is above data, and line is mostly horizontal (low std in y)
                if y_mean > y_max and y_std < 0.05 * y_data_range:
                    # Set to bracket_linewidth (if current is different, it means line_width param didn't work)
                    if abs(current_lw - bracket_linewidth) > 0.01:
                        line.set_linewidth(bracket_linewidth)

    # Rotate xtick labels if requested
    if hide_xtick_labels:
        ax.set_xticklabels([])
    elif xtick_rotation != 0:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation, ha='right')

    # Remove the top and right spines (lines) from the plot margin
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Handle legend
    if show_legend:
        # When hue=x, seaborn doesn't create a legend by default
        # We need to create it manually from the boxplot patches
        # Get unique categories in order
        unique_categories = clinsig_order
        # Create legend handles from the palette
        from matplotlib.patches import Patch
        handles = []
        labels = []
        for cat in unique_categories:
            if cat in palette_cleaned:
                color = palette_cleaned[cat]
                # Create a patch (box) for the legend
                patch = Patch(facecolor=color, edgecolor='black', linewidth=0.75)
                handles.append(patch)
                labels.append(cat)
        # Create legend on the right side, outside the plot
        if handles:
            ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    else:
        # Remove legend if it exists
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    # Apply tight layout after all modifications
    # If legend is shown, leave space on the right for it
    if show_legend:
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.tight_layout()

    return {'fig': ax.figure, 'ax': ax, 'data': bar_df}



def add_extra_row_col(df, 
                      fill_value=np.nan, 
                      annot_type="str", 
                      value_fmt=None,
                      extra_col_name = ". . .",
                      extra_row_index = ". . .",
                      ):
    """
    Add an extra column and row to the right and bottom of a DataFrame, filled with fill_value.
    Returns the new DataFrame and an annotation DataFrame with "..." in the extra row/col.
    annot_type: "int", "float", or "str" -- controls how to format the annotation for the main part.
    value_fmt: function or None -- if not None, used to format the main part of the annotation.
    """
    df_with_extra = df.copy()
    df_with_extra[extra_col_name] = fill_value
    df_with_extra.loc[extra_row_index] = fill_value

    # Prepare annotation DataFrame
    annot = df_with_extra.astype(str)
    if annot_type == "int":
        if value_fmt is not None:
            annot.iloc[:-1, :-1] = df_with_extra.iloc[:-1, :-1].applymap(value_fmt)
        else:
            annot.iloc[:-1, :-1] = df_with_extra.iloc[:-1, :-1].astype(int).astype(str)
    elif annot_type == "float":
        if value_fmt is not None:
            annot.iloc[:-1, :-1] = df_with_extra.iloc[:-1, :-1].applymap(value_fmt)
        else:
            annot.iloc[:-1, :-1] = df_with_extra.iloc[:-1, :-1].astype(float).round(1).astype(str)
    else:
        annot.iloc[:-1, :-1] = df_with_extra.iloc[:-1, :-1].astype(str)
    # Fill last row and last col with "..."
    annot.iloc[-1, :] = "..."
    annot.iloc[:-1, -1] = "..."
    annot.iloc[-1, -1] = "..."
    # Set new index and columns
    annot.index = list(df.index) + ["..."]
    annot.columns = list(df.columns) + ["..."]
    return df_with_extra, annot

def plot_variant_sensitization_schematic(
    wtvariants_to_vep_linear_model_out,
    
    figsize=(20, 4),

    n_haplotypes=10,
    n_wt_variants=5,
    n_clinical_variants=5,
    include_any_1_col=True, 
    include_any_1_row=False,
    
    subplot_space_x=(0.06, 0.06),
    big_arrow_width=0.2,
    big_arrow_height=0.2,

    linewidths=0.5,
    linecolor='grey',

    replace_haplotype_prefix=False,
    include_haplotype_suffix=True,
    random_seed=0,
    noise_scale=0.10,
    coef_matrix_scale=20,
    coef_matrix_min=None,

    plot1_title="Haplotype x WT Variant Matrix",
    plot2_title="Haplotype x Clinical Variant VEP Matrix",
    plot3_title="Variant Sensitization Map",
    
    plot1_kwargs={},
    plot2_kwargs={},
    plot3_kwargs={},

    plot1_cmap="binary",
    plot2_cmap="coolwarm_r",
    plot3_cmap="coolwarm_r",
    
    add_grey_box=True,
    grey_box_color='lightgrey',
    grey_box_alpha=0.3,
    grey_box_left_extension=0.125,
    grey_box_top_extension=0.13,
    grey_box_right_extension=0.01,
    grey_box_bottom_extension=0.005,
    big_arrow_horizontal_offset=0.005,
    grey_box_outline_kwargs={'edgecolor': 'black', 'linewidth': 3, 'linestyle': ':'},
    title_pad=20,
    xlabel_pad=10,
    rotate_x_labels=False,
    title_y_position=None,
    heatmap_aspect='auto',
    facecolor='none',
    
    # New arguments
    show_wt_labels_plot3=True,
    show_text_labels=(True, True, True),
    show_train_predictor_arrow=True,
    show_extract_coefficients_arrow=True,
    xlabel=[None, None, None],
    as_formula=False,
    plot_order=[0, 1, 2],
    xlabel_fontsize='medium',
    ticklabel_fontsize=None
):
    """
    Plots the variant sensitization schematic as three heatmaps with arrows.
    Returns the matplotlib Figure and Axes.
    
    Parameters
    ----------
    wtvariants_to_vep_linear_model_out : dict
        Dictionary containing the linear model output data.
    figsize : tuple, optional
        Figure size as (width, height), default (20, 4).
    n_haplotypes : int, optional
        Number of haplotypes to display, default 10.
    n_wt_variants : int, optional
        Number of WT variants to display, default 5.
    n_clinical_variants : int, optional
        Number of clinical variants to display, default 5.
    subplot_space_x : tuple of float, optional
        Spacing between subplots 0-1 (first value) and 1-2 (second value), default (0.06, 0.06).
    big_arrow_width : float, optional
        Width scaling for the big arrow, default 0.2.
    big_arrow_height : float, optional
        Height scaling for the big arrow, default 0.2.
    linewidths : float, optional
        Line width for heatmap grid lines, default 0.5.
    linecolor : str, optional
        Color for heatmap grid lines, default 'grey'.
    replace_haplotype_prefix : bool, optional
        Whether to replace haplotype prefixes, default False.
    random_seed : int, optional
        Random seed for noise generation, default 0.
    noise_scale : float, optional
        Scale of noise to add to VEP matrix, default 0.10.
    coef_matrix_scale : float, optional
        Scaling factor for coefficient matrix, default 20.
    coef_matrix_min : float, optional
        Minimum threshold for coefficient values, default None.
    plot1_title : str, optional
        Title for the first plot, default "Haplotype x WT Variant Matrix".
    plot2_title : str, optional
        Title for the second plot, default "Haplotype x Clinical Variant VEP Matrix".
    plot3_title : str, optional
        Title for the third plot, default "WT x Clinical Variant Interaction Scores".
    plot1_kwargs : dict, optional
        Additional keyword arguments for the first heatmap.
    plot2_kwargs : dict, optional
        Additional keyword arguments for the second heatmap.
    plot3_kwargs : dict, optional
        Additional keyword arguments for the third heatmap.
    plot1_cmap : str, optional
        Colormap for the first plot, default "binary".
    plot2_cmap : str, optional
        Colormap for the second plot, default "coolwarm_r".
    plot3_cmap : str, optional
        Colormap for the third plot, default "coolwarm_r".
    add_grey_box : bool, optional
        Whether to add a grey box around the first two plots, default True.
    grey_box_color : str, optional
        Color of the grey box, default 'lightgrey'.
    grey_box_alpha : float, optional
        Transparency of the grey box, default 0.3.
    grey_box_left_extension : float, optional
        How far left to extend the grey box (in figure coordinates), default 0.125.
    grey_box_top_extension : float, optional
        How far up to extend the grey box (in figure coordinates), default 0.1.
    grey_box_right_extension : float, optional
        How far right to extend the grey box (in figure coordinates), default 0.01.
    grey_box_bottom_extension : float, optional
        How far down to extend the grey box (in figure coordinates), default 0.005.
    big_arrow_horizontal_offset : float, optional
        Horizontal offset for the "Extract Coefficients" arrow and label (in figure coordinates), default 0.005.
    grey_box_outline_kwargs : dict or None, optional
        Keyword arguments for the grey box outline/edge. If None, no outline is drawn. 
        Common options include: {'edgecolor': 'black', 'linewidth': 2, 'linestyle': '--'}, default {'edgecolor': 'black', 'linewidth': 3, 'linestyle': ':'}.
    title_pad : float, optional
        Vertical padding/offset for all plot titles, default 20. 
    rotate_x_labels : bool, optional
        Whether to rotate x-axis tick labels by 45 degrees and left-justify them, default False.
    title_y_position : float or None, optional
        Y-position for all plot titles in figure coordinates (0-1). If None, uses default title_pad.
        Use this to ensure all titles align horizontally, default None.
    heatmap_aspect : str, float, or tuple, optional
        Aspect ratio for the heatmaps. If a single value (str or float), applies to all 3 heatmaps.
        If a tuple of exactly 3 values, applies each value to its respective heatmap.
        Valid values: 'auto', 'equal', or numeric values. Default 'auto'.
        Examples:
        - heatmap_aspect='equal'  # Square cells for all heatmaps
        - heatmap_aspect=2.0      # Width 2x height for all heatmaps  
        - heatmap_aspect=(1.0, 2.0, 0.5)  # Different ratios for each heatmap
    facecolor : str, optional
        Face color for the figure, default 'none'.
    show_wt_labels_plot3 : bool, optional
        Whether to show haplotype row labels in the third (rightmost) heatmap, default True.
    show_text_labels : tuple of bool, optional
        Tuple of three boolean values to control text label display in each heatmap.
        Format: (plot1_show_labels, plot2_show_labels, plot3_show_labels).
        Default (True, True, True) shows labels in all three heatmaps.
    show_train_predictor_arrow : bool, optional
        Whether to show the "Train Predictor" arrows and label between the first and second plots, default True.
    show_extract_coefficients_arrow : bool, optional
        Whether to show the "Extract Coefficients" arrow and label between the second and third plots, default True.
    xlabel_fontsize : str, float, or None, optional
        Font size for x-axis labels. Can be a string (e.g., 'small', 'medium', 'large') or a numeric value.
        Default 'medium'. If None, uses matplotlib's default.
    ticklabel_fontsize : str, float, or None, optional
        Font size for both x and y-axis tick labels. Can be a string (e.g., 'small', 'medium', 'large') or a numeric value.
        Default None, which means it will use 'small' for both axes (matching the default behavior).
        If None, uses 'small' for both axes.
    
    Returns
    -------
    dict
        Dictionary containing the figure, axes, and data.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import gridspec
    from matplotlib.patches import FancyArrowPatch, Rectangle
    import pandas as pd

    fig = plt.figure(figsize=figsize, facecolor='none')
    
    # Calculate scaling factor based on figure size relative to default (20, 4)
    # Use minimum dimension to ensure proper scaling for both small and large figures
    default_figsize = (20, 4)
    scale_factor = min(figsize[0] / default_figsize[0], figsize[1] / default_figsize[1])
    
    # Scale padding values (originally in points, now scale with figure size)
    title_pad_scaled = title_pad * scale_factor
    xlabel_pad_scaled = xlabel_pad * scale_factor
    
    # Scale arrow sizes
    arrow_linewidth_scaled = 1.5 * scale_factor
    arrow_shrink_scaled = 6 * scale_factor
    arrow_mutation_scale_scaled = 15 * scale_factor
    
    # Scale grey box linewidth if provided (create a copy to avoid modifying original)
    grey_box_outline_kwargs_scaled = None
    if grey_box_outline_kwargs is not None:
        grey_box_outline_kwargs_scaled = grey_box_outline_kwargs.copy()
        if 'linewidth' in grey_box_outline_kwargs_scaled:
            grey_box_outline_kwargs_scaled['linewidth'] = grey_box_outline_kwargs_scaled['linewidth'] * scale_factor
    
    # Validate subplot_space_x
    if not isinstance(subplot_space_x, (tuple, list)) or len(subplot_space_x) != 2:
        raise ValueError("subplot_space_x must be a tuple or list of 2 values")
    
    # Use average spacing for gridspec (will adjust positions manually)
    avg_space = (subplot_space_x[0] + subplot_space_x[1]) / 2
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=avg_space)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    
    # If as_formula is True, automatically set plot_order to [0, 2, 1] (formula layout)
    if as_formula:
        plot_order = [0, 2, 1]
    
    # Use plot_order to reorder plots: [0, 1, 2] means original order, [0, 2, 1] means swap middle and right
    # Validate plot_order
    if sorted(plot_order) != [0, 1, 2]:
        raise ValueError("plot_order must contain exactly [0, 1, 2] in any order")
    
    # Reorder axes, titles, and aspect values based on plot_order
    axes_reordered = [axes[i] for i in plot_order]
    all_titles = [plot1_title, plot2_title, plot3_title]
    titles_reordered = [all_titles[i] for i in plot_order]
    
    # Reorder aspect values
    if isinstance(heatmap_aspect, (list, tuple)):
        if len(heatmap_aspect) != 3:
            raise ValueError("heatmap_aspect tuple must contain exactly 3 values")
        aspect_values = [heatmap_aspect[i] for i in plot_order]
    else:
        # Single value - apply to all 3 heatmaps
        aspect_values = [heatmap_aspect, heatmap_aspect, heatmap_aspect]

    # First heatmap: Binarized WT Variant Matrix
    Xwt = wtvariants_to_vep_linear_model_out['X_wt_clean'].copy()
   
    Xwt, row_range, col_range = mc.find_dense_submatrix(Xwt, 
                                                            window_height=n_haplotypes, 
                                                            window_width=n_wt_variants, 
                                                            include_any_1_col=include_any_1_col, 
                                                            include_any_1_row=include_any_1_row,
                                                            plot=False)
    
    # Get the haplotypes to plot
    haplotypes_to_plot = Xwt.index.tolist()
        

    
    if replace_haplotype_prefix:
        Xwt.index = [f"Hap{i+1}"+":"+x.split(":")[-1] for i, x in enumerate(Xwt.index)]
    if not include_haplotype_suffix:
        Xwt.index = [x.split(":")[0] for x in Xwt.index]

    Xwt_with_extra, annot = add_extra_row_col(Xwt, fill_value=np.nan, annot_type="int")

    # Use show_text_labels parameter to control annotation display
    annot_to_use = annot if show_text_labels[0] else False

    # Determine which axis to use for each plot based on plot_order
    # plot_order[i] tells us which plot (0, 1, or 2) goes to position i in the figure
    # So plot1 (index 0) goes to axes[plot_order.index(0)]
    plot1_position = plot_order.index(0)
    ax_plot1 = axes[plot1_position]
    sns.heatmap(
        Xwt_with_extra,
        annot=annot_to_use,
        fmt="",
        cbar=False,
        cmap=plot1_cmap,
        ax=ax_plot1,
        linewidths=linewidths,
        linecolor=linecolor,
        **plot1_kwargs
    )
    # Set aspect ratio for first heatmap
    ax_plot1.set_aspect(aspect_values[0]) 
    ax_plot1.set_ylabel("Haplotype", fontsize='medium', rotation=90)
    ax_plot1.set_xlabel("WT Variant", fontsize=xlabel_fontsize)
    # Title will be set later to ensure consistent height
    ax_plot1.xaxis.set_label_position('top')
    ax_plot1.xaxis.tick_top()
    
    # Rotate x-axis tick labels if requested
    if rotate_x_labels:
        ax_plot1.tick_params(axis='x', labelrotation=45)
        # Get current tick labels and set horizontal alignment to left
        labels = ax_plot1.get_xticklabels()
        ax_plot1.set_xticklabels(labels, ha='left')

    # Second heatmap: VEP Matrix
    interaction_df = wtvariants_to_vep_linear_model_out['interaction_df']
    Xvep = wtvariants_to_vep_linear_model_out['y_vep_clean'].copy()
    # Sort interaction_df by 'clinical_position'
    interaction_df_sorted = interaction_df.sort_values('clinical_position')
    # Take the top N clinical variants from the 'clinical_variant' column
    selected_clinical_variants = interaction_df_sorted['site'].unique().tolist()[:n_clinical_variants]
    Xvep = Xvep.loc[haplotypes_to_plot].loc[:, selected_clinical_variants].round(1)
    Xvep.columns = Xvep.columns.str.split(":").str[1]
    np.random.seed(random_seed)
    noise = np.random.normal(loc=0, scale=noise_scale, size=Xvep.shape)
    Xvep_noisy = Xvep + noise
    Xvep_noisy = Xvep_noisy.round(1)
    Xvep = Xvep_noisy

    Xvep_with_extra, annot_vep = add_extra_row_col(Xvep, fill_value=np.nan, annot_type="float")

    # Use show_text_labels parameter to control annotation display
    annot_vep_to_use = annot_vep if show_text_labels[1] else False

    # Determine which axis to use for plot2 based on plot_order
    # plot2 is originally at index 1, find where it is in plot_order
    plot2_position = plot_order.index(1)
    ax_plot2 = axes[plot2_position]
    sns.heatmap(
        Xvep_with_extra,
        annot=annot_vep_to_use,
        fmt="",
        cbar=False,
        ax=ax_plot2,
        linewidths=linewidths,
        linecolor=linecolor,
        cmap=plot2_cmap, 
        **plot2_kwargs
    )
    # Set aspect ratio for second heatmap  
    ax_plot2.set_aspect(aspect_values[plot2_position]) 
    # Title will be set later to ensure consistent height
    ax_plot2.xaxis.set_label_position('top')
    ax_plot2.set_ylabel("Haplotype", fontsize='medium', rotation=90)
    ax_plot2.set_xlabel("Clinical Variant", fontsize=xlabel_fontsize)
    ax_plot2.xaxis.tick_top()
    
    # Rotate x-axis tick labels if requested
    if rotate_x_labels:
        ax_plot2.tick_params(axis='x', labelrotation=45)
        # Get current tick labels and set horizontal alignment to left
        labels = ax_plot2.get_xticklabels()
        ax_plot2.set_xticklabels(labels, ha='left')
    
    # Only hide yticklabels for plot2 if it's in the middle position (original behavior)
    if plot2_position == 1:  # plot2 is in middle position
        ax_plot2.set_yticklabels([])

    # Third heatmap: WT x Clinical Variant Interaction Score Matrix
    coef_matrix_abs = wtvariants_to_vep_linear_model_out['coef_matrix_signed'].copy()
    coef_matrix_abs = coef_matrix_abs.iloc[:n_wt_variants, :].loc[:, selected_clinical_variants]
    coef_matrix_abs.columns = coef_matrix_abs.columns.str.split(":").str[-1]
    coef_matrix_abs.index = coef_matrix_abs.index.str.split(":").str[-1]
    coef_matrix_abs *= coef_matrix_scale
    coef_matrix_abs = coef_matrix_abs.clip(upper=1)

    if "vmin" not in plot3_kwargs and "vmax" not in plot3_kwargs:
        plot3_kwargs["vmin"] = -coef_matrix_abs.abs().max().max()
        plot3_kwargs["vmax"] = coef_matrix_abs.abs().max().max()
 

    # Set values below coef_min to 0
    if coef_matrix_min is not None:
        coef_matrix_abs[coef_matrix_abs.abs() < coef_matrix_min] = 0

    
    coef_matrix_abs_with_extra, annot_coef = add_extra_row_col(coef_matrix_abs, 
                                                               fill_value=np.nan, 
                                                               annot_type="float")

    # Convert annotation DataFrame to numeric, coercing errors to NaN
    if isinstance(annot_coef, pd.DataFrame):
        annot_coef_numeric = annot_coef.apply(pd.to_numeric, errors='coerce')
    else:
        annot_coef_numeric = pd.to_numeric(annot_coef, errors='coerce')
 
    # Use show_text_labels parameter to control annotation display
    annot_coef_to_use = annot_coef_numeric if show_text_labels[2] else False

    # Determine which axis to use for plot3 based on plot_order
    # plot3 is originally at index 2, find where it is in plot_order
    plot3_position = plot_order.index(2)
    ax_plot3 = axes[plot3_position]
    sns.heatmap(
        coef_matrix_abs_with_extra,
        annot=annot_coef_to_use, 
        cbar=False,
        ax=ax_plot3,
        linewidths=linewidths,
        linecolor=linecolor,
        cmap=plot3_cmap, 
        **plot3_kwargs
    )
    # Set aspect ratio for third heatmap
    ax_plot3.set_aspect(aspect_values[plot3_position])
    ax_plot3.set_ylabel("WT Variant", fontsize='medium', rotation=90)
    ax_plot3.set_xlabel("Clinical Variant", fontsize=xlabel_fontsize)
    # Title will be set later to ensure consistent height
    ax_plot3.xaxis.set_label_position('top')
    ax_plot3.xaxis.tick_top()
    
    # Rotate x-axis tick labels if requested
    if rotate_x_labels:
        ax_plot3.tick_params(axis='x', labelrotation=45)
        # Get current tick labels and set horizontal alignment to left
        labels = ax_plot3.get_xticklabels()
        ax_plot3.set_xticklabels(labels, ha='left')

    # Control haplotype labels in the third plot
    if not show_wt_labels_plot3:
        ax_plot3.set_yticklabels([])

    # Add grey box around first two plots if requested (skip if as_formula, we use dashed rectangles instead)
    if add_grey_box and not as_formula:
        # Get the positions of the first two axes in figure coordinates
        pos1 = axes[0].get_position()
        pos2 = axes[1].get_position()
        
        # Calculate the bounding box that encompasses both plots
        # Use the extension parameters to control box size on all sides
        x_min = min(pos1.x0, pos2.x0) - grey_box_left_extension
        x_max = max(pos1.x1, pos2.x1) + grey_box_right_extension
        y_min = min(pos1.y0, pos2.y0) - grey_box_bottom_extension
        y_max = max(pos1.y1, pos2.y1) + grey_box_top_extension
        
        # Add some padding around the plots
        padding = 0.02
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        
        # Create the grey box and add it to the figure
        grey_box_kwargs = {
            'facecolor': grey_box_color,
            'alpha': grey_box_alpha,
            'zorder': -10,  # Place far behind the plots
            'transform': fig.transFigure
        }
        
        # Add outline kwargs if provided
        if grey_box_outline_kwargs_scaled is not None:
            grey_box_kwargs.update(grey_box_outline_kwargs_scaled)
        else:
            grey_box_kwargs['edgecolor'] = 'none'
        
        grey_box = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            **grey_box_kwargs
        )
        fig.patches.append(grey_box)

    # Add arrows between the first and second plots, aligned with each row
    # Note: add_extra_row_col adds an extra spacer row, so total rows = n_haplotypes + 1
    # Skip old arrows if as_formula is True (we'll use new formula arrows instead)
    if show_train_predictor_arrow and not as_formula:
        fig.canvas.draw()
        total_rows = len(haplotypes_to_plot) + 1
        for i in range(total_rows):
            y_frac = (i + 0.5) / total_rows
            x0_fig, y0_fig = axes[0].transAxes.transform((1.0, y_frac))
            x1_fig, y1_fig = axes[1].transAxes.transform((0.0, y_frac))
            inv = fig.transFigure.inverted()
            x0_fig_frac, y0_fig_frac = inv.transform((x0_fig, y0_fig))
            x1_fig_frac, y1_fig_frac = inv.transform((x1_fig, y1_fig))
            arrow = FancyArrowPatch(
                (x0_fig_frac, y0_fig_frac), (x1_fig_frac, y1_fig_frac),
                transform=fig.transFigure,
                arrowstyle="->",
                color="black",
                linewidth=arrow_linewidth_scaled,
                shrinkA=arrow_shrink_scaled,
                shrinkB=arrow_shrink_scaled,
                mutation_scale=arrow_mutation_scale_scaled,
                clip_on=False
            )
            fig.patches.append(arrow)
            if i == total_rows - 1:
                label_y_offset = 0.02
                fig.text(
                    (x0_fig_frac + x1_fig_frac) / 2,
                    y0_fig_frac + label_y_offset,
                    "Train\nPredictor",
                    ha="center",
                    va="bottom",
                    fontsize='large',
                    fontweight="bold"
                )

    # Add one big arrow between the middle plot and the rightmost plot
    # Skip old arrows if as_formula is True (we'll use new formula arrows instead)
    if show_extract_coefficients_arrow and not as_formula:
        y_center = 0.5
        x1_fig, y1_fig = axes[1].transAxes.transform((1.0, y_center))
        x2_fig, y2_fig = axes[2].transAxes.transform((0.0, y_center))
        inv = fig.transFigure.inverted()
        x1_fig_frac, y1_fig_frac = inv.transform((x1_fig, y1_fig))
        x2_fig_frac, y2_fig_frac = inv.transform((x2_fig, y2_fig))

        arrowstyle = (
            f"simple,"
            f"head_length={15*big_arrow_height*scale_factor},"
            f"head_width={15*big_arrow_width*scale_factor},"
            f"tail_width={5*big_arrow_width*scale_factor}"
        )
        linewidth = 2.5 * big_arrow_width * scale_factor
        mutation_scale = 30 * big_arrow_height * scale_factor
        shrinkA = 10 * big_arrow_width * scale_factor
        shrinkB = 10 * big_arrow_width * scale_factor

        # Apply horizontal offset to the arrow and label positions
        x1_fig_frac_offset = x1_fig_frac + big_arrow_horizontal_offset
        x2_fig_frac_offset = x2_fig_frac + big_arrow_horizontal_offset
        
        big_arrow = FancyArrowPatch(
            (x1_fig_frac_offset, y1_fig_frac), (x2_fig_frac_offset, y2_fig_frac),
            transform=fig.transFigure,
            arrowstyle=arrowstyle,
            color="black",
            linewidth=linewidth,
            mutation_scale=mutation_scale,
            shrinkA=shrinkA,
            shrinkB=shrinkB,
            clip_on=False,
            zorder=20
        )
        fig.patches.append(big_arrow)

        # Add a label above the big arrow
        label_x = (x1_fig_frac_offset + x2_fig_frac_offset) / 2
        label_y = max(y1_fig_frac, y2_fig_frac) + 0.2 * big_arrow_height
        fig.text(
            label_x,
            label_y,
            "Extract\nCoefficients",
            ha="center",
            va="bottom",
            fontsize='large',
            fontweight="bold",
            zorder=30
        )

    # Add formula-specific elements
    if as_formula:
        # Add dashed rectangles around each heatmap
        for ax in axes_reordered:
            pos = ax.get_position()
            # Add padding around the heatmap
            padding = 0.01
            dashed_rect = Rectangle(
                (pos.x0 - padding, pos.y0 - padding),
                pos.width + 2 * padding,
                pos.height + 2 * padding,
                transform=fig.transFigure,
                fill=False,
                edgecolor='grey',
                linestyle='--',
                linewidth=1.5 * scale_factor,
                zorder=5
            )
            fig.patches.append(dashed_rect)
        
        # Add "Surrogate Model" box above the middle heatmap (plot3, which is axes_reordered[1])
        middle_ax = axes_reordered[1]  # This is plot3 (W x C) in formula layout
        middle_pos = middle_ax.get_position()
        
        # Calculate box position above the middle heatmap
        box_height = 0.08 * scale_factor
        box_width = middle_pos.width * 0.6
        box_x = middle_pos.x0 + (middle_pos.width - box_width) / 2
        box_y = middle_pos.y1 + 0.05 * scale_factor
        
        # Draw the Surrogate Model box
        surrogate_box = Rectangle(
            (box_x, box_y),
            box_width,
            box_height,
            transform=fig.transFigure,
            fill=True,
            facecolor='white',
            edgecolor='black',
            linewidth=1.5 * scale_factor,
            zorder=25
        )
        fig.patches.append(surrogate_box)
        
        # Add "Surrogate Model" text
        fig.text(
            box_x + box_width / 2,
            box_y + box_height / 2,
            "Surrogate Model",
            ha='center',
            va='center',
            fontsize='medium',
            fontweight='bold',
            transform=fig.transFigure,
            zorder=26
        )
        
        # Add arrows: x -> Surrogate Model -> y
        arrow_y = box_y + box_height / 2
        arrow_x_left = box_x - 0.15 * scale_factor
        arrow_x_right = box_x + box_width + 0.15 * scale_factor
        
        # Left arrow: x -> Surrogate Model
        left_arrow = FancyArrowPatch(
            (arrow_x_left, arrow_y),
            (box_x, arrow_y),
            transform=fig.transFigure,
            arrowstyle="->",
            color="black",
            linewidth=2 * scale_factor,
            mutation_scale=20 * scale_factor,
            shrinkA=5 * scale_factor,
            shrinkB=5 * scale_factor,
            clip_on=False,
            zorder=24
        )
        fig.patches.append(left_arrow)
        
        # Add "x" label
        fig.text(
            arrow_x_left - 0.05 * scale_factor,
            arrow_y,
            "x",
            ha='right',
            va='center',
            fontsize='large',
            fontweight='bold',
            transform=fig.transFigure,
            zorder=27
        )
        
        # Right arrow: Surrogate Model -> y
        right_arrow = FancyArrowPatch(
            (box_x + box_width, arrow_y),
            (arrow_x_right, arrow_y),
            transform=fig.transFigure,
            arrowstyle="->",
            color="black",
            linewidth=2 * scale_factor,
            mutation_scale=20 * scale_factor,
            shrinkA=5 * scale_factor,
            shrinkB=5 * scale_factor,
            clip_on=False,
            zorder=24
        )
        fig.patches.append(right_arrow)
        
        # Add "y" label
        fig.text(
            arrow_x_right + 0.05 * scale_factor,
            arrow_y,
            "y",
            ha='left',
            va='center',
            fontsize='large',
            fontweight='bold',
            transform=fig.transFigure,
            zorder=27
        )
        
        # Add dimension labels below each heatmap: "(H x W)", "(W x C)", "(H x C)"
        dimension_labels = ["(H x W)", "(W x C)", "(H x C)"]
        for i, (ax, dim_label) in enumerate(zip(axes_reordered, dimension_labels)):
            pos = ax.get_position()
            fig.text(
                pos.x0 + pos.width / 2,
                pos.y0 - 0.03 * scale_factor,
                dim_label,
                ha='center',
                va='top',
                fontsize='small',
                transform=fig.transFigure,
                zorder=10
            )
    
    # If title_y_position is set and > 1.0, adjust top margin to accommodate titles
    # Calculate rect for tight_layout to preserve space for titles
    if title_y_position is not None and title_y_position > 1.0:
        # Reserve extra space at the top for titles positioned above axes
        extra_top = (title_y_position - 1.0) * 0.15  # Convert to figure fraction
        top_margin = 1.0 - extra_top
        plt.tight_layout(rect=[0, 0, 1, top_margin])
    else:
        plt.tight_layout()
    
    # Adjust subplot spacing manually AFTER tight_layout to allow different spacing between 0-1 and 1-2
    # Get current positions after tight_layout
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    pos2 = axes[2].get_position()
    
    # Calculate desired spacing
    # Space between 0 and 1: subplot_space_x[0]
    # Space between 1 and 2: subplot_space_x[1]
    
    # Adjust position of axis 1 based on spacing from axis 0
    new_x1 = pos0.x1 + subplot_space_x[0]
    axes[1].set_position([new_x1, pos1.y0, pos1.width, pos1.height])
    
    # Update pos1 after adjustment
    pos1 = axes[1].get_position()
    
    # Adjust position of axis 2 based on spacing from axis 1
    new_x2 = pos1.x1 + subplot_space_x[1]
    axes[2].set_position([new_x2, pos2.y0, pos2.width, pos2.height])
    
    # Set all titles at the same height after layout is finalized
    # Find the maximum y-position of all axes to ensure titles align
    max_y_position = max(ax.get_position().y1 for ax in axes_reordered)
    
    if title_y_position is not None:
        # Use the specified y-position
        desired_y_fig = title_y_position
    else:
        # Calculate desired y-position based on the highest axis plus padding
        desired_y_fig = max_y_position + (title_pad_scaled / (fig.get_figheight() * 72))
    
    # Set all titles at the EXACT same y-position in figure coordinates
    # All titles should be at the same absolute y position
    # Use fig.text to place titles at exact same height, then set on axes for consistency
    for ax, title in zip(axes_reordered, titles_reordered):
        # Get axis position
        pos = ax.get_position()
        # Calculate title x position (center of axis)
        title_x = pos.x0 + pos.width / 2
        
        # Place title at exact same y position for all plots
        # Use fig.text to ensure EXACT same height for all titles
        fig.text(
            title_x,
            desired_y_fig,
            title,
            ha='center',
            va='bottom',
            fontsize='large',
            transform=fig.transFigure,
            zorder=100
        )
        
        # Set empty title on axis to avoid any default title positioning issues
        ax.set_title('', fontsize='large')
    
    # Set tick label sizes and rotate y-axis tick labels
    # All y-axis tick labels should be horizontal (0 degrees)
    for ax in axes_reordered:
        # Set tick label size for both axes
        if ticklabel_fontsize is not None:
            ax.tick_params(axis='both', labelsize=ticklabel_fontsize)
        else:
            # Default to small for both axes
            ax.tick_params(axis='both', labelsize='small')
        # Set y-axis tick label rotation to 0 (horizontal)
        ax.tick_params(axis='y', labelrotation=0)
        # Move x-tick labels closer to ticks by reducing pad
        ax.tick_params(axis='x', pad=0.5)
    
    # Pad x-tick labels to ensure all heatmaps have the same label height
    # This prevents misalignment of x-axis titles when labels have different lengths
    # Get all x-tick labels from all three axes
    all_xtick_labels = []
    for ax in axes_reordered:
        labels = ax.get_xticklabels()
        label_texts = [label.get_text() for label in labels]
        all_xtick_labels.append(label_texts)
    
    # Find the maximum width across all labels
    max_width = 0
    for label_list in all_xtick_labels:
        for label in label_list:
            if label:  # Skip empty labels
                max_width = max(max_width, len(label))
    
    # Pad labels to match maximum width (if max_width > 0)
    if max_width > 0:
        for i, ax in enumerate(axes_reordered):
            labels = ax.get_xticklabels()
            padded_labels = []
            for label in labels:
                text = label.get_text()
                if text:
                    # Pad with spaces to match max width, center the text
                    padding_needed = max_width - len(text)
                    left_pad = padding_needed // 2
                    right_pad = padding_needed - left_pad
                    padded_text = ' ' * left_pad + text + ' ' * right_pad
                    padded_labels.append(padded_text)
                else:
                    padded_labels.append(text)
            
            # Get current rotation and alignment settings
            rotation = 45 if rotate_x_labels else 0
            ha = 'left' if rotate_x_labels else 'center'
            
            # Re-set the tick labels with padding
            ax.set_xticklabels(padded_labels, rotation=rotation, ha=ha)
    
    # Apply pad to all x-tick labels to move them closer to ticks (after all label operations)
    for ax in axes_reordered:
        ax.tick_params(axis='x', pad=0.5)
    
    # Store original xlabel texts before clearing, then position them manually using fig.text for perfect alignment
    xlabel_texts = []
    for i, ax in enumerate(axes_reordered):
        # Get current xlabel text from axis if xlabel[i] is None
        if xlabel[i] is not None:
            xlabel_texts.append(xlabel[i])
        else:
            # Get the current xlabel from the axis
            current_xlabel = ax.get_xlabel()
            xlabel_texts.append(current_xlabel if current_xlabel else None)
        # Clear the xlabel on the axis - we'll position it manually later
        ax.set_xlabel('')
        ax.xaxis.set_label_position('top')
    
    # FINALLY: Shift plot2 and plot3 heatmaps down so their tops align with plot1
    # This must happen after all other positioning is complete
    # Use plot1 (index 0) as the reference since it's always present
    plot1_index_in_reordered = plot_order.index(0)
    ax_reference = axes_reordered[plot1_index_in_reordered]
    reference_top = ax_reference.get_position().y1
    
    # Shift plot2 (Haplotype x Clinical Variant VEP Matrix) down
    plot2_index_in_reordered = plot_order.index(1)
    ax_plot2 = axes_reordered[plot2_index_in_reordered]
    pos2 = ax_plot2.get_position()
    shift_amount_plot2 = pos2.y1 - reference_top
    if abs(shift_amount_plot2) > 1e-6:
        ax_plot2.set_position([
            pos2.x0,
            pos2.y0 - shift_amount_plot2,
            pos2.width,
            pos2.height
        ])
        # Verify and correct if needed
        new_pos2 = ax_plot2.get_position()
        if abs(new_pos2.y1 - reference_top) > 1e-6:
            final_shift = new_pos2.y1 - reference_top
            ax_plot2.set_position([
                new_pos2.x0,
                new_pos2.y0 - final_shift,
                new_pos2.width,
                new_pos2.height
            ])
    
    # Shift plot3 (WT-Clinical heatmap) down
    plot3_index_in_reordered = plot_order.index(2)
    ax_plot3 = axes_reordered[plot3_index_in_reordered]
    pos3 = ax_plot3.get_position()
    shift_amount_plot3 = pos3.y1 - reference_top
    if abs(shift_amount_plot3) > 1e-6:
        ax_plot3.set_position([
            pos3.x0,
            pos3.y0 - shift_amount_plot3,
            pos3.width,
            pos3.height
        ])
        # Verify and correct if needed
        new_pos3 = ax_plot3.get_position()
        if abs(new_pos3.y1 - reference_top) > 1e-6:
            final_shift = new_pos3.y1 - reference_top
            ax_plot3.set_position([
                new_pos3.x0,
                new_pos3.y0 - final_shift,
                new_pos3.width,
                new_pos3.height
            ])

    # Position all x-axis labels at the exact same y-coordinate using fig.text
    # This ensures perfect horizontal alignment regardless of tick label sizes
    # All axes should have the same top position after shifting, so use reference_top
    # First, force a draw to get accurate tick label positions
    fig.canvas.draw_idle()
    
    # Find the maximum top position of all x-tick labels across all axes
    max_tick_label_top = reference_top
    for ax in axes_reordered:
        # Get all x-tick labels
        tick_labels = ax.get_xticklabels()
        if tick_labels:
            # Get the bounding box of tick labels in figure coordinates
            for label in tick_labels:
                try:
                    bbox = label.get_window_extent(fig.canvas.get_renderer())
                    bbox_fig = bbox.transformed(fig.transFigure.inverted())
                    max_tick_label_top = max(max_tick_label_top, bbox_fig.y1)
                except:
                    pass
    
    # Position xlabels at a fixed offset above the maximum tick label top
    # Convert padding from points to figure coordinates (72 points per inch)
    # Add extra padding to ensure labels don't overlap with tick labels
    extra_padding = 0.02  # Additional padding in figure coordinates
    xlabel_y_position = max_tick_label_top + (xlabel_pad_scaled / (fig.get_figheight() * 72)) + extra_padding
    
    for i, ax in enumerate(axes_reordered):
        if xlabel_texts[i] is not None:
            # Get axis position in figure coordinates
            pos = ax.get_position()
            # Calculate xlabel x position (center of axis)
            xlabel_x = pos.x0 + pos.width / 2
            # Position xlabel at the exact same y-coordinate for all plots using fig.text
            # DO NOT set on axis to avoid duplicates - only use fig.text
            fig.text(xlabel_x, xlabel_y_position, xlabel_texts[i], 
                    fontsize=xlabel_fontsize, ha='center', va='bottom',
                    transform=fig.transFigure, zorder=100)

    # Return axes in visual order (reordered based on plot_order)
    return {'fig': fig, 'axes': axes_reordered, 'data': {'wtvariants_to_vep_linear_model_out': wtvariants_to_vep_linear_model_out,
                                                'Xwt_with_extra': Xwt_with_extra,
                                                'Xvep_with_extra': Xvep_with_extra,
                                                'coef_matrix_abs_with_extra': coef_matrix_abs_with_extra}}

def safe_int(val):
    try:
        if pd.isna(val):
            return None
        return int(val)
    except Exception:
        return None

def get_contact_score(x, contact_map):
    wt_pos = safe_int(x["wt_position"])
    clin_pos = safe_int(x["clinical_position"])
    if wt_pos is None or clin_pos is None:
        return np.nan
    return contact_map[wt_pos-1, clin_pos-1]



def plot_wt_clinical_interaction_vs_angstroms(
    ridge_df,
    vep_prot,
    N=5,
    figsize=(10, 5),
    x_var="Angstroms",
    y_var="interaction_strength_signed",
    hue_var="clinsig",
    size_var="interaction_strength",
    style_var="is_contact",
    palette=None,
    show=True,
    x_title="3D Distance (Ångstroms)",
    y_title=r"$E_{\text{WT-Clin}}$",
    title="WT-Clinical Variant Joint Effect Size vs. 3D Distance",
    legend_outside=False,
    label_fontsize=10,
    adjust_text_kwargs={},
    rasterize_scatter=True,
):
    """
    Plot WT-Clinical Variant Interaction Strength vs. 3D Distance.

    Parameters
    ----------
    ridge_df : pd.DataFrame
        DataFrame containing ridge regression interaction results.
    vep_prot : pd.DataFrame
        DataFrame containing clinical variant annotations (must have 'mutant' and 'clinsig').
    N : int
        Number of top interactions to label.
    figsize : tuple
        Figure size.
    x_var, y_var, hue_var, size_var, style_var : str
        Column names for plot axes and aesthetics.
    palette : dict or None
        Color palette for clinical significance.
    show : bool
        Whether to call plt.show().
    rasterize_scatter : bool
        If True, rasterize only the scatter points (not text labels or other elements).
        Useful for reducing file size when saving vector graphics with many points.
    """
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    import matplotlib as mpl

    # Prepare DataFrame
    ridge_df = ridge_df.copy()
    vep_prot = vep_prot.copy()
    
    ridge_df['protein'] = ridge_df['site'].str.split(":").str[0]
    ridge_df = utils.add_hgvsp_id(ridge_df)

    vep_prot_tmp = vep_prot[["mutant", "clinsig"]].drop_duplicates().copy()
    vep_prot_tmp["mutant"] = vep_prot_tmp["mutant"].apply(utils.standardize_variant)
    if "clinsig" not in ridge_df.columns:
        ridge_df = ridge_df.merge(
            vep_prot_tmp,
            left_on="clinical_variant",
            right_on="mutant",
            how="left"
        )
    # Convert to string first to handle NaN values, then do replacements
    ridge_df['clinsig'] = ridge_df['clinsig'].astype(str).replace(
        to_replace=r'path$', value='pathogenic', regex=True
    )
    ridge_df['clinsig'] = ridge_df['clinsig'].str.replace("_", " ", regex=False)
    # Replace 'nan' string back to actual NaN
    ridge_df['clinsig'] = ridge_df['clinsig'].replace('nan', np.nan)
    
    vep_prot['clinsig'] = vep_prot['clinsig'].astype(str).replace(
        to_replace=r'path$', value='pathogenic', regex=True
    )
    vep_prot['clinsig'] = vep_prot['clinsig'].str.replace("_", " ", regex=False)
    # Replace 'nan' string back to actual NaN
    vep_prot['clinsig'] = vep_prot['clinsig'].replace('nan', np.nan)


    top_interactions = ridge_df.reindex(
        ridge_df[y_var].abs().sort_values(ascending=False).index
    ).head(N)

    ridge_df = utils.sort_by_clinsig(ridge_df, clinsig_col="clinsig")
    top_interactions = utils.sort_by_clinsig(top_interactions, clinsig_col="clinsig")

    ridge_df["Angstroms_inverted"] = ridge_df["Angstroms"].max() - ridge_df["Angstroms"]

    if palette is None:
        palette = utils.get_clinsig_palette()

    plt.figure(figsize=figsize)

    ax = sns.scatterplot(
        data=ridge_df,
        x=x_var,
        y=y_var,
        hue=hue_var,
        size=size_var,
        style=style_var,
        sizes=(0.001, 100),
        alpha=0.75,
        palette=palette,
        edgecolor="grey",
        linewidth=0.01,
    )

    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)

    # Rasterize scatter points if requested (but not text labels or other elements)
    if rasterize_scatter:
        from matplotlib.collections import PathCollection
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_rasterized(True)

    # Option to put legend inside or outside the plot
    if legend_outside:
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    else:
        legend = ax.legend(loc='upper right')
    # Always remove the border around the legend
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_edgecolor('none')

    for leg in legend.get_texts():
        if "contact_score" in leg.get_text().lower() or "size" in leg.get_text().lower():
            leg.set_text("Contact Score")
        if "interaction_strength" in leg.get_text().lower():
            leg.set_text(r"$|E_{\text{WT-Clin}}|$")
        if "is_contact" in leg.get_text().lower():
            leg.set_text("Contact")
        if "Angstroms_inverted" in leg.get_text().lower():
            leg.set_text(r"Ångstroms")
        if "clinsig" in leg.get_text().lower():
            leg.set_text("Clinical Significance")

    # Prepare label texts for top N points
    texts = []
    for _, row in top_interactions.iterrows():
        label = (
            f"{row['wt_variant']} | {row['clinical_variant_fmt']}"
            # + f"Interaction: {row['interaction_strength_signed']:.2f}\n"
            # + f"$\\mathrm{{\\AA}}$: {row['Angstroms']:.2f}"
        )
        # Set horizontalalignment to 'left' and anchor to 'left' for left-side anchoring
        texts.append(
            plt.text(
                row[x_var],
                row[y_var],
                label,
                fontsize=label_fontsize,
                color='black',  
                # bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'),
                # ha='left',  # horizontal alignment
                # va='center',  # vertical alignment
            )
        )
    # Remove top and right spines (margin lines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust text to avoid overlap, anchor arrows to left side of text

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle='->', color='black', lw=0.5, alpha=0.5),
        ax=ax, 
        # expand_text=(1.05, 1.2),  # slightly expand text box for better arrow placement
        # only_move={'points':'y', 'text':'xy'},  # allow text to move in x and y
        #  only_move='x+',
        explode_radius=0,
        va='top',
        ha='left', 
        **adjust_text_kwargs
    )

    # plt.tight_layout()  # To make room for the legend outside the plot
    if show:
        plt.show()
    return {'fig': ax.figure, 'ax': ax, 'data': top_interactions}

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

def plot_vep_scatter_kde(
    vep_prot, 
    ridge_df=None,
    position_axis="x",   # "y" for Protein_position on y-axis, "x" for Protein_position on x-axis
    vep_col="VEP", 
    position_col="Protein_position", 
    clinsig_col="clinsig",
    title=None,
    palette = None,
    figsize=(5,5),
    ridge_kwargs=None,  # NOTE: changed default to None, see below
    ridge_y_label=r"$|\beta_{Clin}|$",
    wt_ridge_y_label = r"$|\beta_{WT}|$",
    grid_row_ratios=(3, 1, 1, 1),  # default: [scatter=3, kde=1, wt_ridge=1, clin_ridge=1], ignored if no ridge_df
    grid_spacer=0.07, 
    rasterize=True,
    show_wt_margin=True,  # If True and ridge_df has wt_position, show WT margin effect row
    wt_margin_position="bottom",  # "top" or "bottom" - position of WT margin effect relative to clinical margin effect
    title_y=None,  # Y position for title (0-1, figure coordinates). If None, auto-calculated based on number of rows
    s=12,  # Point size for the main scatter plot
    show_legend=True,  # If True, show the legend for clinical significance
    scatter_y_label=None  # Y-axis label for the first scatter plot. If None, uses default based on position_axis
):
    """
    Plot scatter + KDE comparing VEP vs. position, with clinical significance colormaps.
    Optionally, if ridge_df is provided, plots margin effect facets for interactions vs. clinical_position
    and/or wt_position.

    Parameters
    ----------
    vep_prot : pd.DataFrame
        Must contain columns: position_col, vep_col, clinsig_col
    position_axis : {"y", "x"}
        Whether position is on y ("y", default) or x ("x") axis.
    vep_col : str
        Column name for VEP scores.
    position_col : str
        Column name to use for protein position.
    clinsig_col : str
        Column name for clinical significance.
    palette : dict, optional
        Color palette for clinical significance. Defaults to utils.get_clinsig_palette().
    figsize : tuple
        Figure size.
    title : str
        Title for the plot.
    ridge_kwargs : dict
        Keyword arguments for the interaction (ridge) scatterplot.
        Must contain at least "x" and "y" for which columns to use in ridge_df.
    ridge_y_label : str
        Label for the clinical margin effect facet (default: r"$|E_{\text{Clin}}|$").
    grid_row_ratios : tuple of int
        Proportions for the facet rows. Default (3, 1, 1, 1) for [scatter, kde, margin1, margin2].
        Order depends on wt_margin_position. Ignored if no ridge_df.
    grid_spacer : float
        Spacing parameter for the grid facet specification.
    rasterize : bool
        Whether to rasterize the plots for better performance.
    show_wt_margin : bool
        If True and ridge_df has wt_position, show WT margin effect row.
    wt_margin_position : {"top", "bottom"}
        Position of WT margin effect relative to clinical margin effect.
        "bottom" (default): scatter, kde, clinical, WT
        "top": scatter, kde, WT, clinical
    title_y : float, optional
        Y position for the title in figure coordinates (0-1). If None, automatically
        calculated based on the number of rows (0.98 for 4 rows, 0.97 for 3 rows, 0.95 for 2 rows).
    s : float, optional
        Point size for the main scatter plot. Default is 12.
    show_legend : bool, optional
        If True, show the legend for clinical significance. Default is True.
    scatter_y_label : str, optional
        Y-axis label for the first scatter plot. If None, uses default based on position_axis:
        - "Residue Position" when position_axis="y"
        - "VEP" when position_axis="x"
    """
    clinsig_order = ["path", "likely_path", "likely_benign", "benign"]
    if palette is None:
        palette = utils.get_clinsig_palette()

    clinsig_legend_labels = {
        "path": "pathogenic",
        "likely_path": "likely pathogenic",
        "likely_benign": "likely benign",
        "benign": "benign"
    }
    vep_prot = vep_prot.copy()

    vep_prot["site"] = vep_prot["site"].apply(utils.standardize_variant)
    # Setup figure orientation and grid
    sharey = position_axis == "y"
    sharex = position_axis == "x"
    has_interaction_ax = ridge_df is not None
    has_wt_position = has_interaction_ax and "wt_position" in ridge_df.columns and show_wt_margin
    
    # Validate wt_margin_position
    if wt_margin_position not in ["top", "bottom"]:
        raise ValueError(f"wt_margin_position must be 'top' or 'bottom', got '{wt_margin_position}'")

    # Define axes grid depending on requested plot
    if position_axis == "x":
        if has_wt_position:
            nrows = 4  # scatter, kde, and two margin plots
        elif has_interaction_ax:
            nrows = 3  # scatter, kde, clin_margin
        else:
            nrows = 2  # scatter, kde
        height_ratios = grid_row_ratios[:nrows]
        fig, axs = plt.subplots(
            nrows=nrows, ncols=1,
            figsize=figsize,
            sharex=sharex,
            sharey=False,  # Each axis has independent y-axis (especially for bottom two scatter plots)
            gridspec_kw={'height_ratios': height_ratios, 'hspace': grid_spacer}
        )
        if nrows == 2:
            ax_scatter, ax_kde = axs
            ax_wt_interaction = None
            ax_interaction = None
        elif nrows == 3:
            ax_scatter, ax_kde, ax_interaction = axs
            ax_wt_interaction = None
        else:  # nrows == 4
            ax_scatter, ax_kde = axs[0], axs[1]
            if wt_margin_position == "bottom":
                # Order: scatter, kde, clinical, WT
                ax_interaction, ax_wt_interaction = axs[2], axs[3]
            else:  # wt_margin_position == "top"
                # Order: scatter, kde, WT, clinical
                ax_wt_interaction, ax_interaction = axs[2], axs[3]
        bbox_to_anchor = (0.5, 0.88) if nrows == 2 else (0.5, 0.78) if nrows == 3 else (0.5, 0.72)
    else:
        if has_wt_position:
            ncols = 4  # scatter, kde, and two margin plots
        elif has_interaction_ax:
            ncols = 3  # scatter, kde, clin_margin
        else:
            ncols = 2  # scatter, kde
        width_ratios = grid_row_ratios[:ncols]
        fig, axs = plt.subplots(
            nrows=1, ncols=ncols, 
            figsize=figsize,
            sharex=False,
            sharey=sharey,
            gridspec_kw={'width_ratios': width_ratios, 'wspace': grid_spacer}
        )
        if ncols == 2:
            ax_scatter, ax_kde = axs
            ax_wt_interaction = None
            ax_interaction = None
        elif ncols == 3:
            ax_scatter, ax_kde, ax_interaction = axs
            ax_wt_interaction = None
        else:  # ncols == 4
            ax_scatter, ax_kde = axs[0], axs[1]
            if wt_margin_position == "bottom":
                # Order: scatter, kde, clinical, WT
                ax_interaction, ax_wt_interaction = axs[2], axs[3]
            else:  # wt_margin_position == "top"
                # Order: scatter, kde, WT, clinical
                ax_wt_interaction, ax_interaction = axs[2], axs[3]
        bbox_to_anchor = (0.88, 0.5) if ncols == 2 else (1.0, 0.5) if ncols == 3 else (1.12, 0.5)

    # --- Scatter ---
    scatter_kwargs = {
        "data": vep_prot,
        "hue": clinsig_col,
        "hue_order": clinsig_order,
        "palette": palette,
        # "edgecolor": 'none',
        "alpha": 0.4,
        "s": s,
        "ax": ax_scatter,
        "legend": False
    }
    if position_axis == "y":
        scatter_kwargs["y"] = position_col
        scatter_kwargs["x"] = vep_col
    else:
        scatter_kwargs["x"] = position_col
        scatter_kwargs["y"] = vep_col
    sns.scatterplot(**scatter_kwargs)

    # Reverse the scores axis so more negative is right/bottom
    # (convention: higher VEP is more benign)
    if position_axis == "y":
        ax_scatter.invert_xaxis()
    else:
        ax_scatter.invert_yaxis()

    # Set ticks and labels
    ticks = np.linspace(
        vep_prot[position_col].min(), 
        vep_prot[position_col].max(), 
        4, 
        dtype=int
    )
    if position_axis == "y":
        ax_scatter.set_yticks(ticks)
        ax_scatter.set_yticklabels([str(y) for y in ticks], rotation=0)
        ax_scatter.set_xlabel('VEP', fontsize='medium')
        y_label = scatter_y_label if scatter_y_label is not None else 'Residue Position'
        ax_scatter.set_ylabel(y_label, fontsize='medium')
        ax_scatter.invert_yaxis()  # so residue # increases downward, as in seq
    else:
        ax_scatter.set_xticks(ticks)
        ax_scatter.set_xticklabels([str(x) for x in ticks], rotation=0)
        ax_scatter.set_xlabel('Residue Position', fontsize='medium')
        y_label = scatter_y_label if scatter_y_label is not None else 'VEP'
        ax_scatter.set_ylabel(y_label, fontsize='medium')
        ax_scatter.invert_xaxis()  # so residue # increases right-to-left

    # Pretty legend
    if show_legend:
        handles = [
            mpatches.Patch(
                color=palette[c], 
                label=clinsig_legend_labels.get(c, c.replace('_', ' '))
            ) 
            for c in clinsig_order
        ]
        ax_scatter.legend(
            handles=handles, 
            title="Clinical Significance", 
            frameon=False,
            loc='center', 
            bbox_to_anchor=bbox_to_anchor
        )

    # --- KDE ---
    kde_kwargs = {
        "data": vep_prot,
        "hue": clinsig_col,
        "hue_order": clinsig_order,
        "fill": True,
        "multiple": "fill",
        "alpha": 0.40,
        "palette": palette,
        "common_norm": False,
        "cut": 0,
        "bw_adjust": 0.5,
        "linewidth": 1,
        "ax": ax_kde
    }
    if position_axis == "y":
        kde_kwargs["y"] = position_col
    else:
        kde_kwargs["x"] = position_col
    sns.kdeplot(**kde_kwargs)

    # Axis labels and despine for KDE
    if position_axis == "y":
        ax_kde.set_ylabel('Residue Position', fontsize='medium')
        ax_kde.set_xlabel(None)
        ax_kde.invert_yaxis()
    else:
        ax_kde.set_xlabel('Residue Position', fontsize='medium')
        ax_kde.set_ylabel(None)
        ax_kde.invert_xaxis()
    
    # Set consistent tick label sizes for all axes
    tick_label_size = 'medium'  # Medium size for tick labels
    ax_scatter.tick_params(axis='both', labelsize=tick_label_size)
    ax_kde.tick_params(axis='both', labelsize=tick_label_size)
    
    sns.despine(ax=ax_scatter)
    if position_axis == "y":
        sns.despine(ax=ax_kde, left=True)
    else:
        sns.despine(ax=ax_kde, bottom=True)

    # Only show legend in the left plot
    ax_kde.legend().remove()

    # --- WT margin effect plot (if ridge_df provided and has wt_position) ---
    if has_wt_position and ax_wt_interaction is not None:
        # Use ridge_kwargs properly. Set defaults for WT:
        _default_wt_ridge_kwargs = {
            "x": "wt_position",
            "y": "interaction_strength",
            "size": None,
            "s": s,  # Use same point size as main scatter plot
            "alpha": 0.5,
            "color": "black",  # Use black dots for WT
            "hue": None,  # WT variants don't have clinsig
            "palette": palette,
        }
        if ridge_kwargs is not None:
            wt_ridgep = {**_default_wt_ridge_kwargs, **ridge_kwargs}
        else:
            wt_ridgep = _default_wt_ridge_kwargs.copy()
        # Remove hue and palette if hue is None to ensure black color is used
        if wt_ridgep.get("hue") is None:
            wt_ridgep.pop("hue", None)
            wt_ridgep.pop("palette", None)
        # Extract mandatory x/y
        wt_ridge_x = wt_ridgep.pop("x")
        wt_ridge_y = wt_ridgep.pop("y")
       

        if wt_ridge_x not in ridge_df.columns or wt_ridge_y not in ridge_df.columns:
            raise ValueError(f"ridge_df must contain columns: {wt_ridge_x} and {wt_ridge_y}")

        # Aggregate ridge_df to compute mean yval per WT residue
        ridge_df_wt_agg = (
            ridge_df
            .groupby([wt_ridge_x], as_index=False)[wt_ridge_y].mean()
        )
        ridge_df_wt_agg = ridge_df_wt_agg.sort_values(by=wt_ridge_x)

        # scatterplot -- assign coordinates, axis
        if position_axis == "x":
            scatter_args = dict(
                data=ridge_df_wt_agg,
                x=wt_ridge_x,
                y=wt_ridge_y,
                ax=ax_wt_interaction,
                legend=False,
                zorder=2
            )
            scatter_args.update(wt_ridgep)
            sns.scatterplot(**scatter_args)
            ax_wt_interaction.set_xlabel('Residue Position', fontsize='medium')
            ax_wt_interaction.set_ylabel(wt_ridge_y_label, fontsize='medium')
            ax_wt_interaction.set_xlim(vep_prot[position_col].min(), vep_prot[position_col].max())
        else:
            scatter_args = dict(
                data=ridge_df_wt_agg,
                x=wt_ridge_y,
                y=wt_ridge_x,
                ax=ax_wt_interaction,
                legend=False,
                zorder=2
            )
            scatter_args.update(wt_ridgep)
            sns.scatterplot(**scatter_args)
            ax_wt_interaction.set_ylabel('Residue Position', fontsize='medium')
            ax_wt_interaction.set_xlabel(wt_ridge_y_label, fontsize='medium')
            ax_wt_interaction.set_ylim(vep_prot[position_col].min(), vep_prot[position_col].max()) 

        # Layout and ticks
        if position_axis == "x":
            ax_wt_interaction.set_xticks(ticks)
            ax_wt_interaction.set_xticklabels([str(x) for x in ticks], rotation=0)
            # Center y-axis around middle of observed values and set only two ticks: 0 and middle
            data_min = ridge_df_wt_agg[wt_ridge_y].min()
            data_max = ridge_df_wt_agg[wt_ridge_y].max()
            data_middle = (data_min + data_max) / 2
            data_range = data_max - data_min
            # Add 1% padding to prevent points from being cut off
            padding = data_range * 0.01
            # Center axis around the middle of the data with padding
            ax_wt_interaction.set_ylim(data_middle - data_range/2 - padding, data_middle + data_range/2 + padding)
            # Set only two y-ticks: 0 (if in range) and the middle value
            y_lim = ax_wt_interaction.get_ylim()
            y_ticks = []
            if y_lim[0] <= 0 <= y_lim[1]:
                y_ticks.append(0)
            y_ticks.append(data_middle)
            y_ticks.sort()
            ax_wt_interaction.set_yticks(y_ticks)
            # Format y-axis labels in scientific notation (except 0), rounded to 1 decimal place
            formatter = ticker.FuncFormatter(lambda x, p: '0' if x == 0 else f'{x:.1e}')
            ax_wt_interaction.yaxis.set_major_formatter(formatter)
        else:
            ax_wt_interaction.set_yticks(ticks)
            ax_wt_interaction.set_yticklabels([str(y) for y in ticks], rotation=0)
        # Set consistent tick label size for WT interaction plot
        ax_wt_interaction.tick_params(axis='both', labelsize=tick_label_size)
        sns.despine(ax=ax_wt_interaction)

    # --- Interaction plot (if ridge_df provided) ---
    if has_interaction_ax and ax_interaction is not None:
        # Use ridge_kwargs properly. Set defaults:
        _default_ridge_kwargs = {
            "x": "clinical_position",
            "y": "interaction_strength",
            "size": None,
            "s": s,  # Use same point size as main scatter plot
            "alpha": 0.5,
            "hue": "clinsig",
            "palette": palette,
            # "sizes": (1e-8, 100)
        }
        if ridge_kwargs is not None:
            ridgep = {**_default_ridge_kwargs, **ridge_kwargs}
        else:
            ridgep = _default_ridge_kwargs.copy()
        # Extract mandatory x/y
        ridge_x = ridgep.pop("x")
        ridge_y = ridgep.pop("y")
        # ridge_y_label is kept as argument

        if ridge_x not in ridge_df.columns or ridge_y not in ridge_df.columns:
            raise ValueError(f"ridge_df must contain columns: {ridge_x} and {ridge_y}")

        # First, aggregate ridge_df to compute mean yval per residue (ridge_x)
        
        clinsig_order = utils.get_clinsig_order() 
        ridge_df_agg = (
            ridge_df
            .merge(vep_prot[["site", "clinsig"]].drop_duplicates(), on="site", how="left")
            .groupby(["site", "clinsig", ridge_x], as_index=False)[ridge_y].mean()
        )
        # Create Categorical for clinsig with correct order (just those present)
        present_clinsigs = [c for c in clinsig_order if c in ridge_df_agg["clinsig"].unique()]
        ridge_df_agg["clinsig"] = pd.Categorical(ridge_df_agg["clinsig"], categories=present_clinsigs, ordered=True)
        ridge_df_agg = ridge_df_agg.sort_values(by=["site", "clinsig", ridge_x])

        # scatterplot -- assign coordinates, axis
        if position_axis == "x":
            scatter_args = dict(
                data=ridge_df_agg,
                x=ridge_x,
                y=ridge_y,
                ax=ax_interaction,
                legend=False,
                zorder=2
            )
            scatter_args.update(ridgep)
            sns.scatterplot(**scatter_args)
            ax_interaction.set_xlabel('Residue Position', fontsize='medium')
            ax_interaction.set_ylabel(ridge_y_label, fontsize='medium')
            ax_interaction.set_xlim(vep_prot[position_col].min(), vep_prot[position_col].max())
        else:
            scatter_args = dict(
                data=ridge_df_agg,
                x=ridge_y,
                y=ridge_x,
                ax=ax_interaction,
                legend=False,
                zorder=2
            )
            scatter_args.update(ridgep)
            sns.scatterplot(**scatter_args)
            ax_interaction.set_ylabel('Residue Position', fontsize='medium')
            ax_interaction.set_xlabel(ridge_y_label, fontsize='medium')
            ax_interaction.set_ylim(vep_prot[position_col].min(), vep_prot[position_col].max()) 

        # Layout and ticks
        if position_axis == "x":
            ax_interaction.set_xticks(ticks)
            ax_interaction.set_xticklabels([str(x) for x in ticks], rotation=0)
            # Center y-axis around middle of observed values and set only two ticks: 0 and middle
            data_min = ridge_df_agg[ridge_y].min()
            data_max = ridge_df_agg[ridge_y].max()
            data_middle = (data_min + data_max) / 2
            data_range = data_max - data_min
            # Add 1% padding to prevent points from being cut off
            padding = data_range * 0.01
            # Center axis around the middle of the data with padding
            ax_interaction.set_ylim(data_middle - data_range/2 - padding, data_middle + data_range/2 + padding)
            # Set only two y-ticks: 0 (if in range) and the middle value
            y_lim = ax_interaction.get_ylim()
            y_ticks = []
            if y_lim[0] <= 0 <= y_lim[1]:
                y_ticks.append(0)
            y_ticks.append(data_middle)
            y_ticks.sort()
            ax_interaction.set_yticks(y_ticks)
            # Format y-axis labels in scientific notation (except 0), rounded to 1 decimal place
            formatter = ticker.FuncFormatter(lambda x, p: '0' if x == 0 else f'{x:.1e}')
            ax_interaction.yaxis.set_major_formatter(formatter)
        else:
            ax_interaction.set_yticks(ticks)
            ax_interaction.set_yticklabels([str(y) for y in ticks], rotation=0)
        # Set consistent tick label size for interaction plot
        ax_interaction.tick_params(axis='both', labelsize=tick_label_size)
        sns.despine(ax=ax_interaction)

    # Set plot title if provided
    if title is not None:
        # Adjust title position based on number of rows if not explicitly provided
        if title_y is None:
            if has_wt_position:
                title_y = 0.98  # 4 rows
            elif has_interaction_ax:
                title_y = 0.97  # 3 rows
            else:
                title_y = 0.95  # 2 rows
        fig.suptitle(title, y=title_y, fontsize='large')

    if rasterize:
        if ax_interaction is not None:
            ax_interaction = utils.rasterize_figure(ax_interaction)
        if ax_wt_interaction is not None:
            ax_wt_interaction = utils.rasterize_figure(ax_wt_interaction)
        ax_scatter = utils.rasterize_figure(ax_scatter) 

    plt.tight_layout()
    plt.show()
    
    # Prepare return data
    return_data = {"vep_prot": vep_prot}
    if has_interaction_ax and ax_interaction is not None:
        return_data["ridge_df_agg"] = ridge_df_agg
    if has_wt_position and ax_wt_interaction is not None:
        return_data["ridge_df_wt_agg"] = ridge_df_wt_agg
    
    return {"fig": fig, "axs": axs, "data": return_data}

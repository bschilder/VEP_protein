
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import src.utils as utils


LABEL_MAP = {"benign": 0, 
             "Benign": 0,
             "likely_benign": 0,
             "Likely_Benign": 0,
             "likely_path": 1,
             "likely_pathogenic": 1,
             "Likely_Pathogenic": 1,
             "path": 1,
             "pathogenic": 1,
             "Pathogenic": 1}

def run_logreg(X, y, group_id, random_state=42, max_iter=1000, verbose=True):
    """
    Perform logistic regression on the provided dataset and return classification metrics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix. If 1D, will be reshaped to 2D.
    y : np.ndarray
        Binary target vector (0 for benign, 1 for pathogenic).
    group_id : str
        Group identifier for the dataset (e.g., "REF" or "non-REF").
    random_state : int, optional
        Random seed for reproducibility (default: 42).
    max_iter : int, optional
        Maximum number of iterations for the logistic regression solver (default: 1000).

    Returns
    -------
    dict
        Dictionary containing classification metrics, including accuracy, precision, recall, F1 scores,
        average precision, AUPRC, and the PR curve (precision, recall).
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score, auc
    from sklearn.model_selection import train_test_split

    if verbose:
        print(f"Running logistic regression for {group_id}")
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    # y should be binary: 0 (benign), 1 (pathogenic)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    # For PR curve
    y_score = clf.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)
    auprc = auc(recall, precision)
    # Return as a DataFrame row
    results = {
        "group_id": group_id,
        "accuracy": round(report.get("accuracy", 0), 6),
        "macro_precision": round(report.get("macro avg", {}).get("precision", 0), 6),
        "macro_recall": round(report.get("macro avg", {}).get("recall", 0), 6),
        "macro_f1": round(report.get("macro avg", {}).get("f1-score", 0), 6),
        "weighted_precision": round(report.get("weighted avg", {}).get("precision", 0), 6),
        "weighted_recall": round(report.get("weighted avg", {}).get("recall", 0), 6),
        "weighted_f1": round(report.get("weighted avg", {}).get("f1-score", 0), 6),
        "support": int(report.get("macro avg", {}).get("support", 0)),
        "average_precision": round(ap, 6),
        "AUPRC": round(auprc, 6)
    }
    # Also store the PR curve for plotting
    results["pr_curve"] = (precision, recall)
    return results

def run_clinvar_logreg_analysis(
    vep_df,
    vep_col="VEP",
    clinsig_col="CLNSIG_simplified",
    site_col="site",
    label_map=LABEL_MAP,
    verbose=True,
    min_labels=2,
    min_values_per_label=2,
    agg_func="mean"
):
    """
    Perform logistic regression analysis on ClinVar VEP data.

    This function processes a DataFrame containing variant effect predictor (VEP) scores and ClinVar
    simplified clinical significance labels, and runs logistic regression to distinguish between
    benign and pathogenic variants for both reference (REF) and non-reference (non-REF) samples.

    Parameters
    ----------
    vep_df : pd.DataFrame
        DataFrame containing at least the columns: 'sample', 'site', 'VEP', and 'CLNSIG_simplified'.

    Returns
    -------
    logreg_results_df : pd.DataFrame
        DataFrame summarizing the logistic regression results for REF and non-REF samples.
    results_dict : dict
        Dictionary containing the detailed results for "REF" and "non-REF" analyses.
    """
    vep_df = vep_df.copy()

    if "is_ref" not in vep_df.columns:
        if "sample" in vep_df.columns:
            vep_df["is_ref"] = vep_df["sample"] == "REF"
        elif "haplotype" in vep_df.columns:
            vep_df["is_ref"] = vep_df["haplotype"].str.endswith(":REF")
        else:
            raise ValueError("is_ref not found in vep_df")

    vep_df.dropna(subset=[vep_col, clinsig_col], inplace=True)

    vep_df["label"] = vep_df[clinsig_col].map(label_map)
    # Get sites that are present in both is_ref groups (REF and non-REF)
    overlapping_sites = set(vep_df.loc[vep_df["is_ref"], site_col]) & set(vep_df.loc[~vep_df["is_ref"], site_col])
    # Only use rows where CLNSIG_simplified is in label_map
    mask = vep_df[clinsig_col].isin(list(label_map.keys())) & vep_df[site_col].isin(overlapping_sites)

    # REF samples: each row is a variant, with VEP and CLNSIG_simplified
    ref_df = vep_df.loc[mask & (vep_df["is_ref"]), [site_col, vep_col, clinsig_col, "label"]].dropna()

    def check_data(df, min_labels=2, min_values_per_label=2, group_id="REF", verbose=True):
        """
        Checks if the DataFrame has at least min_labels unique mapped values in clinsig_col.
        Returns True if valid, False otherwise.
        """
        # If 'label' is not in columns, try to map it from clinsig_col
        if 'label' not in df.columns:
            if clinsig_col in df.columns:
                df = df.copy()
                df["label"] = df[clinsig_col].map(label_map)
            else:
                if verbose:
                    print(f"DataFrame for {group_id} samples does not contain 'label' or '{clinsig_col}' columns.")
                return False
        if df.empty:
            if verbose:
                print(f"No {group_id} samples found")
            return False
        if df['label'].nunique() < min_labels:
            if verbose:
                print(f"Not enough unique labels for {group_id} samples (found {df['label'].nunique()}, need {min_labels}). Skipping.")
            return False
        for label_val in df['label'].unique():
            if df[df['label'] == label_val].shape[0] < min_values_per_label:
                if verbose:
                    print(f"Not enough {group_id} samples for {label_val} (found {df[df['label'] == label_val].shape[0]}, need {min_values_per_label}). Skipping.")
                return False
        return True

    def can_split_for_logreg(y, min_test_size=2):
        """
        Checks if stratified train_test_split can be performed for binary classification.
        Ensures that each class has at least min_test_size samples.
        """
        import numpy as np
        y = np.asarray(y)
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            return False
        if np.any(counts < min_test_size):
            return False
        return True

    # Check and run for REF
    if not check_data(
        df=ref_df,
        min_labels=min_labels,
        min_values_per_label=min_values_per_label,
        group_id="REF",
        verbose=verbose,
    ):
        if verbose:
            print("REF group failed data checks. Skipping.")
        ref_results = None
    elif not can_split_for_logreg(ref_df['label'].values, min_test_size=2):
        if verbose:
            print("REF group does not have enough samples per class for stratified split. Skipping.")
        ref_results = None
    else:
        try:
            ref_results = run_logreg(
                X=ref_df[vep_col].values,
                y=ref_df['label'].values,
                group_id="REF",
                verbose=verbose
            )
        except ValueError as e:
            if verbose:
                print(f"REF group logistic regression failed: {e}")
            ref_results = None

    # non-REF samples: aggregate by site and CLNSIG_simplified, take mean VEP per group
    nonref_df = (
        vep_df.loc[mask & (~vep_df["is_ref"]), [site_col, vep_col, clinsig_col, "label"]]
        .dropna()
    )
    if agg_func is not None:
        # Group and aggregate, then reset index to get a flat DataFrame
        nonref_df = (
            nonref_df
            .groupby([site_col, clinsig_col, "label"], observed=True, as_index=False)
            .agg({vep_col: agg_func})
        )

    # Check and run for non-REF
    if not check_data(
        df=nonref_df,
        min_labels=min_labels,
        min_values_per_label=min_values_per_label,
        group_id="non-REF",
        verbose=verbose,
    ):
        if verbose:
            print("non-REF group failed data checks. Skipping.")
        nonref_results = None
    elif not can_split_for_logreg(nonref_df['label'].values, min_test_size=2):
        if verbose:
            print("non-REF group does not have enough samples per class for stratified split. Skipping.")
        nonref_results = None
    else:
        try:
            nonref_results = run_logreg(
                X=nonref_df[vep_col].values,
                y=nonref_df['label'].values,
                group_id="non-REF",
                verbose=verbose
            )
        except ValueError as e:
            if verbose:
                print(f"non-REF group logistic regression failed: {e}")
            nonref_results = None

    # Combine all results and create a clean DataFrame
    results = []
    if ref_results is not None:
        results.append(ref_results)
    if nonref_results is not None:
        results.append(nonref_results)
    if results:
        lr_df = pd.DataFrame(results)
        return lr_df
    else:
        return None


def run_clinvar_logreg_analysis_by_group(
        vep_df,
        group_by,
        vep_col="VEP",
        clinsig_col="CLNSIG_simplified",
        site_col="site",
        label_map=LABEL_MAP,
        verbose=True, 
        min_values_per_label=2,
        **kwargs
    ):
        """
        Iteratively runs run_clinvar_logreg_analysis for each group in group_by and concatenates the results.

        Parameters
        ----------
        vep_df : pd.DataFrame
            DataFrame containing at least the columns for grouping, VEP, and CLNSIG_simplified.
        group_by : str or list of str
            Column(s) to group by (e.g., "protein").
        veq_col : str
            Name of the column with VEP scores.
        clinsig_col : str
            Name of the column with clinical significance labels.
        site_col : str
            Name of the column with site identifiers.
        label_map : dict
            Mapping from CLNSIG_simplified values to binary labels.
        **kwargs : dict
            Additional arguments to pass to run_clinvar_logreg_analysis.

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame of logistic regression results for all groups.
        """
        import pandas as pd

        if isinstance(group_by, str):
            group_by = [group_by]

        results = []
        from tqdm.auto import tqdm

        group_iter = vep_df.groupby(group_by, observed=True)
        total_groups = len(vep_df[group_by].drop_duplicates())
        for group_vals, group_df in tqdm(group_iter, total=total_groups, desc="Groups"):
            # group_vals is a tuple if group_by is a list, else a scalar
            group_label = (
                group_vals if isinstance(group_vals, str) or not isinstance(group_vals, tuple)
                else "_".join(str(v) for v in group_vals)
            )

            lr_df = run_clinvar_logreg_analysis(
                group_df,
                vep_col=vep_col,
                clinsig_col=clinsig_col,
                site_col=site_col,
                label_map=label_map, 
                min_values_per_label=min_values_per_label,
                verbose=verbose>1,
                **kwargs
            )
            if lr_df is None:
                continue
            
            # Add group info to the results
            for i in range(len(lr_df)):
                for j, col in enumerate(group_by):
                    lr_df.loc[i, col] = group_vals[j] if isinstance(group_vals, tuple) else group_vals
            results.append(lr_df)

        if results:
            results_df = pd.concat(results, ignore_index=True)
        else:
            raise ValueError("No results found")
        return results_df


def plot_logreg_results(
    lr_df,  
    group_id="group_id",
    palette=utils.get_ref_nonref_palette(),
    figsize=(10, 5),
    gridspec_kw={'width_ratios': [1.5, 1]},
    title1="Pathogenic/Benign Classification Results",
    title2="Precision-Recall Curve",
    remove_macro_metrics=True,
    auprc_fmt="{:.4f}",
    legend_kwargs={"loc": "upper center",  "ncol": 2},
    bar_edge=True,
    bar_edgecolor="black",
    bar_linewidth=1,
    pr_linewidth=2,
    pr_linestyle="-"
):
    """
    Much faster version: avoids Python loops, uses vectorized and batch plotting.

    Parameters
    ----------
    lr_df : pd.DataFrame
        DataFrame with logreg results.
    group_id : str
        Column name for group labels.
    palette : dict
        Color palette mapping group labels to colors.
    figsize : tuple
        Figure size.
    gridspec_kw : dict
        GridSpec kwargs for subplots.
    title1 : str
        Title for barplot.
    title2 : str
        Title for PR curve plot.
    remove_macro_metrics : bool
        If True, remove macro metrics from barplot.
    auprc_fmt : str
        Format string for AUPRC in legend.
    legend_kwargs : dict
        Keyword arguments for legend.
    bar_edge : bool
        Whether to draw lines around the bars in the barplot.
    bar_edgecolor : str
        Color of the bar edges.
    bar_linewidth : float
        Width of the bar edges.
    pr_linewidth : float
        Line width for PR curves.
    pr_linestyle : str
        Line style for PR curves (e.g., '-', '--', '-.', ':').
    """
    import matplotlib.pyplot as plt
    import seaborn as sns  

    # Precompute unique labels and palette
    unique_labels = lr_df[group_id].unique()   

    # Barplot data preparation (vectorized)
    barplot_df = lr_df.melt(
        id_vars=[group_id, "pr_curve", "support"], 
        var_name="metric",
        value_name="value"
    )
    if remove_macro_metrics:
        barplot_df = barplot_df[~barplot_df["metric"].str.startswith("macro_")]
    
    barplot_df['metric'] = barplot_df['metric'].str.replace("_", "\n")

    # Plot setup
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw=gridspec_kw)

    # Barplot (left)
    barplot = sns.barplot(
        data=barplot_df,
        x="metric",
        y="value",
        hue=group_id,
        ax=axes[0],
        palette=palette,
        edgecolor=bar_edgecolor if bar_edge else None,
        linewidth=bar_linewidth if bar_edge else 0
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(range(len(axes[0].get_xticklabels())))
    axes[0].set_xticklabels(
        [tick.get_text() for tick in axes[0].get_xticklabels()],
        rotation=0
    )
    axes[0].set_title(title1)
    axes[0].legend(**legend_kwargs)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # If bar_edge is False, remove bar outlines after plotting (for seaborn >=0.12)
    if not bar_edge:
        for patch in axes[0].patches:
            patch.set_linewidth(0)

    # Precision-Recall curves (right) - batch plotting, no Python for-loop
    # Prepare all PR curves as lists of arrays
    pr_curves = lr_df["pr_curve"].to_list()
    labels = lr_df[group_id].to_numpy()
    auprcs = lr_df["AUPRC"].to_numpy()
    # Build a legend label for each
    legend_labels = [f"{label} (AUPRC={auprc_fmt.format(ap)})" for label, ap in zip(labels, auprcs)]
    # Plot all curves at once using matplotlib for speed
    for i, (pr, label, legend_label) in enumerate(zip(pr_curves, labels, legend_labels)):
        precision, recall = pr
        color = palette[label]
        axes[1].plot(
            recall, precision, 
            label=legend_label, 
            color=color, 
            linewidth=pr_linewidth, 
            linestyle=pr_linestyle
        )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(title2)
    axes[1].legend()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

    return {"fig": fig, "axes": axes, 'data': {'barplot_df': barplot_df, 'pr_curves': pr_curves, 'labels': labels, 'auprcs': auprcs}}


def agg_logreg_results(lr_df,
                       lr_gene_df, 
                       group_by="protein",
                       support_threshold=0):
    """
    Aggregate logistic regression results by filtering for proteins present in both REF and non-REF groups,
    and merge with overall PR/AUPRC curves.

    Parameters
    ----------
    lr_gene_df : pd.DataFrame
        DataFrame with per-gene (or per-protein) logistic regression results, must include columns:
        'support', 'group_id', 'protein', 'AUPRC', etc.
    lr_df : pd.DataFrame
        DataFrame with overall logistic regression results, must include columns:
        'group_id', 'pr_curve', 'AUPRC', etc.
    support_threshold : int, optional
        Minimum support (number of samples) required to include a protein, by default 0.

    Returns
    -------
    pd.DataFrame
        Aggregated results, merged with overall PR/AUPRC curves.
    """
    # Filter by support threshold
    filtered = lr_gene_df.loc[lr_gene_df["support"] >= support_threshold]
    # Find proteins present in both groups
    common_proteins = set(filtered.loc[filtered["group_id"] == "REF", group_by]).intersection(
        filtered.loc[filtered["group_id"] == "non-REF", group_by]
    )
    # Only keep rows with those proteins
    lr_gene_grouped = (
        filtered[filtered[group_by].isin(common_proteins)]
        .groupby("group_id")
        .mean(numeric_only=True)
        .reset_index()
    )
    plot_data = lr_gene_grouped.drop(columns=["AUPRC"]).merge(
        lr_df[["group_id", "pr_curve", "AUPRC"]],
        on="group_id"
    )
    return plot_data


def plot_logreg_results_subplots(
    data_dict,
    group_id="group_id",
    palette=utils.get_ref_nonref_palette(),
    figsize=(15, 5),
    gridspec_kw={'width_ratios': [1.5, 1]},
    title1="Pathogenic/Benign Classification Results",
    title2="Precision-Recall Curve",
    remove_macro_metrics=True,
    auprc_fmt="{:.4f}",
    legend_kwargs={"loc": "upper center", "ncol": 2},
    bar_edge=True,
    bar_edgecolor="black",
    bar_linewidth=1,
    pr_linewidth=2,
    pr_linestyle="-",
    model_names=None
):
    """
    Create subplots for multiple models' logistic regression results.
    Each row represents one model with barplot (left) and PR curve (right).

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping model names to their respective lr_df DataFrames.
    group_id : str
        Column name for group labels.
    palette : dict
        Color palette mapping group labels to colors.
    figsize : tuple
        Figure size (width, height).
    gridspec_kw : dict
        GridSpec kwargs for subplots.
    title1 : str
        Title for barplot.
    title2 : str
        Title for PR curve plot.
    remove_macro_metrics : bool
        If True, remove macro metrics from barplot.
    auprc_fmt : str
        Format string for AUPRC in legend.
    legend_kwargs : dict
        Keyword arguments for legend.
    bar_edge : bool
        Whether to draw lines around the bars in the barplot.
    bar_edgecolor : str
        Color of the bar edges.
    bar_linewidth : float
        Width of the bar edges.
    pr_linewidth : float
        Line width for PR curves.
    pr_linestyle : str
        Line style for PR curves (e.g., '-', '--', '-.', ':').
    model_names : list, optional
        List of model names to plot in order. If None, uses keys from data_dict.

    Returns
    -------
    dict
        Dictionary containing the figure, axes, and data used for plotting.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns  

    if model_names is None:
        model_names = list(data_dict.keys())
    
    n_models = len(model_names)
    
    # Create figure with subplots: n_models rows, 2 columns
    fig, axes = plt.subplots(n_models, 2, figsize=(figsize[0], figsize[1] * n_models))
    
    # Handle case where there's only one model (axes will be 1D)
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each model in its own row
    for i, model_name in enumerate(model_names):
        lr_df = data_dict[model_name]
        
        # Precompute unique labels and palette
        unique_labels = lr_df[group_id].unique()   

        # Barplot data preparation (vectorized)
        barplot_df = lr_df.melt(
            id_vars=[group_id, "pr_curve", "support"], 
            var_name="metric",
            value_name="value"
        )
        if remove_macro_metrics:
            barplot_df = barplot_df[~barplot_df["metric"].str.startswith("macro_")]
        
        barplot_df['metric'] = barplot_df['metric'].str.replace("_", "\n")

        # Barplot (left column)
        barplot = sns.barplot(
            data=barplot_df,
            x="metric",
            y="value",
            hue=group_id,
            ax=axes[i, 0],
            palette=palette,
            edgecolor=bar_edgecolor if bar_edge else None,
            linewidth=bar_linewidth if bar_edge else 0
        )
        axes[i, 0].set_ylim(0, 1)
        axes[i, 0].set_xticks(range(len(axes[i, 0].get_xticklabels())))
        axes[i, 0].set_xticklabels(
            [tick.get_text() for tick in axes[i, 0].get_xticklabels()],
            rotation=0
        )
        axes[i, 0].set_title(f"{title1}: {model_name}")
        axes[i, 0].legend(**legend_kwargs)
        axes[i, 0].spines['top'].set_visible(False)
        axes[i, 0].spines['right'].set_visible(False)

        # If bar_edge is False, remove bar outlines after plotting
        if not bar_edge:
            for patch in axes[i, 0].patches:
                patch.set_linewidth(0)

        # Precision-Recall curves (right column)
        pr_curves = lr_df["pr_curve"].to_list()
        labels = lr_df[group_id].to_numpy()
        auprcs = lr_df["AUPRC"].to_numpy()
        
        # Build a legend label for each
        legend_labels = [f"{label} (AUPRC={auprc_fmt.format(ap)})" for label, ap in zip(labels, auprcs)]
        
        # Plot all curves
        for j, (pr, label, legend_label) in enumerate(zip(pr_curves, labels, legend_labels)):
            precision, recall = pr
            color = palette[label]
            axes[i, 1].plot(
                recall, precision, 
                label=legend_label, 
                color=color, 
                linewidth=pr_linewidth, 
                linestyle=pr_linestyle
            )
        
        axes[i, 1].set_xlabel("Recall")
        axes[i, 1].set_ylabel("Precision")
        axes[i, 1].set_title(f"{title2}: {model_name}")
        axes[i, 1].legend()
        axes[i, 1].spines['top'].set_visible(False)
        axes[i, 1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

    return {"fig": fig, "axes": axes, "model_names": model_names}


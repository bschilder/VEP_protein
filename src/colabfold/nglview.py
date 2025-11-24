import nglview as nv
import numpy as np
import matplotlib
import pandas as pd
from Bio.PDB import PDBParser
import gzip
import shutil
import os
import nglview
# Can color by any of these: https://nglviewer.org/ngl/api/manual/coloring.html
# atomindex         # Colors by the atom serial number in the structure file.
# bfactor           # Colors by the B-factor (temperature factor) from the structure file, representing atomic displacement.
# chainid           # Colors by the chain identifier; different polypeptide/protein/RNA/DNA chains.
# chainindex        # Colors each chain by its index (order as loaded in file).
# chainname         # Colors by the name/label of each chain.
# densityfit        # Colors by the quality of fit to electron density map (if available).
# electrostatic     # Colors by electrostatic (charge) potential.
# element           # Colors by the atomic element type (C, N, O, etc.).
# entityindex       # Colors by entity index (distinct molecules/entities in the file).
# entitytype        # Colors by the type of molecular entity (protein, DNA, RNA, ligand, etc.).
# geoquality        # Colors by geometric quality score of the structure (e.g., from validation).
# hydrophobicity    # Colors according to residue hydrophobicity scale (hydrophobic or hydrophilic).
# modelindex        # Colors by model index (for ensembles or NMR structures).
# moleculetype      # Colors by molecule type (protein, nucleic acid, ligand, etc.).
# occupancy         # Colors by occupancy field of atoms in the structure file (value between 0 and 1).
# random            # Colors atoms/residues randomly.
# residueindex      # Colors by residue index (position in the sequence/structure).
# resname           # Colors by residue or ligand name (e.g., ALA for Alanine, HEM for heme).
# sstruc            # Colors by secondary structure (helix, sheet, loop).
# uniform           # Colors everything the same (single color).
# value             # Colors by a user-provided numeric value for each atom/residue.
# volume            # Colors by the molecular surface/volume (e.g., as in a surface rendering).


### Color key for sstruc:
# sstruc (secondary structure) coloring color keys:
#     - helix: red/pink
#     - sheet: yellow
#     - turn (loop/coil): blue
#     - undefined/none: white  


def get_mean_score_df(df, pos_col, protein_length):
    """Compute mean signed interaction score for a residue position col, 
    restricted to valid (PDB-present) residues."""
    available_indices = np.arange(1, protein_length + 1)
    mean_scores = (
        df.groupby(pos_col)["interaction_strength_signed"].mean()
        .rename_axis("pos").reset_index(name="score")
    )
    mean_scores_filtered = mean_scores[mean_scores["pos"].isin(available_indices)].copy()
    return mean_scores_filtered

def get_normed_colors_separate(df, colormap="bwr_r"):
    """Assign distinct red to all values < 0, blue to all values > 0, white at 0. Skewed distributions, binarize at 0."""
    scores = df["score"].values
    cm = matplotlib.colormaps[colormap]

    vmin = min(0, np.min(scores))
    vmax = max(0, np.max(scores))

    scores = np.asarray(scores)
    normed = np.zeros_like(scores, dtype=float)
    negatives = scores < 0
    positives = scores > 0
    zeros = scores == 0

    if vmin == 0:
        normed[negatives] = 0.5
    else:
        normed[negatives] = 0.5 * (scores[negatives] - vmin) / (0 - vmin)
    if vmax == 0:
        normed[positives] = 0.5
    else:
        normed[positives] = 0.5 + 0.5 * (scores[positives] / vmax)
    normed[zeros] = 0.5

    rgb = cm(normed)
    colors = [matplotlib.colors.rgb2hex(row) for row in rgb]
    return colors

def make_color_df(mean_scores_df, color_list, protein_length=None, default_color="grey"):
    """
    Create DataFrame for NGLView color mapping.
    Pass protein_length to cover all residues (1-based index); positions not in mean_scores_df will get default_color.
    """
    if protein_length is not None:
        all_indices = pd.Index(np.arange(1, protein_length + 1))
        df_full = pd.DataFrame({'residue_index': all_indices})
        df_merged = df_full.merge(
            mean_scores_df[["pos", "score"]], left_on='residue_index', right_on='pos', how='left'
        )
        color_array = np.full(len(df_merged), default_color, dtype='U7')
        mask = ~df_merged['score'].isnull()
        if hasattr(color_list, "tolist"):
            color_list = color_list.tolist()
        color_array[mask] = np.array(color_list)
        out = pd.DataFrame({
            'residue_index': df_merged['residue_index'].astype(int),
            'raw_score': df_merged['score'],
            'color': color_array
        })
        return out
    else:
        return pd.DataFrame({
            'residue_index': mean_scores_df["pos"].astype(int),
            'raw_score': mean_scores_df["score"],
            'color': color_list
        })

def make_selection_scheme(color_df):
    return [[color, str(idx)+".CA"] for idx, color in zip(color_df['residue_index'], color_df['color'])]

def get_interaction_color(val, cmap, norm):
    col = cmap(norm(val))
    return matplotlib.colors.rgb2hex(col)

def prepare_pdb_file(pdb_file):
    decompressed_pdb_file = pdb_file.replace(".gz", "")
    if not os.path.exists(decompressed_pdb_file):
        with gzip.open(pdb_file, 'rb') as f_in, open(decompressed_pdb_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return decompressed_pdb_file

def get_protein_chain_and_length(pdb_file_to_use):
    p = PDBParser(QUIET=True)
    structure = p.get_structure("protein", pdb_file_to_use)
    chain = next(structure.get_chains())
    residues = [res for res in chain if res.id[0] == ' ']
    protein_length = len(residues)
    return chain, protein_length

def visualize_interactions_on_structure(
    pdb_files_ref, 
    ridge_df, 
    vep_prot, 
    label_pos=None, 
    n_top_pairs=100, 
    struct_label="protein",
    show=True,
    sort_interactions="deviation_from_additive",
    filter_interactions=None,
    include_all_labeled_pairs=False
):
    """
    Visualize functional interaction patterns and (optionally) clinical mean on structure using NGLView.
    Returns the nglview.View widget.
    
    Parameters
    ----------
    pdb_files_ref : list
        List of PDB file paths
    ridge_df : pd.DataFrame
        DataFrame containing interaction data with columns like 'wt_variant', 'site', 
        'wt_position', 'clinical_position', 'interaction_strength', etc.
    vep_prot : pd.DataFrame
        DataFrame containing VEP data with 'Protein_position' and 'VEP' columns
    label_pos : list or None, optional
        List of residue positions to label. If None, labels all positions (both WT and clinical).
        Default: None (labels all)
    n_top_pairs : int or None, default 100
        Number of top interaction pairs to display. If None, displays all pairs (after filtering and sorting).
    struct_label : str, default "protein"
        Label for the protein structure in NGLView
    show : bool, default True
        Whether to show the view
    filter_interactions : callable, dict, pd.Series, or None, optional
        Flexible filtering for ridge_df. Can be:
        - A callable function: filter_interactions(ridge_df) -> boolean mask or filtered DataFrame
        - A dictionary: {column: value} or {column: [values]} for filtering
        - A pandas boolean Series: boolean mask with same length as ridge_df
        - None: no filtering (default)
    include_all_labeled_pairs : bool, default False
        If True and label_pos is provided (not None), includes all interaction pairs where both
        wt_position and clinical_position are in label_pos, even if they're not in the top n_top_pairs.
        These pairs are added to the top_pairs for visualization.
    
    Returns
    -------
    nglview.View
        The NGLView widget
    """
    # If label_pos is None, we'll label all positions (handled in the labeling loops)
    label_all = (label_pos is None)
    
    # Apply filtering if provided
    ridge_df_filtered = ridge_df.copy()
    if filter_interactions is not None:
        if callable(filter_interactions):
            # If it's a callable, apply it
            result = filter_interactions(ridge_df)
            if isinstance(result, pd.Series):
                # Boolean mask
                ridge_df_filtered = ridge_df[result]
            elif isinstance(result, pd.DataFrame):
                # Filtered DataFrame
                ridge_df_filtered = result
            else:
                raise ValueError("filter_interactions callable must return a boolean Series or DataFrame")
        elif isinstance(filter_interactions, dict):
            # Dictionary of column: value conditions
            for col, val in filter_interactions.items():
                if col not in ridge_df.columns:
                    raise ValueError(f"Column '{col}' not found in ridge_df")
                if isinstance(val, list):
                    ridge_df_filtered = ridge_df_filtered[ridge_df_filtered[col].isin(val)]
                else:
                    ridge_df_filtered = ridge_df_filtered[ridge_df_filtered[col] == val]
        elif isinstance(filter_interactions, pd.Series):
            # Boolean mask
            if len(filter_interactions) != len(ridge_df):
                raise ValueError(f"filter_interactions Series length ({len(filter_interactions)}) must match ridge_df length ({len(ridge_df)})")
            ridge_df_filtered = ridge_df[filter_interactions]
        else:
            raise ValueError("filter_interactions must be callable, dict, pd.Series, or None")
    
    # Print report before plotting
    n_wt_variants = ridge_df_filtered['wt_variant'].nunique() if 'wt_variant' in ridge_df_filtered.columns else 0
    n_clinical_variants = ridge_df_filtered['site'].nunique() if 'site' in ridge_df_filtered.columns else 0
    n_interactions = len(ridge_df_filtered)
    
    print("=" * 60)
    print("Interaction Visualization Report")
    print("=" * 60)
    print(f"Total WT variants:        {n_wt_variants}")
    print(f"Total clinical variants:  {n_clinical_variants}")
    print(f"Total interactions:       {n_interactions}")
    if filter_interactions is not None:
        print(f"Filtering applied:        Yes")
        print(f"Original interactions:    {len(ridge_df)}")
        print(f"Filtered interactions:    {n_interactions} ({100*n_interactions/len(ridge_df):.1f}%)")
    else:
        print(f"Filtering applied:        No")
    if include_all_labeled_pairs and not label_all and label_pos is not None:
        print(f"Include labeled pairs:     Yes (label_pos: {len(label_pos)} positions)")
    else:
        print(f"Include labeled pairs:     No")
    print("=" * 60)
    
    pdb_file = pdb_files_ref[0]
    pdb_file_to_use = prepare_pdb_file(pdb_file)
    chain, protein_length = get_protein_chain_and_length(pdb_file_to_use)

    # --- Mean scores and colors ---
    mean_scores_wt = get_mean_score_df(ridge_df, "wt_position", protein_length)
    mean_scores_clin = vep_prot.groupby(["Protein_position"])["VEP"].mean().reset_index(name="score").rename(columns={"Protein_position":"pos"})

    color_hex_wt = get_normed_colors_separate(mean_scores_wt)
    color_hex_clin = get_normed_colors_separate(mean_scores_clin)
    color_df_wt = make_color_df(mean_scores_wt, color_hex_wt, protein_length=protein_length)
    color_df_clin = make_color_df(mean_scores_clin, color_hex_clin)

    # --- NGLView coloring ---
    view = nv.show_file(pdb_file_to_use, default=False)

    selection_scheme_wt = make_selection_scheme(color_df_wt)
    selection_scheme_clin = make_selection_scheme(color_df_clin)
    nv.color.ColormakerRegistry.add_selection_scheme("mean_score_wt_scheme", selection_scheme_wt)
    nv.color.ColormakerRegistry.add_selection_scheme("mean_score_clin_scheme", selection_scheme_clin)

    view.add_cartoon(selection=struct_label, color='mean_score_wt_scheme', opacity=1.0)
    # Optionally show clinical as backbone (commented, as in original)
    # view.add_backbone(selection=struct_label, color='mean_score_clin_scheme', opacity=1.0)

    # Add colored backbone at each clinical_position and label selected residues
    for _, row in color_df_clin.iterrows():
        idx = row['residue_index']
        color = row['color']
        view.add_backbone(selection=str(idx)+".CA", color=color, radius=.5)
        if idx is not None and (label_all or idx in label_pos):
            view.add_label(selection=str(idx)+".CA", color="black")

    # Add labels for wt_position not overlapping with clin positions
    for _, row in color_df_wt.loc[~color_df_wt["residue_index"].isin(color_df_clin["residue_index"].unique())].dropna(subset=["raw_score"]).iterrows():
        idx = row['residue_index']
        if idx is not None and (label_all or idx in label_pos):
            color = row['color']
            view.add_label(selection=str(idx)+".CA", color="black", outlineColor="white", outlineWidth=0.6)

    # --- Add connections for top functional interaction pairs ---
    # Handle absolute value sorting if column name starts with "abs("
    if isinstance(sort_interactions, str) and sort_interactions.startswith("abs(") and sort_interactions.endswith(")"):
        # Extract column name from abs(column_name)
        col_name = sort_interactions[4:-1]
        if col_name in ridge_df_filtered.columns:
            # Create temporary column with absolute values for sorting
            ridge_df_filtered = ridge_df_filtered.copy()
            ridge_df_filtered['_temp_abs_sort'] = ridge_df_filtered[col_name].abs()
            sorted_df = ridge_df_filtered.sort_values('_temp_abs_sort', ascending=False)
            top_pairs = sorted_df.head(n_top_pairs) if n_top_pairs is not None else sorted_df
            top_pairs = top_pairs.drop(columns=['_temp_abs_sort'])
        else:
            raise ValueError(f"Column '{col_name}' not found in ridge_df for abs() sorting")
    else:
        sorted_df = ridge_df_filtered.sort_values(sort_interactions, ascending=False)
        top_pairs = sorted_df.head(n_top_pairs) if n_top_pairs is not None else sorted_df
    
    # If include_all_labeled_pairs is True and label_pos is provided, add all pairs between labeled positions
    n_labeled_pairs_added = 0
    if include_all_labeled_pairs and not label_all and label_pos is not None:
        # Find all interactions where both wt_position and clinical_position are in label_pos
        labeled_pairs = ridge_df_filtered[
            (ridge_df_filtered['wt_position'].isin(label_pos)) & 
            (ridge_df_filtered['clinical_position'].isin(label_pos))
        ]
        
        # Combine with top_pairs, removing duplicates (based on wt_position and clinical_position)
        if len(labeled_pairs) > 0:
            # Create a unique identifier for each pair
            top_pairs['_pair_id'] = (
                top_pairs['wt_position'].astype(str) + '_' + 
                top_pairs['clinical_position'].astype(str)
            )
            labeled_pairs['_pair_id'] = (
                labeled_pairs['wt_position'].astype(str) + '_' + 
                labeled_pairs['clinical_position'].astype(str)
            )
            
            # Get pairs that are not already in top_pairs
            new_pairs = labeled_pairs[~labeled_pairs['_pair_id'].isin(top_pairs['_pair_id'])]
            n_labeled_pairs_added = len(new_pairs)
            
            # Combine and remove the temporary column
            if len(new_pairs) > 0:
                top_pairs = pd.concat([top_pairs, new_pairs], ignore_index=True)
                top_pairs = top_pairs.drop(columns=['_pair_id'])
            else:
                top_pairs = top_pairs.drop(columns=['_pair_id'])
        else:
            # No labeled pairs found, just remove the temp column if it exists
            if '_pair_id' in top_pairs.columns:
                top_pairs = top_pairs.drop(columns=['_pair_id'])
    
    # Print final visualization summary
    print(f"Pairs being visualized:    {len(top_pairs)}")
    if n_labeled_pairs_added > 0:
        print(f"  - Top pairs:             {len(top_pairs) - n_labeled_pairs_added}")
        print(f"  - Labeled pairs added:   {n_labeled_pairs_added}")
    print("=" * 60)
    
    # Calculate color normalization based on all pairs being displayed
    cmap = matplotlib.colormaps['bwr']
    cabs = max(abs(top_pairs["interaction_strength_signed"].min()), 
               abs(top_pairs["interaction_strength_signed"].max()))
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-cabs, vcenter=0, vmax=cabs)

    for _, row in top_pairs.iterrows():
        interaction_strength = row["interaction_strength_signed"]
        color = get_interaction_color(interaction_strength, cmap, norm)
        atom_pair = [[f"{row['wt_position']}.CA", f"{row['clinical_position']}.CA"]]
        label_text = f"{interaction_strength:.2f}"
        view.add_distance(atom_pair=atom_pair, label_color=color, color=color, radius=1, label=label_text, label_visible=False)

    view.center()
    view.blur()
    view._remote_call("setSize", target="Widget", args=["1200px", "900px"])
    if show:
        return view
    else:
        return view


def write_html(view, path, frame_range=None, width='90vw', height='90vh'):
    from ipywidgets import Box

    box = Box([view])
    box.layout.width = width
    box.layout.height = height

    nglview.write_html(path, views=[box], frame_range=frame_range)
import nglview as nv
import numpy as np
import matplotlib
import pandas as pd
from Bio.PDB import PDBParser
import gzip
import shutil
import os

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
    show=True
):
    """
    Visualize functional interaction patterns and (optionally) clinical mean on structure using NGLView.
    Returns the nglview.View widget.
    """
    if label_pos is None:
        label_pos = [1684, 1687, 1706, 1710, 1729, 1722]
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
        if idx is not None and not idx in label_pos:
            continue
        view.add_label(selection=str(idx)+".CA", color="black")

    # Add labels for wt_position not overlapping with clin positions
    for _, row in color_df_wt.loc[~color_df_wt["residue_index"].isin(color_df_clin["residue_index"].unique())].dropna(subset=["raw_score"]).iterrows():
        idx = row['residue_index']
        if idx is not None and not idx in label_pos:
            continue
        color = row['color']
        view.add_label(selection=str(idx)+".CA", color="black", outlineColor="white", outlineWidth=0.6)

    # --- Add connections for top functional interaction pairs ---
    top_pairs = ridge_df.sort_values("interaction_strength", ascending=False).head(n_top_pairs)
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

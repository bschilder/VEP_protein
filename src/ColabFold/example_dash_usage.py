#!/usr/bin/env python3
"""
Example showing how to use the Dash version for dynamic contact map updates on hover.

The Dash version provides true interactivity where hovering over points
in the UMAP scatter plot will update the contact map display.
"""

from src.colabfold import create_dash_interactive_umap_app

# Assuming you have af2_meta and contact_maps ready
# Create the Dash app for dynamic contact map updates
app = create_dash_interactive_umap_app(
    af2_meta=af2_meta,
    contact_maps=contact_maps,
    color_col='sample_id',  # or any other column
    size_col='confidence',  # optional
    hover_cols=['sample_id', 'edit_distance', 'model_rank'],
    contact_map_params={
        'bin_size': 2,
        'pow': 4,
        'cmap': 'viridis',  # or 'plasma', 'inferno', etc.
        'normalize_scale': True
    },
    title="Interactive UMAP with Dynamic Contact Maps"
)

# Run the Dash app
if app:
    print("Starting Dash app...")
    print("Open your browser to: http://localhost:8050")
    print("Hover over points in the scatter plot to see contact maps update!")
    app.run_server(debug=True, port=8050)
else:
    print("Failed to create Dash app. Make sure Dash is installed:")
    print("pip install dash") 
#!/usr/bin/env python3
"""
Example showing how to use Bokeh for interactive UMAP visualization in Jupyter notebooks.

Bokeh provides excellent interactive capabilities and works seamlessly in Jupyter.
This version supports dynamic contact map updates when clicking on points.
"""

# In a Jupyter notebook, run this cell first:
# %matplotlib inline
# from bokeh.io import output_notebook
# output_notebook()

from src.colabfold import create_bokeh_interactive_umap
from bokeh.plotting import show
from bokeh.io import output_notebook

# Enable Bokeh in Jupyter notebook
output_notebook()

# Assuming you have af2_meta and contact_maps ready
# Create the Bokeh interactive visualization
layout = create_bokeh_interactive_umap(
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
    title="Interactive UMAP with Dynamic Contact Maps",
    width=1200,
    height=600
)

# Display the interactive plot
if layout:
    show(layout)
    print("Click on points in the scatter plot to see contact maps update!")
else:
    print("Failed to create Bokeh plot. Make sure Bokeh is installed:")
    print("pip install bokeh")

# Alternative: Save to HTML file
# from bokeh.io import save
# save(layout, "interactive_umap_bokeh.html") 
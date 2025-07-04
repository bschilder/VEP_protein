#!/usr/bin/env python3
"""
Comprehensive example showing all three interactive visualization options:

1. Plotly static HTML (default) - Good for sharing, but no dynamic updates
2. Dash web app - Full interactivity in web browser, requires server
3. Bokeh Jupyter - Full interactivity in Jupyter notebooks

Choose the option that best fits your use case!
"""

import numpy as np
import pandas as pd
from src.colabfold import create_interactive_umap_with_contact_maps

# Create sample data for demonstration
np.random.seed(42)

# Sample UMAP coordinates and metadata
n_samples = 50
af2_meta = pd.DataFrame({
    'umap_1': np.random.normal(0, 1, n_samples),
    'umap_2': np.random.normal(0, 1, n_samples),
    'sample_id': [f'sample_{i:03d}' for i in range(n_samples)],
    'edit_distance': np.random.randint(0, 10, n_samples),
    'model_rank': np.random.randint(1, 6, n_samples),
    'confidence': np.random.uniform(0.5, 1.0, n_samples),
    'category': np.random.choice(['A', 'B', 'C'], n_samples)
})

# Sample contact maps (random matrices for demonstration)
contact_maps = {}
for i in range(n_samples):
    size = np.random.randint(50, 200)
    # Create a random contact map with some structure
    contact_map = np.random.rand(size, size)
    # Make it symmetric
    contact_map = (contact_map + contact_map.T) / 2
    # Add some diagonal structure
    for j in range(size):
        for k in range(size):
            distance = abs(j - k)
            if distance < 10:
                contact_map[j, k] *= 2
    contact_maps[f'sample_{i:03d}'] = contact_map

print("Sample data created:")
print(f"- {len(af2_meta)} samples with UMAP coordinates")
print(f"- {len(contact_maps)} contact maps")
print(f"- Columns: {list(af2_meta.columns)}")

# Common parameters for all visualizations
common_params = {
    'color_col': 'category',
    'size_col': 'confidence',
    'hover_cols': ['sample_id', 'edit_distance', 'model_rank', 'confidence'],
    'contact_map_params': {
        'bin_size': 2,
        'pow': 4,
        'cmap': 'viridis',
        'normalize_scale': True
    },
    'title': "Interactive UMAP with Dynamic Contact Maps"
}

print("\n" + "="*60)
print("OPTION 1: Plotly Static HTML (Default)")
print("="*60)
print("Good for: Sharing static plots, embedding in documents")
print("Limitation: Contact map doesn't update on hover")

fig_static = create_interactive_umap_with_contact_maps(
    af2_meta=af2_meta,
    contact_maps=contact_maps,
    output_path='interactive_umap_static.html',
    **common_params
)

print("\n" + "="*60)
print("OPTION 2: Dash Web App")
print("="*60)
print("Good for: Full interactivity in web browser")
print("Requires: pip install dash")
print("Usage: app.run_server(debug=True, port=8050)")

try:
    app_dash = create_interactive_umap_with_contact_maps(
        af2_meta=af2_meta,
        contact_maps=contact_maps,
        use_dash=True,
        **common_params
    )
    if app_dash:
        print("Dash app created successfully!")
        print("To run: app_dash.run_server(debug=True, port=8050)")
        print("Then open: http://localhost:8050")
except ImportError:
    print("Dash not installed. Install with: pip install dash")

print("\n" + "="*60)
print("OPTION 3: Bokeh Jupyter Notebook")
print("="*60)
print("Good for: Interactive exploration in Jupyter notebooks")
print("Requires: pip install bokeh")
print("Usage: show(layout) in Jupyter cell")

try:
    layout_bokeh = create_interactive_umap_with_contact_maps(
        af2_meta=af2_meta,
        contact_maps=contact_maps,
        use_bokeh=True,
        **common_params
    )
    if layout_bokeh:
        print("Bokeh layout created successfully!")
        print("In Jupyter notebook, run:")
        print("from bokeh.plotting import show")
        print("from bokeh.io import output_notebook")
        print("output_notebook()")
        print("show(layout_bokeh)")
except ImportError:
    print("Bokeh not installed. Install with: pip install bokeh")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("1. Static Plotly: Saved as 'interactive_umap_static.html'")
print("2. Dash Web App: Run with app_dash.run_server(debug=True, port=8050)")
print("3. Bokeh Jupyter: Use show(layout_bokeh) in notebook")
print("\nChoose based on your needs:")
print("- Static sharing: Option 1")
print("- Web browser interactivity: Option 2")
print("- Jupyter notebook exploration: Option 3") 
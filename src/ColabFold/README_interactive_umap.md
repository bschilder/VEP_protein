# Interactive UMAP Visualization with Contact Maps

This module provides interactive visualization tools for AlphaFold2 embeddings with dynamic contact map display. When you hover over points in the UMAP scatter plot, the corresponding contact map is displayed in a subplot to the right.

## Features

- **Interactive 2D Scatter Plot**: UMAP embeddings with customizable colors, sizes, and hover information
- **Dynamic Contact Map Display**: Contact maps that update when hovering over different points
- **Multiple Visualization Options**: Static Plotly figures, interactive Dash web applications, and Bokeh Jupyter notebooks
- **Customizable Parameters**: Control contact map processing (binning, power transformation, normalization)
- **Export Capabilities**: Save interactive plots as HTML files

## Installation

### Required Packages

```bash
pip install pandas numpy plotly matplotlib seaborn
```

### For Full Interactivity (Optional)

```bash
# For Dash web applications
pip install dash

# For Bokeh Jupyter notebooks
pip install bokeh
```

## Quick Start

### Basic Usage

```python
from src.colabfold import create_interactive_umap_with_contact_maps

# Create interactive plot
fig = create_interactive_umap_with_contact_maps(
    af2_meta=af2_meta,           # Your DataFrame with umap_1, umap_2 columns
    contact_maps=contact_maps,    # Dictionary mapping IDs to contact map arrays
    color_col='sample_id',       # Column to use for point colors
    hover_cols=['sample_id', 'edit_distance', 'model_rank']  # Hover information
)

# Save to HTML file
fig.write_html('interactive_umap.html')
```

### Full Interactive Dash App

```python
from src.colabfold import create_dash_interactive_umap_app

# Create Dash app
app = create_dash_interactive_umap_app(
    af2_meta=af2_meta,
    contact_maps=contact_maps,
    color_col='population',
    size_col='confidence',
    hover_cols=['sample_id', 'edit_distance']
)

# Run the app
app.run_server(debug=True, port=8050)
# Open browser to http://localhost:8050
```

### Interactive Bokeh for Jupyter Notebooks

```python
from src.colabfold import create_interactive_umap_with_contact_maps
from bokeh.plotting import show
from bokeh.io import output_notebook

# Enable Bokeh in Jupyter
output_notebook()

# Create Bokeh interactive plot
layout = create_interactive_umap_with_contact_maps(
    af2_meta=af2_meta,
    contact_maps=contact_maps,
    use_bokeh=True,  # Use Bokeh instead of Plotly
    color_col='population',
    size_col='confidence',
    hover_cols=['sample_id', 'edit_distance']
)

# Display in notebook
show(layout)
```

## Data Format

### af2_meta DataFrame

Your DataFrame should contain at least these columns:

```python
af2_meta = pd.DataFrame({
    'umap_1': [...],        # UMAP first dimension
    'umap_2': [...],        # UMAP second dimension
    'sample_id': [...],     # Unique identifier for each sequence
    'edit_distance': [...], # Number of edits from reference (optional)
    'model_rank': [...],    # AlphaFold2 model rank (optional)
    'population': [...],    # Population information (optional)
    # ... other metadata columns
})
```

### contact_maps Dictionary

A dictionary mapping sequence identifiers to contact map arrays:

```python
contact_maps = {
    'SAMPLE_001_contact_map.npy': np.array(...),  # Contact map array
    'SAMPLE_002_contact_map.npy': np.array(...),  # Contact map array
    # ... more contact maps
}
```

The keys should contain the sample_id or be related to the sample_id in your af2_meta DataFrame for proper matching.

## Function Reference

### `create_interactive_umap_with_contact_maps()`

Main function for creating interactive visualizations.

**Parameters:**
- `af2_meta`: DataFrame with UMAP coordinates and metadata
- `contact_maps`: Dictionary of contact map arrays
- `x_col`: Column name for x-axis (default: 'umap_1')
- `y_col`: Column name for y-axis (default: 'umap_2')
- `color_col`: Column for point colors (optional)
- `size_col`: Column for point sizes (optional)
- `hover_cols`: List of columns for hover tooltip (optional)
- `contact_map_params`: Dictionary of contact map processing parameters
- `output_path`: Path to save HTML file (optional)
- `use_dash`: Whether to use Dash for full interactivity (default: False)
- `use_bokeh`: Whether to use Bokeh for Jupyter notebook interactivity (default: False)
- `figsize`: Figure size in pixels (default: (1200, 600))
- `title`: Plot title

**Returns:** Plotly figure object, Dash app, or Bokeh layout

### `create_dash_interactive_umap_app()`

Creates a full interactive Dash web application.

**Parameters:** Same as above (except output_path and use_dash)

**Returns:** Dash app object

### `process_contact_map_for_display()`

Processes contact maps for visualization.

**Parameters:**
- `contact_map`: Contact map array
- `bin_size`: Size of bins for matrix binning (default: 2)
- `pow`: Power to raise contact map to (default: 4)
- `normalize_scale`: Whether to normalize to [0,1] scale (default: True)
- `cmap`: Colormap name (default: 'gnuplot2')

**Returns:** Processed contact map array

## Contact Map Parameters

You can customize how contact maps are displayed:

```python
contact_map_params = {
    'bin_size': 2,           # Bin size for matrix reduction
    'pow': 4,                # Power transformation
    'cmap': 'gnuplot2',      # Colormap
    'normalize_scale': True  # Normalize to [0,1] scale
}
```

## Examples

### Example 1: Basic Interactive Plot

```python
import pandas as pd
import numpy as np
from src.colabfold import create_interactive_umap_with_contact_maps

# Load your data
af2_meta = pd.read_csv('your_umap_data.csv')
contact_maps = np.load('your_contact_maps.npy', allow_pickle=True).item()

# Create interactive plot
fig = create_interactive_umap_with_contact_maps(
    af2_meta=af2_meta,
    contact_maps=contact_maps,
    color_col='population',
    size_col='confidence',
    hover_cols=['sample_id', 'edit_distance', 'model_rank'],
    contact_map_params={
        'bin_size': 2,
        'pow': 4,
        'cmap': 'viridis',
        'normalize_scale': True
    },
    title="AlphaFold2 Embeddings with Contact Maps"
)

# Save and display
fig.write_html('interactive_umap.html')
fig.show()
```

### Example 2: Dash Web Application

```python
from src.colabfold import create_dash_interactive_umap_app

# Create Dash app
app = create_dash_interactive_umap_app(
    af2_meta=af2_meta,
    contact_maps=contact_maps,
    color_col='population',
    size_col='confidence',
    hover_cols=['sample_id', 'edit_distance'],
    contact_map_params={
        'bin_size': 1,  # No binning for clearer visualization
        'pow': 1,       # No power transformation
        'cmap': 'plasma',
        'normalize_scale': True
    },
    title="Interactive UMAP with Dynamic Contact Maps"
)

# Run the app
if app:
    app.run_server(debug=True, port=8050)
```

### Example 3: Bokeh Jupyter Notebook

```python
from src.colabfold import create_interactive_umap_with_contact_maps
from bokeh.plotting import show
from bokeh.io import output_notebook

# Enable Bokeh in Jupyter
output_notebook()

# Create Bokeh interactive plot
layout = create_interactive_umap_with_contact_maps(
    af2_meta=af2_meta,
    contact_maps=contact_maps,
    use_bokeh=True,
    color_col='population',
    size_col='confidence',
    hover_cols=['sample_id', 'edit_distance'],
    contact_map_params={
        'bin_size': 2,
        'pow': 4,
        'cmap': 'viridis',
        'normalize_scale': True
    },
    title="Interactive UMAP with Dynamic Contact Maps"
)

# Display in notebook
if layout:
    show(layout)
    print("Click on points to see contact maps update!")
```

### Example 4: Custom Contact Map Processing

```python
from src.colabfold import process_contact_map_for_display
import matplotlib.pyplot as plt

# Process contact map with different parameters
contact_map = your_contact_map_array

# Original
plt.subplot(2, 2, 1)
plt.imshow(contact_map, cmap='viridis')
plt.title('Original')

# Binned
contact_map_binned = process_contact_map_for_display(
    contact_map, bin_size=2, pow=1, normalize_scale=False
)
plt.subplot(2, 2, 2)
plt.imshow(contact_map_binned, cmap='viridis')
plt.title('Binned')

# Powered
contact_map_powered = process_contact_map_for_display(
    contact_map, bin_size=1, pow=2, normalize_scale=False
)
plt.subplot(2, 2, 3)
plt.imshow(contact_map_powered, cmap='viridis')
plt.title('Powered')

# Normalized
contact_map_norm = process_contact_map_for_display(
    contact_map, bin_size=1, pow=1, normalize_scale=True
)
plt.subplot(2, 2, 4)
plt.imshow(contact_map_norm, cmap='viridis')
plt.title('Normalized')

plt.tight_layout()
plt.show()
```

## Troubleshooting

### Common Issues

1. **Contact maps not updating on hover**: Make sure the keys in your `contact_maps` dictionary contain the sample_id or are related to the sample_id in your DataFrame.

2. **Dash app not working**: Install Dash with `pip install dash`

3. **Bokeh not working in Jupyter**: Install Bokeh with `pip install bokeh` and make sure to call `output_notebook()`

4. **Large contact maps causing slow performance**: Use `bin_size` parameter to reduce matrix size.

5. **Memory issues with many contact maps**: Consider processing contact maps in batches or using smaller bin sizes.

### Performance Tips

- Use `bin_size > 1` for large contact maps to improve performance
- Limit the number of hover columns for faster rendering
- Use `normalize_scale=False` if you want to preserve original contact map scales
- For very large datasets, consider using the Dash app or Bokeh instead of static HTML

## Integration with Existing Workflow

If you already have UMAP embeddings and contact maps computed:

```python
# Assuming you have:
# - af2_meta: DataFrame with umap_1, umap_2 columns
# - contact_maps: Dictionary of contact map arrays

# Create interactive visualization
fig = create_interactive_umap_with_contact_maps(
    af2_meta=af2_meta,
    contact_maps=contact_maps,
    color_col='your_color_column',
    hover_cols=['sample_id', 'other_metadata'],
    output_path='your_interactive_plot.html'
)
```

This will create an interactive HTML file that you can share with colleagues or embed in web applications. 
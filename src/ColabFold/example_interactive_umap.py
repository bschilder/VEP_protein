#!/usr/bin/env python3
"""
Example script demonstrating interactive UMAP visualization with contact maps.

This script shows how to create an interactive 2D scatterplot of AlphaFold2 embeddings
with dynamic contact map visualization that updates when hovering over points.

Requirements:
- pandas
- numpy
- plotly
- dash (for full interactivity)
- matplotlib
- seaborn

Usage:
    python example_interactive_umap.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.colabfold import (
    create_interactive_umap_with_contact_maps,
    create_dash_interactive_umap_app,
    process_contact_map_for_display
)

def create_sample_data(n_samples=50, n_residues=100):
    """
    Create sample data for demonstration purposes.
    
    Args:
        n_samples: Number of samples/sequences
        n_residues: Number of residues in each protein
        
    Returns:
        tuple: (af2_meta, contact_maps)
    """
    # Create sample UMAP coordinates
    np.random.seed(42)
    umap_1 = np.random.normal(0, 5, n_samples)
    umap_2 = np.random.normal(0, 5, n_samples)
    
    # Create sample metadata
    af2_meta = pd.DataFrame({
        'umap_1': umap_1,
        'umap_2': umap_2,
        'sample_id': [f'SAMPLE_{i:03d}' for i in range(n_samples)],
        'edit_distance': np.random.randint(0, 5, n_samples),
        'model_rank': np.random.randint(1, 6, n_samples),
        'population': np.random.choice(['AFR', 'EAS', 'EUR', 'SAS', 'AMR'], n_samples),
        'confidence': np.random.uniform(0.5, 1.0, n_samples)
    })
    
    # Create sample contact maps
    contact_maps = {}
    for i in range(n_samples):
        # Create a realistic contact map with some structure
        contact_map = np.random.exponential(0.1, (n_residues, n_residues))
        
        # Add some diagonal structure (contacts between nearby residues)
        for j in range(n_residues):
            for k in range(max(0, j-10), min(n_residues, j+11)):
                if j != k:
                    contact_map[j, k] += np.random.exponential(0.5)
        
        # Make it symmetric
        contact_map = (contact_map + contact_map.T) / 2
        
        # Set diagonal to 0
        np.fill_diagonal(contact_map, 0)
        
        # Store with sample ID as key
        sample_id = af2_meta.iloc[i]['sample_id']
        contact_maps[f"{sample_id}_contact_map.npy"] = contact_map
    
    return af2_meta, contact_maps

def example_basic_interactive():
    """Example of basic interactive plot without Dash."""
    print("Creating basic interactive plot...")
    
    # Create sample data
    af2_meta, contact_maps = create_sample_data(n_samples=30, n_residues=80)
    
    # Create interactive plot
    fig = create_interactive_umap_with_contact_maps(
        af2_meta=af2_meta,
        contact_maps=contact_maps,
        color_col='population',
        size_col='confidence',
        hover_cols=['sample_id', 'edit_distance', 'model_rank', 'population'],
        contact_map_params={
            'bin_size': 2,
            'pow': 2,
            'cmap': 'viridis',
            'normalize_scale': True
        },
        title="Sample AlphaFold2 Embeddings with Contact Maps"
    )
    
    # Save to HTML
    fig.write_html('example_interactive_umap.html')
    print("Interactive plot saved to: example_interactive_umap.html")
    
    return fig

def example_dash_interactive():
    """Example of full interactive Dash app."""
    print("Creating Dash interactive app...")
    
    # Create sample data
    af2_meta, contact_maps = create_sample_data(n_samples=20, n_residues=60)
    
    # Create Dash app
    app = create_dash_interactive_umap_app(
        af2_meta=af2_meta,
        contact_maps=contact_maps,
        color_col='population',
        size_col='confidence',
        hover_cols=['sample_id', 'edit_distance', 'model_rank'],
        contact_map_params={
            'bin_size': 1,  # No binning for clearer visualization
            'pow': 1,       # No power transformation
            'cmap': 'plasma',
            'normalize_scale': True
        },
        title="Interactive UMAP with Dynamic Contact Maps"
    )
    
    if app:
        print("Dash app created successfully!")
        print("To run the app, use: app.run_server(debug=True, port=8050)")
        print("Then open your browser to: http://localhost:8050")
        return app
    else:
        print("Failed to create Dash app. Make sure Dash is installed:")
        print("pip install dash")
        return None

def example_contact_map_processing():
    """Example of contact map processing functions."""
    print("Demonstrating contact map processing...")
    
    # Create a sample contact map
    np.random.seed(42)
    contact_map = np.random.exponential(0.1, (50, 50))
    
    # Add some structure
    for i in range(50):
        for j in range(max(0, i-5), min(50, i+6)):
            if i != j:
                contact_map[i, j] += np.random.exponential(0.5)
    
    contact_map = (contact_map + contact_map.T) / 2
    np.fill_diagonal(contact_map, 0)
    
    # Process with different parameters
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original
    axes[0, 0].imshow(contact_map, cmap='viridis')
    axes[0, 0].set_title('Original Contact Map')
    
    # Binned
    contact_map_binned = process_contact_map_for_display(
        contact_map, bin_size=2, pow=1, normalize_scale=False
    )
    axes[0, 1].imshow(contact_map_binned, cmap='viridis')
    axes[0, 1].set_title('Binned (bin_size=2)')
    
    # Powered
    contact_map_powered = process_contact_map_for_display(
        contact_map, bin_size=1, pow=2, normalize_scale=False
    )
    axes[1, 0].imshow(contact_map_powered, cmap='viridis')
    axes[1, 0].set_title('Powered (pow=2)')
    
    # Normalized
    contact_map_norm = process_contact_map_for_display(
        contact_map, bin_size=1, pow=1, normalize_scale=True
    )
    axes[1, 1].imshow(contact_map_norm, cmap='viridis')
    axes[1, 1].set_title('Normalized')
    
    plt.tight_layout()
    plt.savefig('contact_map_processing_example.png', dpi=150, bbox_inches='tight')
    print("Contact map processing example saved to: contact_map_processing_example.png")
    plt.show()

def main():
    """Main function to run all examples."""
    print("=== Interactive UMAP with Contact Maps Examples ===\n")
    
    # Example 1: Basic interactive plot
    print("1. Basic Interactive Plot")
    print("-" * 30)
    try:
        fig = example_basic_interactive()
        print("✓ Basic interactive plot created successfully\n")
    except Exception as e:
        print(f"✗ Error creating basic plot: {e}\n")
    
    # Example 2: Contact map processing
    print("2. Contact Map Processing")
    print("-" * 30)
    try:
        example_contact_map_processing()
        print("✓ Contact map processing example completed\n")
    except Exception as e:
        print(f"✗ Error in contact map processing: {e}\n")
    
    # Example 3: Dash interactive app
    print("3. Dash Interactive App")
    print("-" * 30)
    try:
        app = example_dash_interactive()
        if app:
            print("✓ Dash app created successfully\n")
        else:
            print("✗ Dash app creation failed\n")
    except Exception as e:
        print(f"✗ Error creating Dash app: {e}\n")
    
    print("=== Examples completed ===")
    print("\nTo use with your own data:")
    print("1. Prepare your af2_meta DataFrame with 'umap_1' and 'umap_2' columns")
    print("2. Prepare your contact_maps dictionary mapping file paths to arrays")
    print("3. Call create_interactive_umap_with_contact_maps() with your data")
    print("\nFor full interactivity, install Dash: pip install dash")

if __name__ == "__main__":
    main() 
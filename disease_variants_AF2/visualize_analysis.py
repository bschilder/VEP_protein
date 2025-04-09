import json
import matplotlib.pyplot as plt
import numpy as np

def load_analysis_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def create_visualization(results):
    # Extract data
    proteins = []
    mean_rmsds = []
    max_rmsds = []
    ref_access = []
    var_access = []
    mutations = []
    sec_structs = []
    
    for protein, data in results.items():
        if 'local_structure_analysis' in data:
            for mut_id, mut_data in data['local_structure_analysis'].items():
                proteins.append(protein)
                mean_rmsds.append(mut_data['local_rmsd_mean'])
                max_rmsds.append(mut_data['local_rmsd_max'])
                ref_access.append(mut_data['ref_solvent_accessibility'])
                var_access.append(mut_data['var_solvent_accessibility'])
                mutations.append(mut_id.replace('mutation_', ''))
                sec_structs.append(f"{mut_data['ref_secondary_structure']} → {mut_data['var_secondary_structure']}")

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 1, height_ratios=[1.5, 1])

    # Plot 1: RMSD values
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(len(proteins))
    width = 0.35
    
    # Create bars for mean and max RMSD
    mean_bars = ax1.bar(x - width/2, mean_rmsds, width, label='Mean RMSD', color='lightblue')
    max_bars = ax1.bar(x + width/2, max_rmsds, width, label='Max RMSD', color='lightcoral')
    
    # Customize RMSD plot
    ax1.set_ylabel('Local RMSD (Å)')
    ax1.set_title('Local Structural Changes Around Mutation Sites')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{p}\n{m}" for p, m in zip(proteins, mutations)], rotation=45, ha='right')
    ax1.legend()
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
    
    autolabel(mean_bars)
    autolabel(max_bars)

    # Plot 2: Accessibility and Secondary Structure
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(len(proteins))
    width = 0.35
    
    # Plot accessibility changes
    ax2.bar(x - width/2, ref_access, width, label='Reference', color='lightblue')
    ax2.bar(x + width/2, var_access, width, label='Variant', color='lightcoral')
    
    # Add secondary structure annotations
    for i, (ss, acc) in enumerate(zip(sec_structs, var_access)):
        ax2.text(i, max(ref_access[i], var_access[i]) + 5, ss,
                ha='center', va='bottom')
    
    # Customize accessibility plot
    ax2.set_ylabel('Solvent Accessibility (%)')
    ax2.set_title('Solvent Accessibility and Secondary Structure Changes')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{p}\n{m}" for p, m in zip(proteins, mutations)], rotation=45, ha='right')
    ax2.legend()

    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # Load and visualize results
    results = load_analysis_results('structure_comparison_results/advanced_structure_analysis.json')
    fig = create_visualization(results)
    
    # Save the plot
    plt.savefig('structure_comparison_results/structural_analysis_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as structural_analysis_visualization.png")

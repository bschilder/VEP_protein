import os
import sys
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import PDB
from Bio.PDB import *
import itertools
from datetime import datetime
import shutil

class StructureComparison:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, 'structure_comparison_results')
        self.ensure_results_dir()
        
    def ensure_results_dir(self):
        """Create results directory if it doesn't exist"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
    def get_pdb_files(self):
        """Get all variant PDB files"""
        pdb_files = []
        for variant_dir in sorted([d for d in os.listdir(self.base_dir) if d.startswith('variant')]):
            pdb_file = os.path.join(self.base_dir, variant_dir, f"{variant_dir}_structure.pdb")
            if os.path.exists(pdb_file):
                pdb_files.append(pdb_file)
        return pdb_files
    
    def calculate_rmsd(self, struct1, struct2):
        """Calculate RMSD between two structures"""
        atoms1 = [atom for atom in struct1.get_atoms() if atom.get_name() == 'CA']
        atoms2 = [atom for atom in struct2.get_atoms() if atom.get_name() == 'CA']
        
        coords1 = np.array([atom.get_coord() for atom in atoms1])
        coords2 = np.array([atom.get_coord() for atom in atoms2])
        
        super_imposer = PDB.Superimposer()
        super_imposer.set_atoms(atoms1, atoms2)
        super_imposer.apply(struct2.get_atoms())
        
        return super_imposer.rms

    def run_tm_align(self, pdb1, pdb2):
        """Run TM-align and parse results"""
        try:
            cmd = f"TMalign {pdb1} {pdb2}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # Parse TM-score from output
            for line in result.stdout.split('\n'):
                if "TM-score" in line and "normalized by length of Chain_1" in line:
                    tm_score = float(line.split('=')[1].strip().split()[0])
                    return tm_score
            return None
        except Exception as e:
            print(f"Error running TM-align: {str(e)}")
            return None

    def perform_analysis(self):
        """Perform both RMSD and TM-align analysis"""
        pdb_files = self.get_pdb_files()
        parser = PDB.PDBParser()
        
        # Initialize results dictionaries
        rmsd_results = {}
        tm_results = {}
        
        print(f"Found {len(pdb_files)} PDB files")
        print("\nPerforming structural alignments...")
        
        # Compare all pairs of structures
        for file1, file2 in itertools.combinations(pdb_files, 2):
            name1 = os.path.basename(file1).replace('_structure.pdb', '')
            name2 = os.path.basename(file2).replace('_structure.pdb', '')
            
            # Calculate RMSD
            struct1 = parser.get_structure('struct1', file1)
            struct2 = parser.get_structure('struct2', file2)
            rmsd = self.calculate_rmsd(struct1, struct2)
            
            # Store RMSD results
            if name1 not in rmsd_results:
                rmsd_results[name1] = {}
            if name2 not in rmsd_results:
                rmsd_results[name2] = {}
            rmsd_results[name1][name2] = rmsd
            rmsd_results[name2][name1] = rmsd
            
            # Calculate TM-score
            tm_score = self.run_tm_align(file1, file2)
            
            # Store TM-score results
            if name1 not in tm_results:
                tm_results[name1] = {}
            if name2 not in tm_results:
                tm_results[name2] = {}
            if tm_score is not None:
                tm_results[name1][name2] = tm_score
                tm_results[name2][name1] = tm_score
        
        return rmsd_results, tm_results

    def save_results(self, rmsd_results, tm_results):
        """Save results to files and create visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to DataFrames
        rmsd_df = pd.DataFrame(rmsd_results)
        tm_df = pd.DataFrame(tm_results)
        
        # Save raw results
        rmsd_df.to_csv(os.path.join(self.results_dir, f'rmsd_results_{timestamp}.csv'))
        tm_df.to_csv(os.path.join(self.results_dir, f'tm_results_{timestamp}.csv'))
        
        # Create heatmaps
        plt.figure(figsize=(12, 10))
        sns.heatmap(rmsd_df, annot=True, cmap='viridis', fmt='.3f', vmin=0, vmax=1)
        plt.title('RMSD Comparison Matrix (Å)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'rmsd_heatmap_{timestamp}.png'))
        plt.close()
        
        if not tm_df.empty:
            plt.figure(figsize=(12, 10))
            sns.heatmap(tm_df, annot=True, cmap='viridis', fmt='.3f', vmin=0, vmax=1)
            plt.title('TM-score Comparison Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'tm_score_heatmap_{timestamp}.png'))
            plt.close()
        
        # Save summary report
        with open(os.path.join(self.results_dir, f'analysis_summary_{timestamp}.txt'), 'w') as f:
            f.write("Structure Comparison Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("RMSD Analysis:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Min RMSD: {rmsd_df.min().min():.3f} Å\n")
            f.write(f"Max RMSD: {rmsd_df.max().max():.3f} Å\n")
            f.write(f"Mean RMSD: {rmsd_df.mean().mean():.3f} Å\n\n")
            
            if not tm_df.empty:
                f.write("TM-score Analysis:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Min TM-score: {tm_df.min().min():.3f}\n")
                f.write(f"Max TM-score: {tm_df.max().max():.3f}\n")
                f.write(f"Mean TM-score: {tm_df.mean().mean():.3f}\n")

    def backup_original_folder(self):
        """Create a backup of the original FoldXcan folder"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{self.base_dir}_backup_{timestamp}"
        shutil.copytree(self.base_dir, backup_dir)
        print(f"Created backup at: {backup_dir}")

def main():
    base_dir = '/Users/cc/Desktop/FoldXcan'
    
    # Create analysis object
    analysis = StructureComparison(base_dir)
    
    # Create backup
    analysis.backup_original_folder()
    
    # Perform analysis
    print("Starting structural analysis...")
    rmsd_results, tm_results = analysis.perform_analysis()
    
    # Save results
    print("\nSaving results...")
    analysis.save_results(rmsd_results, tm_results)
    
    print(f"\nAnalysis complete. Results saved in: {analysis.results_dir}")

if __name__ == '__main__':
    main()

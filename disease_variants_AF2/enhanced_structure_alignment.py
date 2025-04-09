import os
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from Bio import PDB
from Bio.PDB.Superimposer import Superimposer
import json
import subprocess
from pathlib import Path
import re

class StructureComparison:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, 'structure_comparison_results')
        self.parser = PDB.MMCIFParser(QUIET=True)
        self.pdb_parser = PDB.PDBParser(QUIET=True)
        self.ensure_results_dir()
        
    def convert_cif_to_pdb(self, cif_file):
        """Convert CIF file to PDB format"""
        try:
            # Parse the CIF file
            structure = self.parser.get_structure('structure', cif_file)
            
            # Create output PDB path
            output_pdb = os.path.join(os.path.dirname(cif_file), 
                                    os.path.splitext(os.path.basename(cif_file))[0] + '.pdb')
            
            # Write as PDB
            io = PDB.PDBIO()
            io.set_structure(structure)
            io.save(output_pdb)
            return output_pdb
        except Exception as e:
            print(f"Error converting {cif_file} to PDB: {str(e)}")
            return None
    
    def ensure_results_dir(self):
        """Create results directory if it doesn't exist"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def get_structure_files(self):
        """Get all structure files organized by gene"""
        structures = {}
        folds_dir = os.path.join(self.base_dir, 'folds')
        
        for gene in os.listdir(folds_dir):
            gene_dir = os.path.join(folds_dir, gene)
            if not os.path.isdir(gene_dir) or gene == 'terms_of_use.md':
                continue
            
            ref_dir = os.path.join(gene_dir, 'ref')
            var_dir = os.path.join(gene_dir, 'var')
            
            if os.path.exists(ref_dir) and os.path.exists(var_dir):
                ref_file = os.path.join(ref_dir, 'structure.pdb')
                var_file = os.path.join(var_dir, 'structure.pdb')
                
                if os.path.exists(ref_file) and os.path.exists(var_file):
                    structures[gene] = {
                        'ref': ref_file,
                        'var': var_file
                    }
        
        return structures
    
    def calculate_rmsd(self, ref_atoms, var_atoms):
        """Calculate RMSD between two sets of atoms"""
        if not ref_atoms or not var_atoms:
            return None, 0
            
        # Convert atoms to numpy arrays for coordinates
        ref_coords = np.array([atom.get_coord() for atom in ref_atoms])
        var_coords = np.array([atom.get_coord() for atom in var_atoms])
        
        # Find the shorter sequence
        min_length = min(len(ref_coords), len(var_coords))
        ref_coords = ref_coords[:min_length]
        var_coords = var_coords[:min_length]
        
        # Center the coordinates
        ref_center = np.mean(ref_coords, axis=0)
        var_center = np.mean(var_coords, axis=0)
        ref_coords = ref_coords - ref_center
        var_coords = var_coords - var_center
        
        # Calculate covariance matrix
        covariance_matrix = np.dot(var_coords.T, ref_coords)
        V, S, W = np.linalg.svd(covariance_matrix)
        
        # Ensure right-handed coordinate system
        if np.linalg.det(V) * np.linalg.det(W) < 0:
            V[:, -1] *= -1
        
        # Calculate rotation matrix
        rotation = np.dot(V, W)
        
        # Apply rotation
        var_coords_aligned = np.dot(var_coords, rotation)
        
        # Calculate RMSD
        rmsd = float(np.sqrt(np.mean(np.sum((ref_coords - var_coords_aligned) ** 2, axis=1))))
        
        return rmsd, min_length
    
    def compare_structures(self, ref_file, var_file):
        """Compare reference and variant structures using both BioPython and TM-align"""
        try:
            # Load structures using PDB parser
            ref_struct = self.pdb_parser.get_structure('ref', ref_file)
            var_struct = self.pdb_parser.get_structure('var', var_file)
            
            # Get first model
            ref_model = ref_struct[0]
            var_model = var_struct[0]
            
            # Get all CA atoms from both models
            ref_atoms = [atom for atom in ref_model.get_atoms() if atom.get_name() == 'CA']
            var_atoms = [atom for atom in var_model.get_atoms() if atom.get_name() == 'CA']
            
            # Calculate RMSD using BioPython
            bio_rmsd, bio_aligned_length = self.calculate_rmsd(ref_atoms, var_atoms)
            
            # Run TM-align analysis
            tm_results = self.run_tm_align(ref_file, var_file)
            
            results = {
                'biopython_rmsd': bio_rmsd,
                'biopython_aligned_length': bio_aligned_length
            }
            
            if tm_results:
                results.update({
                    'tm_score': tm_results['tm_score'],
                    'tm_aligned_length': tm_results['aligned_length'],
                    'tm_rmsd': tm_results['rmsd']
                })
            
            return results
        except Exception as e:
            print(f"Error comparing structures: {str(e)}")
            return None
    
    def analyze_all_structures(self):
        """Analyze all structures and generate comparison results"""
        structures = self.get_structure_files()
        results = {}
        
        for gene, files in structures.items():
            if files['ref'] and files['var']:
                print(f"Analyzing {gene}...")
                comparison = self.compare_structures(files['ref'], files['var'])
                if comparison:
                    results[gene] = comparison
                else:
                    print(f"Failed to analyze {gene}")
        
        if results:
            # Save results
            results_file = os.path.join(self.results_dir, 'structure_comparison_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Create visualization
            self.visualize_results(results)
        else:
            print("No valid comparisons were made")
        
        return results
    
    def visualize_results(self, results):
        """Create visualization of structural comparisons"""
        genes = list(results.keys())
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot RMSD comparison
        bio_rmsds = [results[gene]['biopython_rmsd'] for gene in genes]
        tm_rmsds = [results[gene].get('tm_rmsd', 0) for gene in genes]
        
        x = np.arange(len(genes))
        width = 0.35
        
        ax1.bar(x - width/2, bio_rmsds, width, label='BioPython RMSD')
        ax1.bar(x + width/2, tm_rmsds, width, label='TM-align RMSD')
        
        ax1.set_title('RMSD Comparison')
        ax1.set_xlabel('Gene')
        ax1.set_ylabel('RMSD (Å)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(genes, rotation=45)
        ax1.legend()
        
        # Plot TM-score
        tm_scores = [results[gene].get('tm_score', 0) for gene in genes]
        ax2.bar(genes, tm_scores)
        ax2.set_title('TM-scores')
        ax2.set_xlabel('Gene')
        ax2.set_ylabel('TM-score')
        ax2.set_xticklabels(genes, rotation=45)
        
        # Add value labels
        for i, score in enumerate(tm_scores):
            ax2.text(i, score, f'{score:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'structure_comparison_plot.png'))
        plt.close()

    def run_tm_align(self, ref_file, var_file):
        """Run TM-align and parse results"""
        try:
            # Use local TMalign executable
            cmd = f"./TMalign {ref_file} {var_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error running TMalign: {result.stderr}")
                return None
            
            tm_score_1 = None
            tm_score_2 = None
            rmsd = None
            aligned_length = None
            
            output_lines = result.stdout.split('\n')
            for i, line in enumerate(output_lines):
                if "TM-score" in line:
                    if "Chain_1" in line:
                        tm_score_1 = float(line.split('=')[1].strip().split()[0])
                    elif "Chain_2" in line:
                        tm_score_2 = float(line.split('=')[1].strip().split()[0])
                elif "Aligned length=" in line:
                    # Parse line like "Aligned length= 147, RMSD=   0.26"
                    parts = line.split(',')
                    aligned_length = int(parts[0].split('=')[1].strip())
                    rmsd = float(parts[1].split('=')[1].strip())
            
            if tm_score_1 and tm_score_2 and rmsd and aligned_length:
                # Use the average TM-score as recommended in the paper
                avg_tm_score = (tm_score_1 + tm_score_2) / 2
                return {
                    'tm_score': float(avg_tm_score),
                    'tm_score_chain1': float(tm_score_1),
                    'tm_score_chain2': float(tm_score_2),
                    'rmsd': float(rmsd),
                    'aligned_length': int(aligned_length)
                }
            return None
        except Exception as e:
            print(f"Error running TM-align: {str(e)}")
            return None
        except FileNotFoundError:
            print("TMalign command not found. Please ensure TMalign is installed and in your PATH")
            return None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    comparison = StructureComparison(base_dir)
    results = comparison.analyze_all_structures()
    
    # Print summary
    print("\nStructural Comparison Results:")
    print("-" * 50)
    for gene, data in results.items():
        print(f"{gene}:")
        print(f"  BioPython RMSD: {data['biopython_rmsd']:.2f} Å")
        print(f"  BioPython aligned residues: {data['biopython_aligned_length']}")
        if 'tm_score' in data:
            print(f"  TM-align RMSD: {data['tm_rmsd']:.2f} Å")
            print(f"  TM-align score: {data['tm_score']:.3f}")
            print(f"  TM-align aligned residues: {data['tm_aligned_length']}")
        print()

if __name__ == '__main__':
    main()

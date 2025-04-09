import os
import json
import numpy as np
from Bio import PDB
from Bio.PDB.Polypeptide import protein_letters_3to1
import subprocess
import matplotlib.pyplot as plt

class AdvancedStructureAnalysis:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, 'structure_comparison_results')
        self.pdb_parser = PDB.PDBParser(QUIET=True)
        self.window_size = 10  # Size of sliding window for local analysis
        
    def get_mutation_position(self, ref_file, var_file):
        """Get the position of the mutation by comparing sequences"""
        ref_struct = self.pdb_parser.get_structure('ref', ref_file)
        var_struct = self.pdb_parser.get_structure('var', var_file)
        
        # Get sequences
        ref_seq = ""
        var_seq = ""
        ref_residues = []
        var_residues = []
        
        for model in ref_struct:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        try:
                            ref_seq += protein_letters_3to1[residue.get_resname()]
                            ref_residues.append(residue)
                        except KeyError:
                            continue
                            
        for model in var_struct:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        try:
                            var_seq += protein_letters_3to1[residue.get_resname()]
                            var_residues.append(residue)
                        except KeyError:
                            continue
        
        # Find mutation position
        mutations = []
        for i, (ref, var) in enumerate(zip(ref_seq, var_seq)):
            if ref != var:
                mutations.append({
                    'position': i + 1,
                    'ref_aa': ref,
                    'var_aa': var,
                    'ref_residue': ref_residues[i],
                    'var_residue': var_residues[i]
                })
        
        return mutations
    
    def calculate_solvent_accessibility(self, structure, residue):
        """Calculate approximate solvent accessibility using atom distances"""
        try:
            # Get all atoms in the residue
            residue_atoms = list(residue.get_atoms())
            if not residue_atoms:
                return None
            
            # Calculate the center of mass
            center = np.mean([atom.get_coord() for atom in residue_atoms], axis=0)
            
            # Count number of neighboring atoms within 10Å
            neighbor_count = 0
            for other_residue in structure[0].get_residues():
                if other_residue != residue:
                    for atom in other_residue.get_atoms():
                        dist = np.linalg.norm(atom.get_coord() - center)
                        if dist < 10.0:  # 10Å cutoff
                            neighbor_count += 1
            
            # Convert to relative accessibility (inverse of neighbor count)
            max_neighbors = 100  # Approximate maximum number of neighbors
            accessibility = max(0, 100 * (1 - neighbor_count / max_neighbors))
            return accessibility
            
        except Exception as e:
            print(f"Error calculating accessibility: {str(e)}")
            return None

    def get_secondary_structure(self, structure, residue):
        """Predict secondary structure using simple geometric criteria"""
        try:
            # Get CA atoms of neighboring residues
            chain = residue.get_parent()
            residue_id = residue.id[1]
            
            # Get 4 residues before and after
            window = range(max(1, residue_id - 4), min(len(list(chain.get_residues())), residue_id + 5))
            ca_coords = []
            
            for i in window:
                try:
                    r = chain[(' ', i, ' ')]
                    ca = r['CA'].get_coord()
                    ca_coords.append(ca)
                except:
                    continue
            
            if len(ca_coords) < 5:
                return 'C'  # Coil
            
            # Calculate angles between consecutive CA atoms
            angles = []
            for i in range(len(ca_coords)-2):
                v1 = ca_coords[i+1] - ca_coords[i]
                v2 = ca_coords[i+2] - ca_coords[i+1]
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                angles.append(np.degrees(angle))
            
            # Simple criteria for structure assignment
            avg_angle = np.mean(angles)
            if 85 <= avg_angle <= 95:  # Close to 90 degrees
                return 'H'  # Helix
            elif avg_angle >= 110:  # More extended
                return 'E'  # Sheet
            else:
                return 'C'  # Coil
            
        except Exception as e:
            print(f"Error calculating secondary structure: {str(e)}")
            return None

    def calculate_local_rmsd(self, ref_file, var_file, mutation_info):
        """Calculate RMSD in sliding windows around mutation sites"""
        ref_struct = self.pdb_parser.get_structure('ref', ref_file)
        var_struct = self.pdb_parser.get_structure('var', var_file)
        
        local_analysis = {}
        for mutation in mutation_info:
            pos = mutation['position']
            start = max(0, pos - self.window_size)
            end = pos + self.window_size
            
            # Get CA atoms in window
            ref_atoms = []
            var_atoms = []
            
            for model in ref_struct:
                for chain in model:
                    for residue in chain:
                        if residue.id[1] >= start and residue.id[1] <= end:
                            if 'CA' in residue:
                                ref_atoms.append(residue['CA'])
                                
            for model in var_struct:
                for chain in model:
                    for residue in chain:
                        if residue.id[1] >= start and residue.id[1] <= end:
                            if 'CA' in residue:
                                var_atoms.append(residue['CA'])
            
            # Calculate local RMSD
            if len(ref_atoms) == len(var_atoms):
                ref_coords = np.array([atom.get_coord() for atom in ref_atoms])
                var_coords = np.array([atom.get_coord() for atom in var_atoms])
                
                # Center coordinates
                ref_center = np.mean(ref_coords, axis=0)
                var_center = np.mean(var_coords, axis=0)
                ref_coords = ref_coords - ref_center
                var_coords = var_coords - var_center
                
                # Optimal rotation matrix
                covariance_matrix = np.dot(var_coords.T, ref_coords)
                V, S, W = np.linalg.svd(covariance_matrix)
                rotation = np.dot(V, W)
                
                # Apply rotation and calculate RMSD
                var_coords_aligned = np.dot(var_coords, rotation)
                rmsd_values = np.sqrt(np.sum((ref_coords - var_coords_aligned) ** 2, axis=1))
                mean_rmsd = np.mean(rmsd_values)
                max_rmsd = np.max(rmsd_values)
                
                # Calculate solvent accessibility and secondary structure
                ref_asa = self.calculate_solvent_accessibility(ref_struct, mutation['ref_residue'])
                ref_ss = self.get_secondary_structure(ref_struct, mutation['ref_residue'])
                var_asa = self.calculate_solvent_accessibility(var_struct, mutation['var_residue'])
                var_ss = self.get_secondary_structure(var_struct, mutation['var_residue'])
                
                local_analysis[f"mutation_{mutation['ref_aa']}{mutation['position']}{mutation['var_aa']}"] = {
                    'local_rmsd_mean': float(mean_rmsd),
                    'local_rmsd_max': float(max_rmsd),
                    'window_size': self.window_size,
                    'residues_in_window': len(ref_atoms),
                    'ref_solvent_accessibility': ref_asa,
                    'var_solvent_accessibility': var_asa,
                    'ref_secondary_structure': ref_ss,
                    'var_secondary_structure': var_ss
                }
        
        return local_analysis
    
    def run_foldx(self, pdb_file):
        """Run FoldX stability analysis"""
        # This is a placeholder - actual FoldX implementation would go here
        # FoldX needs to be installed and available in the system
        try:
            # Example FoldX command (needs to be adjusted based on your FoldX installation)
            cmd = f"foldx --command=Stability --pdb={pdb_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                # Parse FoldX output
                # This needs to be adjusted based on actual FoldX output format
                return {'stability_score': 0.0}  # placeholder
            return None
        except Exception as e:
            print(f"Error running FoldX: {str(e)}")
            return None
    
    def analyze_all_structures(self):
        """Perform advanced analysis on all structures"""
        results = {}
        folds_dir = os.path.join(self.base_dir, 'folds')
        
        for gene in os.listdir(folds_dir):
            gene_dir = os.path.join(folds_dir, gene)
            if not os.path.isdir(gene_dir):
                continue
                
            ref_pdb = os.path.join(gene_dir, 'ref', 'structure.pdb')
            var_pdb = os.path.join(gene_dir, 'var', 'structure.pdb')
            
            if os.path.exists(ref_pdb) and os.path.exists(var_pdb):
                print(f"Analyzing {gene}...")
                
                # Get mutation information
                mutations = self.get_mutation_position(ref_pdb, var_pdb)
                
                # Calculate local RMSD
                local_analysis = self.calculate_local_rmsd(ref_pdb, var_pdb, mutations)
                
                # Run FoldX analysis
                ref_energy = self.run_foldx(ref_pdb)
                var_energy = self.run_foldx(var_pdb)
                
                results[gene] = {
                    'mutations': [{'position': m['position'], 
                                 'ref_aa': m['ref_aa'], 
                                 'var_aa': m['var_aa']} for m in mutations],
                    'local_structure_analysis': local_analysis,
                }
                
                if ref_energy and var_energy:
                    results[gene]['energy_analysis'] = {
                        'ref_stability': ref_energy['stability_score'],
                        'var_stability': var_energy['stability_score'],
                        'delta_stability': var_energy['stability_score'] - ref_energy['stability_score']
                    }
        
        # Save results
        output_file = os.path.join(self.results_dir, 'advanced_structure_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    analyzer = AdvancedStructureAnalysis(base_dir)
    results = analyzer.analyze_all_structures()
    
    # Print summary
    print("\nAdvanced Structural Analysis Results:")
    print("-" * 50)
    for gene, data in results.items():
        print(f"\n{gene}:")
        for mutation in data['mutations']:
            mut_id = f"{mutation['ref_aa']}{mutation['position']}{mutation['var_aa']}"
            print(f"  Mutation: {mut_id}")
            if f"mutation_{mut_id}" in data['local_structure_analysis']:
                local_data = data['local_structure_analysis'][f"mutation_{mut_id}"]
                print(f"    Local RMSD (±{local_data['window_size']} residues): mean = {local_data['local_rmsd_mean']:.3f} Å, max = {local_data['local_rmsd_max']:.3f} Å")
            if local_data['ref_solvent_accessibility'] is not None:
                print(f"    Solvent accessibility: {local_data['ref_solvent_accessibility']:.1f}% -> {local_data['var_solvent_accessibility']:.1f}%")
            if local_data['ref_secondary_structure'] is not None:
                print(f"    Secondary structure: {local_data['ref_secondary_structure']} -> {local_data['var_secondary_structure']}")
        
        if 'energy_analysis' in data:
            print(f"  Energy analysis:")
            print(f"    ΔΔG: {data['energy_analysis']['delta_stability']:.2f} kcal/mol")

if __name__ == '__main__':
    main()

import os
from Bio import PDB
from Bio.PDB import *
import itertools
import numpy as np

def load_structure(pdb_file):
    parser = PDB.PDBParser()
    structure_id = os.path.basename(pdb_file).replace('.pdb', '')
    return parser.get_structure(structure_id, pdb_file)

def calculate_rmsd(struct1, struct2):
    # Get all CA atoms from both structures
    atoms1 = [atom for atom in struct1.get_atoms() if atom.get_name() == 'CA']
    atoms2 = [atom for atom in struct2.get_atoms() if atom.get_name() == 'CA']
    
    # Get coordinates
    coords1 = np.array([atom.get_coord() for atom in atoms1])
    coords2 = np.array([atom.get_coord() for atom in atoms2])
    
    # Superimpose and get RMSD
    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(atoms1, atoms2)
    super_imposer.apply(struct2.get_atoms())
    
    return super_imposer.rms

def main():
    base_dir = '/Users/cc/Desktop/FoldXcan'
    
    # Get all variant PDB files
    pdb_files = []
    for variant_dir in sorted([d for d in os.listdir(base_dir) if d.startswith('variant')]):
        pdb_file = os.path.join(base_dir, variant_dir, f"{variant_dir}_structure.pdb")
        if os.path.exists(pdb_file):
            pdb_files.append(pdb_file)
    
    print(f"Found {len(pdb_files)} PDB files")
    print("\nStructural alignment results:")
    print("-" * 60)
    print(f"{'Structure 1':<15} {'Structure 2':<15} {'RMSD (Å)':<10}")
    print("-" * 60)
    
    # Compare all pairs of structures
    for file1, file2 in itertools.combinations(pdb_files, 2):
        struct1 = load_structure(file1)
        struct2 = load_structure(file2)
        
        rmsd = calculate_rmsd(struct1, struct2)
        
        name1 = os.path.basename(file1).replace('_structure.pdb', '')
        name2 = os.path.basename(file2).replace('_structure.pdb', '')
        print(f"{name1:<15} {name2:<15} {rmsd:.3f}")

if __name__ == '__main__':
    main()

from Bio import PDB
from Bio.PDB.Polypeptide import protein_letters_3to1
import os

def get_sequence(pdb_file):
    """Extract sequence from PDB file"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_file)
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:  # Only consider amino acid residues
                    try:
                        sequence += protein_letters_3to1[residue.get_resname()]
                    except KeyError:
                        # Skip non-standard amino acids
                        continue
    return sequence

def compare_sequences(ref_seq, var_seq):
    """Find differences between sequences"""
    differences = []
    for i, (ref, var) in enumerate(zip(ref_seq, var_seq)):
        if ref != var:
            differences.append((i+1, ref, var))  # 1-based position
    return differences

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folds_dir = os.path.join(base_dir, 'folds')
    
    for gene in os.listdir(folds_dir):
        gene_dir = os.path.join(folds_dir, gene)
        if not os.path.isdir(gene_dir):
            continue
            
        ref_pdb = os.path.join(gene_dir, 'ref', 'structure.pdb')
        var_pdb = os.path.join(gene_dir, 'var', 'structure.pdb')
        
        if os.path.exists(ref_pdb) and os.path.exists(var_pdb):
            print(f"\nAnalyzing {gene}:")
            ref_seq = get_sequence(ref_pdb)
            var_seq = get_sequence(var_pdb)
            
            if len(ref_seq) != len(var_seq):
                print(f"  Sequence length mismatch! Ref: {len(ref_seq)}, Var: {len(var_seq)}")
            
            differences = compare_sequences(ref_seq, var_seq)
            if differences:
                print("  Amino acid differences:")
                for pos, ref, var in differences:
                    print(f"    Position {pos}: {ref} -> {var}")
            else:
                print("  No amino acid differences found")

if __name__ == '__main__':
    main()

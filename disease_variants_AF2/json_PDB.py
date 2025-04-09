import os
from Bio.PDB import MMCIFParser, PDBIO

def convert_cif_to_pdb(cif_file, output_pdb):
    # Parse the CIF file
    parser = MMCIFParser()
    try:
        structure = parser.get_structure('structure', cif_file)
        
        # Write as PDB
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_pdb)
        return True
    except Exception as e:
        print(f"Error processing {cif_file}: {str(e)}")
        return False

def process_all_structures(base_dir):
    # Process each gene directory
    for gene_dir in os.listdir(base_dir):
        gene_path = os.path.join(base_dir, gene_dir)
        if not os.path.isdir(gene_path):
            continue
        
        # Process reference structure
        ref_dir = os.path.join(gene_path, 'ref')
        if os.path.exists(ref_dir):
            ref_cif = os.path.join(ref_dir, 'model_0.cif')
            if os.path.exists(ref_cif):
                output_pdb = os.path.join(ref_dir, 'structure.pdb')
                print(f"Processing {gene_dir} reference...")
                if convert_cif_to_pdb(ref_cif, output_pdb):
                    print(f"Created PDB file: {output_pdb}")
        
        # Process variant structure
        var_dir = os.path.join(gene_path, 'var')
        if os.path.exists(var_dir):
            var_cif = os.path.join(var_dir, 'model_0.cif')
            if os.path.exists(var_cif):
                output_pdb = os.path.join(var_dir, 'structure.pdb')
                print(f"Processing {gene_dir} variant...")
                if convert_cif_to_pdb(var_cif, output_pdb):
                    print(f"Created PDB file: {output_pdb}")

if __name__ == '__main__':
    base_dir = '/home/caom/VEP_protein/disease_variants/folds'
    process_all_structures(base_dir)

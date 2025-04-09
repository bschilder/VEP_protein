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

def process_all_variants(base_dir):
    # Process each variant directory
    for variant_dir in sorted([d for d in os.listdir(base_dir) if d.startswith('variant')]):
        variant_path = os.path.join(base_dir, variant_dir)
        if not os.path.isdir(variant_path):
            continue
            
        # Find all CIF files
        cif_files = [f for f in os.listdir(variant_path) if f.endswith('model_0.cif')]
        if not cif_files:
            continue
            
        cif_file = os.path.join(variant_path, cif_files[0])
        output_pdb = os.path.join(variant_path, f"{variant_dir}_structure.pdb")
        
        print(f"Processing {variant_dir}...")
        if convert_cif_to_pdb(cif_file, output_pdb):
            print(f"Created PDB file: {output_pdb}")

# Process all variants in the FoldXcan directory
if __name__ == '__main__':
    foldxcan_dir = '/Users/cc/Desktop/FoldXcan'
    process_all_variants(foldxcan_dir)

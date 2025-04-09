import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import ensembl_rest
from src import haplosaurus as hs
from src import get_gene_sequences as gs
from Bio.Seq import Seq
from pathlib import Path

# Dictionary of disease variants with their details
DISEASE_VARIANTS = {
    "Sickle Cell Anemia": {
        "gene": "HBB",
        "position": 6,
        "ref_aa": "E",  # Glu
        "var_aa": "V",  # Val
        "ref_nt": "A",
        "var_nt": "T",
        "nt_pos": 20,
        "rs_id": "rs334"
    },
    "Factor V Leiden": {
        "gene": "F5",
        "position": 506,
        "ref_aa": "R",  # Arg
        "var_aa": "Q",  # Gln
        "ref_nt": "G",
        "var_nt": "A",
        "nt_pos": 1691,
        "rs_id": "rs6025"
    },
    "Hereditary Hemochromatosis": {
        "gene": "HFE",
        "position": 282,
        "ref_aa": "C",  # Cys
        "var_aa": "Y",  # Tyr
        "ref_nt": "G",
        "var_nt": "A",
        "nt_pos": 845,
        "rs_id": "rs1800562"
    },
    "Achondroplasia": {
        "gene": "FGFR3",
        "position": 380,
        "ref_aa": "G",  # Gly
        "var_aa": "R",  # Arg
        "ref_nt": "G",
        "var_nt": "A",
        "nt_pos": 1138,
        "rs_id": "rs28931614"
    },
    "Alpha-1 Antitrypsin Deficiency": {
        "gene": "SERPINA1",
        "position": 342,
        "ref_aa": "E",  # Glu
        "var_aa": "K",  # Lys
        "ref_nt": "G",
        "var_nt": "A",
        "nt_pos": 1096,
        "rs_id": "rs28929474"
    },
    "Hereditary Fructose Intolerance": {
        "gene": "ALDOB",
        "position": 149,
        "ref_aa": "A",  # Ala
        "var_aa": "P",  # Pro
        "ref_nt": "G",
        "var_nt": "C",
        "nt_pos": 448,
        "rs_id": "rs1800546"
    },
    "Familial Hypercholesterolemia": {
        "gene": "APOB",
        "position": 3500,
        "ref_aa": "R",  # Arg
        "var_aa": "Q",  # Gln
        "ref_nt": "G",
        "var_nt": "A",
        "nt_pos": 10580,
        "rs_id": "rs5742904"
    },
    "Parkinson's Disease": {
        "gene": "LRRK2",
        "position": 2019,
        "ref_aa": "G",  # Gly
        "var_aa": "S",  # Ser
        "ref_nt": "G",
        "var_nt": "A",
        "nt_pos": 6055,
        "rs_id": "rs34637584"
    },
    "MEN2B": {
        "gene": "RET",
        "position": 918,
        "ref_aa": "M",  # Met
        "var_aa": "T",  # Thr
        "ref_nt": "T",
        "var_nt": "C",
        "nt_pos": 2753,
        "rs_id": None
    }
}

def get_variant_sequences(disease_name):
    """Get reference and variant sequences for a disease variant."""
    if disease_name not in DISEASE_VARIANTS:
        raise ValueError(f"Disease {disease_name} not found in database")
    
    variant = DISEASE_VARIANTS[disease_name]
    gene = variant["gene"]
    
    try:
        # Get gene info using the Ensembl REST client
        client = ensembl_rest.EnsemblClient()
        gene_info = client.symbol_lookup(species="homo_sapiens", symbol=gene)
        if not gene_info:
            print(f"No gene info found for {gene}")
            return None
        
        # Get canonical transcript
        tx_id = gene_info.get('canonical_transcript', '').split('.')[0]
        if not tx_id:
            print(f"No transcript found for {gene}")
            return None
            
        # Get haplotypes directly using haplosaurus
        haplotypes = hs.get_haplotypes(
            tx_ids=[tx_id],
            species="homo_sapiens",
            params=hs.config.PARAMS_HAPLOTYPES,
            force=True,
            cache_only=False,
            verbose=True
        )
        
        if tx_id not in haplotypes:
            print(f"No haplotypes found for {gene}")
            return None
            
        # Get reference sequences
        protein_haps = haplotypes[tx_id]['protein_haplotypes']
        cds_haps = haplotypes[tx_id]['cds_haplotypes']
        
        ref_protein = next((h['seq'] for h in protein_haps if ':REF' in h['name']), None)
        ref_cds = next((h['seq'] for h in cds_haps if ':REF' in h['name']), None)
        
        if not ref_protein or not ref_cds:
            print(f"No reference sequences found for {gene}")
            return None
        
        # Create variant sequences
        var_protein = list(ref_protein)
        var_protein[variant['position'] - 1] = variant['var_aa']
        var_protein = ''.join(var_protein)
        
        var_cds = list(ref_cds)
        var_cds[variant['nt_pos'] - 1] = variant['var_nt']
        var_cds = ''.join(var_cds)
        
        return {
            "disease": disease_name,
            "gene": gene,
            "transcript_id": tx_id,
            "reference_protein": ref_protein,
            "variant_protein": var_protein,
            "reference_cds": ref_cds,
            "variant_cds": var_cds
        }
        
    except Exception as e:
        print(f"Error getting sequences for {gene}: {str(e)}")
        return None

def write_results(results, output_dir="disease_variants"):
    """Write the results to individual FASTA files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for result in results:
        if result:
            gene = result['gene']
            disease = result['disease']
            
            # Write protein sequences
            protein_file = output_dir / f"{gene}_protein.fasta"
            with open(protein_file, 'w') as f:
                f.write(f">{gene}_REF\n{result['reference_protein']}\n")
                f.write(f">{gene}_VAR\n{result['variant_protein']}\n")
            
            # Write CDS sequences
            cds_file = output_dir / f"{gene}_cds.fasta"
            with open(cds_file, 'w') as f:
                f.write(f">{gene}_REF\n{result['reference_cds']}\n")
                f.write(f">{gene}_VAR\n{result['variant_cds']}\n")
            
            print(f"Wrote sequences for {gene} to {protein_file} and {cds_file}")

def main():
    """Main function to get sequences for all disease variants."""
    results = []
    for disease in DISEASE_VARIANTS:
        try:
            result = get_variant_sequences(disease)
            if result:
                results.append(result)
            else:
                print(f"Could not find sequences for {disease}")
        except Exception as e:
            print(f"Error processing {disease}: {str(e)}")
    
    write_results(results)
    print("Results have been written to disease_variants directory")

if __name__ == "__main__":
    main()

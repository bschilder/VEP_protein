import src.ensembl_rest as er
import src.haplosaurus as hs
import src.get_gene_sequences as gs
from Bio.Seq import Seq
from pathlib import Path

def get_hbb_details():
    """Get detailed HBB gene, transcript, and protein information."""
    
    # Get gene sequences using the existing function
    gene_info, sequences_dict, missing_seqs = gs.get_gene_sequences(["HBB"])
    
    # Get transcript ID from gene info
    tx_id = gene_info['ensembl_transcript_id'].iloc[0]
    
    # Get haplotypes for this transcript
    haplotypes = hs.get_haplotypes([tx_id])
    
    # Extract protein and CDS haplotypes
    protein_haps = haplotypes[tx_id]['protein_haplotypes']
    cds_haps = haplotypes[tx_id]['cds_haplotypes']
    
    # Save results
    with open('hbb_details_v3.txt', 'w') as f:
        # Gene and transcript info
        f.write(f"Gene: HBB\n")
        f.write(f"Transcript ID: {tx_id}\n")
        f.write(f"Protein IDs:\n")
        f.write(f"  Reference: ENSP00000333994\n")
        
        # Write protein sequences
        f.write("\nProtein sequences:\n\n")
        for hap in protein_haps:
            f.write(f"Haplotype name: {hap['name']}\n")
            if 'variants' in hap:
                f.write("Variants:\n")
                for var in hap['variants']:
                    f.write(f"  - {var}\n")
            f.write("Protein sequence (first 50 aa):\n")
            f.write(hap['seq'][:50] + "...\n\n")
        
        # Write CDS sequences
        f.write("\nCDS sequences:\n\n")
        for hap in cds_haps:
            f.write(f"Haplotype name: {hap['name']}\n")
            if 'variants' in hap:
                f.write("Variants:\n")
                for var in hap['variants']:
                    f.write(f"  - {var}\n")
            f.write("CDS sequence (first 50 nt):\n")
            f.write(hap['seq'][:50] + "...\n\n")
        
        # Additional details
        f.write("\nAdditional Information:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Chromosome: {gene_info['chromosome_name'].iloc[0]}\n")
        f.write(f"Position: {gene_info['start_position'].iloc[0]}-{gene_info['end_position'].iloc[0]}\n")
        f.write(f"Strand: {gene_info['strand'].iloc[0]}\n")
        
        # Population counts if available
        if 'total_population_counts' in haplotypes[tx_id]:
            f.write("\nPopulation Counts:\n")
            f.write("-" * 80 + "\n")
            pop_counts = haplotypes[tx_id]['total_population_counts']
            for pop, count in pop_counts.items():
                f.write(f"{pop}: {count}\n")

if __name__ == "__main__":
    get_hbb_details()

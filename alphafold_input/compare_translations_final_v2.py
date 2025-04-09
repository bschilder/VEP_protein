import src.ensembl_rest as er
import src.haplosaurus as hs
import src.get_gene_sequences as gs
from Bio.Seq import Seq
from pathlib import Path
import re

def get_variant_details(variant):
    """Extract position and amino acid change from variant notation."""
    match = re.match(r'(\d+)([A-Z])>([A-Z*])', variant)
    if match:
        pos = int(match.group(1))
        from_aa = match.group(2)
        to_aa = match.group(3)
        return pos, from_aa, to_aa
    return None, None, None

def find_matching_protein_variant(translated_seq, protein_haps):
    """Find matching protein variant based on exact sequence match."""
    translated_seq = translated_seq.rstrip('*')
    
    # First try exact match
    for protein_hap in protein_haps:
        if protein_hap['seq'].rstrip('*') == translated_seq:
            return protein_hap
    
    # If no exact match, try prefix match for truncated sequences
    if len(translated_seq) < 147:  # If sequence is truncated
        for protein_hap in protein_haps:
            if protein_hap['seq'].startswith(translated_seq):
                return protein_hap
    
    return None

def compare_translations():
    """Compare CDS translations with provided protein sequences."""
    
    # Get gene sequences
    gene_info, sequences_dict, missing_seqs = gs.get_gene_sequences(["HBB"])
    tx_id = gene_info['ensembl_transcript_id'].iloc[0]
    haplotypes = hs.get_haplotypes([tx_id])
    
    # Extract protein and CDS haplotypes
    protein_haps = haplotypes[tx_id]['protein_haplotypes']
    cds_haps = haplotypes[tx_id]['cds_haplotypes']
    
    # Keep track of matched protein variants
    matched_proteins = set()
    
    with open('translation_comparison_final_v2.txt', 'w') as f:
        f.write("HBB CDS Translation Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        for hap in cds_haps:
            f.write(f"Haplotype: {hap['name']}\n")
            f.write("-" * 80 + "\n")
            
            # Get CDS sequence
            cds_seq = hap['seq']
            
            # Create Seq object and translate
            seq_obj = Seq(cds_seq)
            translated = str(seq_obj.translate(table=1, to_stop=True))
            
            # Find matching protein variant
            matching_protein = None
            if hap['name'] == 'ENST00000335295:REF':
                # For reference sequence, match with reference protein
                matching_protein = next((p for p in protein_haps if p['name'].endswith(':REF')), None)
            else:
                matching_protein = find_matching_protein_variant(translated, protein_haps)
            
            f.write("\nCDS sequence (first 50 nt):\n")
            f.write(cds_seq[:50] + "...\n")
            
            f.write("\nTranslated protein (first 50 aa):\n")
            f.write(translated[:50] + "...\n")
            
            if matching_protein:
                matched_proteins.add(matching_protein['name'])
                f.write(f"\nMatching protein variant: {matching_protein['name']}\n")
                f.write("Protein sequence (first 50 aa):\n")
                f.write(matching_protein['seq'][:50] + "...\n")
                
                # Compare sequences
                translated = translated.rstrip('*')
                protein_seq = matching_protein['seq'].rstrip('*')
                
                if translated == protein_seq:
                    f.write("\nMatch Status: EXACT MATCH\n")
                else:
                    f.write("\nMatch Status: MISMATCH\n")
                    # Find where they differ
                    min_len = min(len(translated), len(protein_seq))
                    differences = []
                    for i in range(min_len):
                        if translated[i] != protein_seq[i]:
                            differences.append(f"Position {i+1}: {translated[i]} vs {protein_seq[i]}")
                    if differences:
                        f.write("First few differences:\n")
                        for diff in differences[:5]:
                            f.write(f"  {diff}\n")
                
                f.write(f"\nSequence lengths:\n")
                f.write(f"Translated CDS: {len(translated)} aa\n")
                f.write(f"Protein: {len(protein_seq)} aa\n")
            else:
                f.write("\nNo matching protein variant found\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
        
        # Check for unmatched protein variants
        f.write("\nUnmatched Protein Variants Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        unmatched_proteins = []
        for protein_hap in protein_haps:
            if protein_hap['name'] not in matched_proteins:
                unmatched_proteins.append(protein_hap)
        
        if unmatched_proteins:
            f.write(f"Found {len(unmatched_proteins)} unmatched protein variants:\n\n")
            for protein_hap in unmatched_proteins:
                f.write(f"Variant: {protein_hap['name']}\n")
                f.write("Sequence (first 50 aa):\n")
                f.write(protein_hap['seq'][:50] + "...\n\n")
                
                # Try to find CDS variants that produce similar sequences
                f.write("Looking for similar CDS variants:\n")
                for cds_hap in cds_haps:
                    seq_obj = Seq(cds_hap['seq'])
                    translated = str(seq_obj.translate(table=1, to_stop=True))
                    if translated.rstrip('*') == protein_hap['seq'].rstrip('*'):
                        f.write(f"Found matching CDS variant: {cds_hap['name']}\n")
                f.write("\n")
        else:
            f.write("All protein variants were matched with CDS variants!\n")

if __name__ == "__main__":
    compare_translations()

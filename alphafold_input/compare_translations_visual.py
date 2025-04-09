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

def create_sequence_visualization(ref_seq, var_seq, var_name):
    """Create HTML visualization of sequence differences."""
    html = []
    html.append("""
    <div style="font-family: monospace; margin: 20px; padding: 10px; background-color: #f8f9fa;">
        <div style="margin-bottom: 10px; font-weight: bold;">Variant: {}</div>
        <div style="display: flex;">
            <div style="width: 100px;">Reference:</div>
            <div style="letter-spacing: 1px;">
    """.format(var_name))
    
    # Add reference sequence with position markers
    for i, aa in enumerate(ref_seq):
        if (i + 1) % 10 == 1:
            html.append(f'<span style="color: #666; font-size: 0.8em;">{i+1}</span>')
        
        if i < len(var_seq) and aa != var_seq[i]:
            html.append(f'<span style="background-color: #ffcdd2;">{aa}</span>')
        else:
            html.append(aa)
        
        if (i + 1) % 10 == 0:
            html.append(' ')
    
    html.append("""
        </div>
    </div>
    <div style="display: flex;">
        <div style="width: 100px;">Variant:</div>
        <div style="letter-spacing: 1px;">
    """)
    
    # Add variant sequence
    for i, aa in enumerate(var_seq):
        if (i + 1) % 10 == 1:
            html.append(f'<span style="color: #666; font-size: 0.8em;">{i+1}</span>')
        
        if i < len(ref_seq) and aa != ref_seq[i]:
            html.append(f'<span style="background-color: #c8e6c9;">{aa}</span>')
        else:
            html.append(aa)
        
        if (i + 1) % 10 == 0:
            html.append(' ')
    
    html.append("""
            </div>
        </div>
    </div>
    <hr style="margin: 20px 0;">
    """)
    
    return ''.join(html)

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
    
    # Create text output file
    with open('translation_comparison_visual.txt', 'w') as f:
        f.write("HBB CDS Translation Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        # Create HTML visualization file
        with open('translation_visualization.html', 'w') as html_f:
            html_f.write("""
            <html>
            <head>
                <title>HBB Variant Analysis</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .legend { margin: 20px; padding: 10px; background-color: #f8f9fa; }
                    .legend-item { display: inline-block; margin-right: 20px; }
                </style>
            </head>
            <body>
                <h1>HBB Variant Analysis</h1>
                <div class="legend">
                    <div class="legend-item">
                        <span style="background-color: #ffcdd2; padding: 2px 5px;">Red</span>
                        = Reference amino acid at variant position
                    </div>
                    <div class="legend-item">
                        <span style="background-color: #c8e6c9; padding: 2px 5px;">Green</span>
                        = Variant amino acid
                    </div>
                </div>
            """)
            
            # Get reference sequence
            ref_protein = next((p for p in protein_haps if p['name'].endswith(':REF')), None)
            ref_seq = ref_protein['seq'] if ref_protein else ""
            
            for hap in cds_haps:
                f.write(f"Haplotype: {hap['name']}\n")
                f.write("-" * 80 + "\n")
                
                # Get CDS sequence and translate
                cds_seq = hap['seq']
                seq_obj = Seq(cds_seq)
                translated = str(seq_obj.translate(table=1, to_stop=True))
                
                # Find matching protein variant
                matching_protein = None
                if hap['name'] == 'ENST00000335295:REF':
                    matching_protein = ref_protein
                else:
                    matching_protein = find_matching_protein_variant(translated, protein_haps)
                
                # Write text output
                f.write("\nCDS sequence (first 50 nt):\n")
                f.write(cds_seq[:50] + "...\n")
                
                f.write("\nTranslated protein (first 50 aa):\n")
                f.write(translated[:50] + "...\n")
                
                if matching_protein:
                    matched_proteins.add(matching_protein['name'])
                    f.write(f"\nMatching protein variant: {matching_protein['name']}\n")
                    
                    # Create visualization for this variant
                    if matching_protein['name'] != 'ENSP00000333994:REF':
                        html_visualization = create_sequence_visualization(
                            ref_seq,
                            matching_protein['seq'],
                            matching_protein['name']
                        )
                        html_f.write(html_visualization)
                    
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
            
            html_f.write("</body></html>")

if __name__ == "__main__":
    compare_translations()

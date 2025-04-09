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

def format_sequence_with_numbers(seq, chunk_size=10):
    """Format sequence with position numbers."""
    result = []
    for i in range(0, len(seq), chunk_size):
        chunk = seq[i:i+chunk_size]
        pos = str(i + 1).rjust(4)
        result.append(f"{pos} {chunk}")
    return "\n".join(result)

def create_sequence_visualization(ref_cds, var_cds, ref_aa, var_aa, var_name):
    """Create HTML visualization of sequence differences."""
    html = []
    html.append(f"""
    <div style="font-family: monospace; margin: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3 style="color: #2c3e50; margin-bottom: 20px;">Variant: {var_name}</h3>
        
        <div style="margin-bottom: 20px;">
            <h4 style="color: #34495e;">CDS Sequences</h4>
            <div style="display: grid; grid-template-columns: 100px 1fr; gap: 10px; margin-bottom: 10px;">
                <div style="font-weight: bold;">Reference:</div>
                <div style="letter-spacing: 1px; white-space: pre-wrap;">{format_sequence_with_numbers(ref_cds)}</div>
            </div>
            <div style="display: grid; grid-template-columns: 100px 1fr; gap: 10px;">
                <div style="font-weight: bold;">Variant:</div>
                <div style="letter-spacing: 1px; white-space: pre-wrap;">{format_sequence_with_numbers(var_cds)}</div>
            </div>
        </div>

        <div style="margin-bottom: 20px;">
            <h4 style="color: #34495e;">Amino Acid Sequences</h4>
            <div style="display: grid; grid-template-columns: 100px 1fr; gap: 10px; margin-bottom: 10px;">
                <div style="font-weight: bold;">Reference:</div>
                <div style="letter-spacing: 1px; white-space: pre-wrap;">{format_sequence_with_numbers(ref_aa)}</div>
            </div>
            <div style="display: grid; grid-template-columns: 100px 1fr; gap: 10px;">
                <div style="font-weight: bold;">Variant:</div>
                <div style="letter-spacing: 1px; white-space: pre-wrap;">{format_sequence_with_numbers(var_aa)}</div>
            </div>
        </div>
        
        <div style="margin-top: 20px;">
            <h4 style="color: #34495e;">Differences</h4>
            <div style="background-color: white; padding: 10px; border-radius: 3px;">
    """)
    
    # Find CDS differences
    cds_diffs = []
    for i, (ref, var) in enumerate(zip(ref_cds, var_cds)):
        if ref != var:
            pos = i + 1
            cds_diffs.append(f"CDS position {pos}: {ref} → {var}")
    
    # Find AA differences
    aa_diffs = []
    for i, (ref, var) in enumerate(zip(ref_aa, var_aa)):
        if ref != var:
            pos = i + 1
            aa_diffs.append(f"AA position {pos}: {ref} → {var}")
    
    if cds_diffs:
        html.append("<div style='margin-bottom: 10px;'><b>CDS changes:</b></div>")
        for diff in cds_diffs[:5]:  # Show first 5 differences
            html.append(f"<div style='margin-left: 20px;'>{diff}</div>")
        if len(cds_diffs) > 5:
            html.append(f"<div style='margin-left: 20px;'>... and {len(cds_diffs)-5} more changes</div>")
    
    if aa_diffs:
        html.append("<div style='margin-top: 10px; margin-bottom: 10px;'><b>Amino acid changes:</b></div>")
        for diff in aa_diffs[:5]:  # Show first 5 differences
            html.append(f"<div style='margin-left: 20px;'>{diff}</div>")
        if len(aa_diffs) > 5:
            html.append(f"<div style='margin-left: 20px;'>... and {len(aa_diffs)-5} more changes</div>")
    
    if not cds_diffs and not aa_diffs:
        html.append("<div>No differences found (sequences are identical)</div>")
    
    html.append("""
            </div>
        </div>
    </div>
    <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
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
    
    # Get reference sequences
    ref_cds_hap = next((h for h in cds_haps if h['name'].endswith(':REF')), None)
    ref_protein = next((p for p in protein_haps if p['name'].endswith(':REF')), None)
    ref_cds = ref_cds_hap['seq'] if ref_cds_hap else ""
    ref_aa = ref_protein['seq'] if ref_protein else ""
    
    # Create HTML visualization file
    with open('translation_visualization_v2.html', 'w') as html_f:
        html_f.write("""
        <html>
        <head>
            <title>HBB Variant Analysis</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 40px; 
                    background-color: #f5f6fa;
                    color: #2c3e50;
                }
                .header {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 30px;
                }
                .header h1 {
                    margin: 0;
                }
                .header p {
                    margin: 10px 0 0 0;
                    opacity: 0.8;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>HBB Variant Analysis</h1>
                <p>Detailed comparison of CDS and protein sequences</p>
            </div>
        """)
        
        for hap in cds_haps:
            if hap['name'] == ref_cds_hap['name']:
                continue  # Skip reference sequence
                
            # Get CDS sequence and translate
            cds_seq = hap['seq']
            seq_obj = Seq(cds_seq)
            translated = str(seq_obj.translate(table=1, to_stop=True))
            
            # Find matching protein variant
            matching_protein = find_matching_protein_variant(translated, protein_haps)
            
            if matching_protein:
                matched_proteins.add(matching_protein['name'])
                
                # Create visualization for this variant
                html_visualization = create_sequence_visualization(
                    ref_cds,
                    cds_seq,
                    ref_aa,
                    matching_protein['seq'],
                    f"{hap['name']} → {matching_protein['name']}"
                )
                html_f.write(html_visualization)
        
        html_f.write("</body></html>")

if __name__ == "__main__":
    compare_translations()

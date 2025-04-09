from Bio import SeqIO
from pathlib import Path
import os
from collections import defaultdict

def load_disease_variants():
    """Load all disease variant sequences from FASTA files."""
    variants = defaultdict(dict)
    input_dir = Path("disease_variants")
    
    # Get all gene names from protein files
    genes = set(f.stem.split("_")[0] for f in input_dir.glob("*_protein.fasta"))
    
    for gene in genes:
        protein_file = input_dir / f"{gene}_protein.fasta"
        cds_file = input_dir / f"{gene}_cds.fasta"
        
        if protein_file.exists() and cds_file.exists():
            protein_records = list(SeqIO.parse(protein_file, "fasta"))
            cds_records = list(SeqIO.parse(cds_file, "fasta"))
            
            if len(protein_records) == 2 and len(cds_records) == 2:
                variants[gene] = {
                    'protein_ref': protein_records[0],
                    'protein_var': protein_records[1],
                    'cds_ref': cds_records[0],
                    'cds_var': cds_records[1]
                }
    
    return variants

def create_position_markers(seq_length):
    """Create position markers for the sequence."""
    markers = []
    for i in range(0, seq_length, 10):
        markers.append(str(i+1).rjust(10))
    return "Position: " + "".join(markers)

def create_sequence_html(seq_str, variant_pos=None, highlight_color="yellow"):
    """Create HTML for a sequence with optional highlighting."""
    if variant_pos is None:
        return seq_str
    
    # Convert to 0-based index
    pos = variant_pos - 1
    
    # Split sequence into before, variant, and after
    before = seq_str[:pos]
    variant = seq_str[pos]
    after = seq_str[pos+1:]
    
    return f"{before}<span style='background-color: {highlight_color}'>{variant}</span>{after}"

def create_gene_visualization(gene, variants, variant_info):
    """Create HTML visualization for a single gene's sequences."""
    if gene not in variants or gene not in variant_info:
        return ""
    
    data = variants[gene]
    info = variant_info[gene]
    protein_pos = info['position']
    cds_pos = info['nt_pos']
    
    html = []
    html.append(f"<h2>{gene} Sequences</h2>")
    html.append(f"<p>Variant: {info['ref_aa']}{protein_pos}{info['var_aa']} ({info['ref_nt']}>{info['var_nt']} at position {cds_pos})</p>")
    
    # Add CDS sequences
    html.append("<h3>CDS Sequences</h3>")
    html.append("<div class='sequence-block'>")
    html.append("<pre>")
    html.append(create_position_markers(len(str(data['cds_ref'].seq))))
    html.append(f"Reference: {create_sequence_html(str(data['cds_ref'].seq), cds_pos, '#ffcccb')}")
    html.append(f"Variant:   {create_sequence_html(str(data['cds_var'].seq), cds_pos, '#90EE90')}")
    html.append("</pre>")
    html.append("</div>")
    
    # Add protein sequences
    html.append("<h3>Protein Sequences</h3>")
    html.append("<div class='sequence-block'>")
    html.append("<pre>")
    html.append(create_position_markers(len(str(data['protein_ref'].seq))))
    html.append(f"Reference: {create_sequence_html(str(data['protein_ref'].seq), protein_pos, '#ffcccb')}")
    html.append(f"Variant:   {create_sequence_html(str(data['protein_var'].seq), protein_pos, '#90EE90')}")
    html.append("</pre>")
    html.append("</div>")
    
    return "\n".join(html)

def main():
    # Dictionary of variant information
    variant_info = {
        'HBB': {
            'position': 6,
            'ref_aa': 'E',
            'var_aa': 'V',
            'ref_nt': 'A',
            'var_nt': 'T',
            'nt_pos': 20
        },
        'F5': {
            'position': 506,
            'ref_aa': 'R',
            'var_aa': 'Q',
            'ref_nt': 'G',
            'var_nt': 'A',
            'nt_pos': 1691
        },
        'HFE': {
            'position': 282,
            'ref_aa': 'C',
            'var_aa': 'Y',
            'ref_nt': 'G',
            'var_nt': 'A',
            'nt_pos': 845
        },
        'FGFR3': {
            'position': 380,
            'ref_aa': 'G',
            'var_aa': 'R',
            'ref_nt': 'G',
            'var_nt': 'A',
            'nt_pos': 1138
        },
        'SERPINA1': {
            'position': 342,
            'ref_aa': 'E',
            'var_aa': 'K',
            'ref_nt': 'G',
            'var_nt': 'A',
            'nt_pos': 1096
        },
        'ALDOB': {
            'position': 149,
            'ref_aa': 'A',
            'var_aa': 'P',
            'ref_nt': 'G',
            'var_nt': 'C',
            'nt_pos': 448
        },
        'LRRK2': {
            'position': 2019,
            'ref_aa': 'G',
            'var_aa': 'S',
            'ref_nt': 'G',
            'var_nt': 'A',
            'nt_pos': 6055
        },
        'RET': {
            'position': 918,
            'ref_aa': 'M',
            'var_aa': 'T',
            'ref_nt': 'T',
            'var_nt': 'C',
            'nt_pos': 2753
        }
    }
    
    # Load all variants
    variants = load_disease_variants()
    
    # Create HTML file
    output_file = "disease_variants_visualization.html"
    with open(output_file, "w") as f:
        # Write HTML header
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Disease Variants Visualization</title>
    <style>
        body {
            font-family: monospace;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }
        h2 {
            color: #444;
            margin-top: 30px;
            background-color: #e0e0e0;
            padding: 10px;
            border-radius: 5px;
        }
        h3 {
            color: #666;
            margin-top: 20px;
        }
        .sequence-block {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin: 10px 0;
            overflow-x: auto;
        }
        pre {
            margin: 0;
            white-space: pre;
        }
        .legend {
            margin: 20px 0;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
        }
        .legend-item {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <h1>Disease Variants Visualization</h1>
    <div class="legend">
        <h3>Legend</h3>
        <div class="legend-item">Reference variant position: <span style="background-color: #ffcccb">highlighted in red</span></div>
        <div class="legend-item">Modified variant position: <span style="background-color: #90EE90">highlighted in green</span></div>
    </div>
""")
        
        # Add each gene's visualization
        for gene in sorted(variants.keys()):
            f.write(create_gene_visualization(gene, variants, variant_info))
        
        # Close HTML
        f.write("\n</body>\n</html>")
    
    print(f"HTML visualization has been saved to {output_file}")

if __name__ == "__main__":
    main()

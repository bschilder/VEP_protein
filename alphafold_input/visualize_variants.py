from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path
import os

def load_variants():
    """Load all HBB variant sequences from FASTA files."""
    variants = []
    input_dir = Path("alphafold_input")
    
    # Sort files to ensure consistent ordering
    fasta_files = sorted([f for f in input_dir.glob("HBB_variant_*.fasta")])
    
    for fasta_file in fasta_files:
        record = next(SeqIO.parse(fasta_file, "fasta"))
        # Extract variant number from filename
        variant_num = int(fasta_file.stem.split("_")[-1])
        variants.append((variant_num, record))
    
    return sorted(variants)

def create_alignment_visualization(variants):
    """Create a visualization of the sequence alignment with variant markers."""
    # Get reference sequence (variant 1)
    ref_seq = next(seq for num, seq in variants if num == 1).seq
    
    # Create the position marker line
    positions = []
    for i in range(0, len(ref_seq), 10):
        positions.append(str(i+1).rjust(10))
    position_line = "Position:  " + "".join(positions)
    
    # Create the position number line
    number_line = "           " + "".join("." * 9 + "|" for _ in range(len(ref_seq)//10 + 1))
    
    # Initialize the alignment visualization
    alignment = [position_line, number_line]
    
    # Add each sequence with markers for variants
    for num, record in variants:
        # Extract variant info from header
        header = record.description
        
        # Create sequence line with variant markers
        seq_str = str(record.seq)
        name = f"Variant {str(num).rjust(2)}"
        
        # Add the sequence line
        alignment.append(f"{name}:    {seq_str}")
        
        # Add marker line for variants
        if num > 1:  # Skip reference sequence
            markers = [" "] * len(seq_str)
            # Extract variant positions from header
            if "REF" not in header:  # Skip reference sequence
                variant_info = header.split("|")[-1].strip()
                if ":" in variant_info:
                    mutations = variant_info.split(":")[-1].split(",")
                    for mut in mutations:
                        # Parse position from mutation (format: 7E>V)
                        pos = int(''.join(c for c in mut if c.isdigit())) - 1
                        markers[pos] = "^"
            
            alignment.append("           " + "".join(markers))
    
    return "\n".join(alignment)

def main():
    # Load all variants
    variants = load_variants()
    
    # Create alignment visualization
    alignment_viz = create_alignment_visualization(variants)
    
    # Print the alignment
    print("\nHBB Variant Alignment Visualization")
    print("=" * 80)
    print("^ marks positions of amino acid changes\n")
    print(alignment_viz)
    
    # Save to file
    output_file = "hbb_variants_alignment.txt"
    with open(output_file, "w") as f:
        f.write("HBB Variant Alignment Visualization\n")
        f.write("=" * 80 + "\n")
        f.write("^ marks positions of amino acid changes\n\n")
        f.write(alignment_viz)
    
    print(f"\nAlignment saved to {output_file}")

if __name__ == "__main__":
    main()

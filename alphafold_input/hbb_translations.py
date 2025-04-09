from Bio.Seq import Seq
from pathlib import Path

def analyze_hbb_sequence():
    """Analyze HBB sequence translations."""
    
    # Read the sequence from the existing file
    with open('alphafold_input/HBB_sequences.txt', 'r') as f:
        content = f.read()
        # Extract sequence from the content
        seq_lines = content.split('\n')
        for i, line in enumerate(seq_lines):
            if line.startswith("=== Ensembl Protein Sequence ==="):
                sequence = seq_lines[i+1].strip()
                break
    
    # Create sequence object
    seq_obj = Seq(sequence)
    
    # Different translation attempts
    # 1. Direct translation
    direct_trans = seq_obj.translate()
    
    # 2. Find first ATG and translate from there
    start_codon_pos = sequence.upper().find('ATG')
    if start_codon_pos != -1:
        from_start = seq_obj[start_codon_pos:].translate(to_stop=True)
    else:
        from_start = "No ATG found"
    
    # 3. Find all ORFs (Open Reading Frames)
    def find_orfs(seq):
        orfs = []
        seq_len = len(seq)
        for frame in range(3):
            for i in range(frame, seq_len, 3):
                if str(seq[i:i+3]).upper() == 'ATG':
                    # Found start codon, look for stop codon
                    for j in range(i, seq_len, 3):
                        codon = str(seq[j:j+3]).upper()
                        if codon in ['TAA', 'TAG', 'TGA']:
                            orf = seq[i:j+3]
                            if len(orf) >= 30:  # Only keep ORFs of at least 10 amino acids
                                orfs.append({
                                    'start': i,
                                    'end': j+3,
                                    'sequence': str(orf),
                                    'translation': str(orf.translate())
                                })
                            break
        return orfs
    
    orfs = find_orfs(seq_obj)
    
    # Save results
    with open('hbb_translation_analysis.txt', 'w') as f:
        f.write("HBB Sequence Translation Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. Original DNA Sequence\n")
        f.write("-" * 80 + "\n")
        f.write(f"Length: {len(sequence)} bp\n")
        f.write("Sequence:\n")
        for i in range(0, len(sequence), 80):
            f.write(sequence[i:i+80] + '\n')
        
        f.write("\n2. Direct Translation (all frames)\n")
        f.write("-" * 80 + "\n")
        f.write(str(direct_trans) + '\n')
        
        f.write("\n3. Translation from first ATG to stop\n")
        f.write("-" * 80 + "\n")
        f.write(str(from_start) + '\n')
        
        f.write("\n4. All Possible Open Reading Frames (>= 10 amino acids)\n")
        f.write("-" * 80 + "\n")
        for i, orf in enumerate(orfs, 1):
            f.write(f"\nORF {i}:\n")
            f.write(f"Position: {orf['start']}-{orf['end']}\n")
            f.write(f"DNA sequence:\n{orf['sequence']}\n")
            f.write(f"Protein sequence:\n{orf['translation']}\n")

if __name__ == "__main__":
    analyze_hbb_sequence()

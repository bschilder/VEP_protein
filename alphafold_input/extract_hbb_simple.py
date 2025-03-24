import src.haplosaurus as hs
import src.ensembl_rest as er
from pathlib import Path
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

# Get HBB transcript ID
tx_id = "ENST00000335295"  # This is the canonical transcript for HBB

# Get transcript sequence
print("\nGetting transcript sequence...")
client = er.get_ensembl_client()
transcript_response = client.sequence_id(id=tx_id, species='homo_sapiens', object_type="transcript")

# Get protein sequence for comparison
print("\nGetting protein sequence...")
protein_response = client.sequence_id(id=tx_id, species='homo_sapiens', object_type="protein")
if transcript_response and 'seq' in transcript_response:
    transcript_seq = transcript_response['seq']
    print(f"\nTranscript sequence ({len(transcript_seq)} nucleotides):\n{transcript_seq}")
    
    # Find the first ATG and translate to first stop codon
    start_idx = transcript_seq.upper().find('ATG')
    if start_idx >= 0:
        coding_seq = transcript_seq[start_idx:]
        translated = str(Seq(coding_seq).translate(to_stop=True))
        print(f"\nTranslated from first ATG to stop ({len(translated)} amino acids):\n{translated}")

if protein_response and 'seq' in protein_response:
    protein_seq = protein_response['seq']
    print(f"\nEnsembl protein sequence ({len(protein_seq)} amino acids):\n{protein_seq}")

# Get the variant sequences for comparison
print("\nGetting variant sequences from haplosaurus...")

# Get haplotypes
print("\nGetting haplotypes for HBB...")
haplotypes = hs.get_haplotypes(
    tx_ids=[tx_id],
    species="homo_sapiens",
    force=True,
    cache_only=False,
    verbose=True
)

print("\nGetting protein sequences...")
seqs = hs.get_haplotype_seqs(haplotypes, verbose=True)

# Get protein IDs and names
protein_ids = {}
for hap in haplotypes[tx_id]['protein_haplotypes']:
    seq = hap['aligned_sequences'][1].rstrip('*')
    protein_ids[seq] = {
        'protein_id': hap.get('protein_id', 'Unknown'),
        'name': hap.get('name', 'Unknown')
    }

print("\nAvailable transcript IDs:", list(seqs.keys()))
if tx_id in seqs:
    print(f"\n{'='*80}")
    print("HBB Sequences for AlphaFold Prediction")
    print(f"{'='*80}")
    
    # Create output directory
    output_dir = Path("alphafold_input")
    output_dir.mkdir(exist_ok=True)
    
    # Get the sequence data
    sequences = seqs[tx_id]
    if isinstance(sequences, list):
        # Filter out truncated sequences (those with '*' in the middle)
        valid_sequences = [seq for seq in sequences if '*' not in seq[:-1]]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sequences = []
        for seq in valid_sequences:
            clean_seq = seq.rstrip('*')
            if clean_seq not in seen:
                seen.add(clean_seq)
                unique_sequences.append(clean_seq)
        
        print(f"\nFound {len(unique_sequences)} unique full-length sequences")
        
        # Save each sequence as FASTA
        for i, sequence in enumerate(unique_sequences, 1):
            # Get protein ID and haplotype info
            info = protein_ids.get(sequence, {'protein_id': 'Unknown', 'name': f'variant_{i}'})
            
            # Create a descriptive name
            name = f"variant_{i}"
            filename = f"HBB_{name}.fasta"
            filepath = output_dir / filename
            
            # Write FASTA file
            with open(filepath, 'w') as f:
                header = f">HBB {name}"
                header += f" | ENSP: {info['protein_id']}"
                header += f" | Haplotype: {info['name']}"
                f.write(header + "\n")
                # Write sequence in chunks of 80 characters
                for j in range(0, len(sequence), 80):
                    f.write(sequence[j:j+80] + '\n')
            
            print(f"\n{'-'*40}")
            print(f"Variant: {name}")
            print(f"ENSP ID: {info['protein_id']}")
            print(f"Haplotype name: {info['name']}")
            print(f"FASTA file: {filepath}")
            print(f"Sequence length: {len(sequence)} amino acids")
            print("First 50 aa:", sequence[:50] + "...")
    else:
        print("\nUnexpected sequence data format")
else:
    print("\nNo sequences found for HBB")

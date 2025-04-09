import src.ensembl_rest as er
import src.haplosaurus as hs
import src.get_gene_sequences as gs
from Bio.Seq import Seq
from pathlib import Path

def get_hbb_transcript():
    """Get HBB transcript and protein sequences using src functions."""
    
    # Get gene sequences using the existing function
    gene_info, sequences_dict, missing_seqs = gs.get_gene_sequences(["HBB"])
    
    # Get transcript ID from gene info
    tx_id = gene_info['ensembl_transcript_id'].iloc[0]
    
    # Get haplotypes for this transcript
    haplotypes = hs.get_haplotypes([tx_id])
    
    # Get sequences from haplotypes
    hap_seqs = hs.get_haplotype_seqs(haplotypes)
    
    # Save results
    with open('hbb_transcript_analysis.txt', 'w') as f:
        f.write("HBB Transcript Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        # Gene and transcript info
        f.write("1. Gene Information\n")
        f.write("-" * 80 + "\n")
        f.write(f"Gene Symbol: HBB\n")
        f.write(f"Ensembl Gene ID: {gene_info['ensembl_gene_id'].iloc[0]}\n")
        f.write(f"Ensembl Transcript ID: {tx_id}\n")
        f.write(f"Chromosome: {gene_info['chromosome_name'].iloc[0]}\n")
        f.write(f"Position: {gene_info['start_position'].iloc[0]}-{gene_info['end_position'].iloc[0]}\n")
        f.write(f"Strand: {gene_info['strand'].iloc[0]}\n\n")
        
        # Transcript sequence
        f.write("2. Transcript Sequences\n")
        f.write("-" * 80 + "\n")
        if tx_id in sequences_dict:
            tx_data = sequences_dict[tx_id]
            if 'transcript' in tx_data:
                f.write("\nTranscript sequence:\n")
                seq = tx_data['transcript']
                for i in range(0, len(seq), 80):
                    f.write(seq[i:i+80] + '\n')
                
                # Try translating
                seq_obj = Seq(seq)
                trans = seq_obj.translate(to_stop=True)
                f.write("\nTranslated sequence:\n")
                for i in range(0, len(str(trans)), 80):
                    f.write(str(trans)[i:i+80] + '\n')
            
            if 'protein' in tx_data:
                f.write("\nProtein sequence from Ensembl:\n")
                for name, seq in tx_data['protein']:
                    f.write(f"\n{name}:\n")
                    for i in range(0, len(seq), 80):
                        f.write(seq[i:i+80] + '\n')
        
        # Haplotype sequences
        f.write("\n3. Haplotype Sequences\n")
        f.write("-" * 80 + "\n")
        if tx_id in hap_seqs:
            for i, seq in enumerate(hap_seqs[tx_id], 1):
                f.write(f"\nHaplotype {i}:\n")
                for j in range(0, len(seq), 80):
                    f.write(seq[j:j+80] + '\n')

if __name__ == "__main__":
    get_hbb_transcript()

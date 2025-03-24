from src.get_gene_sequences import get_gene_sequences, format_for_alphafold

# Get BRCA1 sequences
genes = ["BRCA1"]
gene_info, sequences_dict, _ = get_gene_sequences(genes)

# Format and save sequences
if sequences_dict:
    for gene_symbol in genes:
        gene_rows = gene_info[gene_info['external_gene_name'] == gene_symbol]
        if not gene_rows.empty:
            tx_id = gene_rows.iloc[0]['ensembl_transcript_id']
            if tx_id in sequences_dict:
                # Format sequences for AlphaFold
                fasta_files = format_for_alphafold(
                    {tx_id: sequences_dict[tx_id]}, 
                    gene_symbol
                )
                print("\nFASTA files created:")
                for hap_name, filepath in fasta_files.items():
                    print(f"{hap_name}: {filepath}")

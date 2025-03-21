import src.ensembl_rest as er
import src.haplosaurus as hs
import pandas as pd
import time
import warnings
import ensembl_rest
import gzip
import json
from pathlib import Path

def get_gene_sequences(gene_symbols, species="homo_sapiens", max_retries=3, retry_delay=5):
    """
    Get gene information and their corresponding protein/CDS sequences.
    
    Args:
        gene_symbols (list): List of gene symbols (e.g., ["BRCA1", "BRCA2", "TP53"])
        species (str): Species name (default: "homo_sapiens")
        max_retries (int): Maximum number of retries for API calls
        retry_delay (int): Delay in seconds between retries
        
    Returns:
        tuple: (gene_data, sequences_dict, missing_seqs)
            - gene_data: DataFrame with gene information including all transcripts
            - sequences_dict: Dictionary with protein and CDS sequences
            - missing_seqs: List of transcript IDs with no sequences
    """
    # Initialize Ensembl client
    client = ensembl_rest.EnsemblClient()
    
    # Process each gene symbol
    gene_info_list = []
    transcript_ids = set()  # Use set to avoid duplicates
    
    for gene_symbol in gene_symbols:
        try:
            # Get gene info using symbol_lookup
            gene_info = client.symbol_lookup(species=species, symbol=gene_symbol)
            
            if gene_info:
                # Get canonical transcript ID
                canonical_transcript = gene_info.get('canonical_transcript')
                if canonical_transcript:
                    transcript_id = canonical_transcript.split('.')[0]  # Remove version number
                    gene_info_list.append({
                        'external_gene_name': gene_symbol,
                        'ensembl_gene_id': gene_info['id'],
                        'ensembl_transcript_id': transcript_id,
                        'chromosome_name': gene_info.get('seq_region_name'),
                        'start_position': gene_info.get('start'),
                        'end_position': gene_info.get('end'),
                        'strand': gene_info.get('strand'),
                        'version': gene_info.get('version'),
                        'biotype': gene_info.get('biotype')
                    })
                    transcript_ids.add(transcript_id)
        except Exception as e:
            warnings.warn(f"Error getting info for gene {gene_symbol}: {str(e)}")
    
    # Create DataFrame from gene info
    gene_info_df = pd.DataFrame(gene_info_list)
    
    # 2. Get haplotypes for all transcripts with retry logic
    sequences_dict = {}
    missing_seqs = []
    
    if transcript_ids:
        transcript_ids = list(transcript_ids)  # Convert set to list
        haplotypes = hs.get_haplotypes(
            tx_ids=transcript_ids,
            species=species,
            params=hs.config.PARAMS_HAPLOTYPES,
            use_protein_ids=False,
            verbose=True,
            force=False,  # Use cached data if available
            cache_only=True  # Only use cached data
        )
        
        # Process each transcript's sequences
        for tx_id, haplotype_data in haplotypes.items():
            sequences_dict[tx_id] = {
                'protein': [],
                'protein_ids': {'ref': None, 'hap1': None},  # Store reference and first haplotype IDs
                'cds': []
            }
            
            # Get protein sequences and IDs
            if 'protein_haplotypes' in haplotype_data:
                for i, hap in enumerate(haplotype_data['protein_haplotypes']):
                    name = hap.get('name', 'unknown')
                    protein_seq = hap.get('seq')
                    
                    # Extract protein ID from name (format is usually ENSP00000350283:REF)
                    protein_id = name.split(':')[0] if ':' in name else None
                    
                    if protein_seq:
                        sequences_dict[tx_id]['protein'].append((name, protein_seq))
                        
                        # Store reference and first haplotype IDs
                        if ':REF' in name:
                            sequences_dict[tx_id]['protein_ids']['ref'] = protein_id
                        elif i == 1:  # First non-reference haplotype
                            sequences_dict[tx_id]['protein_ids']['hap1'] = protein_id
            
            # Get CDS sequences
            if 'cds_haplotypes' in haplotype_data:
                for hap in haplotype_data['cds_haplotypes']:
                    name = hap.get('name', 'unknown')
                    cds_seq = hap.get('seq')
                    if cds_seq:
                        sequences_dict[tx_id]['cds'].append((name, cds_seq))
            
            # If no sequences found, add to missing
            if not sequences_dict[tx_id]['protein'] and not sequences_dict[tx_id]['cds']:
                missing_seqs.append(tx_id)
                del sequences_dict[tx_id]
    
    return gene_info_df, sequences_dict, missing_seqs

# Example usage:
if __name__ == "__main__":
    genes = ["BRCA1", "BRCA2", "TP53"]
    gene_info, sequences_dict, missing_seqs = get_gene_sequences(genes)

    # Print gene information
    print("\nGene Information:")
    print(gene_info.to_string())

    # Print sequence information for all genes
    if sequences_dict:
        for gene_symbol in genes:
            # Find the transcript ID for this gene from gene_info
            gene_rows = gene_info[gene_info['external_gene_name'] == gene_symbol]
            if not gene_rows.empty:
                gene_row = gene_rows.iloc[0]
                tx_id = gene_row['ensembl_transcript_id']
                
                if tx_id in sequences_dict:
                    print(f"\n{'='*80}")
                    print(f"Gene: {gene_symbol}")
                    print(f"Transcript ID: {tx_id}")
                    print(f"Protein IDs:")
                    print(f"  Reference: {sequences_dict[tx_id]['protein_ids']['ref']}")
                    print(f"  Haplotype 1: {sequences_dict[tx_id]['protein_ids']['hap1']}")
                    
                    seq_data = sequences_dict[tx_id]
                    
                    # Print only reference and first haplotype sequences
                    if seq_data['protein']:
                        print("\nProtein sequences:")
                        for name, seq in seq_data['protein'][:2]:  # Only first two sequences
                            print(f"\nHaplotype name: {name}")
                            print(f"Protein sequence (first 50 aa):")
                            print(seq[:50] + "...")
                    
                    # Print only reference and first haplotype CDS
                    if seq_data['cds']:
                        print("\nCDS sequences:")
                        for name, seq in seq_data['cds'][:2]:  # Only first two sequences
                            print(f"\nHaplotype name: {name}")
                            print(f"CDS sequence (first 50 nt):")
                            print(seq[:50] + "...")
                else:
                    print(f"\nNo sequences found for {gene_symbol} (transcript {tx_id})")
            else:
                print(f"\nNo information found for gene {gene_symbol}")

    # Print missing sequences
    if missing_seqs:
        print("\nTranscripts with no sequences found:")
        for tx_id in missing_seqs:
            print(f"- {tx_id}")

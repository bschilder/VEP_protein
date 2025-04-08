 # %%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.haplosaurus as hs
import src.biopython as bp
 # %%
haplotypes = hs.get_haplotypes(
    # max_tx_ids=5,
    cache_only=True
    )

 # %%
hap_seqs = hs.get_haplotype_seqs(
    haplotypes, 
    aligned=2,
    add_haplotype_names=True,
    )

 # %%
# Get haplotype reference sequences
haplotypes_ref = hs.get_haplotype_ref(haplotypes)

 # %%
# Get haplotype sequences
hap_seqs_ref, missing_seqs = hs.get_haplotype_seqs(
    haplotypes_ref,
    return_missing=True,
    # aligned=2,
    # add_haplotype_names=True,
    verbose=True)


(len(haplotypes), 
 len(haplotypes_ref), 
 len(hap_seqs_ref),
 len(missing_seqs))

# %%
# Within each transcript, check if the haplotype sequence is the same as the reference sequence
tx_ref_len = {}
for tx_id, tx_hap_seqs in hap_seqs_ref.items():
    tx_ref_len[tx_id] = []
    processed_seq = bp.preprocess_sequence(tx_hap_seqs)
    tx_ref_len[tx_id].append(len(processed_seq))


tx_len = {}
for tx_id, tx_hap_seqs in hap_seqs.items():
    tx_len[tx_id] = []
    for seq_name, seqs in tx_hap_seqs:
        processed_seq = bp.preprocess_sequence(seqs[0])
        tx_len[tx_id].append(len(processed_seq))


tx_len_correct = {tx_id:tx_ref_len[tx_id] == list(set(tx_len[tx_id])) for tx_id in tx_ref_len.keys()}
print(f"{sum(tx_len_correct.values()) / len(tx_len_correct)*100}% of transcripts have all their haplotypes with the expected length")



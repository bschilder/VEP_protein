import src.haplosaurus as hs
import src.biopython as bp

haplotypes = hs.get_haplotypes(
    max_tx_ids=5,
    cache_only=True
    )

# Get haplotype reference sequences
haplotypes_ref = hs.get_haplotype_ref(haplotypes)

# Get haplotype sequences
haplotypes_ref_seqs, missing_seqs = hs.get_haplotype_seqs(
    haplotypes_ref,
    return_missing=True,
    verbose=True)


(len(haplotypes), 
 len(haplotypes_ref), 
 len(haplotypes_ref_seqs),
 len(missing_seqs))

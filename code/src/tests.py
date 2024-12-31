

## Insertion
ref_seq = list("ACCGC")
seq = ref_seq.copy()
allele = "TTT"
ref_start = 1
ref_stop = 3
rec_ref = ref_seq[ref_start:ref_stop]

length_diff = len(allele) - (ref_stop - ref_start) 

print(length_diff)
if length_diff > 0: #Insertion
    print("Insertion")
    seq[ref_start:ref_stop] = [allele] + ['-'] * (length_diff-1)
elif length_diff < 0: #Deletion
    print("Deletion")
    seq[ref_start:ref_stop] = [allele] + ['-'] * length_diff
else: #Substitution
    print("Substitution")
    seq[ref_start:ref_stop] = list(allele)

print(ref_seq)
print(seq)  
print(rec_ref)
len(seq) == len(ref_seq)


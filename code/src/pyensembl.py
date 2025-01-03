import sys
sys.path.append("code")
from src.utils import as_list, intersect, add_codon_buffer, is_VariantFile
from src.variant_annotation import filter_variants

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
import os
from Bio.Seq import Seq
import numpy as np

def get_db(release=111, species="homo_sapiens"):
    from pyensembl import EnsemblRelease
    db = EnsemblRelease(release=release, species=species) 
    return db

def get_ids(objects):
    return [x.id for x in objects]


def transcript_to_gene(transcript_ids,
                       db=None):
    transcript_ids = as_list(transcript_ids)
    if db is None:
        db = get_db()
    return {k:db.gene_name_of_transcript_id(k) for k in transcript_ids}

def get_mane_transcripts(protein_coding_only=True,
                         db_only=True,
                         db=None):
    import pandas as pd
    mane = pd.read_csv("https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/MANE.GRCh38.v1.4.summary.txt.gz", 
                       sep="\t")
    mane['TranscriptId'] = mane['Ensembl_nuc'].str.split('.').str[0]
    if protein_coding_only:
        mane = mane[mane['Ensembl_prot'].notna()]
    if db_only:
        if db is None:
            db = get_db()
        # if protein_coding_only:
        #     ids = get_protein_coding_transcripts(db, ids_only=True)
        # else: 
        ids = db.transcript_ids()
        mane = mane.loc[mane['TranscriptId'].isin(ids)]
    print(f"{len(mane)} MANE transcripts found.")
    return mane

def get_canonical_transcripts(db=None, 
                              protein_coding_only=True):
    import pandas as pd
    knownCanonical = pd.read_csv("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/knownCanonical.txt.gz", 
                                 sep="\t", 
                                 header=None)
    # Extract the TranscriptId from the knownCanonical file
    knownCanonical['TranscriptId'] = knownCanonical[4].str.split('.').str[0]
    if protein_coding_only:
        coding_transcripts = get_protein_coding_transcripts(db)
        knownCanonical = knownCanonical[knownCanonical['TranscriptId'].isin(coding_transcripts)]
    return knownCanonical

def filter_transcripts(transcripts, 
                       id=[],
                       biotype=[],
                       contig=[],
                       max_transcripts=None,
                       verbose=True):
    from tqdm.auto import tqdm
    contig = as_list(contig)
    biotype = as_list(biotype)
    if id is None:
        id = []
    id = as_list(id)
    if len(contig)>0:
        contig = [str(x).replace("chr", "") for x in contig]
        transcripts = [x for x in tqdm(transcripts, desc="Filtering by contig", disable=not verbose) if x.contig in contig]
    if len(biotype)>0:
        transcripts = [x for x in tqdm(transcripts, desc="Filtering by biotype", disable=not verbose) if x.biotype in biotype]
    if len(id)>0:
        ids = intersect([x.id for x in transcripts], id)
        transcripts = [x for x in tqdm(transcripts, desc="Filtering by id", disable=not verbose) if x.id in ids]
    if max_transcripts is not None:
        transcripts = transcripts[:max_transcripts]
    if verbose:
        print(f"{len(transcripts)} transcripts remaining.")
    return transcripts

def get_protein_coding_transcripts(db=None, 
                                   ids_only=False,
                                   verbose=True):
    from tqdm.auto import tqdm
    if db is None:
        db = get_db()
    protein_ids = db.protein_ids()
    transcripts = [db.transcript_by_protein_id(x) for x in tqdm(protein_ids,"Retrieving transcripts.", disable=not verbose)]
    # transcripts = filter_transcripts(transcripts, biotype=['protein_coding'])
    if ids_only:
        return [x.id for x in transcripts]
    else:
        return transcripts

def get_transcripts(db, 
                    biotypes=[], 
                    complete=True):
    print("Getting all transcripts.")
    def get_transcripts_inner():
        return db.transcripts()
    all_transcripts = get_transcripts_inner()

    if len(biotypes) > 0:
        all_transcripts = [x for x in all_transcripts if x.biotype in biotypes]
    if complete:
        all_transcripts = [x for x in all_transcripts if x.complete is True]
    # [x.id for x in all_transcripts]
    return all_transcripts

def get_transcript_seqs(db, transcript_ids, translate=True):
    from Bio.Seq import Seq
    from tqdm.auto import tqdm
    seqs = {}
    aa_seqs = {}
    for transcript_id in tqdm(transcript_ids):
        tx = db.transcript_by_id(transcript_id) 
        seqs[transcript_id] = tx.coding_sequence
        if translate:
            aa_seqs[transcript_id] = Seq(tx.coding_sequence).translate()
    if translate:
        return seqs, aa_seqs
    else:
        return seqs
 
def get_transcript_exons(db, transcript_id):
    exon_ids = db.exon_ids_of_transcript_id(transcript_id)
    exons = [db.exon_by_id(x) for x in exon_ids]
    return exons

# def add_variant(seq, 
#                 allele, 
#                 rec_start, 
#                 rec_stop): 
#     if len(allele) == 1:  
#         seq[rec_start:rec_stop] = allele
#     else:
#         seq[rec_start:(rec_stop)] = list(allele) 
#     return seq
def add_variant(seq, allele, rec_start, rec_stop, rec): 
    """
    Inserts a variant into the reference sequence ensuring the sequence length remains unchanged.

    Parameters:
    - seq (list of str): The reference genome sequence as a list of single-character strings.
    - allele (str): The allele to be inserted (can be an insertion, deletion, or substitution).
    - rec_start (int): Start index of the variant in the sequence (0-based).
    - rec_stop (int): Stop index of the variant in the sequence (exclusive).

    Returns:
    - list of str: The modified sequence with the variant applied.
    """
    # rec.start and rec.stop refer to the start and stop of the variant in the sequence, 
    # NOT the start/stop of the reference allele

    # Get reference length
    ref_length = len("".join(rec.ref)) #rec_stop - rec_start
    ref_length2 = rec_stop - rec_start
    if ref_length != ref_length2:
        raise ValueError(f"Reference length mismatch: {ref_length} != {ref_length2}")
    # Get allele length
    allele = "".join(allele).replace("-", "").replace(".", "")
    allele_length = len(allele)
    # Get allele/reference length difference
    length_diff = allele_length - ref_length

    # NOTE: Must insert allele using rec_start/rec_start variables (not original coordinates) because
    # 1. These variable are relative to the reference sequence (as opposed to absolute genomic coordinates)
    # 2. These variables account for strand orientation
    if length_diff > 0:  # Insertion
        # Insert allele and remove extra bases to maintain length
        # Example: Insert 'TTT' where ref_length is 1
        # Replace with 'TTT' and remove 2 additional bases
        # seq[rec_start:rec_stop] = [allele] + ['-'] * (length_diff)
        seq[rec_start] = allele
    elif length_diff < 0:  # Deletion
        # Replace with allele and pad with '-' to maintain length
        # Example: Delete 2 bases, insert 'A' (length_diff = -1)
        seq[rec_start:rec_stop] = [allele] + ['-'] * (-length_diff)
    else:  # Substitution
        # Direct replacement
        seq[rec_start:rec_stop] = list(allele)
    return seq

def check_allele_length(allele, rec):
    if rec.alleles_variant_types[0] != "REF":
        raise ValueError(f"First alleles_variant_type should be REF, but is {rec.alleles_variant_types[0]}")
    elif rec.alleles_variant_types[1] == "SNP":
        if len(allele) != 1:
            raise ValueError(f"Allele length should be 1, but is {len(allele)}: {allele}")
    elif rec.alleles_variant_types[1] == "DEL":
        if len(allele) < 1:
            raise ValueError(f"Allele length should be greater than 0, but is {len(allele)}: {allele}")
    elif rec.alleles_variant_types[1] == "INS":
        if len(allele) < 2:
            raise ValueError(f"Allele length should be greater than 1, but is {len(allele)}: {allele}")
    # elif rec.alleles_variant_types[1] == "INDEL":
    #     if len(allele) == len(rec.ref):
    #         raise ValueError(f"Allele length should be different than reference length, but is {len(allele)}: {allele}")

def get_rec_start_stop(tx, 
                       rec, 
                       ref_seq_len=None,
                       relative=True,
                       strand_aware=True):
    """
    Calculate the start and stop positions of a variant record relative to the transcript.

    Parameters:
    - tx: Transcript object containing genomic coordinates and strand information.
    - rec: Variant record from VCF.
    - relative: If True, return positions relative to the transcript.

    Returns:
    - (rec_start, rec_stop): Tuple of start and stop positions.
    """
    # Genomic coordinates from the variant record
    variant_start = rec.start  # 1-based position
    variant_end = rec.stop #rec.pos + len(rec.ref) - 1  # Inclusive end

    if relative:
        # Calculate relative to transcript start
        rec_start = variant_start - tx.start
        rec_stop = variant_end - tx.start
    else:
        rec_start = variant_start
        rec_stop = variant_end

    # Adjust for strand orientation
    if strand_aware and tx.strand == "-":
        if ref_seq_len is None:
            ref_seq_len = len(tx.sequence)
        # For negative strand, reverse the positions
        rec_start, rec_stop = ref_seq_len - rec_stop, ref_seq_len - rec_start

    return rec_start, rec_stop

def run_check_ref_mismatch(rec, 
                           rec_start, 
                           rec_stop, 
                           ref_seq, 
                           ref_genome, 
                           chrom,
                           tx,
                           strand_aware=True):
    # Get reference allele
    rec_ref = rec.ref
    if strand_aware and tx.strand == "-":
        rec_ref = reverse_complement(rec_ref)
    # Check 1
    ref_seq_rec = "".join(ref_seq[rec_start:rec_stop])
    if (rec_ref != ref_seq_rec):
        raise ValueError(f"Reference allele mismatch: {rec_ref} != {ref_seq_rec}")
    # Check 2
    ref_seq_rec = "".join(ref_genome[chrom][rec.start:rec.stop].seq) 
    if strand_aware and tx.strand == "-":
        ref_seq_rec = reverse_complement(ref_seq_rec)
    if (rec_ref != ref_seq_rec):
        raise ValueError(f"Reference allele mismatch: {rec_ref} != {ref_seq_rec}")
    # Check 3
    # if (rec.ref != rec.alleles[0]):
    #     raise ValueError(f"Reference allele is not the same as the first allele: {rec.ref} == {rec.alleles[0]}")

def get_variant_id(rec):
    return f"{rec.chrom}:{rec.start}_{rec.stop}_{rec.alleles[0]}_{rec.alleles[1]}"

def process_allele(tx, 
                   allele, 
                   rec, 
                   strand_aware=True): 
    # Complement alleles if on negative strand
    if strand_aware and tx.strand == "-":
        allele = reverse_complement(allele)
    check_allele_length(allele, rec)
    return allele


def personalize_nuc_seqs(tx,
                         vcf_in,
                         ref_genome,
                         ref_seq=None,
                         nuc_seqs=None,
                         sample=None,
                         variant_types=None,#["SNP", "DEL", "INS"],
                         check_ref_mismatch=True,
                         strand_aware=True,
                         as_list=True,
                         extra_variants=None,
                         variant_recorder = None,
                         verbose=True):
    if variant_recorder is None:
        variant_recorder = []
    chrom = "chr"+tx.contig.replace("chr","")  
    # Get the reference sequence
    if ref_seq is None:
        ref_seq = get_ref_seq(tx=tx, 
                              strand_aware=strand_aware)
    # Convert ref_seq to list of strings
    ref_seq = ["".join(x) for x in list(ref_seq)]
    ref_seq_len=len(ref_seq)
    # Return reference sequence if sample is "REFERENCE"
    if sample == "REFERENCE":
        return [ref_seq, ref_seq], variant_recorder
    # Get personalized sequences
    if nuc_seqs is None:
        seq1 = ref_seq.copy()
        seq2 = ref_seq.copy()
    else:
        seq1 = nuc_seqs[0]
        seq2 = nuc_seqs[1]
    # Get variant records
    if is_VariantFile(vcf_in):
        records = vcf_in.fetch(chrom, tx.start, tx.end)
    else:
        records = vcf_in
    
    # Iterate over variants within the transcript
    for rec in records: 
        
        # Filter by variant type
        if variant_types is not None:
            if rec.alleles_variant_types[1] not in variant_types:
                continue
        
        # compute relative range of the variant in the transcript (manually, without using tx.offset_range)
        # VCF records already add 1 to the end position, so no need to add 1 to the end position
        rec_start, rec_stop = get_rec_start_stop(tx=tx,
                                                 rec=rec,
                                                 ref_seq_len=ref_seq_len,
                                                 strand_aware=strand_aware,
                                                 relative=True)
        
        # Confirm that the reference allele is the same as the reference allele in the VCF  
        if check_ref_mismatch:
            run_check_ref_mismatch(rec=rec, 
                                   rec_start=rec_start, 
                                   rec_stop=rec_stop, 
                                   ref_seq=ref_seq, 
                                   ref_genome=ref_genome, 
                                   strand_aware=strand_aware,
                                   tx=tx,
                                   chrom=chrom)
        
        # Get sample data
        if sample is not None:
            sample_data = rec.samples[sample]  # Get sample by name
            gt = sample_data.allele_indices  # Tuple of genotype indices for a sample (phase1, phase2) 
        else:
            gt = [1,1] # For ClinVar records, assume both alleles are alternate
        alleles = rec.alleles # tuple of reference allele followed by alt alleles
        
        # sample_data.phased

        # Skip if both alleles are same as reference
        if gt[0]==0 and gt[1]==0:
            continue
        
        # Phase 1 allele
        allele1 = process_allele(tx, 
                                 alleles[gt[0]], 
                                 rec, 
                                 strand_aware=strand_aware)
        seq1 = add_variant(seq1, allele1, rec_start, rec_stop, rec)
        
        # Phase 2 allele
        allele2 = process_allele(tx, 
                                 alleles[gt[1]], 
                                 rec, 
                                 strand_aware=strand_aware)
        seq2 = add_variant(seq2, allele2, rec_start, rec_stop, rec)

        # Record variant (only if the variant is not already recorded)
        variant_id = get_variant_id(rec)
        if variant_id not in variant_recorder:
                variant_recorder += [variant_id]

        # Check that the sequences are the same length
        if len(seq1) != len(ref_seq):
            raise ValueError(f"Sequences are not the same length: {len(seq1)} != {len(ref_seq)}: {variant_id}")
        if len(seq2) != len(ref_seq):
            raise ValueError(f"Sequences are not the same length: {len(seq2)} != {len(ref_seq)}: {variant_id}")
    
    # Use recursive function to add extra variants
    if extra_variants is not None:
        if verbose:
            print(f"Adding {len(extra_variants)} extra variants")
        nuc_seqs, variant_recorder = personalize_nuc_seqs(
            tx=tx,
            vcf_in=extra_variants,
            nuc_seqs=[seq1, seq2],
            ref_genome=ref_genome,
            ref_seq=ref_seq,
            variant_types=variant_types,
            variant_recorder=variant_recorder,
            as_list=as_list,
            verbose=verbose
        )
        return nuc_seqs, variant_recorder
    
    # Return as list or string
    if as_list is True:
        return [seq1, seq2], variant_recorder
    else:
        return ["".join(seq1), "".join(seq2)], variant_recorder


def personalize_aa_seqs(tx, 
                        sample,
                        nuc_seqs=None,
                        offset_ranges=None,
                        to_stop=False, 
                        codon_buffer=None,
                        variant_types=None,
                        extra_variants=None,
                        verbose=True,
                        **kwargs):
    """
    Get personalized protein sequences for both haplotypes of a transcript.
    
    Args:
        tx: Transcript object
        vcf_in: VCF file object containing variants
        ref_genome: Reference genome object
        sample: Sample name
        transcript_id: Transcript ID
        to_stop: Whether to stop at the first stop codon
        codon_buffer: Buffer to add to the end of the sequence
        
    Returns:
        tuple: (protein_seq1, protein_seq2) - Protein sequences for both haplotypes
    """
    transcript_id = tx.id
    if nuc_seqs is None:
        nuc_seqs, variant_recorder = personalize_nuc_seqs(
            tx=tx,
            sample=sample,
            variant_types=variant_types,
            extra_variants=extra_variants,
            verbose=verbose,
            **kwargs
        )
    # Translate CDS to Protein Sequence 
    if offset_ranges is None:
        offset_ranges = get_offset_ranges(tx)
    aa_seqs = ['', '']
    problem_seqs = []
    # Iterate over both phases
    for i,phase in enumerate(["phase1", "phase2"]):
        if phase == "phase1":
            cds_seq = subset_seq(nuc_seqs[0], offset_ranges)
        else:
            cds_seq = subset_seq(nuc_seqs[1], offset_ranges)
        # Clean sequence and/or add codon buffer
        cds_seq = clean_seq(cds_seq, codon_buffer=codon_buffer)
        try:
            if len(cds_seq) % 3 != 0:
                problem_seqs += [(transcript_id, sample, "phase1")]
            aa_seqs[i] = translate_seq(cds_seq, to_stop=to_stop)
        except:
            problem_seqs += [(transcript_id, sample, "phase1")]
            aa_seqs[i] = None
    return nuc_seqs, aa_seqs, variant_recorder, problem_seqs

def process_vcf_file(f, 
                     ref_genome, 
                     max_transcripts, 
                     max_samples, 
                     samples, 
                     include_reference, 
                     save_dir, 
                     force, 
                     transcript_ids, 
                     codon_buffer, 
                     to_stop,
                     verbose,
                     extra_variants):
    from tqdm.auto import tqdm
    import pickle
    import os
    import pysam
    import pyfaidx

    # Initialize a separate db connection within the thread
    db = get_db()
    # Fetch transcripts relevant to this VCF file
    transcripts = get_protein_coding_transcripts(db, 
                                                 verbose=verbose)
    # Initialize reference genome if it's a file path
    if isinstance(ref_genome, str): 
        ref_genome = pyfaidx.Fasta(ref_genome) 
    chrom = os.path.basename(f).split('.')[1] 
    # Create new file
    if verbose:
        print(f"Processing VCF: {f}")
    # Import VCF file
    vcf_in = pysam.VariantFile(f)
    # Create storage variable for all transcripts in this VC
    results = {}
    # Get relevant transcript coordinates
    chr_transcripts = filter_transcripts(
        transcripts, 
        contig=chrom,
        biotype=['protein_coding'],
        id=transcript_ids,
        max_transcripts=max_transcripts,
        verbose=verbose > 1
    )
    if len(chr_transcripts) == 0:
        print(f"No transcripts found for {chrom}")
        return chrom, {}
    # List samples
    if samples is None:
        samples = list(vcf_in.header.samples)
    else:
        samples = list(set(samples).intersection(["REFERENCE"] + list(vcf_in.header.samples)))
    # Include reference sequence
    if include_reference:
        samples = list(set(["REFERENCE"] + samples))
        if max_samples is not None:
            max_samples = max_samples+1
    # Ensure REFERENCE is first in list if present
    if "REFERENCE" in samples:
        samples.remove("REFERENCE")
        samples = ["REFERENCE"] + samples
    if max_samples is not None:
        samples = samples[:max_samples]
    # Process each transcript
    for tx in tqdm(chr_transcripts, desc=f"Processing {chrom} transcripts"): 
        transcript_id = tx.id
        # Create storage variables
        tx_variant_recorder = {}
        nuc_seqs = {}
        aa_seqs = {}
        problem_seqs = [] 
        # Load existing file
        save_path = f"{save_dir}/{transcript_id}.pkl"
        if os.path.exists(save_path) and not force:
            if verbose > 1:
                print(f"Loading existing file: {save_path}")
            with open(save_path, 'rb') as handle:
                tx_results = pickle.load(handle)
        else: 
            # Get transcript-level variables
            offset_ranges = get_offset_ranges(tx)
            ref_seq = get_ref_seq(tx=tx)
            add_sample_progress_bar  = True
            # Filter extra variants
            if extra_variants is not None:
                tx_extra_variants = filter_variants(
                    recs=extra_variants, 
                    regions=[f"{chrom}:{tx.start}-{tx.end}"],
                    verbose=verbose>1)
            else:
                tx_extra_variants = None
            # Add progress bar
            if add_sample_progress_bar:
                samples_iterator = tqdm(
                    samples, 
                    desc=f"{transcript_id}: Processing {len(samples)} samples",
                    leave=False,
                    colour="orange")
            else:
                samples_iterator = samples
            # Get personalized sequences
            for sample in samples_iterator:
                (nuc_seqs_i, 
                 aa_seqs_i, 
                 variant_recorder_i, 
                 problem_seqs_i) = personalize_aa_seqs(
                    tx=tx, 
                    vcf_in=vcf_in,
                    ref_genome=ref_genome,
                    ref_seq=ref_seq,
                    sample=sample,
                    offset_ranges=offset_ranges,
                    codon_buffer=codon_buffer,
                    to_stop=to_stop,
                    extra_variants=tx_extra_variants,
                    verbose=verbose>1
                )   
                nuc_seqs[sample] = nuc_seqs_i
                aa_seqs[sample] = aa_seqs_i
                tx_variant_recorder[sample] = variant_recorder_i
                problem_seqs += problem_seqs_i
            # Save results per transcript
            if save_path is not None:
                if verbose > 1:
                    print(f"Saving results: {save_path}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                tx_results = {
                    # Sample-level
                    'nuc_seqs': nuc_seqs,
                    'aa_seqs': aa_seqs,
                    # Transcript-level
                    'offset_ranges': offset_ranges,
                    'variant_recorder': tx_variant_recorder,
                    'problem_seqs': problem_seqs
                }
                with open(save_path, 'wb') as handle:
                    pickle.dump(tx_results, handle)
        results[transcript_id] = tx_results
    # Return dictionary of results
    return chrom, results

def personalize_seqs(vcf_files,
                         ref_genome,
                         max_files = None,
                         max_transcripts = None,
                         transcript_ids = None,
                         max_samples = None,
                         samples = None,
                         save_dir = "1KG/sequence_dict",
                         max_workers = 1,
                         force = False,
                         codon_buffer = 'N',
                         to_stop = False,
                         include_reference = True,
                         executor = "ThreadPoolExecutor",
                         group_by_chrom = False,
                         extra_variants = None,
                         verbose = 0):
    if max_files is not None:
        vcf_files = vcf_files[:max_files]
        if verbose:
            print(f"Only {len(vcf_files)} files being processed.")
    os.makedirs(save_dir, exist_ok=True)
    if executor == "ProcessPoolExecutor":
        executor = ProcessPoolExecutor
    elif executor == "ThreadPoolExecutor":
        executor = ThreadPoolExecutor
    # Process VCF files in parallel
    with executor(max_workers=max_workers) as executor:
        process_func = partial(process_vcf_file,
                               max_transcripts=max_transcripts,
                               transcript_ids=transcript_ids,
                               max_samples=max_samples,
                               ref_genome=ref_genome,
                               samples=samples,
                               include_reference=include_reference,
                               codon_buffer=codon_buffer,
                               to_stop=to_stop,
                               save_dir=save_dir,
                               force=force,
                               verbose=verbose,
                               extra_variants=extra_variants)
        results_all = list(tqdm(executor.map(process_func, vcf_files), 
                           desc="Processing VCF files",
                           colour="black",
                           total=len(vcf_files)))
    # convert into dict
    print("Converting all results to dict.")
    results_all = {x[0]: x[1] for x in results_all}
    # Flatten so that the top-level key in the dict is the transcript ID
    if group_by_chrom is False:
        flattened_results = {}
        for chrom, chrom_results in results_all.items():
            for transcript_id, tx_data in chrom_results.items():
               flattened_results[transcript_id] = tx_data
        results_all = flattened_results
    # Return results
    return results_all

def download(url, path):
    import os
    import requests
    if not os.path.exists(path):
        print(f"Downloading {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        response = requests.get(url)
        with open(path, 'wb') as f:
            f.write(response.content)

def download_ref_genome(ref_genome = "GRCh38/GRCh38_full_analysis_set_plus_decoy_hla.fa"):
    download(f"https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa",
              ref_genome)
    download(f"https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai",
              ref_genome+".fai")

def get_ref_genome(ref_genome = "GRCh38/GRCh38_full_analysis_set_plus_decoy_hla.fa"):
    import os
    import pyfaidx
    if isinstance(ref_genome, str):
        if not os.path.exists(ref_genome):
            download_ref_genome(ref_genome)
        return pyfaidx.Fasta(ref_genome)
    elif isinstance(ref_genome, pyfaidx.Fasta):
        return ref_genome
    else:
        raise ValueError(f"Reference genome must be a file path or a pyfaidx.Fasta object: {type(ref_genome)}")

def get_ref_seq(tx=None,
                transcript_id=None,
                ref_genome = "GRCh38/GRCh38_full_analysis_set_plus_decoy_hla.fa",
                db = None,
                coding_only = False, 
                strand_aware = True):
    if tx is None:
        if db is None:
            db = get_db()
        tx = db.transcript_by_id(transcript_id)
    if coding_only:
        ref_seq = tx.coding_sequence
    else:
        chrom = "chr"+str(tx.contig).replace("chr","")
        ref_genome = get_ref_genome(ref_genome)
        ref_seq = ref_genome[chrom][tx.start:tx.end].seq
    # import copy
    # ref_seq = copy.deepcopy(ref_seq)
    if strand_aware:
        if tx.strand == "-":
            ref_seq = reverse_complement(ref_seq)
    return ref_seq

def get_offset_ranges(tx, 
                      ranges=None):
   if ranges is None:
       ranges = tx.coding_sequence_position_ranges
   offset_ranges = [tx.offset_range(x[0],x[1]) for x in ranges]
   return offset_ranges

def subset_seq(seq, 
               offset_ranges,
               as_list=False): 
    if as_list:
        seq_subset = []
        for range in offset_ranges:
            seq_subset += seq[range[0]:range[1]]
        return seq_subset
    else:
        return "".join(["".join(seq[x[0]:x[1]]) for x in offset_ranges])

def translate_seq(seq, 
                  to_stop=False,
                  as_string=True,  
                  **kwargs):
    seq = as_seq(seq)
    seq = seq.translate(to_stop=to_stop, **kwargs)
    if as_string:
        return str(seq)
    else:
        return seq

def get_translated_seq(tx, 
                       **kwargs):
    # return translate_seq(tx.coding_sequence, **kwargs)
    return tx.protein_sequence

def as_seq(seq):
    from Bio.Seq import Seq
    if isinstance(seq, list):
        seq = "".join(seq)
    return Seq(seq)

def clean_seq(seq, 
              replace=["-", ".", "="],
              codon_buffer=None):
    if isinstance(seq, list):
        seq = "".join(seq) 
    for x in replace:
        seq = seq.replace(x,"")
    seq = add_codon_buffer(seq, codon_buffer)
    return seq

def sequence_similarity(seq1, seq2):
    import numpy as np
    # Get max length and pad shorter sequence with spaces which will count as mismatches
    if isinstance(seq1, str):
        seq1 = list(seq1)
    if isinstance(seq2, str):
        seq2 = list(seq2)
    max_len = max(len(seq1), len(seq2))
    seq1 = seq1 + [' '] * (max_len - len(seq1))
    seq2 = seq2 + [' '] * (max_len - len(seq2))
    
    # Convert sequences to NumPy arrays
    arr1 = np.array(seq1, dtype='U1')
    arr2 = np.array(seq2, dtype='U1')
    
    # Calculate the number of matching elements
    matches = np.sum(arr1 == arr2)
    
    return matches / max_len

def get_ref_seq_keys(d):
    return [sample for sample in d.keys() if sample.startswith("REFERENCE")]

def get_sequence_similarity(results_all, 
                            seq_type=["aa", "nuc", "nuc_subset"][0],
                            seqs=None,
                            split_samples="_",
                            db=None,
                            as_df=True,
                            unique=False):
    from tqdm.auto import tqdm
    import pandas as pd
    if db is None:
        db = get_db()
    # For each transcript get the sequence similarity between the reference and each of the two haplotypes within each sample 
    seq_similarity = {} 
    if seqs is None:
        if seq_type == "aa":
            seqs = get_aa_seqs(results_all, db=db, unique=unique)
        elif seq_type == "nuc":
            seqs = get_nuc_seqs(results_all, db=db, unique=unique)
        elif seq_type == "nuc_subset":
            seqs = get_nuc_seqs(results_all, db=db)
    unique_sequences = {}
    for transcript_id, x in tqdm(seqs.items(),"Processing transcripts"):
        offset_ranges = get_offset_ranges(db.transcript_by_id(transcript_id))
        unique_sequences[transcript_id] = []
        # Add reference sequence
        # Find the reference sequence key
        ref_seq_keys = get_ref_seq_keys(x)
        if len(ref_seq_keys) > 0:
            ref_seq = x[ref_seq_keys[0]][0]
        else:
            if seq_type == "aa":
                ref_seq = db.transcript_by_id(transcript_id).protein_sequence
            elif seq_type == "nuc":
                ref_seq = db.transcript_by_id(transcript_id).coding_sequence
        if seq_type == "nuc_subset":
            ref_seq = subset_seq(ref_seq, offset_ranges)
        unique_sequences[transcript_id] += [ref_seq]
        # Iterate over samples
        seq_similarity[transcript_id] = []
        for sample, seq_list in tqdm(x.items(),
                                 desc=f"{transcript_id}: Processing samples",    
                                 leave=False,
                                 colour="orange"):
            if sample in ref_seq_keys:
                continue
            # Phase 1
            if seq_list[0] is not None:
                seq = seq_list[0]
                if seq_type == "nuc_subset":
                    seq = subset_seq(seq, offset_ranges)
                seq1_sim = sequence_similarity(ref_seq, seq)
                if seq not in unique_sequences[transcript_id]:
                    unique_sequences[transcript_id] += [seq]
            else:
                seq1_sim = None
            # Phase 2
            if seq_list[1] is not None:
                seq = seq_list[1]
                if seq_type == "nuc_subset":
                    seq = subset_seq(seq, offset_ranges)
                seq2_sim = sequence_similarity(ref_seq, seq)
            else:
                seq2_sim = None
            # Add unique sequences
            for seq in [seq_list[0], seq_list[1]]:
                if seq not in unique_sequences[transcript_id]:
                    unique_sequences[transcript_id] += [seq]
            seq_similarity[transcript_id] += [seq1_sim, seq2_sim]
    # Convert to dataframe
    if as_df:
        seq_similarity_df = pd.DataFrame(seq_similarity)
        samples = get_samples(results_all, include_reference=False, per_phase=True)
        # phases = get_phases(seqs)
        # seq_similarity_df.insert(0, "phase", phases)
        seq_similarity_df.insert(0, "sample", samples)
        seq_similarity_df = seq_similarity_df.melt(ignore_index=False, 
                                                   id_vars=["sample"],
                            var_name="transcript_id",
                            value_name="similarity")  
        seq_similarity_df['sample_id'] = seq_similarity_df['sample'].str.split("_", expand=True)[0]
        seq_similarity_df['phase'] = seq_similarity_df.groupby(['transcript_id','sample']).cumcount()  
        seq_similarity_df['similarity_mean'] = seq_similarity_df.groupby('transcript_id')['similarity'].transform('mean')
        # Add the number of unique sequences per transcript
        unique_sequences_df = pd.DataFrame({k:len(v) for k,v in unique_sequences.items()}, index=['unique_sequences']).T.reset_index()
        unique_sequences_df.columns = ['transcript_id', 'unique_sequences']
        seq_similarity_df = pd.merge(seq_similarity_df, 
                                     unique_sequences_df, 
                                     on='transcript_id', 
                                     how="left")
        if split_samples:
            seq_similarity_df['sample_group'] = seq_similarity_df['sample'].str.split(split_samples, expand=True)[1]
        return seq_similarity_df
    else:
        return seq_similarity
    
def correlate_sequence_similarity(seq_similarity_df,
                                  group_col="sample_group"):
    groups = seq_similarity_df[group_col].unique()
    df = seq_similarity_df.pivot(index=["transcript_id","sample_id",'phase'], columns=group_col, values="similarity")
    from scipy import stats
    corr, pval = stats.pearsonr(df[groups[0]], df[groups[1]])
    return {'r':corr, 'pval':pval}
    
def plot_sequence_similarity(seq_similarity_df, 
                             title=None,
                             sort=True, 
                             interact=False,
                             facet_col="sample_group",
                             ylim=None,
                             return_fig=False   
                             ):
    if sort:
         seq_similarity_df.sort_values(by=["sample_group","similarity_mean"], 
                                        ascending=[False,False], 
                                        inplace=True)
    if title is None:
        title = f"Similarity to reference sequence<br>{seq_similarity_df.transcript_id.nunique()} transcripts<br>{seq_similarity_df['sample'].nunique()} samples"
    if interact: 
        import plotly.express as px
        # Add unique sequences count to hover text
        fig = px.box(seq_similarity_df,
                    x="transcript_id", y="similarity",
                    title=title,
                    facet_col=facet_col,
                    labels={"transcript_id": "transcript_id (n unique sequences)"})
        # Update x-axis labels to include unique sequence counts
        fig.update_xaxes(ticktext=[f"{tid} (n={n})" for tid, n in 
                        seq_similarity_df.groupby('transcript_id')['unique_sequences'].first().items()],
                        tickvals=seq_similarity_df['transcript_id'].unique())
        if ylim is not None:
            fig.update_layout(yaxis_range=ylim)
    else:
        # With seaborn
        import seaborn as sns
        import matplotlib.pyplot as plt
        if title is not None:
            title = title.replace("<br>", "\n")
        if facet_col is not None:
            # Create figure with subplots for each group
            groups = seq_similarity_df[facet_col].unique()
            fig, axes = plt.subplots(1, len(groups), figsize=(5*len(groups), 5), 
                                     sharex=True, sharey=True)
            if len(groups) == 1:
                axes = [axes]
            for ax, group in zip(axes, groups):
                group_data = seq_similarity_df[seq_similarity_df[facet_col] == group]
                sns.boxplot(x="transcript_id", y="similarity",
                          data=group_data, ax=ax)
                ax.set_title(group)
                ax.tick_params(axis='x', rotation=90)
                # Update x-axis labels to include unique sequence counts
                ticks = ax.get_xticks()
                ax.xaxis.set_major_locator(plt.FixedLocator(ticks))
                ax.set_xticklabels([f"{tid} (n={n})" for tid, n in 
                                group_data.groupby('transcript_id')['unique_sequences'].first().items()])
                ax.set_xlabel("transcript_id (n unique sequences)")
                # Add mean similarity line
                mean_similarity = group_data['similarity'].mean()
                ax.axhline(y=mean_similarity, color='black', linestyle=':', alpha=0.8)
                if ylim is not None:
                    ax.set_ylim(ylim)
            fig.show()
        else:
            fig = plt.figure()
            sns.boxplot(x="transcript_id", y="similarity",
                       data=seq_similarity_df)
            plt.xticks(rotation=90, ha='right')
            # Update x-axis labels to include unique sequence counts
            ax = plt.gca()
            ticks = ax.get_xticks()
            ax.xaxis.set_major_locator(plt.FixedLocator(ticks))
            ax.set_xticklabels([f"{tid} (n={n})" for tid, n in 
                             seq_similarity_df.groupby('transcript_id')['unique_sequences'].first().items()])
            plt.xlabel("transcript_id (n unique sequences)")
            plt.title(title)
            # Add mean similarity line
            mean_similarity = seq_similarity_df['similarity'].mean()
            ax.axhline(y=mean_similarity, color='black', linestyle=':', alpha=0.8)
            ax.legend()
            if ylim is not None:
                plt.ylim(ylim)
        plt.tight_layout()
        plt.show()
    if return_fig:
        return fig
    
def get_unique_seqs(seq_dict):
    return {k: list(set([y for v in seq_dict[k].values() for y in v if y is not None])) for k in seq_dict.keys()}

def get_nuc_seqs(results_all, 
                 db=None,
                 unique=False,
                 add_reference=False,
                 seqs_as_list=True):
    if db is None:
        db = get_db()
    nuc_seqs = {k: dict([('REFERENCE', list(db.transcript_by_id(k).coding_sequence))] + list(v['nuc_seqs'].items())) if add_reference else v['nuc_seqs'] for k,v in results_all.items()}
    if unique:
        nuc_seqs = get_unique_seqs(nuc_seqs)
    if seqs_as_list is False:
        nuc_seqs = {transcript_id: [["".join(s[0]), "".join(s[1])] for s in v.values()] for transcript_id,v in nuc_seqs.items()}
    return nuc_seqs

def get_aa_seqs(results_all, 
                db=None, 
                unique=False,
                add_reference=False):
    if db is None:
        db = get_db()
    aa_seqs = {k: dict([('REFERENCE', list(db.transcript_by_id(k).protein_sequence))] + list(v['aa_seqs'].items())) if add_reference else v['aa_seqs'] for k,v in results_all.items()}
    if unique:
        aa_seqs = get_unique_seqs(aa_seqs)
    return aa_seqs

def get_positions(tx):
    return [x for y in [list(range(x[0]-1, x[1])) for x in tx.coding_sequence_position_ranges] for x in y]

def get_exon_indices(tx):
    return [x for y in [[i]*(x[1]-x[0]+1) for i,x in enumerate(tx.coding_sequence_position_ranges)] for x in y]

def compare_nuc_seqs(results_all, 
                     transcript_id,
                     sample,
                     db=None,
                     coding_only=False,
                     difference_only=False):
    import pandas as pd
    ref_seq = results_all[transcript_id]['nuc_seqs']['REFERENCE'][0]
    seq1 = results_all[transcript_id]['nuc_seqs'][sample][0]
    seq2 = results_all[transcript_id]['nuc_seqs'][sample][1]
    if coding_only:
        if db is None:
            db = get_db()
        tx = db.transcript_by_id(transcript_id)
        # find rows where seq1 and seq2 are not identical to ref_seq
        offset_ranges = get_offset_ranges(tx)
        ref_seq = subset_seq(ref_seq, offset_ranges, as_list=True)
        seq1 = subset_seq(seq1, offset_ranges, as_list=True)
        seq2 = subset_seq(seq2, offset_ranges, as_list=True)
        positions = get_positions(tx)
        exon_indices = get_exon_indices(tx)
        # subset seq_df to a list of tuples, where each tuple is a start and end index
        # seq_df = pd.concat([seq_df.iloc[start:end] for start,end in offset_ranges])
    seq_df =  pd.DataFrame({"ref_seq":ref_seq, 
                            "seq1":seq1, 
                            "seq2":seq2})
    seq_df.insert(0, "exon_index", exon_indices)
    seq_df.insert(0, "position", positions)
    seq_df.insert(0, "chrom", tx.contig)
    
    if difference_only:
        seq_df = seq_df[(seq_df.seq1 != seq_df.ref_seq) | (seq_df.seq2 != seq_df.ref_seq)]
    return seq_df

def get_samples(results_all, 
                include_reference=False,
                per_phase=False):
    # Assumes all transcripts have the same samples
    samples = list(list(results_all.values())[0]['nuc_seqs'].keys())
    ref_seq_keys = get_ref_seq_keys(list(results_all.values())[0]['nuc_seqs'])
    if not include_reference:
        samples = [x for x in samples if x not in ref_seq_keys]
    if per_phase:
        # interleave the samples with itself
        samples = [item for sublist in zip(samples, samples) for item in sublist]
    return samples

def get_phases(seqs, unnest=True):
    phases = [[list(range(len(s))) for s in v.values()] for v in seqs.values()] 
    # Unnest twice
    if unnest:
        return [item for sublist in [item for sublist in phases for item in sublist] for item in sublist]
    else:
        return phases

def count_variants(results_all, as_df=True):
    # get variant counts from chr_variant_recorder: iterating through chromosomes, then transcripts, then samples, then variants
    from itertools import chain
    import pandas as pd 
    variant_counts = {}
    samples = set()
    for transcript_id, results in results_all.items():
        variant_counts[transcript_id] = []
        for sample, variants in results['variant_recorder'].items():
            if sample != 'REFERENCE':
                samples.add(sample)
                # Process one transcript at a time to reduce memory usage
                variant_counts[transcript_id] += [len(set(variants))]
    if as_df:
        variant_counts_df = pd.DataFrame(variant_counts,
                                         index=samples).reset_index().rename(columns={'index':'sample'})
        variant_counts_df = variant_counts_df.melt(id_vars='sample',
                                                   var_name='transcript_id',
                                                   value_name='variant_count')
        return variant_counts_df
    else:
        return variant_counts
    
def plot_variant_counts(variant_counts_df, 
                        interact=False):
    if interact:
        import plotly.express as px
        fig = px.histogram(variant_counts_df, x=1)
        return fig
    else:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        # Plot histogram with seaborn
        fig = sns.histplot(data=variant_counts_df, x='variant_count', bins=100)
        plt.show()
        return fig
    
def count_variant_lengths(ref_seq, 
                          seq1, 
                          seq2, 
                          sort=True, 
                          verbose=True):
    # count frequency of each variant length
    from collections import Counter
    # Get length counts and sort by length
    ref_counts = Counter([len(x) for x in ref_seq])
    seq1_counts = Counter([len(x) for x in seq1]) 
    seq2_counts = Counter([len(x) for x in seq2])
    if sort:
        ref_counts = sorted(ref_counts.keys())
        seq1_counts = sorted(seq1_counts.keys())
        seq2_counts = sorted(seq2_counts.keys())
    if verbose:
        print("Variant length counts:")
        for length in ref_counts:
            print(f"Length {length}: {ref_counts[length]}")
        print("\nSequence 1 lengths:")
        for length in seq1_counts:
            print(f"Length {length}: {seq1_counts[length]}")
        print("\nSequence 2 lengths:")
        for length in seq2_counts:
            print(f"Length {length}: {seq2_counts[length]}")
    return ref_counts, seq1_counts, seq2_counts

def get_sequence_diff_indices(seq1, seq2):
    indices = [i for i, (x, y) in enumerate(zip(seq1, seq2)) if x != y]
    return indices

def get_variant_counts(results_all):
    return {k: len([x for x in len(set(results_all[k]['variant_recorder'])) if x is not None]) for k in results_all.keys()}

def reverse_complement(seq):
    from Bio.Seq import Seq
    return str(Seq(seq).reverse_complement())

def merge_personalized_seqs(results_wt, results_cv, 
                            suffixes=["_WT", "_ClinVar"]):
    import tqdm.auto as tqdm
    results_all = {}
    for transcript_id in tqdm.tqdm(results_wt.keys(), desc="Merging personalized sequences"):
        results_all[transcript_id] = results_wt[transcript_id].copy()
        # Add suffix to each sample name in results_cv
        ## nuc_seqs
        results_all[transcript_id]['nuc_seqs'] = {f"{sample_id}{suffixes[0]}":v for sample_id,v in results_all[transcript_id]['nuc_seqs'].items()}
        results_all[transcript_id]['nuc_seqs'].update({f"{sample_id}{suffixes[1]}":v for sample_id,v in results_cv[transcript_id]['nuc_seqs'].items()})
        ## aa_seqs
        results_all[transcript_id]['aa_seqs'] = {f"{sample_id}{suffixes[0]}":v for sample_id,v in results_all[transcript_id]['aa_seqs'].items()}
        results_all[transcript_id]['aa_seqs'].update({f"{sample_id}{suffixes[1]}":v for sample_id,v in results_cv[transcript_id]['aa_seqs'].items()})
    return results_all
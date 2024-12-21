import sys
sys.path.append("code")
from src.utils import as_list, intersect, codon_buffer
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
import pysam
import pyfaidx
import os
from Bio.Seq import Seq

def get_db(release=111, species="homo_sapiens"):
    from pyensembl import EnsemblRelease
    db = EnsemblRelease(release=release, species=species) 
    return db

def get_ids(objects):
    return [x.id for x in objects]

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

def get_protein_coding_transcripts(db):
    from tqdm.auto import tqdm
    protein_ids = db.protein_ids()
    transcripts = [db.transcript_by_protein_id(x) for x in tqdm(protein_ids,"Retrieving transcripts.")]
    # transcripts = filter_transcripts(transcripts, biotype=['protein_coding'])
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


def get_relative_ranges(tx, ranges=None):
    relative_ranges = []
    current_offset = 0
    if ranges is None:
        ranges = tx.coding_sequence_position_ranges
    for exon_range in ranges:
        start, end = tx.offset_range(exon_range[0], exon_range[1])
        length = end - start
        relative_ranges.append((current_offset, current_offset + length))
        current_offset += length
    return relative_ranges

def get_translated_seq(tx, **kwargs):
    from Bio.Seq import Seq
    relative_ranges = get_relative_ranges(tx)
    final_seq = Seq('')
    for x in relative_ranges:
        final_seq += Seq(tx.coding_sequence[x[0]:x[1]])
    return final_seq.translate(**kwargs)

def personalize_transcript(tx,
                           vcf_in,
                           ref_genome,
                           sample,
                           check_ref_mismatch=True):
    transcript_id = tx.id
    variant_recorder[transcript_id] = {}
    variant_recorder[transcript_id][sample] = []
    chrom = "chr"+tx.contig.replace("chr","")  
    # Get the reference sequence
    ref_seq = ref_genome[chrom][tx.start:tx.end].seq
    seq1 = list(ref_seq)
    seq2 = list(ref_seq)
    if sample != "REFERENCE":
        for rec in vcf_in.fetch(chrom, tx.start, tx.end): 
            # Confirm that the reference allele is the same as the reference allele in the VCF  
            ref_seq_rec = ref_genome[chrom][rec.start:rec.stop].seq
            if check_ref_mismatch and rec.ref != ref_seq_rec:
                raise ValueError(f"Reference allele mismatch: {rec.ref} != {ref_seq_rec}")
            sample_data = rec.samples[sample]  # Get sample by name
            alleles = rec.alleles # tuple of reference allele followed by alt alleles
            gt = sample_data.allele_indices  # Tuple of genotype indices for a sample (phase1, phase2) 
            # sample_data.phased
            # Phase 1 allele
            seq1[rec.start:rec.stop] = alleles[gt[0]]
            # Phase 2 allele
            seq2[rec.start:rec.stop] = alleles[gt[1]] 
            # Record variant
            # variant_recorder[transcript_id][sample] += [f"{rec.chrom}:{rec.pos}_{gt[0]}_{gt[1]}"]
    return seq1, seq2


def personalize_protein(tx, vcf_in, ref_genome, sample, to_stop=True, buffer='N'):
    """
    Get personalized protein sequences for both haplotypes of a transcript.
    
    Args:
        tx: Transcript object
        vcf_in: VCF file object containing variants
        ref_genome: Reference genome object
        sample: Sample name
        transcript_id: Transcript ID
        to_stop: Whether to stop at the first stop codon
        buffer: Buffer to add to the end of the sequence
        
    Returns:
        tuple: (protein_seq1, protein_seq2) - Protein sequences for both haplotypes
    """
    transcript_id = tx.id
    seq1, seq2 = personalize_transcript(tx,
                                       vcf_in,
                                       ref_genome,
                                       sample,
                                       check_ref_mismatch=True)
    
    # Translate CDS to Protein Sequence 
    protein_seqs = ['', '']
    for i,phase in enumerate(["phase1", "phase2"]):
        if phase == "phase1":
            cds_seq = clean_seq(seq1, buffer=buffer)
        else:
            cds_seq = clean_seq(seq2, buffer=buffer)
        try:
            if len(cds_seq) % 3 != 0:
                problem_seqs[transcript_id][sample] += [(transcript_id, sample, "phase1")]
            protein_seqs[i] = translate_seq(cds_seq, to_stop=to_stop)
        except:
            problem_seqs[transcript_id][sample] += [(transcript_id, sample, "phase1")]
            protein_seqs[i] = None
    return protein_seqs[0], protein_seqs[1]

def process_vcf_file(f, 
                     ref_genome, 
                     max_transcripts, 
                     max_samples, 
                     samples, include_reference, save_dir, force, transcript_ids, transcripts, buffer, verbose):
    # Init inside each thread instance in accordance with: https://github.com/mdshw5/pyfaidx/issues/92#issuecomment-230269505
    if isinstance(ref_genome, str): 
        ref_genome = pyfaidx.Fasta(ref_genome) 
    chrom = os.path.basename(f).split('.')[1] 
    # Create new file
    if verbose:
        print(f"Processing VCF: {f}")
    # Import VCF file
    vcf_in = pysam.VariantFile(f) 
    # Create storage variables
    global variant_recorder, problem_seqs
    transcript_seqs = {}
    seq_counter = {}
    variant_recorder = {}
    problem_seqs = {} 
    # Get relevant transcript coordinates
    chr_transcripts = filter_transcripts(transcripts, 
                                         contig=chrom,
                                         biotype=['protein_coding'],
                                         id=transcript_ids,
                                         max_transcripts=max_transcripts,
                                         verbose=verbose>1
                                         )
    # List samples
    if samples is None:
        samples = list(vcf_in.header.samples)
    else:
        samples = [x for x in samples if x in list(vcf_in.header.samples)+["REFERENCE"]]
    # Process each transcript
    for tx in tqdm(chr_transcripts, desc=f"Processing {chrom} transcripts"):
        transcript_id = tx.id
        # Create storage variables
        transcript_seqs[transcript_id] = {}
        variant_recorder[transcript_id] = {}
        seq_counter[transcript_id] = 0
        problem_seqs[transcript_id] = {}
        # Load existing file
        save_path = f"{save_dir}/{transcript_id}.pkl"
        if os.path.exists(save_path) and force is not True:
            if verbose>1:
                print(f"Loading existing file: {save_path}")
            with open(save_path,'rb') as handle:
                results = pickle.load(handle)
            # Add to storage variables
            transcript_seqs[transcript_id] = results['transcript_seqs']
            seq_counter[transcript_id] = results['seq_counter']
            variant_recorder[transcript_id] = results['variant_recorder']
            problem_seqs[transcript_id] = results['problem_seqs'] 
        else: 
            # Include reference sequence
            if include_reference is True:
                samples = list(set(["REFERENCE"]+samples))
            # Get personalized sequences
            for sample in samples[:max_samples]:
                transcript_seqs[transcript_id][sample] = personalize_protein(
                    tx=tx, 
                    vcf_in=vcf_in,
                    ref_genome=ref_genome,
                    sample=sample,
                    buffer=buffer
                ) 
                seq_counter[transcript_id] += len(transcript_seqs[transcript_id][sample])   
            # Save results per transcript
            if save_path is not None:
                if verbose>1:
                    print(f"Saving results: {save_path}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                results = {'transcript_seqs': transcript_seqs[transcript_id],
                           'seq_counter': seq_counter[transcript_id],
                           'variant_recorder': variant_recorder[transcript_id],
                           'problem_seqs': problem_seqs[transcript_id]}
                with open(save_path,'wb') as handle:
                    pickle.dump(results, handle)
    # Report problem sequences per transcript
    if verbose:
        problem_seqs_total = sum([[len(sample) for sample in transcript_id.values()] for transcript_id in problem_seqs.values()])
        seq_counter_total = sum([[len(sample) for sample in transcript_id.values()] for transcript_id in seq_counter.values()])
        print(f"Detected {problem_seqs_total}/{seq_counter_total} problem sequences for {chrom}.")
    return chrom, transcript_seqs, variant_recorder, seq_counter, problem_seqs 

def personalize_proteins(vcf_files,
                         ref_genome,
                         db=None,
                         max_files = None,
                         max_transcripts = None,
                         transcript_ids = None,
                         max_samples = None,
                         samples = None,
                         save_dir = "1KG/sequence_dict",
                         max_workers = 1,
                         force = False,
                         buffer = 'N',
                         include_reference = True,
                         verbose = 0):

    if db is None:
        db = get_db()
    transcripts = get_protein_coding_transcripts(db)
    chr_transcript_seqs = {}
    chr_seq_counter = {}
    chr_variant_recorder = {}
    chr_problem_seqs = {}
    if max_files is not None:
        vcf_files = vcf_files[:max_files]
        if verbose:
            print(f"Only {len(vcf_files)} files being processed.")
    os.makedirs(save_dir, exist_ok=True)

    # Process VCF files in parallel
    with Pool(processes=max_workers) as pool:
        process_func = partial(process_vcf_file,
                               max_transcripts=max_transcripts,
                               transcript_ids=transcript_ids,
                               max_samples=max_samples,
                               ref_genome=ref_genome,
                               samples=samples,
                               include_reference=include_reference,
                               buffer=buffer,
                               save_dir=save_dir,
                               force=force,
                               transcripts=transcripts,
                               verbose=verbose)
        results = list(tqdm(pool.imap(process_func, vcf_files), 
                          desc="Processing VCF files",
                          total=len(vcf_files)))
    # Combine results across chromosomes
    chr_transcript_seqs = {}
    chr_seq_counter = {}
    chr_variant_recorder = {}
    chr_problem_seqs = {}
    for chrom, transcript_seqs, variant_recorder, seq_counter, problem_seqs  in results:
        chr_transcript_seqs[chrom] =   transcript_seqs
        chr_variant_recorder[chrom] = variant_recorder
        chr_seq_counter[chrom] = seq_counter
        chr_problem_seqs[chrom] = problem_seqs
    # Return results
    return chr_transcript_seqs, chr_seq_counter, chr_variant_recorder, chr_problem_seqs

def as_seq(seq):
    from Bio.Seq import Seq
    if isinstance(seq, list):
        seq = "".join(seq)
    return Seq(seq)

def clean_seq(seq, 
              replace=["-", ".", "="],
              buffer=None):
    if isinstance(seq, list):
        seq = "".join(seq)
    for x in replace:
        seq = seq.replace(x,"")
    seq = codon_buffer(seq, buffer)
    return seq

def translate_seq(seq, 
                  to_stop=True,
                  as_string=False,  
                  **kwargs):
    seq = as_seq(seq)
    seq = seq.translate(to_stop=to_stop, **kwargs)
    if as_string:
        return str(seq)
    else:
        return seq

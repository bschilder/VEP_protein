import sys
sys.path.append("code")
from src.utils import as_list, intersect, add_codon_buffer
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

def get_protein_coding_transcripts(db, 
                                   verbose=True):
    from tqdm.auto import tqdm
    protein_ids = db.protein_ids()
    transcripts = [db.transcript_by_protein_id(x) for x in tqdm(protein_ids,"Retrieving transcripts.", disable=not verbose)]
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

def add_variant(seq, 
                allele, 
                rec_start, 
                rec_stop): 
    # By definition, each position in the reference genome is a single base
    # So, if the allele is a deletion, we need to add "-" to the end of the allele
    # If the allele is an insertion, we need to add the allele to the start of the sequence
    # If the allele is a substitution, we need to add the allele to the start of the sequence
    if len(allele) < 1:  # Deletion
        allele = allele + "-" * (abs(rec_stop - rec_start) - len(allele))
        seq[rec_start:rec_stop] = list(allele)
    elif len(allele) > 1:  # Insertion
        seq[rec_start] = "".join(allele)
    else:
        seq[rec_start] = "".join(allele)
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

def personalize_nuc_seqs(tx,
                         vcf_in,
                         ref_genome,
                         ref_seq,
                         sample,
                         variant_types=None,#["SNP", "DEL", "INS"],
                         check_ref_mismatch=True,
                         as_list=True):
    variant_recorder = []
    chrom = "chr"+tx.contig.replace("chr","")  
    # Get the reference sequence
    if ref_seq is None:
        ref_seq = get_ref_seq(tx=tx)
    # Convert ref_seq to list of strings
    ref_seq = ["".join(x) for x in list(ref_seq)]
    # Return reference sequence if sample is "REFERENCE"
    if sample == "REFERENCE":
        return [ref_seq, ref_seq], variant_recorder
    # Get personalized sequences
    seq1 = ref_seq.copy()
    seq2 = ref_seq.copy()
    # Iterate over variants within the transcript
    for rec in vcf_in.fetch(chrom, tx.start, tx.end): 
        if variant_types is not None:
            if rec.alleles_variant_types[1] not in variant_types:
                continue
        # compute relative range of the variant in the transcript (manually, without using tx.offset_range)
        # VCF records already add 1 to the end position, so no need to add 1 to the end position
        rec_start, rec_stop = rec.start - tx.start, rec.stop - tx.start 
        # Confirm that the reference allele is the same as the reference allele in the VCF  
        ref_seq_rec = "".join(ref_seq[rec_start:rec_stop])
        if check_ref_mismatch and (rec.ref != ref_seq_rec):
            raise ValueError(f"Reference allele mismatch: {rec.ref} != {ref_seq_rec}")
        # Confirm that the reference allele is the same as the reference allele in the VCF  
        # Get reference sequence for the variant
        ref_seq_rec = "".join(ref_genome[chrom][rec.start:rec.stop].seq)
        if check_ref_mismatch and (rec.ref != ref_seq_rec):
            raise ValueError(f"Reference allele mismatch: {rec.ref} != {ref_seq_rec}")
        # Get sample data
        sample_data = rec.samples[sample]  # Get sample by name
        alleles = rec.alleles # tuple of reference allele followed by alt alleles
        gt = sample_data.allele_indices  # Tuple of genotype indices for a sample (phase1, phase2) 
        # sample_data.phased
        # Phase 1 allele
        allele1 = alleles[gt[0]]
        check_allele_length(allele1, rec)
        seq1 = add_variant(seq1, allele1, rec_start, rec_stop)
        # Phase 2 allele
        allele2 = alleles[gt[1]]
        check_allele_length(allele2, rec)
        seq2 = add_variant(seq2, allele2, rec_start, rec_stop)
        # Record variant (only if the variant is not already recorded)
        variant_id = f"{rec.chrom}:{rec.pos}_{allele1}_{allele2}"
        if variant_id not in variant_recorder:
                variant_recorder += [variant_id]
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
                     verbose):
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
    # Process each transcript
    for tx in tqdm(chr_transcripts, desc=f"Processing {chrom} transcripts"): 
        transcript_id = tx.id
        # Create storage variables
        variant_recorder = {}
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
            # Get personalized sequences
            for sample in samples[:max_samples]:
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
                    to_stop=to_stop
                )   
                nuc_seqs[sample] = nuc_seqs_i
                aa_seqs[sample] = aa_seqs_i
                variant_recorder[sample] = variant_recorder_i
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
                    'variant_recorder': variant_recorder,
                    'problem_seqs': problem_seqs
                }
                with open(save_path, 'wb') as handle:
                    pickle.dump(tx_results, handle)
        results[transcript_id] = tx_results
    # Return dictionary of results
    return chrom, results

def personalize_proteins(vcf_files,
                         ref_genome,
                         max_files = None,
                         max_transcripts = None,
                         transcript_ids = None,
                         max_samples = None,
                         samples = None,
                         save_dir = "1KG/sequence_dict",
                         max_workers = 1,
                         force = False,
                         codon_buffer = None,
                         to_stop = False,
                         include_reference = True,
                         executor = "ThreadPoolExecutor",
                         group_by_chrom = False,
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
                               verbose=verbose)
        results_all = list(tqdm(executor.map(process_func, vcf_files), 
                           desc="Processing VCF files",
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
        response = requests.get(f"url")
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
                coding_only = False):
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
    return ref_seq

def get_offset_ranges(tx, 
                      ranges=None):
   if ranges is None:
       ranges = tx.coding_sequence_position_ranges
   offset_ranges = [tx.offset_range(x[0],x[1]) for x in ranges]
   return offset_ranges

def subset_seq(seq, 
               offset_ranges): 
    seq_subset = "".join(["".join(seq[x[0]-1:x[1]]) for x in offset_ranges])
    return seq_subset

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

def get_translated_seq(tx, **kwargs):
    from Bio.Seq import Seq
    relative_ranges = get_offset_ranges(tx)
    final_seq = Seq('')
    for x in relative_ranges:
        final_seq += Seq(tx.coding_sequence[x[0]:x[1]])
    return final_seq.translate(**kwargs)

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
    return sum([1 for x,y in zip(seq1, seq2) if x == y]) / len(seq1)


def get_sequence_similarity(results_all, 
                            aa_seqs=None,
                            db=None,
                            as_df=True):
    from tqdm.auto import tqdm
    import pandas as pd
    if db is None:
        db = get_db()
    # For each transcript get the sequence similarity between the reference and each of the two haplotypes within each sample 
    seq_similarity = {} 
    if aa_seqs is None:
        aa_seqs = get_aa_seqs(results_all, db=db)
    for trancript_id, x in tqdm(aa_seqs.items(),"Processing transcripts"):
        if 'REFERENCE' in x.keys():
            aa_ref_seq = x['REFERENCE'][0]
        else:
            aa_ref_seq = db.transcript_by_id(trancript_id).protein_sequence
        seq_similarity[trancript_id] = []
        for sample, seqs in x.items():
            if sample == 'REFERENCE':
                continue
            if seqs[0] is not None:
                seq1_sim = sequence_similarity(aa_ref_seq, seqs[0])
            else:
                seq1_sim = None
            if seqs[1] is not None:
                seq2_sim = sequence_similarity(aa_ref_seq, seqs[1])
            else:
                seq2_sim = None
            seq_similarity[trancript_id] += [seq1_sim, seq2_sim]
    if as_df:
        seq_similarity_df = pd.DataFrame(seq_similarity)
        samples = [x['aa_seqs'].keys() for x in results_all.values()][0]
        # interleave the samples with itself
        samples = [item for sublist in zip(samples, samples) for item in sublist if item != 'REFERENCE']
        seq_similarity_df.index = samples
        seq_similarity_df = seq_similarity_df.melt(ignore_index=False, 
                            var_name="transcript_id",
                                value_name="similarity")   
        return seq_similarity_df
    else:
        return seq_similarity
    
def get_aa_seqs(results_all, db=None):
    if db is None:
        db = get_db()
    return {k: dict([('REFERENCE', db.transcript_by_id(k).protein_sequence)] + list(v['aa_seqs'].items())) if 'REFERENCE' not in v['aa_seqs'] else v['aa_seqs'] for k,v in results_all.items()}

def get_nuc_seqs(results_all, db=None):
    if db is None:
        db = get_db()
    return {k: dict([('REFERENCE', db.transcript_by_id(k).coding_sequence)] + list(v['nuc_seqs'].items())) if 'REFERENCE' not in v['nuc_seqs'] else v['nuc_seqs'] for k,v in results_all.items()}
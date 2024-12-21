
def import_transcript_exons(gtf_file = "GRCh38/gencode.v47.annotation.gtf.gz",
                            save_path = "transcript_exons.pkl", 
                            transcript_ids=None,
                            force = False):
    import pickle
    import os
    import HTSeq
    from tqdm import tqdm
    if save_path is not None and os.path.exists(save_path) and not force:
        print(f"Loading transcript exons from {save_path}")
        with open(save_path, 'rb') as f:
            transcript_exons = pickle.load(f)
        # Skip if transcript_ids is specified and transcript_id is not in transcript_ids
        if transcript_ids is not None:
            # subset transcript_exons to transcript_ids
            transcript_exons = {k: v for k, v in transcript_exons.items() if k in transcript_ids}
    else:
        # Load the GTF Annotation File 
        gtf = HTSeq.GFF_Reader(gtf_file)
        # Collect Exons for Each Protein-Coding Transcript
        transcript_exons = {}
        for feature in tqdm(gtf, desc="Processing GTF features"):
            if feature.type == "CDS" and feature.attr.get("transcript_type") == "protein_coding":
                transcript_id = feature.attr["transcript_id"]
                # Skip if transcript_ids is specified and transcript_id is not in transcript_ids
                if transcript_ids is not None and transcript_id not in transcript_ids:
                    continue
                if transcript_id not in transcript_exons:
                    transcript_exons[transcript_id] = []
                transcript_exons[transcript_id].append(feature)
        print(len(transcript_exons),"transcripts assembled.")
        # save transcript_exons
        if save_path is not None:
            print(f"Saving transcript exons to {save_path}")
            with open(save_path, 'wb') as f:
                pickle.dump(transcript_exons, f)
    print("Returning",len(transcript_exons),"transcripts.")
    return transcript_exons

# transcript_exons = import_transcript_exons(transcript_ids=knownCanonical[4].tolist(), force=False)


def get_chr_transcripts(transcript_exons, 
                        chrom=None,
                        max_transcripts=None,
                        verbose=True):
    """
    Filter transcripts for a specific chromosome and get first transcript's exons.
    
    Args:
        transcript_exons: Dictionary mapping transcript IDs to their exons
        chrom: Chromosome to filter for (default: None)
        
    Returns:
        tuple: (transcript_id, sorted_exons) for first transcript on chromosome
        
    Raises:
        ValueError: If no transcripts found for specified chromosome
    """
    # Get all transcripts if chrom is None
    if chrom is None:
        chr_transcripts = transcript_exons
    else:
        chr_transcripts= {}
        for tid, exons in transcript_exons.items():
            if exons[0].iv.chrom == chrom:
                chr_transcripts[tid] = exons
    if not chr_transcripts:
        raise ValueError(f"No {chrom} transcripts found")
    else:
        if verbose:
            print(len(chr_transcripts), f"transcripts on {chrom} found.")
    if max_transcripts is not None:
        # subset dict to first max_transcripts transcripts
        chr_transcripts = dict(list(chr_transcripts.items())[:max_transcripts])
    return chr_transcripts

def get_transcript_exons(chr_transcripts, transcript_id, verbose=True):
    # Get first transcript
    if transcript_id is None:
        transcript_id = list(chr_transcripts.keys())[0]
        if verbose:
            print(f"Using first transcript ny default: {transcript_id}")
    exons = chr_transcripts[transcript_id]
    # Sort exons by genomic position
    exons.sort(key=lambda x: x.iv.start)
    # Return transcript ID and sorted exons
    return exons


# Function to Get Personalized Sequence for a Genomic Region
def get_personalized_sequence(chrom, start, end, ref_genome, vcf_in, sample, transcript_id, codon_buffer="N"):
    variant_recorder[transcript_id] = {}
    variant_recorder[transcript_id][sample] = []
    # Get the reference sequence
    seq = ref_genome[chrom][start:end].seq  # 0-based indexing
    seq1 = list(seq)  # Convert to list for mutability
    seq2 = list(seq)  # Second sequence for phase 2
    if sample != "REFERENCE":
        # Fetch variants in this region
        for rec in vcf_in.fetch(chrom, start, end):
            # Adjust position
            pos = rec.pos - 1 - start  # Position relative to the start of the sequence
            if pos >= 0 and pos < len(seq1):
                # Get the genotype of the individual
                sample_data = rec.samples[sample]  # Get sample by name
                alleles = [rec.ref] + list(rec.alts)  # Convert tuple to list before concatenating
                gt = sample_data['GT']  # Genotype tuple (phase1, phase2) 
                # Apply both alleles
                if gt[0] is not None and gt[0] != 0:
                    seq1[pos] = alleles[gt[0]]  # Phase 1 allele
                if gt[1] is not None and gt[1] != 0:
                    seq2[pos] = alleles[gt[1]]  # Phase 2 allele
                variant_recorder[transcript_id][sample] += [f"{rec.chrom}:{rec.pos}_{gt[0]}_{gt[1]}"]
    # Add "N" buffer for sequences that are not multiples of 3
    if codon_buffer is not None:
        if len(seq1) % 3 != 0:
            seq1 = seq1 + [codon_buffer] * (3 - len(seq1) % 3)
        if len(seq2) % 3 != 0:
            seq2 = seq2 + [codon_buffer] * (3 - len(seq2) % 3)
    return ''.join(seq1), ''.join(seq2)

def get_protein_sequences(exons, ref_genome, vcf_in, sample, transcript_id, to_stop=True, buffer='N'):
    """
    Get personalized protein sequences for both haplotypes of a transcript.
    
    Args:
        exons: List of exon objects containing genomic coordinates
        ref_genome: Reference genome object
        vcf_in: VCF file object containing variants
        
    Returns:
        tuple: (protein_seq1, protein_seq2) - Protein sequences for both haplotypes
    """
    # Initialize CDS Sequence
    cds_seq1 = ''
    cds_seq2 = '' 
    # Iterate Over Exons to Build CDS
    for exon in exons:
        chrom = exon.iv.chrom
        start = exon.iv.start  # 0-based
        end = exon.iv.end
        exon_seq1, exon_seq2 = get_personalized_sequence(chrom, 
                                                         start, 
                                                         end, 
                                                         ref_genome, 
                                                         vcf_in, sample, 
                                                         transcript_id)
        cds_seq1 += exon_seq1
        cds_seq2 += exon_seq2
    # Adjust for Strand Orientation
    strand = exons[0].iv.strand
    if strand == '-':
        cds_seq1 = str(Seq(cds_seq1).reverse_complement())
        cds_seq2 = str(Seq(cds_seq2).reverse_complement())
    # Add buffer to the end of the sequence
    if buffer is not None:
        cds_seq1 += buffer * (3 - len(cds_seq1) % 3)
        cds_seq2 += buffer * (3 - len(cds_seq2) % 3)
    # Translate CDS to Protein Sequence 
    protein_seqs = ['', '']
    for i,phase in enumerate(["phase1", "phase2"]):
        if phase == "phase1":
            cds_seq = cds_seq1
        else:
            cds_seq = cds_seq2
        try:
            dna_seq = Seq(cds_seq).replace("-","").replace(".","").replace("=","")
            if len(dna_seq) % 3 != 0:
                problem_seqs[transcript_id][sample] += [(transcript_id, sample, "phase1")]
            protein_seqs[i] = str(dna_seq.translate(to_stop=to_stop))
        except:
            problem_seqs[transcript_id][sample] += [(transcript_id, sample, "phase1")]
            protein_seqs[i] = pd.NA 
    return protein_seqs[0], protein_seqs[1]
    

def compare_protein_sequences(transcript_id, protein_seq1, protein_seq2):
    """
    Compare two protein sequences and print differences.
    
    Args:
        transcript_id: ID of the transcript being analyzed
        protein_seq1: First protein sequence
        protein_seq2: Second protein sequence
    """
    print(f"Protein sequence for {transcript_id}:\n{protein_seq1}")
    print(f"Protein sequence for {transcript_id}:\n{protein_seq2}")
    
    if protein_seq1 == protein_seq2:
        print("The two protein sequences are identical")
    else:
        print("The two protein sequences differ")
        # Print positions where they differ
        for i, (aa1, aa2) in enumerate(zip(protein_seq1, protein_seq2)):
            if aa1 != aa2:
                print(f"Position {i+1}: {aa1} vs {aa2}")

def make_personal_seqs(vcf_files, 
                       ref_genome,
                       max_files = None,
                       max_transcripts = None,
                       max_samples = None,
                       samples = None,
                       save_dir = "1KG/sequence_dict",
                       max_workers = 1,
                       force = False,
                       buffer = 'N',
                       include_reference = True,
                       verbose = 0):
    import pickle
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    from functools import partial
    from tqdm.auto import tqdm
    import pyfaidx
    import os
    chr_transcript_seqs = {}
    chr_seq_counter = {}
    chr_variant_recorder = {}
    chr_problem_seqs = {}
    if max_files is not None:
        vcf_files = vcf_files[:max_files]
        if verbose:
            print(f"Only {len(vcf_files)} files being processed.")
    os.makedirs(save_dir, exist_ok=True)

    def process_vcf_file(f, transcript_exons, max_transcripts, max_samples, samples, ref_genome, save_dir, force, position):
        # Init inside each thread instance in accordance with: https://github.com/mdshw5/pyfaidx/issues/92#issuecomment-230269505
        if isinstance(ref_genome, str):
            ref_genome = pyfaidx.Fasta(ref_genome)
        chrom = os.path.basename(f).split('.')[1] 
        # Create new file
        if verbose:
            print(f"Processing VCF: {f}")
        vcf_in = pysam.VariantFile(f) 
        # Create storage variables
        global variant_recorder, problem_seqs
        transcript_seqs = {}
        seq_counter = {}
        variant_recorder = {}
        problem_seqs = {} 
        # Get relevant transcript coordinates
        chr_transcripts = get_chr_transcripts(transcript_exons,
                                             chrom=chrom,
                                             max_transcripts=max_transcripts,
                                             verbose=verbose>1)
        # List samples
        if samples is None:
            samples = list(vcf_in.header.samples)
        else:
            samples = [x for x in samples if x in list(vcf_in.header.samples)+["REFERENCE"]]
        # Create progress bar for this chromosome
        pbar = tqdm(chr_transcripts, 
                   desc=f"Processing {chrom} transcripts",
                   position=position,
                   leave=True) 
        # Process each transcript
        for transcript_id in pbar:
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
                # Process transcript
                # Get exons coordinates
                exons = get_transcript_exons(chr_transcripts, transcript_id, verbose=verbose>1) 
                # Include reference sequence
                if include_reference is True:
                    samples = list(set(["REFERENCE"]+samples))
                # Get personalized sequences
                for sample in samples[:max_samples]:
                    transcript_seqs[transcript_id][sample] = get_protein_sequences(
                        exons=exons,
                        ref_genome=ref_genome,
                        vcf_in=vcf_in,
                        sample=sample,
                        transcript_id=transcript_id,
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
    # Process VCF files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, vcf_file in enumerate(vcf_files):
            process_func = partial(process_vcf_file,
                                   transcript_exons=transcript_exons,
                                   max_transcripts=max_transcripts,
                                   max_samples=max_samples,
                                   samples=samples,
                                   ref_genome=ref_genome,
                                   save_dir=save_dir,
                                   force=force,
                                   position=i)
            futures.append(executor.submit(process_func, vcf_file))
        results = [f.result() for f in futures]
    ref_genome.close()
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



# Load the Reference Genome
# ref_genome = pyfaidx.Fasta( "GRCh38/GRCh38_full_analysis_set_plus_decoy_hla.fa")
(chr_transcript_seqs, 
 chr_seq_counter, 
 chr_variant_recorder, 
 chr_problem_seqs) = make_personal_seqs(vcf_files, 
                                        ref_genome = "GRCh38/GRCh38_full_analysis_set_plus_decoy_hla.fa",
                                        save_dir = "1KG/sequence_dict_all2",
                                        max_files = None,
                                        max_transcripts = None,
                                        max_samples = 3,
                                        # samples = "REFERENCE",
                                        max_workers = 8,
                                        force=True)
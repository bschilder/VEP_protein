# Generate FASTA file for each sample and chromosome

def gatk_FastaAlternateReferenceMaker(chrs, 
                                      max_workers=16,
                                      ref_fa='GRCh38/GRCh38_full_analysis_set_plus_decoy_hla.fa',
                                      in_dir='1KG/vcf_formatted',
                                      out_dir='1KG/gatk_FastaAlternateReferenceMaker',
                                      max_samples=None):
    """
    Generate FASTA files for each sample and chromosome using GATK FastaAlternateReferenceMaker
    NOTE: This tool works only for SNPs and for simple indels (but not for things like complex substitutions).
    Args:
        chrs: List of chromosomes to process
        max_workers: Number of parallel processes to use
        max_samples: Maximum number of samples to process
        ref_fa: Path to reference FASTA file
        in_dir: Path to input VCF directory
        out_dir: Path to output FASTA directory
    """
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    import multiprocessing
    import os

    ncores = multiprocessing.cpu_count()
    print(f"Parallelizing across {max_workers}/{ncores} CPU cores")

    def process_sample(sample, chr):
        out_path = f"{out_dir}/{sample}.{chr}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.fasta"
        if not os.path.exists(out_path) or (os.path.exists(out_path)  and os.path.getsize(out_path) == 0) :
            print(f"Processing Sample: {sample}, Chromosome: {chr}")
            !module load EBModules GATK > /dev/null 2>&1 && \
            gatk FastaAlternateReferenceMaker \
                -R {ref_fa} \
                -V {in_dir}/ALL.{chr}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz \
                -O {out_path} \
                --intervals "{chr}" \
                --use-iupac-sample {sample}
        else:
            print(f"Skipping Sample: {sample}, Chromosome: {chr}")
        return out_path
    # Iterate over chromosome (not parallel)
    for chr in chrs:
        !module load EBModules BCFtools > /dev/null 2>&1 && \
            bcftools query -l {in_dir}/ALL.{chr}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz > samples.txt
        with open('samples.txt', 'r') as f:
            samples = f.read().splitlines()
        if max_samples is not None:
            samples = samples[:max_samples]
        # Process samples in parallel 
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            process_chr = partial(process_sample, chr=chr)
            executor.map(process_chr, samples)

gatk_FastaAlternateReferenceMaker(chrs, max_samples=2)
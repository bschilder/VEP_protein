def bcftools_consensus(chrs, 
                       in_dir='1KG/vcf_norm',
                       out_dir='1KG/bcftools_consensus',
                       ref_fa='GRCh38/GRCh38_full_analysis_set_plus_decoy_hla.fa',
                       bed_file=None, 
                       mask_file=None,
                       max_samples=None, 
                       phased=True,
                       parallel=True,
                       force=False,
                       max_workers=16):
    """
    Generate FASTA files for each sample and chromosome using bcftools consensus.
    
    Args:
        chrs: List of chromosomes to process
        bed_file: Optional BED file to restrict regions
        max_samples: Maximum number of samples to process per chromosome
        phased: Whether to use phased genotypes
        parallel: Whether to process samples in parallel or sequentially
        processes: Number of parallel processes to use
    -H, --haplotype WHICH         
    choose which allele to use from the FORMAT/GT field, note
    the codes are case-insensitive:  
        1: first allele from GT, regardless of phasing  
        2: second allele from GT, regardless of phasing  
        R: REF allele in het genotypes  
        A: ALT allele  
        I: IUPAC code for all genotypes  
        LR,LA: longer allele and REF/ALT if equal length  
        SR,SA: shorter allele and REF/ALT if equal length  
        1pIu,2pIu: first/second allele for phased and IUPAC code for unphased GTs  

    Example:
        bcftools_consensus(chrs, 
                   ref_fa="GRCh38/GRCh38_full_analysis_set_plus_decoy_hla.fa",
                   bed_file="ucsc/ucsc_knownGene_CDS.bed", #"cds_ranges.bed",
                #    mask_file="ucsc/ucsc_knownGene_mask.bed",
                   max_samples=2, 
                   out_dir='1KG/bcftools_consensus_exons', 
                   parallel=False, 
                   force=True)


    """
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    import multiprocessing
    import pandas as pd
    import os

    if parallel:
        ncores = multiprocessing.cpu_count()
        print(f"Parallelizing across {max_workers}/{ncores} CPU cores")
    phases = ['1pIu','2pIu'] if phased else ['I'] 
    
    def process_sample(sample, chr, bed_file, mask_file, phases, ref_fa): 
        for phase in phases:
            output_file = f"{out_dir}/{sample}.{chr}.{phase}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.fasta"
            if not os.path.exists(output_file) or force:
                in_vcf = f"{in_dir}/ALL.{chr}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz"
                print(f"\nProcessing {chr}: Sample={sample}, Phase={phase}", flush=True)
                if bed_file is not None:
                    # Subset the BED file to the relevant chromosome and convert to ranges
                    bed_path, range_path = bed2ranges(bed_file, chr=chr)
                    # Subset the VCF to the relevant regions and sample
                    tmp_vcf = f"/tmp/{sample}.{chr}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.cds.vcf.gz"
                    !module load EBModules BCFtools SAMtools > /dev/null 2>&1 && \
                        bcftools view -R {bed_path} -s {sample} -Oz -o {tmp_vcf} {in_vcf} && \
                        bcftools index \
                            -t {tmp_vcf} 
                    # Create reference sequence
                    if mask_file is not None:
                        !module load EBModules BCFtools SAMtools> /dev/null 2>&1 && \
                        samtools faidx {ref_fa} -r {range_path} | \
                        bcftools consensus \
                            -o {output_file} \
                            --missing "?" \
                            --mask {mask_file} \
                            -H {phase} \
                            -s {sample} \
                            {tmp_vcf}
                    else:
                         !module load EBModules BCFtools SAMtools> /dev/null 2>&1 && \
                        samtools faidx {ref_fa} -r {range_path} | \
                        bcftools consensus \
                            -o {output_file} \
                            --missing "?" \
                            -H {phase} \
                            -s {sample} \
                            {tmp_vcf}
                else:
                    # Create reference sequence
                    !module load EBModules BCFtools SAMtools> /dev/null 2>&1 && \
                    samtools faidx {ref_fa} {chr} | \
                    bcftools consensus \
                        -o {output_file} \
                        --missing "?" \
                        -H {phase} \
                        -s {sample} \
                        {in_vcf}
            else:
                print(f"Skipping Sample: {sample}, Chromosome: {chr}, Phase: {phase}", flush=True)
    for chr in chrs:
        !module load EBModules BCFtools > /dev/null 2>&1 && \
            bcftools query -l {in_dir}/ALL.{chr}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz > samples.txt
        with open('samples.txt', 'r') as f:
            samples = f.read().splitlines()
        if max_samples is not None:
            samples = samples[:max_samples]
        
        if parallel:
            # Process samples in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                process_chr = partial(process_sample, chr=chr, bed_file=bed_file, mask_file=mask_file, phases=phases, ref_fa=ref_fa)
                executor.map(process_chr, samples)
        else:
            # Process samples sequentially
            for sample in samples:
                process_sample(sample, chr=chr, bed_file=bed_file, mask_file=mask_file, phases=phases, ref_fa=ref_fa)


def add_range(bed):
    bed['range'] = bed.apply(lambda x: f"{x[0]}:{x[1]}-{x[2]}", axis=1) 
    
def bed2ranges(bed_file, 
               chr=None,
               temp_dir="/tmp", 
               return_df=False):
    # Create bed file
    import pandas as pd
    bed = pd.read_csv(bed_file, sep="\t", header=None)
    if chr is not None:
        bed = bed.loc[bed[0]==chr]
    else:
            chr = 'ALL'
    sub_path = f"{temp_dir}/cds_ranges.{chr}.bed"
    bed.iloc[:,:3].to_csv(sub_path, sep="\t", index=False, header=None)
    # Create range file
    range_path = sub_path.replace(".bed",".txt")
    add_range(bed)
    bed['range'].to_csv(range_path, sep=" ", index=False, header=None)
    if return_df:
        return bed
    else:
        return sub_path, range_path


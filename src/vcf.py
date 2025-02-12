import sys
sys.path.append("/grid/koo/home/schilder/projects/GenomeEncoder/code/")
from src.onekg import get_chr_1kg
from src.utils import list_vcf

def reformat_vcfs(vcf_files=None,
                  vcf_dir=None, 
                  save_dir=None):
    import os
    if save_dir==vcf_dir:
        raise ValueError("save_dir and vcf_dir cannot be the same as the original VCF would be overwritten")
    if vcf_files is None:
        vcf_files = list_vcf(vcf_dir)
    if save_dir is None:
        raise ValueError("save_dir must be specified")
    os.makedirs(save_dir, exist_ok=True)
    # Reformat VCFs 
    new_vcf_files = []
    for vcf_file in vcf_files:
        chr = get_chr_1kg(vcf_file)
        print("Reformatting",chr)
        out_path = f"{save_dir}/{os.path.basename(vcf_file)}"
        new_vcf_files.append(out_path)
        if not os.path.exists(out_path):
            import subprocess
            cmd = (f"module load EBModules BCFtools > /dev/null 2>&1 && "
                  f"bcftools annotate "
                  f"--rename-chrs chr_name_conv.txt "
                  f"{vcf_file}"
                  f"-Oz -o {out_path} && "
                  f"bcftools index -t {out_path}")
            subprocess.run(cmd, shell=True, check=True)
        else:
            print("Skipping",chr)
    return new_vcf_files

def normalize_vcfs(vcf_files=None,
                   vcf_dir=None,
                   save_dir=None,
                   ref_genome="GRCh38/GRCh38_full_analysis_set_plus_decoy_hla.fa",
                   max_workers = 1):
    # Normalise VCFs in parallel
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    import os

    if vcf_files is None:
        vcf_files = list_vcf(vcf_dir)
    if save_dir is None:
        raise ValueError("save_dir must be specified")
    os.makedirs(save_dir, exist_ok=True)
    ncores = multiprocessing.cpu_count()
    print(f"Parallelizing across {max_workers}/{ncores} CPU cores")
    # Define function to normalize VCF
    def normalize_vcf(vcf_file, save_dir, ref_genome):
        chr = get_chr_1kg(vcf_file)
        out_path = f"{save_dir}/{os.path.basename(vcf_file)}"
        if not os.path.exists(out_path) or not os.path.exists(f"{out_path}.tbi"):
            print("Normalising", chr)
            import subprocess
            cmd = (f"module load EBModules BCFtools > /dev/null 2>&1 && "
                  f"bcftools norm "
                  f"-f {ref_genome} "
                  f"{vcf_file} "
                  f"-Oz "
                  f"-o {out_path} && "
                  f"bcftools index "
                  f"-t {out_path}")
            subprocess.run(cmd, shell=True, check=True)
        else:
            print("Skipping", chr)
        return out_path

    # Process chromosomes in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        new_vcf_files = list(executor.map(normalize_vcf, 
                                          vcf_files, 
                                          save_dir, 
                                          ref_genome))
    return new_vcf_files

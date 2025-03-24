#!/bin/bash
#SBATCH -p general
#SBATCH -t 4:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:4
#SBATCH -o alphafold_hbb_%j.out
#SBATCH -e alphafold_hbb_%j.err

# Load conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate alphafold

# Set paths
INPUT_DIR=~/VEP_protein/alphafold_input
OUTPUT_DIR=~/VEP_protein/alphafold_output
DATA_DIR=/net/scratch2/caom/alphafold_data

# Create output directory
mkdir -p $OUTPUT_DIR

# Function to run AlphaFold for a single variant
run_variant() {
    local fasta=$1
    local variant=$(basename "$fasta" .fasta)
    local variant_dir="$OUTPUT_DIR/$variant"
    
    echo "Processing $variant..."
    mkdir -p "$variant_dir"
    
    python /home/caom/AI_Genomics/models/alphafold/run_alphafold.py \
        --fasta_paths="$fasta" \
        --output_dir="$variant_dir" \
        --model_preset=monomer \
        --db_preset=reduced_dbs \
        --max_template_date=2024-1-1 \
        --use_gpu_relax=True \
        --data_dir=$DATA_DIR \
        --uniref90_database_path=$DATA_DIR/uniref90/uniref90.fasta \
        --mgnify_database_path=$DATA_DIR/mgnify/mgy_clusters_2022_05.fa \
        --template_mmcif_dir=$DATA_DIR/pdb_mmcif/mmcif_files \
        --obsolete_pdbs_path=$DATA_DIR/pdb_mmcif/obsolete.dat \
        --pdb70_database_path=$DATA_DIR/pdb70/pdb70 \
        --small_bfd_database_path=$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta
}

# Function to get GPU ID for parallel processing
get_gpu_id() {
    echo $(($1 % 4))  # Using modulo to cycle through 4 GPUs
}

# Process variants in parallel
echo "Starting parallel processing of variants..."
variant_count=0
for fasta in $INPUT_DIR/HBB_variant_*.fasta; do
    gpu_id=$(get_gpu_id $variant_count)
    export CUDA_VISIBLE_DEVICES=$gpu_id
    run_variant "$fasta" &
    
    # Increment counter
    variant_count=$((variant_count + 1))
    
    # Wait if we've started 4 jobs (one per GPU)
    if [ $((variant_count % 4)) -eq 0 ]; then
        wait
    fi
done

# Wait for any remaining jobs
wait

# Create a summary of the results
echo -e "\nSummary of AlphaFold predictions:"
echo "=================================="
for variant_dir in $OUTPUT_DIR/HBB_variant_*; do
    if [ -d "$variant_dir" ]; then
        variant=$(basename "$variant_dir")
        pdb_file=$(find "$variant_dir" -name "ranked_0.pdb" | head -n 1)
        if [ -f "$pdb_file" ]; then
            confidence=$(grep "pLDDT" "$pdb_file" | head -n 1 | awk '{print $NF}')
            echo "$variant: pLDDT = $confidence"
        else
            echo "$variant: No prediction found"
        fi
    fi
done

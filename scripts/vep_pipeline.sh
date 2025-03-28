#!/bin/bash
#SBATCH --job-name=vep_pipeline
#SBATCH --output=$TMPDIR/logs/vep_pipeline/vep_pipeline_%j.out
#SBATCH --error=$TMPDIR/logs/vep_pipeline/vep_pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p $TMPDIR/logs/vep_pipeline

# Activate conda environment
source $HOME/.bashrc
conda_init
conda activate esm2

# Set variables
MODELS="esm1v_t33_650M_UR90S_1"
SCORING_STRATEGIES="esm1v_t33_650M_UR90S_1:wt-marginals,masked-marginals"
HAP_DIR="$HOME/projects/VEP_protein/data/1KG/haplotypes"
SAVE_DIR="$HOME/projects/VEP_protein/data/1KG/vep"
VERBOSE=True

# Run the VEP pipeline
python -m $HOME/projects/VEP_protein/src/vep_pipeline \
  --models "${MODELS}" \
  --scoring_strategies "${SCORING_STRATEGIES}" \
  --hap_dir "${HAP_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --verbose "${VERBOSE}"

echo "VEP pipeline completed at $(date)"

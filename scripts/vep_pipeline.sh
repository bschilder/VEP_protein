#!/bin/bash
#$ -cwd
#$ -N vep_pipeline
#$ -l m_mem_free=128G
#$ -l gpu=1
#$ -pe threads 4

# Create logs directory if it doesn't exist
mkdir -p $TMPDIR/logs/vep_pipeline

# Activate conda environment
source $HOME/.bashrc
conda_init
conda activate esm2

# Set variables
MODELS="esm2_t12_35M_UR50D,esm2_t33_650M_UR50D"

SCORING_STRATEGIES="esm2_t12_35M_UR50D:wt-marginals,masked-marginals;esm2_t33_650M_UR50D:wt-marginals,masked-marginals"
HAP_DIR="$HOME/projects/VEP_protein/data/1KG/haplotypes"
SAVE_DIR="$HOME/projects/VEP_protein/data/1KG/vep"
VERBOSE=True

# Run the VEP pipeline
python -m $HOME/projects/VEP_protein/src/vep_pipeline.py \
  --models "${MODELS}" \
  --scoring_strategies "${SCORING_STRATEGIES}" \
  --hap_dir "${HAP_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --verbose "${VERBOSE}"

echo "VEP pipeline completed at $(date)"

#!/bin/bash
#SBATCH --job-name=enf_FT_CT
#SBATCH --output=out/CT_FT_%A_%a.out  # %A is the job ID, %a is the array index
#SBATCH --error=out/CT_FT_%A_%a.err   # %A is the job ID, %a is the array index
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amurphy@cshl.edu
#SBATCH --export=ALL
#SBATCH --array=0-13  # Define the array size based on number of model variants

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

# Load modules or activate conda environment
source /grid/it/data/elzar/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate clg

# Navigate to your project directory
cd /grid/koo/home/amurphy/projects/clg

# Make the training script executable
chmod 777 ./bin/catastrophic_forgetting_FT_training.py
chmod 777 ./*
chmod 777 ../clg

# Define model combinations (repo_id and model_variant)
# Format: "repo_id:model_variant"
COMBINATIONS=(
#    "anikethjr/finetuning-enformer:joint_regression_data_seed_7_lr_0.0005_wd_0.005_rcprob_0.5_rsmax_3"
    "anikethjr/finetuning-enformer:joint_regression_data_seed_97_lr_0.0005_wd_0.005_rcprob_0.5_rsmax_3"
    "anikethjr/finetuning-enformer:joint_regression_data_seed_42_lr_0.0005_wd_0.005_rcprob_0.5_rsmax_3"
    "anikethjr/finetuning-enformer:regression_data_seed_42_lr_0.0001_wd_0.001_rcprob_0.5_rsmax_3"
    "anikethjr/finetuning-enformer:regression_data_seed_7_lr_0.0001_wd_0.001_rcprob_0.5_rsmax_3"
    "anikethjr/finetuning-enformer:regression_data_seed_97_lr_0.0001_wd_0.001_rcprob_0.5_rsmax_3"
    "anikethjr/finetuning-enformer:classification_data_seed_42_lr_0.0001_wd_0.001_rcprob_0.5_rsmax_3"
    "anikethjr/finetuning-enformer:classification_data_seed_7_lr_0.0001_wd_0.001_rcprob_0.5_rsmax_3"
    "anikethjr/finetuning-enformer:classification_data_seed_97_lr_0.0001_wd_0.001_rcprob_0.5_rsmax_3"
    "anikethjr/finetuning-enformer:joint_regression_with_Malinois_MPRA_data_seed_42_lr_0.0001_wd_0.001_rcprob_0.5_rsmax_3"
    "anikethjr/finetuning-enformer:joint_regression_with_Malinois_MPRA_data_seed_7_lr_0.0001_wd_0.001_rcprob_0.5_rsmax_3"
    "anikethjr/finetuning-enformer:joint_regression_with_Malinois_MPRA_data_seed_97_lr_0.0001_wd_0.001_rcprob_0.5_rsmax_3"
    "anikethjr/finetuning-enformer:single_regression_counts_data_seed_42_lr_0.0001_wd_0.001_rcprob_0.5_rsmax_3"
    "anikethjr/finetuning-enformer:single_regression_counts_data_seed_7_lr_0.0001_wd_0.001_rcprob_0.5_rsmax_3"
    "anikethjr/finetuning-enformer:single_regression_counts_data_seed_97_lr_0.0001_wd_0.001_rcprob_0.5_rsmax_3"
)

# Get the combination for this array task
if [ $SLURM_ARRAY_TASK_ID -lt ${#COMBINATIONS[@]} ]; then
    COMBINATION=${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}
    
    # Split the combination into repo_id and model_variant
    REPO_ID=$(echo $COMBINATION | cut -d':' -f1)
    MODEL_VARIANT=$(echo $COMBINATION | cut -d':' -f2)
    
    echo "Running with REPO_ID: $REPO_ID"
    echo "Running with MODEL_VARIANT: $MODEL_VARIANT"
    
    # Run the training script with the selected parameters
    python ./bin/catastrophic_forgetting_FT_training.py \
        --seq_len 49152 \
        --data_path "/grid/koo/home/shared/enformer_data_npz/human/" \
        --freeze_base "transformer" \
        --repo_id "$REPO_ID" \
        --model_variant "$MODEL_VARIANT" \
        --wandb_api_key "751fa83d2b3b5f96687d0581696401cc8146fa94" \
	--use_scheduler

else
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of range"
    exit 1
fi

echo "End time: $(date)" 

# Deactivate the environment
conda deactivate
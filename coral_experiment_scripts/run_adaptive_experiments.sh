#!/bin/bash

#SBATCH --job-name=adaptive_seeds
#SBATCH --output=logs/adaptive_%A_%a.out
#SBATCH --error=logs/adaptive_%A_%a.err
#SBATCH --partition=p_mischaik_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --array=0-149          # 5 datasets x 30 seeds
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --export=ALL

cd /home/bg545/latent_dynamics

DATASETS=("train_500_100_adaptive" "train_500_200_adaptive" "train_500_300_adaptive" "train_500_400_adaptive" "train_500_500_adaptive")
N_SEEDS=30

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
DATASET=${DATASETS[$DATASET_IDX]}
OUTPUT_SUBDIR="${DATASET}/seed_${SEED}"

echo "Dataset: ${DATASET}, Seed: ${SEED}, Output subdir: ${OUTPUT_SUBDIR}"

srun python3 train.py \
    --config coral.yaml \
    --train_file ${DATASET} \
    --seed ${SEED} \
    --output_subdir ${OUTPUT_SUBDIR}

srun python3 morse_graph.py \
    --config coral.yaml \
    --train_file ${DATASET} \
    --output_subdir ${OUTPUT_SUBDIR}

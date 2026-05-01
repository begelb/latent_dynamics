#!/bin/bash

#SBATCH --job-name=coral_seeds
#SBATCH --output=logs/coral_%A_%a.out
#SBATCH --error=logs/coral_%A_%a.err
#SBATCH --partition=p_mischaik_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --array=0-179          # 6 datasets x 30 seeds
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --export=ALL

cd /home/bg545/latent_dynamics

DATASETS=("train_100" "train_200" "train_500" "train_1000" "train_2000" "train_5000")
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

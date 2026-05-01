#!/bin/bash
# Run once before submitting the job array.
# Fits and saves one scaler per dataset size.

cd /home/bg545/latent_dynamics

DATASETS=("train_100" "train_200" "train_500" "train_1000" "train_2000" "train_5000")

for DATASET in "${DATASETS[@]}"; do
    echo "Fitting scaler for ${DATASET}..."
    python3 scale_data.py --config coral.yaml --train_file ${DATASET}
done

echo "All scalers saved."

#!/bin/bash
# Run once before submitting the adaptive job array.
# Fits and saves one scaler per adaptively sampled dataset.

cd /home/bg545/latent_dynamics

DATASETS=("train_500_100_adaptive" "train_500_200_adaptive" "train_500_300_adaptive" "train_500_400_adaptive" "train_500_500_adaptive")

for DATASET in "${DATASETS[@]}"; do
    echo "Fitting scaler for ${DATASET}..."
    python3 scale_data.py --config coral.yaml --train_file ${DATASET}
done

echo "All scalers saved."

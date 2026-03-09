import pandas as pd
import numpy as np
import torch
import joblib
import os
from src.true_dynamics_models import RedCoralModel
from src.config import Config

def is_within_bounds(point, lower, upper):
    """Checks if a 13D point is within the specified lower and upper bounds."""
    return np.all(point >= lower) and np.all(point <= upper)

def run_hybrid_adaptive_sampling():
    # --- 1. Setup and Paths ---
    # Paths configured for the current iteration
    model_dir = 'output/coral_hybrid3/models'
    morse_sets_path = 'output/coral_hybrid3/MG/morse_sets'
    scaler_path = 'output/coral_hybrid3/data/scalers/scaler.gz'
    decoder_path = os.path.join(model_dir, 'decoder.pt')
    old_dataset_path = 'data/coral_hybrid3/train.csv'
    new_dataset_path = 'data/coral_hybrid4/train.csv'

    os.makedirs(os.path.dirname(new_dataset_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(scaler_path)
    decoder = torch.load(decoder_path, map_location=device, weights_only=False)
    decoder.eval()

    model = RedCoralModel()
    lower_bounds = np.array(model.lower_bounds)
    upper_bounds = np.array(model.upper_bounds)

    # --- 2. Sample 100 points in the Original 13D Space ---
    print("Sampling 100 points uniformly in the original 13D space...")
    initial_conditions_original = np.random.uniform(lower_bounds, upper_bounds, (100, 13))

    # --- 3. Sample 100 points using the Latent Gap Procedure ---
    print("Sampling 100 points in the latent gap [min(MS2), max(MS1)]...")
    df_morse = pd.read_csv(morse_sets_path)
    if 'a' not in df_morse.columns:
        df_morse = pd.read_csv(morse_sets_path, names=['a', 'b', 'label'])

    val_a = df_morse[df_morse['label'] == 2]['a'].min()  # Min of label 2
    val_b = df_morse[df_morse['label'] == 1]['b'].max()  # Max of label 1
    
    interval_min, interval_max = min(val_a, val_b), max(val_a, val_b)
    
    initial_conditions_latent = []
    while len(initial_conditions_latent) < 100:
        # Sample in latent space
        z_sample = np.random.uniform(interval_min, interval_max, (1, 1))
        
        with torch.no_grad():
            z_tensor = torch.from_numpy(z_sample).float().to(device)
            # Decode and inverse transform
            scaled_pt = decoder(z_tensor).cpu().numpy()
            unscaled_pt = scaler.inverse_transform(scaled_pt).flatten()
            
            # Validity Check: Only accept if within 13D domain bounds
            if is_within_bounds(unscaled_pt, lower_bounds, upper_bounds):
                initial_conditions_latent.append(unscaled_pt)

    initial_conditions_latent = np.array(initial_conditions_latent)

    # Combine all 200 initial points
    all_initial_points = np.vstack([initial_conditions_original, initial_conditions_latent])

    # --- 4. Generate Trajectories (20 steps each) ---
    n_iterations = 20
    X_new, Y_new = [], []

    print(f"Generating trajectories for {len(all_initial_points)} total points...")
    for point in all_initial_points:
        curr_pt = point.copy()
        for _ in range(n_iterations):
            next_pt = model.f(curr_pt)
            X_new.append(curr_pt)
            Y_new.append(next_pt)
            curr_pt = next_pt

    # --- 5. Combine and Save ---
    new_data_pts = np.hstack((np.array(X_new), np.array(Y_new)))
    old_df = pd.read_csv(old_dataset_path)
    new_df = pd.DataFrame(new_data_pts, columns=old_df.columns)

    augmented_df = pd.concat([old_df, new_df], ignore_index=True)
    augmented_df.to_csv(new_dataset_path, index=False)

    print(f"Success! Added {len(new_df)} transition pairs to the dataset.")
    print(f"Dataset size: {len(augmented_df)} rows.")
    print(f"Saved to: {new_dataset_path}")

if __name__ == "__main__":
    run_hybrid_adaptive_sampling()
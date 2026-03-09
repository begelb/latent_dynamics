import pandas as pd
import numpy as np
import torch
import joblib
import os
from src.true_dynamics_models import RedCoralModel
from src.config import Config

def run_adaptive_sampling():
    # --- 1. Setup and Paths ---
    # Maintaining paths from your adaptive_sampling.py script
    model_dir = 'output/coral_adaptive_roa2/models'
    morse_sets_path = 'output/coral_adaptive_roa2/MG/morse_sets'
    scaler_path = 'output/coral_adaptive_roa2/data/scalers/scaler.gz'
    decoder_path = os.path.join(model_dir, 'decoder.pt')
    old_dataset_path = 'data/coral_adaptive_roa2/train.csv'
    new_dataset_path = 'data/coral_adaptive_roa3/train.csv'

    # Ensure the target directory for the new training data exists
    os.makedirs(os.path.dirname(new_dataset_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load necessary assets
    scaler = joblib.load(scaler_path)
    decoder = torch.load(decoder_path, map_location=device)
    decoder.eval()

    # --- 2. Define Sampling Interval [min(MS2), max(MS1)] ---
    try:
        # Load Morse sets data (assumed CSV format from CMGDB)
        df = pd.read_csv(morse_sets_path)
        if 'a' not in df.columns:
            df = pd.read_csv(morse_sets_path, names=['a', 'b', 'label'])
    except Exception as e:
        print(f"Error loading Morse sets file: {e}")
        return

    # Filter for the specific labels you identified
    ms2_data = df[df['label'] == 2] # Separatrix
    ms1_data = df[df['label'] == 1] # Extinction

    if ms2_data.empty or ms1_data.empty:
        print("Error: Could not find required labels 1 and 2 in the Morse sets file.")
        print(f"Available labels are: {df['label'].unique()}")
        return

    # Boundary values as per your instructions
    val_a = ms2_data['a'].min()  # Min of label 2
    val_b = ms1_data['b'].max()  # Max of label 1

    # Define the range (using min/max to handle whichever bound is numerically smaller)
    interval_min = min(val_a, val_b)
    interval_max = max(val_a, val_b)

    print(f"Morse set 2 (separatrix) min boundary: {val_a:.6f}")
    print(f"Morse set 1 (extinction) max boundary: {val_b:.6f}")
    print(f"Sampling Interval: [{interval_min:.6f}, {interval_max:.6f}]")
    # --- 3. Uniform Random Sampling ---
    n_initial_samples = 200
    latent_samples = np.random.uniform(interval_min, interval_max, n_initial_samples).reshape(-1, 1)

    # --- 4. Decode to High-Dimensional State Space ---
    with torch.no_grad():
        z_tensor = torch.from_numpy(latent_samples).float().to(device)
        # Decode latent z back into the 13D population vector (scaled)
        initial_conditions_scaled = decoder(z_tensor).cpu().numpy()
        # Inverse transform to get actual colony counts for the RedCoralModel
        initial_conditions = scaler.inverse_transform(initial_conditions_scaled)

    # --- 5. Generate Trajectories (No Mortality Logic) ---
    model = RedCoralModel()
    n_iterations = 20
    X_new, Y_new = [], []

    print(f"Generating 20-step trajectories for {n_initial_samples} initial points...")
    for point in initial_conditions:
        curr_pt = point.copy()
        for _ in range(n_iterations):
            # Step the system forward
            next_pt = model.f(curr_pt)
            
            # Store transition pair (x_t, x_t+1)
            X_new.append(curr_pt)
            Y_new.append(next_pt)
            
            curr_pt = next_pt

    # --- 6. Combine with Old Dataset and Save ---
    new_data_pts = np.hstack((np.array(X_new), np.array(Y_new)))
    old_df = pd.read_csv(old_dataset_path)

    # Create new DataFrame with matching column headers
    new_df = pd.DataFrame(new_data_pts, columns=old_df.columns)

    # Append to existing training data
    augmented_df = pd.concat([old_df, new_df], ignore_index=True)
    augmented_df.to_csv(new_dataset_path, index=False)

    print(f"Success!")
    print(f" - Transitions added: {len(new_df)}")
    print(f" - Total training samples: {len(augmented_df)}")
    print(f" - New dataset saved to: {new_dataset_path}")

if __name__ == "__main__":
    run_adaptive_sampling()
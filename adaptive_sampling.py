import pandas as pd
import numpy as np
import torch
import joblib
import os
from src.true_dynamics_models import RedCoralModel # Adjust if mortality variant used
from src.config import Config

# --- 1. Setup and Paths ---
model_dir = 'output/coral_adaptive_second_iter/models'
save_path = 'output/coral_adaptive_second_iter/MG/'
morse_sets_path = 'output/coral_adaptive_second_iter/MG/morse_sets'
scaler_path = 'output/coral_adaptive_second_iter/data/scalers/scaler.gz'
decoder_path = os.path.join(model_dir, 'decoder.pt')
old_dataset_path = 'data/coral_adaptive_second_iter/train.csv'
new_dataset_path = 'data/coral_adaptive_third_iter/train.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load(scaler_path)
decoder = torch.load(decoder_path, map_location=device, weights_only=False)
decoder.eval()

# --- 2. Define Complement of Morse Set 0 ---
df = pd.read_csv(morse_sets_path)
if 'a' not in df.columns:
    df = pd.read_csv(morse_sets_path, names=['a', 'b', 'label'])

# Domain Z is the total extent of all Morse sets
z_min, z_max = df[['a', 'b']].min().min(), df[['a', 'b']].max().max()

# Intervals for Morse Set 0
ms0_intervals = df[df['label'] == 0][['a', 'b']].sort_values(by='a').values

# Calculate complement intervals within [z_min, z_max]
complement_intervals = []
curr = z_min
for start, end in ms0_intervals:
    if start > curr:
        complement_intervals.append((curr, start))
    curr = max(curr, end)
if curr < z_max:
    complement_intervals.append((curr, z_max))

# --- 3. Uniform Random Sampling in Complement ---
total_comp_length = sum(b - a for a, b in complement_intervals)
n_samples = 200
latent_samples = []

for a, b in complement_intervals:
    # Proportionally allocate samples to this specific gap
    interval_n = int(round(n_samples * (b - a) / total_comp_length))
    if interval_n > 0:
        samples = np.random.uniform(a, b, interval_n)
        latent_samples.extend(samples)

# Ensure exactly 200 samples (adjust for rounding)
latent_samples = np.array(latent_samples[:n_samples]).reshape(-1, 1)

# --- 4. Decode to High-Dimensional Initial Conditions ---
with torch.no_grad():
    # Decoder expects latent z, returns scaled high-dim vector
    z_tensor = torch.from_numpy(latent_samples).float().to(device)
    initial_conditions_scaled = decoder(z_tensor).cpu().numpy()
    
    # Inverse transform to get actual population counts
    initial_conditions = scaler.inverse_transform(initial_conditions_scaled)

# --- 5. Generate Trajectories (Simulation) ---
# Parameters consistent with make_data.py for coral system
model = RedCoralModel()
n_iterations = 20
X_new, Y_new = [], []

for point in initial_conditions:
    curr_pt = point.copy()
    for _ in range(n_iterations):
        next_pt = model.f(curr_pt)
        X_new.append(curr_pt)
        Y_new.append(next_pt)
        curr_pt = next_pt

# --- 6. Combine with Old Dataset and Save ---
new_data_pts = np.hstack((np.array(X_new), np.array(Y_new)))
old_df = pd.read_csv(old_dataset_path)

# Convert new points to DataFrame with matching headers
headers = old_df.columns
new_df = pd.DataFrame(new_data_pts, columns=headers)

# Final Augmented Dataset
augmented_df = pd.concat([old_df, new_df], ignore_index=True)
augmented_df.to_csv(new_dataset_path, index=False)

print(f"Sampling complete. {len(new_df)} new transitions added.")
print(f"New dataset saved to: {new_dataset_path}")
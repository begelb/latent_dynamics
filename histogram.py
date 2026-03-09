import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse

def plot_final_population_histograms(csv_path, save_path, steps_per_trajectory=20):
    """
    Plots histograms of the total population sum at the final step 
    of each trajectory in the dataset.
    """
    # 1. Load the dataset
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # 2. Identify the 'y' columns (the state after the transition)
    y_cols = [col for col in df.columns if col.startswith('y')]
    
    # 3. Extract the final row of every trajectory block
    # Trajectories are stored as contiguous blocks of 'steps_per_trajectory'
    # The last 'y' state of the trajectory is in the last row of the block.
    final_states_df = df.iloc[steps_per_trajectory - 1 :: steps_per_trajectory]
    
    # 4. Calculate the sum of all elements for the final populations
    final_populations = final_states_df[y_cols].sum(axis=1).values
    
    print(f"Processing {len(final_populations)} trajectories...")

    # --- Plotting Configuration ---
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.serif": ["STIXGeneral"]
    })

    plt.figure(figsize=(8, 6))
    ax1 = plt.gca()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color palette
    color_linear = '#648FFF' # Blue
    color_log = '#DC267F'    # Magenta

    # --- Plot 1: Linear Scale ---
    ax1.hist(final_populations, bins=10, color=color_linear, edgecolor='black', alpha=0.8)
  #  ax1.set_title("Final Population Distribution: Augumented Dataset", fontsize=22)
    ax1.set_xlabel("Total Final Population", fontsize=18)
    ax1.set_ylabel("Number of Trajectories", fontsize=18)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # # --- Plot 2: Logarithmic Scale ---
    # # We define log-spaced bins to ensure the bars look even on a log axis
    # # Handle the case where population might be 0 for the log scale
    # min_val = max(1e-6, final_populations.min())
    # max_val = final_populations.max()
    # log_bins = np.logspace(np.log10(min_val), np.log10(max_val), 10)

    # ax2.hist(final_populations, bins=log_bins, color=color_log, edgecolor='black', alpha=0.8)
    # ax2.set_xscale('log')
    # ax2.set_title("Final Population Sum (Log Scale)", fontsize=22)
    # ax2.set_xlabel("Total Population (Log)", fontsize=18)
    # ax2.set_ylabel("Number of Trajectories", fontsize=18)
    # ax2.tick_params(axis='both', labelsize=14)
    # ax2.grid(axis='y', linestyle='--', alpha=0.4)

    # plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"Histogram saved as: {save_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/coral',
                        help='Directory containing the training CSV files')
    parser.add_argument('--train_file', type=str, default='train',
                        help='Training CSV base name (without .csv)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save the histogram PDF')
    parser.add_argument('--steps_per_trajectory', type=int, default=20)
    args = parser.parse_args()

    csv_path      = os.path.join(args.data_dir, args.train_file + '.csv')
    metadata_path = os.path.join(args.data_dir, args.train_file + '_metadata.json')
    save_path     = os.path.join(args.output_dir, f'histogram_{args.train_file}.pdf')

    steps = args.steps_per_trajectory
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            steps = json.load(f)['n_iterations']
        print(f"Using n_iterations={steps} from {metadata_path}")

    plot_final_population_histograms(csv_path, save_path, steps)
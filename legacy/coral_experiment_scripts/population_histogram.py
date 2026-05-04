import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse

def plot_final_population_histograms(csv_path, save_path, steps_per_trajectory=20, ymax=None):
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
    ax1.set_yscale('log')

    # --- Plot ---
    ax1.hist(final_populations, bins=10, color=color_linear, edgecolor='black', alpha=0.8)
    ax1.set_xlabel("Total Final Population", fontsize=28)
    ax1.set_ylabel("Number of Trajectories", fontsize=28)
    ax1.tick_params(axis='both', labelsize=24)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # Y-axis: more tick marks and labels
    ax1.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
    ax1.yaxis.set_minor_locator(plt.LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=50))
    ax1.yaxis.set_minor_formatter(plt.LogFormatter(base=10, labelOnlyBase=False))
    ax1.tick_params(axis='y', which='minor', labelsize=10)

    # Record or apply ymax
    current_ymax = ax1.get_ylim()[1]
    if ymax is not None:
        ax1.set_ylim(top=ymax)
    else:
        print(f"y-axis max: {current_ymax:.4g}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Histogram saved as: {save_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/coral',
                        help='Directory containing the training CSV files')
    parser.add_argument('--train_file', type=str, default='train_500',
                        help='Training CSV base name (without .csv)')
    parser.add_argument('--output_dir', type=str, default='output/histogram',
                        help='Directory to save the histogram PDF')
    parser.add_argument('--ymax', type=float, default=1000,
                        help='Fix the y-axis maximum (use value printed on first run)')
    args = parser.parse_args()

    csv_path      = os.path.join(args.data_dir, args.train_file + '.csv')
    metadata_path = os.path.join(args.data_dir, args.train_file + '_metadata.json')
    save_path     = os.path.join(args.output_dir, f'histogram_{args.train_file}.pdf')

    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            steps = json.load(f)['n_iterations']
        print(f"Using n_iterations={steps} from {metadata_path}")
    else:
        print("Number of steps not found!")

    plot_final_population_histograms(csv_path, save_path, steps, ymax=args.ymax)
import pandas as pd
import numpy as np
import torch
import joblib
import os
import json
import argparse
import matplotlib.pyplot as plt
from src.true_dynamics_models import RedCoralModel
from src.config import Config


def is_within_bounds(point, lower, upper):
    """Checks if a point is within the specified lower and upper bounds."""
    return np.all(point >= lower) and np.all(point <= upper)


def plot_latent_samples(df_morse, z_samples, save_path):
    """Plot Morse set intervals with sampled latent points overlaid."""
    color_list = ['#FFB000', '#DC267F', '#648FFF', '#FE6100', '#785EF0']

    fig, ax = plt.subplots(figsize=(12, 2))
    for _, row in df_morse.iterrows():
        color = color_list[int(row['label']) % len(color_list)]
        ax.plot([row['a'], row['b']], [0, 0], color=color, linewidth=10,
                solid_capstyle='projecting', zorder=0)

    ax.scatter(z_samples, np.zeros(len(z_samples)), color='black',
               s=20, zorder=10, label='sampled points', alpha=0.5)

    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Latent sample plot saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default='config/')
    parser.add_argument('--config', type=str, default='coral.yaml')
    parser.add_argument('--train_file', type=str, default='train_500',
                        help='Training CSV base name used for the source model and dataset (without .csv)')
    parser.add_argument('--output_subdir', type=str, default='train_500/seed_16',
                        help='Subdirectory within output_dir containing the trained model (e.g. train_500/seed_)')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of initial conditions to sample via the latent gap procedure')
    parser.add_argument('--morse_label_low', type=int, default=0,
                        help='Morse set label defining the lower bound of the latent gap')
    parser.add_argument('--morse_label_high', type=int, default=2,
                        help='Morse set label defining the upper bound of the latent gap')
    args = parser.parse_args()

    config = Config(args.config_dir + args.config)

    # Derive paths from config
    if args.output_subdir is not None:
        subdir_root = os.path.join(config.output_dir, args.output_subdir)
    else:
        subdir_root = config.output_dir

    model_dir        = os.path.join(subdir_root, 'models')
    morse_sets_path  = os.path.join(subdir_root, 'MG', 'morse_sets')
    scaler_path      = os.path.join(config.scaler_dir, args.train_file, 'scaler.gz')
    old_dataset_path = os.path.join(config.data_dir, args.train_file + '.csv')
    new_dataset_path = os.path.join(config.data_dir,
                                    f'{args.train_file}_{args.n_samples}_adaptive.csv')
    metadata_path    = os.path.join(config.data_dir, args.train_file + '_metadata.json')
    decoder_path     = os.path.join(model_dir, 'decoder.pt')

    # Read n_iterations from metadata
    with open(metadata_path) as f:
        n_iterations = json.load(f)['n_iterations']
    print(f"Using n_iterations={n_iterations} from {metadata_path}")

    os.makedirs(os.path.dirname(new_dataset_path), exist_ok=True)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler  = joblib.load(scaler_path)
    decoder = torch.load(decoder_path, map_location=device, weights_only=False)
    decoder.eval()

    model        = RedCoralModel()
    lower_bounds = np.array(model.lower_bounds)
    upper_bounds = np.array(model.upper_bounds)

    # --- Sample initial conditions via the latent gap procedure ---
    print(f"Sampling {args.n_samples} points in the latent gap...")
    df_morse = pd.read_csv(morse_sets_path)
    if 'a' not in df_morse.columns:
        df_morse = pd.read_csv(morse_sets_path, names=['a', 'b', 'label'])

    val_a = df_morse[df_morse['label'] == args.morse_label_high]['b'].max()
    val_b = df_morse[df_morse['label'] == args.morse_label_low]['a'].min()
    interval_min, interval_max = min(val_a, val_b), max(val_a, val_b)

    initial_conditions = []
    z_samples = []
    while len(initial_conditions) < args.n_samples:
        z_sample = np.random.uniform(interval_min, interval_max, (1, 1))
        with torch.no_grad():
            z_tensor    = torch.from_numpy(z_sample).float().to(device)
            scaled_pt   = decoder(z_tensor).cpu().numpy()
            unscaled_pt = scaler.inverse_transform(scaled_pt).flatten()
            if is_within_bounds(unscaled_pt, lower_bounds, upper_bounds):
                initial_conditions.append(unscaled_pt)
                z_samples.append(z_sample.flatten()[0])

    plot_save_path = os.path.join(subdir_root, 'MG', 'latent_samples.pdf')
    # plot_latent_samples(df_morse, z_samples, plot_save_path)

    # --- Generate trajectories ---
    print(f"Generating trajectories for {args.n_samples} points "
          f"({n_iterations} steps each)...")
    X_new, Y_new = [], []
    for point in initial_conditions:
        curr_pt = point.copy()
        for _ in range(n_iterations):
            next_pt = model.f(curr_pt)
            X_new.append(curr_pt)
            Y_new.append(next_pt)
            curr_pt = next_pt

    # --- Combine with old dataset and save ---
    new_data_pts = np.hstack((np.array(X_new), np.array(Y_new)))
    old_df       = pd.read_csv(old_dataset_path)
    new_df       = pd.DataFrame(new_data_pts, columns=old_df.columns)
    augmented_df = pd.concat([old_df, new_df], ignore_index=True)
    augmented_df.to_csv(new_dataset_path, index=False)

    # --- Save metadata ---
    dataset_name  = os.path.splitext(os.path.basename(new_dataset_path))[0]
    metadata = {
        'dataset_name':      dataset_name,
        'system':            config.system,
        'dimension':         config.high_dims,
        'n_samples':         len(augmented_df) // n_iterations,
        'n_iterations':      n_iterations,
        'sampling_method':   'adaptive_latent',
        'source_dataset':    args.train_file,
        'n_adaptive_samples': args.n_samples,
        'source_model_subdir': args.output_subdir,
        'morse_label_low':   args.morse_label_low,
        'morse_label_high':  args.morse_label_high,
        'latent_interval':   [float(interval_min), float(interval_max)],
        'lower_bounds':      lower_bounds.tolist(),
        'upper_bounds':      upper_bounds.tolist(),
    }
    metadata_save_path = new_dataset_path.replace('.csv', '_metadata.json')
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Added {len(new_df)} transition pairs. "
          f"Total dataset size: {len(augmented_df)} rows. "
          f"Saved to: {new_dataset_path}")
    print(f"Metadata saved to: {metadata_save_path}")


if __name__ == '__main__':
    main()

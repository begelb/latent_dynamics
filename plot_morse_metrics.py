"""
Plots the fraction of seeds passing the unique Morse set membership metric
as a function of dataset size (number of initial conditions).

One line per fixed point (a0, a1, r). X-axis is the integer size extracted
from the dataset directory name (e.g. train_100 -> 100).
"""

import os
import argparse
import matplotlib.pyplot as plt
from src.config import Config
from compute_morse_metric import check_unique_membership, find_seed_subdirs


def find_dataset_subdirs(output_dir):
    """
    Return sorted (n_samples, subdir_name) pairs for subdirs of output_dir
    that contain seed results and have an integer size in their name.
    """
    datasets = []
    for entry in os.scandir(output_dir):
        if not entry.is_dir():
            continue
        # Extract integer from name, e.g. train_100 -> 100
        parts = entry.name.split('_')
        try:
            n_samples = int(parts[-1])
        except ValueError:
            continue
        # Only include if it contains seed subdirs
        if find_seed_subdirs(entry.path):
            datasets.append((n_samples, entry.name))
    return sorted(datasets)


def compute_pass_fractions(output_dir, scaler_dir, dataset_subdirs):
    """
    For each dataset, compute the fraction of seeds passing for a0, a1, r.

    Returns
    -------
    x      : list of int   dataset sizes
    fracs  : dict          {point_name: list of fractions}
    """
    x = []
    fracs = {'a0': [], 'a1': [], 'r': []}

    for n_samples, dataset_name in dataset_subdirs:
        dataset_path = os.path.join(output_dir, dataset_name)
        scaler_path  = os.path.join(scaler_dir, dataset_name, 'scaler.gz')
        seed_subdirs = find_seed_subdirs(dataset_path)

        counts = {'a0': 0, 'a1': 0, 'r': 0}
        for seed_subdir in seed_subdirs:
            encoder_path    = os.path.join(dataset_path, seed_subdir, 'models', 'encoder.pt')
            morse_sets_path = os.path.join(dataset_path, seed_subdir, 'MG', 'morse_sets')
            _, metrics = check_unique_membership(encoder_path, scaler_path, morse_sets_path)
            for name in counts:
                counts[name] += int(metrics[name])

        n_seeds = len(seed_subdirs)
        x.append(n_samples)
        for name in fracs:
            fracs[name].append(counts[name] / n_seeds)

    return x, fracs


def main():
    parser = argparse.ArgumentParser(description='Plot Morse metric pass fractions vs dataset size.')
    parser.add_argument('--config_dir', type=str, default='config/')
    parser.add_argument('--config',     type=str, default='coral.yaml')
    args = parser.parse_args()

    config = Config(args.config_dir + args.config)

    dataset_subdirs = find_dataset_subdirs(config.output_dir)
    if not dataset_subdirs:
        print(f'No dataset subdirectories with results found in {config.output_dir}.')
        return

    x, fracs = compute_pass_fractions(config.output_dir, config.scaler_dir, dataset_subdirs)

    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'stix',
        'font.serif': ['STIXGeneral']
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = {'a0': r'$E(a_0)$', 'a1': r'$E(a_1)$', 'r': r'$E(r)$'}
    colors = {'a0': '#648FFF', 'a1': '#DC267F', 'r': '#FFB000'}
    markers = {'a0': 'o', 'a1': 's', 'r': '^'}

    for name in ('a0', 'a1', 'r'):
        ax.plot(x, fracs[name], label=labels[name], color=colors[name],
                marker=markers[name], linewidth=2, markersize=7)

    ax.set_xlabel('Number of initial conditions', fontsize=14)
    ax.set_ylabel('Fraction of seeds passing', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    save_path = os.path.join(config.output_dir, 'morse_metric_plot.pdf')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f'Plot saved to {save_path}')
    plt.show()


if __name__ == '__main__':
    main()

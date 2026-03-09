"""
Computes the unique Morse set membership metric for fixed points a0, a1, r.

A Morse set is the union of all intervals with the same label in the
morse_sets CSV file (columns: a, b, label).

For each point p in {a0, a1, r}, the Boolean metric is:
  E(p) falls in some Morse set M, AND the other two points do NOT fall in M.

Usage
-----
  python compute_morse_metric.py --output_dir output/coral_hybrid4 --scaler_path output/coral_hybrid4/data/scalers/scaler.gz

The script auto-discovers subdirectories (e.g. seed_0, seed_1, ...) that
contain both models/encoder.pt and MG/morse_sets.
"""

import os
import torch
import joblib
import pandas as pd
import argparse


a0 = [0] * 13
a1 = [868.12066371, 771.75927004, 488.52361793, 340.5009617, 176.0389972,
      76.92904178, 22.07863499, 12.60690058, 4.19809789, 3.14857342,
      3.14857342, 1.04847495, 1.04847495]
r  = [321.84389612752153, 286.11922365736666, 181.1134685751131,
      126.23608759685382, 65.26405728757342, 28.52039303466959,
      8.185352800950172, 4.673836449342548, 1.5563875376310683,
      1.1672906532233014, 1.1672906532233014, 0.3887077875233593,
      0.3887077875233593]

FIXED_PTS = {'a0': a0, 'a1': a1, 'r': r}


def find_morse_label(z, morse_df):
    """Return the Morse set label containing scalar z, or None."""
    for _, row in morse_df.iterrows():
        if row['a'] <= z <= row['b']:
            return int(row['label'])
    return None


def check_unique_membership(encoder_path, scaler_path, morse_sets_path):
    """
    For each point p in {a0, a1, r}, compute the Boolean:
      E(p) is in some Morse set M, and the other two points are NOT in M.

    Returns
    -------
    labels  : dict  {point_name: Morse_label_or_None}
    metrics : dict  {point_name: bool}
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder.eval()
    scaler = joblib.load(scaler_path)

    morse_df = pd.read_csv(morse_sets_path)
    if 'a' not in morse_df.columns:
        morse_df = pd.read_csv(morse_sets_path, names=['a', 'b', 'label'])

    labels = {}
    with torch.no_grad():
        for name, pt in FIXED_PTS.items():
            pt_scaled = scaler.transform([pt])
            z = encoder(torch.from_numpy(pt_scaled).float().to(device))
            z_val = z.cpu().numpy().flatten()[0]
            labels[name] = find_morse_label(z_val, morse_df)

    metrics = {}
    for name in FIXED_PTS:
        my_label = labels[name]
        if my_label is None:
            metrics[name] = False
        else:
            others_in_same = any(labels[other] == my_label for other in FIXED_PTS if other != name)
            metrics[name] = not others_in_same

    return labels, metrics


def find_seed_subdirs(base_output_dir):
    """Return sorted subdirectory names that have models/encoder.pt and MG/morse_sets."""
    subdirs = []
    for entry in sorted(os.scandir(base_output_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        has_encoder    = os.path.isfile(os.path.join(entry.path, 'models', 'encoder.pt'))
        has_morse_sets = os.path.isfile(os.path.join(entry.path, 'MG', 'morse_sets'))
        if has_encoder and has_morse_sets:
            subdirs.append(entry.name)
    return subdirs


def main():
    parser = argparse.ArgumentParser(description='Compute unique Morse set membership metric.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory containing per-seed subdirs (e.g. output/coral_hybrid4)')
    parser.add_argument('--scaler_path', type=str, required=True,
                        help='Path to scaler.gz')
    args = parser.parse_args()

    subdirs = find_seed_subdirs(args.output_dir)

    if not subdirs:
        print(f'No seed subdirectories found in {args.output_dir}.')
        return

    all_metrics = {}
    for subdir in subdirs:
        subdir_path     = os.path.join(args.output_dir, subdir)
        encoder_path    = os.path.join(subdir_path, 'models', 'encoder.pt')
        morse_sets_path = os.path.join(subdir_path, 'MG', 'morse_sets')
        labels, metrics = check_unique_membership(encoder_path, args.scaler_path, morse_sets_path)
        all_metrics[subdir] = metrics
        print(f'{subdir}: a0={metrics["a0"]}  a1={metrics["a1"]}  r={metrics["r"]}  |  labels={labels}')

    print(f'Seeds evaluated: {len(all_metrics)}')
    for name in ('a0', 'a1', 'r'):
        n_pass = sum(v[name] for v in all_metrics.values())
        print(f'  {name} passes: {n_pass}/{len(all_metrics)}')


if __name__ == '__main__':
    main()

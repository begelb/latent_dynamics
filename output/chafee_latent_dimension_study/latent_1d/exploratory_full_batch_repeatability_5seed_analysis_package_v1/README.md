# Chafee–Infante d=1 full-batch repeatability audit

This exploratory audit ran the canonical 4,000-epoch, learning-rate 0.003,
full-batch Adam computation three times for each predeclared seed 0 through 4.
Each five-seed repeat ran in a fresh Python process. The repeat index did not
alter the training seed.

## Result

All three checkpoints and loss histories are byte-for-byte identical for every
fixed seed. The trajectory statistic or failure mode also repeats exactly.
These are deterministic replays on this hardware/runtime, not 15 independent
initializations.

| Seed | Repeat 1 | Repeat 2 | Repeat 3 | Valid statistics | Strict singleton cells | Non-singleton cells |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 44.3907% | 44.3907% | 44.3907% | 3/3 | 130/256 | 126/256 |
| 1 | 76.8888% | 76.8888% | 76.8888% | 3/3 | 176/256 | 80/256 |
| 2 | 22.5515% | 22.5515% | 22.5515% | 3/3 | 32/256 | 224/256 |
| 3 | N/A | N/A | N/A | 0/3 | 102/256 | 154/256 |
| 4 | 41.4017% | 41.4017% | 41.4017% | 3/3 | 86/256 | 170/256 |

Seed 3 trained successfully and has complete uniform/adaptive Morse topology
and both full-grid RoA representations. Its Marcio-equivalent trajectory
statistic is undefined because the encoded positive stable root matches no
strict singleton basin.

Across the four statistics-valid unique seeds, the combined trajectory score
has mean 46.3082%, median 42.8962%, sample standard deviation 22.5633
percentage points, and range 22.5515%–76.8888%. Statistics are valid for four
of five unique seeds. No invalid score is imputed into these summaries.

Marcio's archived result is 78.3897%. The best result here is seed 1 at
76.8888%, 1.5009 percentage points lower.

## The earlier “8%”

For seed 0, 8.7128% is only the positive-basin contribution. The negative-basin
contribution is 35.6779%, giving 44.3907% combined. Seed 0 is 1.9175 points
below the valid-seed mean and 1.4945 points above the valid-seed median, so the
five-seed evidence does not make it unusually unlucky. Seed 1 is the favorable
initialization.

## Stored artifacts

The package reports 15/15 complete uniform and adaptive Morse topologies,
15/15 complete full-grid RoA pairs, and 12/15 valid trajectory statistics.

For every trial, `analysis/by_run/<run>/` contains:

- `models/autoencoder.pt` and architecture metadata;
- `MG_uniform_s8/morse_graph` and `MG_uniform_s8/morse_sets`;
- `MG_uniform_s8/regions_of_attraction_strict_singleton.npz`, the
  authoritative all-256-cell strict singleton lookup;
- `MG_uniform_s8/regions_of_attraction_exact.npz`, a separately documented
  blocker/LCA diagnostic with different semantics;
- `MG_adaptive/morse_graph` and `MG_adaptive/morse_sets`;
- trajectory labels, encoded roots, and `basin_statistics.json` whenever root
  association is valid;
- stage, analysis, and topology/RoA augmentation manifests.

The original three training sweeps were verified read-only and copied into
each repeat package before analysis. Their source and copied file trees match
byte-for-byte.

Machine-readable entry points:

- `aggregate_manifest.json`
- `seed_summary.csv`
- `repeat_00/analysis/results_by_run.json`
- `repeat_01/analysis/results_by_run.json`
- `repeat_02/analysis/results_by_run.json`


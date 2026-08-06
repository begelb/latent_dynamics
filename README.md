# `archive/brittany/` — Brittany's autoencoder + CMGDB pipeline

A self-contained pipeline (`make_data → scale_data → train → morse_graph`) and the
two experiments that matter:

1. **`output/Leslie_3D/spurious_attractor_ex/`** — a 3D Leslie run that produces a
   *six*-node Conley-Morse graph with **three minimal Morse sets**, one of which is
   spurious (no preimage in ambient `R^3`).
2. **`output/coral/`** — 30-seed sweeps of the 13D red-coral discrete map at
   different training-set sizes (uniform Sobol *and* adaptive resampling), each
   reduced to a 1D latent map and analyzed with CMGDB.

Everything else (the contents of `Leslie_analysis_scripts/`, the empty
`output/Leslie_3D/28.9_29.8_22.0/scalers/` slot, etc.) is documentation/figure
support for those two experiments.

```
brittany/
├── config/                              # YAMLs
│   ├── Leslie_3D_larger_domain_tail_only.yaml
│   └── coral.yaml
├── src/                                 # shared library
│   ├── config.py                        # YAML -> attribute object
│   ├── models.py                        # Encoder / Decoder / LatentDynamics
│   ├── training.py                      # Training class + losses
│   └── true_dynamics_models.py          # LeslieModel3D, LeslieContraction, RedCoralModel, ...
├── main_scripts/                        # generic 4-step pipeline
│   ├── make_data.py
│   ├── scale_data.py
│   ├── train.py
│   └── morse_graph.py
├── coral_experiment_scripts/            # SLURM + analysis for coral sweeps
│   ├── setup_coral_scalers.sh           # fits scalers for 6 dataset sizes
│   ├── setup_adaptive_scalers.sh        # fits scalers for 5 adaptive sizes
│   ├── run_coral_experiments.sh         # 6 sizes x 30 seeds = 180 jobs
│   ├── run_adaptive_experiments.sh      # 5 sizes x 30 seeds = 150 jobs
│   ├── compute_morse_metric.py          # unique-Morse-set membership pass/fail
│   ├── plot_morse_metrics.py            # success-rate vs dataset size
│   ├── population_histogram.py          # final-population histograms
│   └── 1D_morse_set_plot_for_coral.py   # bespoke 1D Morse-set plot (seed_16)
├── Leslie_analysis_scripts/
│   └── Leslie3D_spurious_attractor_figure.py   # makes the spurious-attractor figure
├── data/                                # generated train/test CSVs (+ metadata.json)
└── output/                              # trained models, Morse graphs, plots
```

---

## 0. Pipeline mechanics (common to every experiment)

### Networks (`src/models.py`)

For `num_layers=L`, `hidden_shape=H`, `high_dims=D`, `low_dims=d`:

```
Encoder:        Linear(D,H) -> ReLU -> [Linear(H,H) -> ReLU] x (L-1) -> Linear(H,d) -> Tanh
Decoder:        Linear(d,H) -> ReLU -> [Linear(H,H) -> ReLU] x (L-1) -> Linear(H,D) -> Sigmoid
LatentDynamics: Linear(d,H) -> ReLU -> [Linear(H,H) -> ReLU] x (L-1) -> Linear(H,d) -> Tanh
```

`Encoder` and `LatentDynamics` produce in `[-1, 1]^d`; `Decoder` produces in `[0, 1]^D`
(matching the MinMax-scaled targets).

### Loss (`src/training.py::dynamics_losses`)

For each `(x_t, x_{t+tau})` pair (both MinMax-scaled to `[0,1]`),

```
loss_ae1  = MSE( D(E(x_t)),         x_t          )       # reconstruction at t
loss_ae2  = MSE( D(LatDyn(E(x_t))),  x_{t+tau}   )       # decoded one-step prediction
loss_dyn  = MSE( LatDyn(E(x_t)),     E(x_{t+tau}))       # latent-space prediction
loss_total = w[0]*loss_ae1 + w[1]*loss_ae2 + w[2]*loss_dyn
```

YAML key `weight: [w0, w1, w2]`. Both experiments use `[10, 10, 1]`.

### Training (`train.py`)

- Adam (`lr=config.learning_rate`)
- `ReduceLROnPlateau(threshold=0.001, patience=config.patience)` driven by **test** loss
- early stop after `patience` epochs without train-loss improvement
- per-epoch loss histories pickled to `<log_dir>/{train_losses,test_losses}.pkl` as
  `dict[str, list[float]]` with keys `loss_ae1 / loss_ae2 / loss_dyn / loss_total`.
- final per-loss values written to `<output_dir>/final_losses.txt` as `L1, L2, L3`,
  divided by their respective weights so they are comparable to the YAML targets.
- `Training.save_models` saves the **whole modules** (`torch.save(self.encoder, ...)`)
  not just `state_dict`s — reloading therefore needs `src.models` on `sys.path` and
  `weights_only=False`.

### Morse-graph (`morse_graph.py`)

- loads encoder + dynamics modules and the saved `scaler.gz`
- encodes every train/test point to find the latent box; **first** latent dim is
  expanded by `0.01 * width` on each side (other dims left tight)
- runs `CMGDB.ComputeConleyMorseGraph` with CLI flags `--init / --smin / --smax` and
  YAML `subdiv_limit`
- `BoxMap(..., padding=True)`
- writes `MG/morse_graph[.pdf,.png]`, `MG/morse_sets[.pdf,.png]` and the binary box
  cover `MG/morse_sets`. For 1D latents an additional `morse_sets2.png` is rendered.
- run params + wall-clock saved to `<output_dir>/mg_params_log.txt`.

### Data files

- CSVs are produced by `make_data.py`, written via `np.savetxt(... fmt="%.8f", header="x0,x1,...,y0,y1,...")`.
  Loaded everywhere with `np.loadtxt(..., skiprows=1)`. Each row is one transition
  pair `(x, f(x))`, dimension `2 * D`.
- A sibling `<train_file>_metadata.json` records `n_samples`, `n_iterations`,
  `skip_initial_steps`, `lower_bounds`, `upper_bounds`, `sampling_method`, and (for
  adaptive datasets) provenance about the model that generated them.
- Scalers are `sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))` fit jointly on
  `vstack([x_train, y_train])`, saved with `joblib.dump` to
  `<scaler_dir>/<train_file>/scaler.gz`.

> **Heads-up on artifacts.** Many of the saved `.pt` and `MG/morse_*` files in
> `output/coral/` are **0 bytes**: only training runs that completed and that yielded
> a non-trivial Morse decomposition produced full artifacts. See the coral section for
> the per-dataset breakdown.

---

## 1. Leslie 3D — `output/Leslie_3D/spurious_attractor_ex/`

The **headline run**: a 3-class Leslie map at parameters where the true map has two
period-4 attractors, but the trained latent map admits a Morse decomposition with a
third, **spurious** minimal node.

### Underlying ground-truth map

`src.true_dynamics_models.LeslieModel3D`:

```
s = x0 + x1 + x2
decay = exp(-0.1 * s)
x'_0 = (th1*x0 + th2*x1 + th3*x2) * decay
x'_1 = p1 * x0
x'_2 = p2 * x1
```

with the parameter set hard-coded in `make_data.py::main()` for `system='leslie3d'`:

| Param | Value |
| --- | --- |
| `th1` | **28.9** |
| `th2` | **29.8** |
| `th3` | **22.0** |
| `survival_p1`, `survival_p2` | 0.7 |
| `lower_bounds`, `upper_bounds` | `(0,0,0)`, `(220,154,108)` |

(Note these differ from `LeslieModel3D`'s defaults `(19.6, 23.68, 23.68)`. The
"`28.9_29.8_22.0`" subdirectory name is the parameter triple.)

### Data set (uniform sampling)

`make_data.py` for `leslie3d`:

| Field | Value |
| --- | --- |
| `n_samples_total` (train pool) | 4000 (commented note: "5000") |
| `n_iterations` | 30 (each IC iterated 30 steps) |
| `skip_initial_steps` | 0 |
| `train_sizes` (saved) | `[2000]` |
| `n_samples_test` | 5000 |
| Sampling | `'uniform'` (`np.random.uniform`) |
| Train CSV | `2train.csv` (rows = `2000 * 30 = 60 000`, 6 columns) |
| Test CSV | `2test.csv` (rows = `5000 * 30 = 150 000`, 6 columns) |

(The `2` prefix on the CSVs is just a manual rename from a re-run.)

### Config: `config/Leslie_3D_larger_domain_tail_only.yaml`

```yaml
system: leslie3d
data_dir:    data/Leslie_3D_larger_domain_tail_only/28.9_29.8_22.0
output_dir:  output/Leslie_3D_larger_domain_tail_only/
model_dir:   output/Leslie_3D_larger_domain_tail_only/28.9_29.8_22.0/models
log_dir:     output/Leslie_3D_larger_domain_tail_only/logs
scaler_dir:  output/Leslie_3D_larger_domain_tail_only/28.9_29.8_22.0/scalers/
subdiv_limit: 10000
num_layers: 3
hidden_shape: 32
non_linearity: ReLU
batch_size: 1024
learning_rate: 0.001
epochs: 1000
high_dims: 3
low_dims: 2
patience: 100
weight: [10, 10, 1]
```

### Network architecture (`high_dims=3, low_dims=2, hidden_shape=32, num_layers=3`)

Four Linear layers per net (`num_layers=3` hidden + 1 output):

```
Encoder:        3  -> 32 -> 32 -> 32 -> 2     (Tanh out)
Decoder:        2  -> 32 -> 32 -> 32 -> 3     (Sigmoid out)
LatentDynamics: 2  -> 32 -> 32 -> 32 -> 2     (Tanh out)
```

Per-layer weight shapes (verified against `models/encoder.pt` etc.):

| Module | linear_0 (W/b) | linear_1 | linear_2 | linear_3 |
| --- | --- | --- | --- | --- |
| Encoder | `[32,3]/[32]` | `[32,32]/[32]` | `[32,32]/[32]` | `[2,32]/[2]` |
| Decoder | `[32,2]/[32]` | `[32,32]/[32]` | `[32,32]/[32]` | `[3,32]/[3]` |
| Dynamics | `[32,2]/[32]` | `[32,32]/[32]` | `[32,32]/[32]` | `[2,32]/[2]` |

### Training result (`final_losses.txt`)

```
L1: 5.27e-06   L2: 1.13e-04   L3: 6.16e-05
```

`logs/{train,test}_losses.pkl` are `dict[str, list[float]]` with the four keys
defined above.

### CMGDB runs

`mg_params_log.txt` (the "main" 6-node Morse graph in `MG/morse_graph`):

| Field | Value |
| --- | --- |
| Latent rectangle (lower) | `[-0.6228695, -0.74216413]` |
| Latent rectangle (upper) | `[ 0.30980384,  0.22416562]` |
| `subdiv_init` | 23 |
| `subdiv_min` | 23 |
| `subdiv_max` | 27 |
| `subdiv_limit` | 10000 |
| Wall-clock | **94 min** |

`mg_params_log_2.txt` (a coarser companion run saved as `MG/morse_graph2`):

| Field | Value |
| --- | --- |
| Same latent rectangle | (identical to within rounding) |
| `subdiv_init / min / max` | 18 / 18 / 20 |
| Wall-clock | **141 min** |
| Resulting graph | 3 nodes, single chain `2 -> 1 -> 0` (1 minimal). Saved alongside as `MG/morse_graph2[.pdf,.png]`, `MG/morse_sets2[.pdf,.png]`, `MG/morse_sets2`. Used for comparison against the over-resolved 6-node graph. |

### Morse graph (the interesting one): `MG/morse_graph`

Six nodes (no Conley polynomials in the saved `.dot`, just IDs and colours):

| Node | Colour | Role |
| ---: | --- | --- |
| 0 | gold `#FFB000` | period-4 attractor (genuine) |
| 1 | magenta `#DC267F` | period-4 attractor (genuine) |
| 2 | orange `#FE6100` | flows into 0 and 1 |
| 3 | blue `#648FFF` | flows into 1 |
| **4** | **purple `#785EF0`** | **spurious minimal Morse set** (also a sink) |
| 5 | teal `#008080` | rank-2 source, flows into 3 and 4 |

```
{rank=same; 0  1  4};            # the three minimal nodes
2 -> 0;
2 -> 1;
3 -> 1;
5 -> 3;
5 -> 4;
```

So the Morse graph reports **three** minimal Morse sets (`0`, `1`, `4`), but the
true 3D Leslie at these parameters has **two** period-4 attractors. Node 4 is
spurious — there is no preimage of it under the encoder back in `R^3`. The hard-coded
period-4 cycles for nodes 0 and 1 (used as plotting markers) live in
`Leslie_analysis_scripts/Leslie3D_spurious_attractor_figure.py::periodic_pts`:

```
node 0:  [102.59,   4.63,  0.59], [0.065, 71.82,  3.24], [1.21, 0.045, 50.27], [6.61, 0.85, 0.032]
node 1:  [ 20.09,   2.26, 21.11], [14.41, 14.06,  1.58], [43.08, 10.09,  9.84], [ 3.23, 30.16, 7.06]
```

### `plot_data/` (regeneration cache for the figure)

| File | Content |
| --- | --- |
| `preimage_samples_k4_20pts.pkl` | 20 ambient-`R^3` samples whose orbits the encoder maps into Morse set `k = 4` after 20 iterations (i.e. samples in the *preimage* of the spurious set). Produced by `find_and_save_preimage_samples(target_k=4, num_samples=20, iterations=20)`. |
| `preimage_plot_data_indexed.pkl` | A grid (`res=120`) over `[0,221] x [0,155] x [0,109]` encoded into latent space and labelled by the 5-column Morse-box CSV. Used to draw the 3D pre-image cloud; `morse_set_data` columns are `(lx, ly, ux, uy, label)`. |

`Leslie3D_spurious_attractor_figure.py` produces `latent_trajectory.PDF`/`.png`
showing the 6-node Morse graph projected to latent space with the period-4 cycles
overlaid and the spurious-set preimage highlighted.

> **Note on the scaler path.** The figure script hard-codes
> `scaler_dir = '/Users/brittany/Documents/GitHub/PCA-Leslie/output/Leslie_3D/28.9_29.8_22.0/scalers'`,
> i.e. an absolute path on Brittany's machine. To re-run locally, redirect this to
> `output/Leslie_3D/28.9_29.8_22.0/scalers/scaler.gz` (which exists in this archive).

### Companion 6-node "new colours" rendering

`MG/morse_graph_new_colors.pdf` is a re-render of the same 6-node graph with a
different palette; same edges, no algorithmic difference.

---

## 2. Coral — `output/coral/`

13D red-coral demographic map → **1D latent**, swept across two design axes:

- **Dataset size sweep** (uniform Sobol sampling): 6 sizes × 30 seeds.
- **Adaptive resampling sweep**: start from the `train_500/seed_16` model, draw an
  additional `n_adaptive` ambient points whose latent encoding falls into the
  one-pre-attractor latent interval, retrain. 5 adaptive budgets × 30 seeds.

### Underlying ground-truth map

`src.true_dynamics_models.RedCoralModel`: 13 size classes with Beverton-Holt-style
density-dependent recruitment.

```
rho = (sum(x) - x[0]) / surface_area              # adult density
L(rho) = 2.94 / (rho + 520 * exp(-0.14 * rho))    # density-dependent larval survival
x'_0 = L(rho) * sum_i b_i * x_i                   # recruits
x'_i = s_{i-1} * x_{i-1}     for i = 1..12
```

Default coefficients (used everywhere in `output/coral/`):

| Quantity | Value |
| --- | --- |
| `surface_area` | 36 |
| `b` (length 13) | `[0, 0, 2.89, 10.03, 21.59, 39.02, 56.41, 77.72, 103.23, 131.87, 164.57, 201.46, 242.65]` |
| `survival_rates` (length 12) | `[0.889, 0.633, 0.697, 0.517, 0.437, 0.287, 0.571, 0.333, 0.75, 1, 0.333, 1]` |
| `lower_bounds` | `[0]*13` |
| `upper_bounds` | `[1300, 1150, 750, 520, 270, 120, 35, 20, 7, 5, 5, 2, 2]` |

The map has three reference fixed points used in the success metric (see below):

```
a0 = 0                             (extinction)
a1 ≈ [868, 772, 489, 341, 176, 77, 22, 13, 4.20, 3.15, 3.15, 1.05, 1.05]   (stable equilibrium)
r  ≈ [322, 286, 181, 126,  65, 29,  8.2, 4.7, 1.56, 1.17, 1.17, 0.39, 0.39] (saddle / Allee-effect repeller)
```

(Exact values in `compute_morse_metric.py`; `a1` is a sink, `r` is a non-sink in the
true Morse decomposition.)

### Data sets

`make_data.py` for `system='coral'`:

| Field | Value |
| --- | --- |
| `n_iterations` | 20 (steps per IC) |
| `skip_initial_steps` | 0 |
| `train_sizes` | `[100, 200, 500, 1000, 2000, 5000]` |
| `n_samples_test` | 10000 |
| Sampling | `'uniform'` for both train and test (note: `make_data.py` *can* do Sobol — `seed=42` for train, `9999` for test — but `main()` passes `'uniform'` here) |

**Adaptive sweep** (note: the resampler script itself is *not* in this archive — only
its outputs and metadata are). The metadata for an adaptive set is, e.g.:

```json
{
  "dataset_name": "train_500_300_adaptive",
  "n_samples": 800,
  "sampling_method": "adaptive_latent",
  "source_dataset": "train_500",
  "source_model_subdir": "train_500/seed_16",
  "n_adaptive_samples": 300,
  "morse_label_low": 0, "morse_label_high": 2,
  "latent_interval": [-1.008, -0.024]
}
```

i.e. start from `train_500` (Sobol-sampled), encode through `seed_16`'s encoder,
keep ambient points whose latent value lies in `[-1.008, -0.024]` (the gap between
the two minimal Morse intervals at labels 0 and 2 in `seed_16`'s output), append
`n_adaptive_samples` to the original 500 to make `500 + n` rows.

| Dataset | Total rows | Adaptive samples |
| --- | ---: | ---: |
| `train_500_100_adaptive` | 600 | 100 |
| `train_500_200_adaptive` | 700 | 200 |
| `train_500_300_adaptive` | 800 | 300 |
| `train_500_400_adaptive` | 900 | 400 |
| `train_500_500_adaptive` | 1000 | 500 |

### Config: `config/coral.yaml`

```yaml
system: coral
data_dir:   data/coral
output_dir: output/coral/
model_dir:  output/coral/models
log_dir:    output/coral/logs
scaler_dir: output/coral/data/scalers/
subdiv_limit: 10000
num_layers: 3
hidden_shape: 64
non_linearity: ReLU
batch_size: 1024
learning_rate: 0.001
epochs: 1000
high_dims: 13
low_dims: 1
patience: 100
weight: [10, 10, 1]
```

### Network architecture (`high_dims=13, low_dims=1, hidden_shape=64, num_layers=3`)

```
Encoder:        13 -> 64 -> 64 -> 64 -> 1     (Tanh out)
Decoder:         1 -> 64 -> 64 -> 64 -> 13    (Sigmoid out)
LatentDynamics:  1 -> 64 -> 64 -> 64 -> 1     (Tanh out)
```

Confirmed against, e.g., `output/coral/train_500_300_adaptive/seed_21/models/*.pt`.

### CMGDB defaults (1D latent)

`run_coral_experiments.sh` calls `python3 morse_graph.py --config coral.yaml ...`
with no `--init/--smin/--smax`, so the `morse_graph.py` argparse defaults take over:

| Field | Value |
| --- | --- |
| `subdiv_init` | 6 |
| `subdiv_min` | 8 |
| `subdiv_max` | 10 |
| `subdiv_limit` | 10000 |

(However, every saved `mg_params_log.txt` I inspected reports `init=8, min=8, max=12`,
indicating Brittany re-ran the morse graph step at coarser-but-deeper settings before
final results were saved.)

`MG/morse_sets` for 1D runs is a CSV: each line `lower, upper, label` describes one
1D box assigned to Morse set `label`. Example (`train_2000/seed_0`):

```
-0.15662,-0.15205,0
...
-0.45386,-0.44929,1
...
```

The companion `morse_sets2.png` is rendered for every 1D run regardless of whether
the graph is non-trivial.

### Sweep layout & artifact survey

```
output/coral/
├── train_<N>/seed_<k>/        # uniform sweep, k=0..29, N in {100,200,500,1000,2000,5000}
│   ├── models/                 (encoder.pt, decoder.pt, dynamics.pt — full module pickles)
│   ├── MG/                     (morse_graph[.pdf,.png], morse_sets, morse_sets2.png)
│   ├── logs/                   (train_losses.pkl, test_losses.pkl)
│   ├── final_losses.txt
│   └── mg_params_log.txt
├── train_500_<n>_adaptive/seed_<k>/   # adaptive sweep, n in {100,200,300,400,500}
└── histograms/                        # final-population histograms per dataset
```

**Artifact completeness** (counts of seeds, out of 30, with non-empty
`MG/morse_graph` and non-empty `models/encoder.pt`):

| Dataset | Non-empty graph | Non-empty checkpoint |
| --- | ---: | ---: |
| `train_100` | 12 | 10 |
| `train_200` | 0 | 0 |
| `train_500` | 0 | 0 |
| `train_1000` | 0 | 0 |
| `train_2000` | 30 | 30 |
| `train_5000` | 0 | 0 |
| `train_500_100_adaptive` | 30 | 30 |
| `train_500_200_adaptive` | 30 | 30 |
| `train_500_300_adaptive` | 30 | 30 |
| `train_500_400_adaptive` | 30 | 0 |
| `train_500_500_adaptive` | 0 | 0 |

(Some runs produced PDFs/PNGs but the source `morse_graph` and `morse_sets` were
overwritten with empty files. The success-rate plots in `coral_experiment_scripts/`
require the non-empty subset to be reproducible.)

### Example Morse graph (`train_2000/seed_0/MG/morse_graph`)

```
0 : (x-1, 0)        gold      sink (attractor)
1 : (0, x-1)        magenta   flows into 0
2 : (x-1, 0)        blue      sink (attractor)
3 : (0, x-1)        orange    flows into 0 and 2
edges: 1 -> 0,  3 -> 0,  3 -> 2
```

Two minimal nodes (`0`, `2`) — these correspond to the latent images of `a0` and
`a1`. Node `r` (the saddle) should map into a non-minimal node such as `1` or `3`.

### Success metric: `coral_experiment_scripts/compute_morse_metric.py`

For each seed, encodes the three reference points `a0, a1, r` and asks:

- which Morse interval contains `E(a0)`, `E(a1)`, `E(r)` (the CSV `(a, b, label)` cover);
- which Morse-graph nodes are sinks (no outgoing edges).

A point passes if it lies in *some* Morse set whose label is **unique** among the three
points, **and** that label is a sink (for `a0`, `a1`) or a non-sink (for `r`).
Aggregated by `plot_morse_metrics.py` into per-dataset pass-fractions; output PDFs
`morse_metric_plot.pdf` (size sweep, log x) and `morse_metric_plot_adaptive.pdf`
(adaptive sweep, linear x) land in `output/coral/`.

### Histograms: `coral_experiment_scripts/population_histogram.py`

For each `train_<N>.csv`, plots the final-row population total (`sum(y)`) per
trajectory. Eight pre-built outputs in `output/coral/histograms/`:

```
histogram_train.pdf
histogram_train_100.pdf, _500.pdf, _1000.pdf, _2000.pdf, _5000.pdf
histogram_train_500_300_adaptive.pdf
histogram_train_500_500_adaptive.pdf
```

### Bespoke 1D plot: `coral_experiment_scripts/1D_morse_set_plot_for_coral.py`

Hard-coded to `output/coral/seed_16` (a path that no longer exists in this archive —
the actual seed 16 outputs live under `output/coral/train_500/seed_16`). Encodes
`a0`, `a1`, `r`, plus the empirical observed coral state and a couple of
"overharvested" perturbations onto the 1D latent axis, overlays them on the
Morse-set intervals. Used to make the latent-axis figure in the paper.

---

## How to (re)run

```bash
cd archive/brittany

# --- Leslie 3D spurious-attractor experiment ---
python3 main_scripts/make_data.py    --config Leslie_3D_larger_domain_tail_only.yaml
python3 main_scripts/scale_data.py   --config Leslie_3D_larger_domain_tail_only.yaml --train_file train
python3 main_scripts/train.py        --config Leslie_3D_larger_domain_tail_only.yaml --train_file train
python3 main_scripts/morse_graph.py  --config Leslie_3D_larger_domain_tail_only.yaml \
                                     --init 23 --smin 23 --smax 27

# --- Coral size sweep (one cell of the grid) ---
python3 main_scripts/make_data.py    --config coral.yaml
bash    coral_experiment_scripts/setup_coral_scalers.sh
python3 main_scripts/train.py        --config coral.yaml --train_file train_2000 \
                                     --seed 0 --output_subdir train_2000/seed_0
python3 main_scripts/morse_graph.py  --config coral.yaml --train_file train_2000 \
                                     --output_subdir train_2000/seed_0

# --- Aggregate metrics over all seeds (after the SLURM array job) ---
python3 coral_experiment_scripts/plot_morse_metrics.py --config coral.yaml --mode size
python3 coral_experiment_scripts/plot_morse_metrics.py --config coral.yaml --mode adaptive
```

Requirements: `torch`, `numpy`, `scipy`, `scikit-learn`, `joblib`, `matplotlib`,
`pandas`, `pydot`, `tqdm`, `pyyaml`, plus the `CMGDB` package (Morse-graph step).
The `.pt` files are full-module pickles, so loading needs `archive/brittany/` on
`sys.path` (so `src.models` resolves) and `weights_only=False`.

---

## Quick-reference cheat sheet

| | Spurious 3D Leslie | Coral |
| --- | --- | --- |
| True map | `LeslieModel3D(28.9, 29.8, 22.0, 0.7, 0.7)` | `RedCoralModel()` (defaults) |
| Ambient dim | 3 | 13 |
| Latent dim | 2 | 1 |
| Hidden (`L`/`H`) | 3 / 32 | 3 / 64 |
| Train pool | 60 000 rows (`2000 ICs × 30` steps) | up to `5000 ICs × 20` steps |
| Test set | 150 000 rows | 200 000 rows |
| Sampling | uniform | uniform (Sobol available) + adaptive resampling |
| Loss weights | `[10, 10, 1]` | `[10, 10, 1]` |
| Epochs / batch / lr | 1000 / 1024 / 1e-3 | 1000 / 1024 / 1e-3 |
| Scaler | `MinMaxScaler(0,1)` joint on (x, y) | same |
| CMGDB box | `[-0.623, 0.310] × [-0.742, 0.224]` | `[lo, hi]` learned per seed (1D first axis) |
| `subdiv_init/min/max` | 23 / 23 / 27 (run 1)<br>18 / 18 / 20 (run 2) | 8 / 8 / 12 (as saved) |
| Headline result | 6 nodes, **3 minimal** (1 spurious: node 4) | success-rate vs dataset size + adaptive resampling |

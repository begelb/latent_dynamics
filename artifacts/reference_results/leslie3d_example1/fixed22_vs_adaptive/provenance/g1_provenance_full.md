# Complete provenance of g_1 (leslie3d_example1, coauthor checkpoint)

> **Sanitized public copy.** References to the
> coauthor's private development archive and to local working-tree
> paths are shown as `<author archive>` / `<private worktree>`, and the
> coauthor's given name has been replaced by "coauthor", including in
> recorded directory names. All sha256 checksums, dates, and measured
> values are unchanged from the original audit.

All paths are relative to the root of the private working tree in which this audit was performed.
Everything below was read from local artifacts; no training, no CMGDB run, no notebook execution.
Python: `.venv/bin/python` (torch 2.11.0, sklearn 1.8.0).

---

## 0. Executive summary of the provenance chain

The active g_1 is **the coauthor's original 2026-05-03 checkpoint, unmodified**. Two *different* CMGDB
computations of `MG(G_1)` exist for it, on two *different* latent domains, and they produce the
**identical six-node Morse graph with minimal nodes {0,1,4}**. The paper uses the second one.

| stage | artifact | date | status |
|---|---|---|---|
| data | `2train.csv` / `2test.csv` | 2026-05-03 | authoritative; matches paper 3200/800 x 20 exactly |
| scaler | `28.9_29.8_22.0/scalers/scaler.gz` | 2026-05-03 | **fit on a 64,000-row CSV that is NOT byte-identical to `2train.csv`** (§3) |
| weights | `spurious_attractor_ex/models/{encoder,decoder,dynamics}.pt` | 2026-05-03 | authoritative, legacy 3-file pickled-module format |
| CMGDB run A (the coauthor) | `<author archive>/.../MG/morse_sets` (52,205 boxes) | 2026-05-03 | tight domain, **NOT used by the paper** |
| CMGDB run B (pipeline) | `replay_sources/.../MG/morse_sets` (122,346 boxes) | 2026-05-21, copied 2026-05-26 | **wide domain; this is the paper's `MG(G_1)`** |
| paper figure | the manuscript's `morse_graph.pdf` panel (`fig:3D_Leslie_latent`) | 2026-05-27 14:53 | render of run B's DOT |

---

## 1. Active checkpoint

`replay_sources/leslie3d_example1/spurious_attractor_ex/models/` (mtime 2026-05-03T12:40:33)

| file | sha256 |
|---|---|
| `encoder.pt` | `211f84456707afb254da1cfa05defa41137f90813f624af420722954763d1f4c` |
| `decoder.pt` | `25fb56dae18de6239bffefad463e0abed126535036f883be123f39ccfc4173e2` |
| `dynamics.pt` | `5d175395081ba8983ac863841fac83e9eab26e82de31121be572a2575b8ed955` |

Format: **legacy 3-file**, and specifically *pickled whole `nn.Module` objects* (not state dicts):
`torch.load` fails with `ModuleNotFoundError: No module named 'src'` unless the legacy package tree is
importable. `legacy/src/models.py` supplies it.

All three are **byte-identical** to `<author archive>/output/Leslie_3D/spurious_attractor_ex/models/*.pt`
(same sha256). The checkpoint has never been overwritten.

**Architecture read from the loaded state dict** (not from any YAML):

```
encoder    : Linear(3,32) ReLU Linear(32,32) ReLU Linear(32,32) ReLU Linear(32,2) Tanh
latent_map : Linear(2,32) ReLU Linear(32,32) ReLU Linear(32,32) ReLU Linear(32,2) Tanh
decoder    : Linear(2,32) ReLU Linear(32,32) ReLU Linear(32,32) ReLU Linear(32,3) Sigmoid
```
Parameter tensor shapes: `net.0.weight (32,3)`, `net.2.weight (32,32)`, `net.4.weight (32,32)`,
`net.6.weight (2,32)` for the encoder, etc. This **matches** `src/latentdynamics/configs/leslie3d_example1.yaml`
(`hidden_shapes: [32,32,32]`, relu/tanh/tanh/sigmoid) and the coauthor's
`<author archive>/config/Leslie_3D_larger_domain_tail_only.yaml` (`num_layers: 3`, `hidden_shape: 32`,
`non_linearity: ReLU`) expanded by `<author archive>/src/models.py`.

### Replayability: YES

```python
sys.path.insert(0, "code/src")
cfg   = load_config("leslie3d_example1_replay")
model, arch = load_any_checkpoint(
    "replay_sources/leslie3d_example1/spurious_attractor_ex/models",
    arch=cfg.arch, legacy_root="legacy")
```
returns a `LatentDynamicsAutoencoder`. Encoder, latent map and decoder all evaluate.
`src/latentdynamics/training.load_any_checkpoint` dispatches to `load_legacy_checkpoint`
because `models/autoencoder.pt` is absent. **g_1 is fully replayable, not merely documented.**

A **new-format conversion** of the same weights exists at
`<private worktree>/old_output/<format-conversion dir>/seed_0/models/autoencoder.pt` (+ `.json`).
Direct parameter comparison against the legacy checkpoint gives **max |Δparam| = 0.0** over all tensors.
Despite the directory name, this is **a format conversion, not a retrain**; the directory's `_retrain` suffix
is a misnomer and is a trap for anyone reading that tree.

---

## 2. Archived training / validation CSVs

`replay_sources/leslie3d_example1/data/Leslie_3D_larger_domain_tail_only/28.9_29.8_22.0/`
(byte-identical to `<author archive>/data/Leslie_3D_larger_domain_tail_only/28.9_29.8_22.0/`)

| file | sha256 | bytes | lines |
|---|---|---|---|
| `2train.csv` | `aee9689d70bca3494ddeca78f3d511c94b8c5c72d466da86c5ab2d5aed4351e5` | 4,383,318 | 64,001 |
| `2test.csv`  | `30f50cbbff44f1088a91d780c1c55ff2d85df9d428727d41cae886f458d18486` | 1,096,566 | 16,001 |

Columns: `x0,x1,x2,y0,y1,y2` — one **transition pair** `(x, f_theta(x))` per row.

Verified structure (computed):

| quantity | 2train.csv | 2test.csv |
|---|---|---|
| data rows (excl. header) | 64,000 | 16,000 |
| chain breaks (`y_i != x_{i+1}`) | 3,199 | 799 |
| **trajectories** | **3,200** | **800** |
| segment lengths | all exactly **20** | all exactly **20** |
| `max abs(Y - f_theta(X))` | 6.09e-08 | 6.02e-08 |
| x-column min | [0.03753427, 0.02627399, 0.01839179] | [0.03757028, 0.02629920, 0.01840944] |
| x-column max | [106.59078, 74.61788, 52.23462] | [106.59081, 74.61527, 52.23069] |

**Reconciliation with the paper.** The manuscript (`sec:3d_leslie`) states: "4,000 uniformly sampled initial
conditions split into 3,200 training and 800 validation trajectories. After discarding the first 10
iterates, the next 20 were retained." The CSVs match this **exactly**: 3,200 x 20 = 64,000 and
800 x 20 = 16,000 transition pairs; 3,200 + 800 = 4,000 ICs.
The manuscript's data table (`tab:data`: N train 3200 / N validation 800) also matches.
**Number of transition pairs: 64,000 train + 16,000 validation.** Discarded transient: 10 iterates
(inferred; the CSVs contain only the tail, so `skip=10` cannot be read off the rows directly — it is
established by the row counts (20 of 30 retained), by the directory name `..._tail_only`, and by the
paper text).

The theta used to generate the data is verified *from the data itself*: `Y = f_theta(X)` to 6.1e-8 with
theta = (28.9, 29.8, 22.0), p1 = p2 = 0.7.

### Initial-condition sampling domain and seed

`<author archive>/main_scripts/make_data.py:21` calls `sample_random_pts(model.lower_bounds,
model.upper_bounds, n_samples)`, i.e. `np.random.uniform` (line 11) on the model's own bounds.
`<author archive>/src/true_dynamics_models.py:16` gives `LeslieModel3D(..., lower_bounds=[0,0,0],
upper_bounds=[220,154,108])`, and `make_data.py:101` instantiates `LeslieModel3D(th1=28.9, th2=29.8,
th3=22.0, ...)` **without overriding bounds**.

> **IC box = [0,220] x [0,154] x [0,108]**, i.e. **8x the volume of the paper's analysis box
> B = [0,110] x [0,77] x [0,54]**. The paper says "4,000 uniformly sampled initial
> conditions" without stating the box; a reader will assume B. This is independently corroborated in §5.

**Seed: NOT RECOVERABLE.** The `'uniform'` branch of `make_data.py` calls bare `np.random.uniform`
with no seeding; only the `'sobol'` branch uses seeds 42/9999 (`make_data.py:26-28`). The
`data.train_seed: 42` / `data.val_seed: 9999` in every generated manifest and in
`src/latentdynamics/configs/leslie3d_example1.yaml` are therefore **meaningless for g_1**.
The **training** seed is likewise unrecoverable: `<author archive>/main_scripts/train.py` takes
`--seed` defaulting to `None`, and no seed is recorded in any surviving artifact for this run.

### The archived `make_data.py` did not produce these CSVs

`make_data.py:101-105` for `leslie3d` sets `n_samples_total = 4000` (**dead variable, never used**),
`n_iterations = 30`, `skip = 0`, `train_sizes = [2000]`, `n_samples_test = 5000`, and writes
`train_2000.csv` / `test.csv` plus `{name}_metadata.json`. The shipped files are named `2train.csv` /
`2test.csv`, have 3,200/800 trajectories of length 20, and **no metadata JSON exists**. Git history in
`<author archive>/.git` contains only two commits touching `main_scripts/make_data.py`
(`bda08fad`, `63c2938e`), both with the same content. **The generator of `2train.csv`/`2test.csv` is
not preserved in this repository.** The surviving `n_samples_total = 4000` and `n_iterations = 30` are
vestiges of it.

---

## 3. Scaler — and a real inconsistency

Path: `replay_sources/leslie3d_example1/28.9_29.8_22.0/scalers/scaler.gz`
sha256 `1d86986b1622f6e57527b852cae5769c412450b7570b82e138e5afbae082d02d`
(byte-identical to `<author archive>/output/Leslie_3D/28.9_29.8_22.0/scalers/scaler.gz`).

Loaded with `joblib.load` (emits `InconsistentVersionWarning`: pickled with sklearn 1.2.1, read with 1.8.0).

```
type                sklearn.preprocessing.MinMaxScaler,  feature_range=(0,1), clip=False
n_features_in_      3
n_samples_seen_     128000
data_min_           [ 0.03754539,  0.02628178,  0.01839724]
data_max_           [106.59079063, 74.62144694, 52.23700842]
data_range_         [106.55324524, 74.59516516, 52.21861118]
scale_              [0.00938498, 0.01340569, 0.01915026]
min_                [-0.00035236, -0.00035233, -0.00035231]
```

**What `n_samples_seen_ = 128,000` implies.** `<author archive>/main_scripts/scale_data.py` fits one
scaler on `np.vstack((x_train, y_train))`, i.e. **2 x (number of training rows)**. Hence the scaler saw a
training CSV with **exactly 64,000 rows** — consistent with 3,200 trajectories x 20 transitions, and
inconsistent with any 4,000/5,000-trajectory or `skip=0` (30-iterate) dataset. This is independent
confirmation of the 3,200 x 20 split.

> **INCONSISTENCY (flagged).** The scaler's `data_min_`/`data_max_` do **not** equal the elementwise
> min/max of `vstack(x,y)` over the shipped `2train.csv`:
>
> | | scaler | 2train.csv combined |
> |---|---|---|
> | min | [0.03754539, 0.02628178, 0.01839724] | [0.03753427, 0.02627399, 0.01839179] |
> | max | [106.59079063, 74.62144694, 52.23700842] | [106.59077995, 74.61787690, 52.23462265] |
>
> Relative differences ~3e-4 (component 0) to ~4.6e-5 — **three orders of magnitude above** the CSV's
> `%.8f` write precision, so this is not rounding. The scaler was fit on a **different 64,000-row draw**
> generated by the same recipe. Consequence: `2train.csv` cannot be certified as byte-exactly the set
> g_1 was trained on. Numerical impact is negligible (`scale_` differs by ~2e-5 relative; scaler max
> still brackets the CSV max, so scaled data stay in [0,1] up to -1e-7). **Trust: scaler +
> `n_samples_seen_` for the *count*; `2train.csv` for the *structure*; neither for byte-exact identity.**

---

## 4. Training configuration and the saved epoch

Sources: `<author archive>/config/Leslie_3D_larger_domain_tail_only.yaml`,
`<author archive>/main_scripts/train.py`, `<author archive>/src/training.py`,
`replay_sources/leslie3d_example1/spurious_attractor_ex/logs/{train,test}_losses.pkl`,
`.../final_losses.txt`.

| item | value | source |
|---|---|---|
| optimizer | Adam | `training.py:106` |
| learning rate | 0.001 | config yaml `learning_rate` |
| batch size | 1024 (63 train batches, 16 val batches) | config yaml |
| epochs (max) | 1000 | config yaml |
| patience | 100 | config yaml |
| loss weights | [10, 10, 1] | config yaml `weight` |
| loss terms | `L1 = MSE(x_t, D(E(x_t)))`, `L2 = MSE(x_tau, D(g(E(x_t))))`, `L3 = MSE(g(E(x_t)), E(x_tau))`; `L = 10 L1 + 10 L2 + 1 L3` | `training.py:81-91` |
| LR scheduler | `ReduceLROnPlateau(threshold=0.001, patience=patience)` stepped on **validation** total | `training.py:107,184` |
| early stopping | on **training** total loss, patience 100, **no best-model restore** | `training.py:185-192` |
| gradient clipping | **none** in the archived code | `training.py` (absent) |
| epochs actually run | **246** (indices 0..245) | `len(train_losses['loss_total'])` |
| **saved weights** | **final epoch (index 245)** — `save_models()` is called after `train()` returns | `train.py:73-75` |
| best validation epoch | index **236** (`test loss_total` argmin) | `test_losses.pkl` |

Training-loss minima (train): `loss_ae1` 1.320584e-05 @240, `loss_ae2` 2.800448e-05 @224,
`loss_dyn` 1.433343e-05 @212, `loss_total` 1.750169e-04 @240.
Validation minima: `loss_ae1` 1.432821e-05 @236, `loss_ae2` 2.964308e-05 @237, `loss_dyn` 1.350210e-05 @236,
`loss_total` 1.871430e-04 @236.

> **The checkpoint is NOT the best-validation model.** There is no best-epoch selection anywhere in the
> pipeline: `save_models()` writes the weights present when the loop exits.

> **INCONSISTENCY (flagged): `final_losses.txt` is not a clean loss report.**
> `.../final_losses.txt` gives L1 = 5.268487836929126e-06, L2 = 1.1298822732896951e-04,
> L3 = 6.159611615430549e-05. These reconcile with `train_losses.pkl` only through **two defects** in
> `<author archive>/src/training.py`:
> 1. `num_batches` is rebound to `len(test_loader)` (16) before lines 194-196, while the accumulators
>    `loss_*_train` are sums over 63 train batches -> everything is inflated by 63/16 = 3.9375.
> 2. Values come from **epoch index 244**, not 245: the `break` at line 191 precedes the `l1,l2,l3`
>    assignment, so the last computed values are one epoch stale.
>
> Check (train epoch 244: ae1 = 1.338029e-05, ae2 = 2.869542e-05, dyn = 1.564346e-05):
> `0.1 * 3.9375 * 1.338029e-05 = 5.268488e-06` = L1 (exact);
> `1.0 * 3.9375 * 1.564346e-05 = 6.159612e-05` = L3 (exact);
> `1.0 * 3.9375 * 2.869542e-05 = 1.129882e-04` = L2 (exact) — but the archived line 195 divides by
> `weight[1] = 10`, which would give 1.129882e-05. **The code that ran used a different divisor for L2
> than the archived `training.py` line 195** (consistent with `weight[2]`).
> **Trust `logs/{train,test}_losses.pkl`; treat `final_losses.txt` as unreliable.**
> This is a second, independent sign that the archived `src/` is not exactly the code that produced g_1
> (the first being `make_data.py`, §2).

Also note the active YAML declares `lr_patience: 10` and `gradient_clip_norm: 1.0`
(`src/latentdynamics/configs/leslie3d_example1.yaml`), neither of which exists in the code that
trained g_1 (scheduler patience = 100 = `config.patience`; no clipping). These are **forward-fitted
defaults, not recovered parameters.**

---

## 5. The CMGDB run(s)

### 5a. Run A — the coauthor's own, 2026-05-03 (NOT the paper's)

`<author archive>/output/Leslie_3D/spurious_attractor_ex/mg_params_log.txt`:
```
Lower bounds: [-0.6228695, -0.74216413]
Upper bounds: [0.30980384, 0.22416562]
Subdivision init/min/max: 23 / 23 / 27 ; limit 10000 ; duration 94 minutes
```
Box map: `CMGDB.BoxMap(g, rect, padding=True)` on `dynamics.pt` only
(`<author archive>/main_scripts/morse_graph.py:110`), device `cuda if available else cpu`, per-call
(no precompute).

I reproduced the domain: encoding `vstack(x_train, x_test, y_train, y_test)` (160,000 rows, the coauthor's
recipe, `morse_graph.py:83`) through the saved encoder + saved scaler gives
latent min `[-0.6228742, -0.7421643]`, max `[0.30976182, 0.22416666]` — **these are the recorded
bounds, with no epsilon applied**, even though `morse_graph.py:94-97` applies `epsilon = 0.01*w`
to dimension 0. (Note `morse_graph.py:100-101` never pads dimension 1 at all — a one-dimension-only
padding bug.) Third independent sign that the archived script post-dates the run.

`<author archive>/.../MG/morse_graph` carries **no Conley indices** (plain labels `"0"..."5"`).

### 5b. Run B — the pipeline run that the paper uses

Identical byte-for-byte in two places:

| file | sha256 |
|---|---|
| `<private worktree>/old_output/<format-conversion dir>/seed_0/MG/morse_sets` | `b9be6b28922f06898f765027980c6106fda40a349077872255db5dac321e05d6` |
| `replay_sources/leslie3d_example1/spurious_attractor_ex/MG/morse_sets` | *same* |
| `<private worktree>/old_output/.../seed_0/MG/morse_graph` | `fb7a57e012b76f5a02cb6e0e591cf194ccd130a80c34909f5cccb1d80233d596` |
| `replay_sources/.../MG/morse_graph` | *same* |

`replay_sources/leslie3d_example1/spurious_attractor_ex/mg_params_log.txt` (mtime 2026-05-26T17:54:39):
```
Lower bounds: [-0.6983771920204163, -0.8291957378387451]
Upper bounds: [0.9562897086143494,  0.7747613787651062]
subdiv_init 23 ; subdiv_min 23 ; subdiv_max 27 ; subdiv_limit 10000
bounds_epsilon_frac 0.01 ; padding True ; box_map_backend auto
compute_roa False ; bounds_source encoded_data ; duration_minutes 6.0179
```
The private-worktree copy (`<private worktree>/old_output/.../seed_0/mg_params_log.txt`) records the same bounds and
`duration_minutes 3.8733`; run created `2026-05-21T11:06:59Z` per its `run_manifest.json`.

These are **the paper's bounds**, quoted in the manuscript (`sec:3d_leslie`) as
`B ≈ [-0.70,0.95] x [-0.83,0.77]`.

> **Where the wide domain comes from — reconstructed.** It is **not** the encoded hull of g_1's own
> training data. Encoding `2train.csv`+`2test.csv` (or the regenerated `replay_sources/leslie3d_example1/data_pairs/`
> CSVs) through the saved encoder + saved scaler gives, with 1% padding on both axes,
> `[-0.6322, -0.7518] -> [0.3191, 0.2342]` — far tighter.
> I reproduced the paper's domain by sampling 4,000 ICs **uniformly on [0,220]x[0,154]x[0,108]**,
> iterating **30 times with skip = 0** (transient included), and encoding all 240,000 points:
>
> | | lower | upper |
> |---|---|---|
> | reconstruction (rng seed 0, 1% pad both axes) | [-0.68383, -0.82962] | [0.95489, 0.77631] |
> | **paper / `mg_params_log.txt`** | **[-0.69838, -0.82920]** | **[0.95629, 0.77476]** |
> | same but ICs on B = [0,110]x[0,77]x[0,54] | [-0.70247, -0.75883] | [0.63177, 0.47401] (**excluded**) |
>
> Agreement is ~1.4e-3 on three of four bounds (the residual is the expected spread of extreme-value
> statistics under a different unseeded draw); the B-hypothesis is off by 0.32 in the x-upper bound and
> is decisively excluded.
>
> **Consequence.** The latent domain on which the paper computes `MG(G_1)` is the encoded hull of
> **skip=0, out-of-distribution data on an 8x-larger IC box**. The encoded hull of the data g_1 was
> actually trained on is `[-0.6229,0.3098] x [-0.7422,0.2242]`, which is
> **0.564 x 0.602 = 34% of the area** of the paper's B. **Roughly two thirds of the paper's latent
> domain is encoder extrapolation.** (Node 4 is *not* in that region — see §7.)

### 5c. Both runs are effectively UNIFORM at level 23; `subdiv_max = 27` never took effect

Parsing the box files with exact arithmetic:

| | run B (active) | run A (the coauthor) |
|---|---|---|
| total boxes | 122,346 | 52,205 |
| distinct box widths | **1** | **1** |
| box size (dx, dy) | (4.039714112878e-04, 7.831821858417e-04) | (2.277034509461e-04, 4.718406999018e-04) |
| dx x 4096 | 1.654666900635 = recorded x-width | 0.93267334 = recorded x-width |
| dy x 2048 | 1.603957116604 = recorded y-width | 0.96632975 = recorded y-width |
| grid-index integrality residual | **0.0 (exact)** | 3.7e-05 (float32 log rounding) |
| occupied index range | x [172,2457]/4096, y [110,1334]/2048 | — |

Level 23 = 12 splits in x + 11 in y. **Every Morse-set box in both runs sits at level 23**; there is no
refined box anywhere. With `subdiv_limit = 10000` and 122,346 (resp. 52,205) boxes, refinement toward
`subdiv_max = 27` was blocked immediately. Reporting "subdiv 23/23/27" overstates the resolution:
**the effective computation is a uniform level-23 cubical cover.**

Note also that run B, despite the same nominal subdivision numbers, is **coarser in absolute terms**
(dx 1.77x, dy 1.66x larger) because its domain is bigger.

---

## 6. The Morse graph — full, both runs

`replay_sources/leslie3d_example1/spurious_attractor_ex/MG/morse_graph` (run B, with Conley indices):

```
digraph {
0 [label="0 : (x^4-1, 0, 0)"]   1 [label="1 : (x^4-1, 0, 0)"]
2 [label="2 : (0, x^4-1, 0)"]   3 [label="3 : (0, x^2+1, 0)"]
4 [label="4 : (x-1, 0, 0)"]     5 [label="5 : (0, x^2-1, 0)"]
{rank=same; 0 1 4 };  {rank=same; 2 5 };
2 -> 1;  2 -> 0;  3 -> 1;  5 -> 4;  5 -> 3;
}
```

**Nodes 6. Edges 5. Minimal nodes 3: {0, 1, 4}.**

| node | Conley index | type | minimal? |
|---|---|---|---|
| 0 | (x^4-1, 0, 0) | period-4 attractor | **yes** |
| 1 | (x^4-1, 0, 0) | period-4 attractor | **yes** |
| 2 | (0, x^4-1, 0) | period-4, degree-1 index (saddle) | no (-> 0, 1) |
| 3 | (0, x^2+1, 0) | degree-1 index, order-4 map | no (-> 1) |
| 4 | **(x-1, 0, 0)** | **fixed-point attractor** | **yes** |
| 5 | (0, x^2-1, 0) | degree-1 index, period-2 | no (-> 3, 4) |

`<author archive>/output/Leslie_3D/spurious_attractor_ex/MG/morse_graph` (run A): same 6 nodes, edge set
`{2->0, 2->1, 3->1, 5->3, 5->4}` — **identical as a set**, `{rank=same; 0 1 4}` — same minimal nodes.
The two files differ only in the fill colours assigned to nodes 2 and 3 (different palette ordering) and
in run A carrying no index labels. Run A's identical labels at commit `e6501f1b` (`git show
e6501f1b:output/Leslie_3D/spurious_attractor_ex/MG/morse_graph`) confirm the graph predates all our work.

`MG/morse_graph2` + `MG/morse_sets2` (both runs, byte-identical, sha `584889ca4efad715c6f3bfa7e0da7b860962a35be3a6300eda0ea59628bc77d1`)
are an **earlier, coarser computation**: `mg_params_log_2.txt` records subdiv 18/18/20, 141 minutes,
3 nodes, chain `2 -> 1 -> 0`, **only 1 minimal node**. This is the low-resolution predecessor and must
not be confused with the paper's graph.

### 6a. Morse-set box counts and bounding boxes (latent coordinates)

Run B — `replay_sources/.../MG/morse_sets` (122,346 rows, `xlo,ylo,xhi,yhi,label`):

| node | n_boxes | x-range | y-range |
|---|---|---|---|
| 0 | 3,594 | [-0.628894109, 0.294584537] | [-0.743045697, 0.216352480] |
| 1 | 116,789 | [-0.382875520, -0.068989733] | [-0.566829706, -0.060110831] |
| 2 | 1,186 | [-0.468113488, 0.028367377] | [-0.559781066, -0.070292200] |
| 3 | 480 | [-0.214419441, -0.186141443] | [-0.572311981, -0.082823115] |
| **4** | **174** | **[-0.218055184, -0.213207527]** | **[-0.350671422, -0.335007779]** |
| 5 | 123 | [-0.218459155, -0.214015470] | [-0.357720062, -0.326392775] |

Run A — `<author archive>/.../MG/morse_sets` (52,205 rows):

| node | n_boxes | x-range | y-range |
|---|---|---|---|
| 0 | 3,955 | [-0.622869492, 0.285894981] | [-0.737917569, 0.210010397] |
| 1 | 46,064 | [-0.338240178, -0.114863093] | [-0.550596811, -0.100932624] |
| 2 | 1,347 | [-0.465526407, 0.024946826] | [-0.558146262, -0.074037704] |
| 3 | 550 | [-0.212547873, -0.187728197] | [-0.571829642, -0.084890040] |
| **4** | **159** | **[-0.217101942, -0.214369501]** | **[-0.346761628, -0.338740337]** |
| 5 | 130 | [-0.217557349, -0.214824907] | [-0.356670283, -0.327888000] |

> **Node 4 scaling behaviour.** Its box *count* is essentially resolution-independent (174 vs 159), while
> its *diameter* shrinks by 1.78x in x and 1.95x in y between run B and run A — matching the box-size
> ratios 1.774 and 1.660. This is the signature of a **genuine small invariant set being resolved**, not of
> a cover artifact (which would either vanish or keep a fixed geometric size). Node 4 also sits far
> inside the training-data encoded hull, so it is not produced by the domain inflation of §5b.

---

## 7. What node 4 is (direct diagnostics on the saved g_1)

Computed with the loaded checkpoint; every number reproducible from the artifacts above.

- **f_theta's positive fixed point** `p_* = (18.73654933, 13.11558453, 9.18090917)`
  (closed form `p_0 = ln(th1 + 0.7 th2 + 0.49 th3)/0.219`; residual `|f(p)-p| = 7.1e-15`).
  `Df(p_*)` eigenvalues `-1.12221934`, `-0.13703263 ± 0.80040840i` -> moduli
  **1.12222, 0.81205, 0.81205**: a **saddle**, 1 unstable + 2 stable directions. In `MG(F)` it is a
  **non-minimal** node.
- **The saved latent map g has a genuine attracting fixed point.** Iterating `g` 2,000 times from node 4's
  box centre converges to `z* = (-0.21573170, -0.34282628)` with `|g(z*) - z*| = 2.68e-07`.
  `Dg(z*)` eigenvalues `-0.62324107 ± 0.29819989i`, modulus **0.69091 < 1** -> hyperbolic **sink**.
  `z*` lies **inside** node 4's box union (node 4 bbox x [-0.218055,-0.213208], y [-0.350671,-0.335008]).
  -> answer **(b) is true**: node 4 is a real attractor of the saved map, not a cubical artifact.
- **It corresponds to p_*.** `decode(z*) = (18.875055, 12.990301, 9.137733)` vs
  `p_* = (18.736549, 13.115585, 9.180909)`: **max component error 0.1385**, i.e. 1.3e-3 of the domain
  extent. `E(p_*) = (-0.21183777, -0.33854002)`; `|z* - E(p_*)| = 5.79e-03`;
  `g(E(p_*)) = (-0.21795474, -0.34289820)`, which is **2.2e-03 from z\***, i.e. **one application of g
  carries `E(p_*)` essentially onto `z*`**.
  -> answer **(c)**: node 4 is a **learned stability change at a real invariant object of f** — the saddle
  `p_*` is represented as a sink.
- **The prior claim of "no counterpart" does not survive.** `docs/figure_contracts/leslie3d_example1.md`
  states `E(p_*)` is `1.369758509e-3` from node 4's box union and concludes "node 4 has no counterpart in
  that detected recurrent-set inventory." I reproduce that distance **to all ten digits**
  (`1.369758509e-03` for run B; `2.619439950e-03` for run A) — but the inference is invalid. That gap is
  **3.4 box widths** (dx = 4.04e-4) and is the same order as the run's own reported
  `max_semiconjugacy_error = 4.47e-4` (`.../metrics.json`). Non-intersection of a finite cubical cover
  with the *image of a single point* is criterion (i); it is not evidence about (vi) invariant-set
  correspondence. Given `decode(z*) ≈ p_*` to 0.14 and `g(E(p_*)) ≈ z*`, the counterpart exists.
- **Mechanism — a local obstruction, not data scarcity.** With `DE(p_*)` (2x3, singular values
  0.012985, 0.004873), `ker DE = (-0.12792, 0.93249, 0.33779)`.
  - `Df(p_*) ker DE` makes an angle of **78.65°** with `ker DE`, so `ker DE` is **not** `Df(p_*)`-invariant.
    `Df(p_*)` has exactly **one real eigendirection** (the unstable one), at **62.24°** to `ker DE`.
    Therefore **no 2x2 matrix `Dg` can satisfy `Dg · DE = DE · Df(p_*)`** — the linearised semiconjugacy
    is *unattainable* at `p_*`. Measured residual `||DE·Df - Dg·DE||_F / ||DE·Df||_F = 0.665`.
  - The encoder does **not** annihilate the unstable direction (`||DE v_u|| = 0.01091` vs
    `||DE v_s|| = 0.00458` — it is the *larger*), so the naive "the encoder projects away the instability"
    story is **false**.
  - The least-squares-optimal induced map `A = DE·Df·DE^+` has eigenvalues `{-0.11713, -1.62334}`,
    **spectral radius 1.62 > 1**: the best available 2D linear surrogate at `p_*` is a **saddle**. The
    trained `Dg` is a sink instead. So a saddle *was* representable and the training did not find it.
  - **Data is not absent near p_\***: `2train.csv` has **123** rows within distance 1 of `p_*`, **356**
    within 2, **1,426** within 5, **6,380** within 10 (of 64,000); **243** training rows encode within
    0.01 of `z*` and **69** within 0.005. The "more data fixes it" narrative is not supported here.
- **The extra fixed-point attractor is universal across retrains.** Conley indices of the minimal nodes in
  every independent retrain in `<private worktree>/output/`:
  | run | minimal-node indices |
  |---|---|
  | `leslie3d_example1/seed_2/MG/morse_graph` | (x^4-1,0,0) **+ (x-1,0,0)** |
  | `leslie3d_example1_test/seed_2` | (x^4-1,0,0) **+ (x-1,0,0)** |
  | `leslie3d_example1_wide/seed_{0,2,3}` | (x^4-1,0,0) **+ (x-1,x-1,0)** |
  | `leslie3d_example1_wide/seed_1` | (x-1,x-1,0) + (x^4-1,0,0) |
  | `leslie3d_example1_search/seed_{7,10,12,14}` | (x^4-1,0,0) **+ (x-1,x-1,0)** |
  | `leslie3d_example1_search/seed_{8,9}` | (x-1,x-1,0) only / + others |
  | `leslie3d_example1_search/seed_{11,13}` | (x^4-1,0,0) only |
  **Not one of 13 retrains reproduces the ground truth's two period-4 attractors.** `seed_2` — the seed
  the active config selects (`configs/leslie3d_example1.yaml: seeds: [2]  # exhibits bistable latent Morse
  structure`) — has the "right" count (2 minimal nodes) with the **wrong types**: one period-4 attractor
  and one fixed-point attractor. Counting minimal nodes is therefore not a validity test.

---

## 8. Manifest-vs-artifact disagreements (which source to trust)

| # | manifest / doc claim | contradicting artifact | trust |
|---|---|---|---|
| 1 | `run_manifest.json` `data`: `n_samples_train 4000, n_samples_val 5000, n_iterations 30, skip 0` (all four manifests: `replay_sources/.../run_manifest.json`, `replay/leslie3d_example1{,_coauthor,_replay}/run_manifest.json`) | CSVs: 3,200/800 trajectories x 20; scaler `n_samples_seen_ = 128000` | **CSVs + scaler.** The manifest block is verbatim the dead defaults of `make_data.py:101-105`; the run was `requested_stages: ["render"]` and never touched data. |
| 2 | same manifests: `data.train_seed: 42`, `val_seed: 9999` | `make_data.py:11,21` — `'uniform'` branch is unseeded | **Neither.** Seed is unrecoverable. |
| 3 | `replay_sources/.../pipeline_summary.json` paths `replay_sources/leslie3d_spurious/...` | directory is `leslie3d_example1` | pre-rename residue (`leslie3d_spurious` -> `leslie3d_example1`) |
| 4 | `replay/leslie3d_example1_coauthor/run_manifest.json` `train_csv_sha256 a31520e7...` | current `replay_sources/leslie3d_example1/data_pairs/train.csv` is `5085e03f...` | file regenerated between 2026-05-27 and 2026-06-11; the recorded sha refers to a file that no longer exists |
| 5 | `replay/leslie3d_example1/metrics.json` and `replay/leslie3d_example1_coauthor/metrics.json` report node box counts **3955 / 46064 / 159** | active `replay_sources/.../MG/morse_sets` has **3594 / 116789 / 174** | **the box file.** Those metrics (mtime 2026-05-26T15:57) predate the 17:54 overwrite of `morse_sets` and describe **run A**, while the sibling `morse_graph.pdf` (14:53 next day) renders **run B**. Only `replay/leslie3d_example1_replay/metrics.json` (2026-06-11) matches the active boxes. |
| 6 | `replay_sources/.../metrics.json` (mtime 2026-05-04): `target_label 0`, `is_spurious_attractor true` | node 4 is never evaluated there | do not use it to label node 4 |
| 7 | `docs/figure_contracts/leslie3d_example1.md`: "node 4 has no counterpart in that detected recurrent-set inventory" | §7 above: `decode(z*) - p_*` = 0.1385, `g(E(p_*))` lands on `z*` | **the diagnostics.** The contract's distance number is correct; its conclusion is not. |
| 8 | directory name `<private worktree>/old_output/<format-conversion dir>/` | max |Δparam| = 0.0 vs the coauthor's checkpoint | **the weights.** It is a format conversion. |
| 9 | active YAML `lr_patience: 10`, `gradient_clip_norm: 1.0` | `training.py:107` uses `patience` (100); no clipping exists | **the code.** YAML values are forward-fitted, not recovered. |
| 10 | `mg_params_log.txt` `bounds_source: encoded_data` (implying g_1's data) | encoding g_1's data gives `[-0.6322,0.3191] x [-0.7518,0.2342]`, not the recorded bounds | **§5b reconstruction**: the bounds are the hull of *skip=0 data on the [0,220]x[0,154]x[0,108] IC box* |
| 11 | `final_losses.txt` | `logs/*.pkl` (§4) | **the pkl logs** |
| 12 | `<author archive>/src/*` and `main_scripts/*` presented as the code of record | 3 independent mismatches (§2, §4, §5a) | the scripts post-date the run; use them for structure only |

## 9. Files that are byte-identical between `<author archive>/` and `replay_sources/leslie3d_example1/`

`models/{encoder,decoder,dynamics}.pt`, `scalers/scaler.gz`, `data/.../2train.csv`, `2test.csv`,
`MG/morse_sets2`, `final_losses.txt`, `mg_params_log_2.txt`, `logs/*.pkl`.
**Differing:** `MG/morse_sets` (b9be6b28 vs 50c9f3bb), `MG/morse_graph` (fb7a57e0 vs the coauthor's unlabelled),
`mg_params_log.txt` (run B overwrote run A's).

## 10. Paper-asset chain (verified by sha256)

The manuscript's `morse_graph.pdf` panel (`fig:3D_Leslie_latent`)
= `<private worktree>/replay/leslie3d_example1_coauthor/MG/morse_graph.pdf`
= `72137014f4f127157cdee75e8a1fab5817b5197abcba05dd5dc19318c07f0782` (mtime 2026-05-27T14:53:10).
Its decompressed content stream contains the strings `0 : (x^4-1, 0, 0)`, `2 : (0, x^4-1, 0)` etc.,
confirming it renders **run B's labelled DOT**, not the coauthor's unlabelled one.
The manuscript's `morse_sets_with_overlay.pdf` panel = `10edb10384f737c68dd8c35f02aa97604404292f91dad0bb2b1601a97f6ed672`
(does **not** match `replay/leslie3d_example1_coauthor/MG/morse_sets_with_overlay.pdf`; overlay was
restyled separately — see `code/paper_figures/restyled/leslie3d_example1_replay/`).

## 11. Needed computations (NOT run here)

1. Re-run CMGDB on g_1 restricted to the **training-data encoded hull** `[-0.6229,0.3098] x [-0.7422,0.2242]`
   with `subdiv_limit` large enough to let `subdiv_max` bite (>= 200,000), to separate "node 4 survives
   refinement" from "refinement was never attempted". Est. tens of minutes.
2. Re-run at level 25/27 uniform to test whether node 4's box count stays O(150) while its diameter keeps
   halving (the decisive artifact-vs-invariant-set test).
3. Compute `MG(F)` node membership of `p_*` in the ground-truth run and the **exact** Morse-set index of
   `p_*`, to state the stability change as a node-to-node correspondence rather than a point comparison.
4. Recover or regenerate the IC draw for `2train.csv` (unseeded) — likely impossible; instead re-derive
   a scaler from `2train.csv` and quantify the (small) downstream effect of the §3 mismatch.

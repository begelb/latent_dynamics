# Paper dataset-table source audit

Date: 2026-08-03  
Scope: read-only audit of the paper against the archived production sources;
no file under `../paper/` was changed.

## Bottom line

The numeric entries in `../paper/main_KM2.tex`, Table `tab:data`, are correct
as counts of sampled initial conditions. No number in the table needs to be
replaced.

The ambiguity is that the table does not say explicitly that `N` counts
initial conditions rather than rows presented to the network. The trajectory
generator stores one pair `(x_t, x_{t+1})` for each retained map step, so

```text
number of transition pairs = N * (T - T0).
```

Here `T` is the number of generated map steps, not the number of states: a
trajectory with `T=20` contains 21 states and supplies 20 one-step pairs.

## Source comparison

| paper example | train / validation initial conditions | `T0`, `T` | final train / validation pairs | source status |
|---|---:|---:|---:|---|
| Leslie2Gen10D | 8,000 / 2,000 | 0, 20 | 160,000 / 40,000 | verified from config, metadata, and CSV rows |
| Leslie3D Example1 (Brittany) | 3,200 / 800 | 10, 30 | 64,000 / 16,000 | verified directly from Brittany's archived CSVs; each is a concatenation of 20-pair trajectories |
| Leslie3D Example2 (Patrick) | 8,000 / 2,000 | 0, 20 | 160,000 / 40,000 | manuscript and reconstruction agree; Patrick's raw CSVs are absent, but his scaler independently implies 160,000 training rows |
| Chafee--Infante (Marcio) | 1,000 / none | 0, 30 | 30,000 / none | verified from Marcio's generator, training loader, and all five archived datasets |
| Coral (Brittany) | `{100,200,500,1000,2000,5000}` / 10,000 | 0, 20 | `{2k,4k,10k,20k,40k,100k}` / 200,000 | verified from Brittany's metadata and CSVs; the archive calls the validation split `test` |

## What needs to be fixed or highlighted

1. **Clarify the table, without changing its numbers.** In the caption of
   `tab:data`, define `N` as sampled initial conditions and state that each
   split contains `N(T-T0)` one-step pairs. Renaming `T` from "trajectory
   length" to "map steps generated" would also remove the off-by-one
   ambiguity. Adding explicit train/validation pair-count rows would make the
   final dataset sizes visible immediately.

2. **Correct the Leslie3D prose terminology.** At
   `../paper/main_KM2.tex:847`, "4,000 and 10,000 data points" means sampled
   initial conditions. Those computations actually used 80,000 and 200,000
   transition pairs. The sentence should say "4,000 and 10,000 sampled initial
   conditions" (and may optionally give the pair totals).

3. **Do not use the current Leslie3D Example1 generation config as paper
   provenance.** `src/latentdynamics/configs/leslie3d_example1.yaml` currently
   specifies 4,000 training and 5,000 validation trajectories with `T=30` and
   `T0=10`, producing 80,000 and 100,000 pairs. The Brittany paper run was
   3,200/800 trajectories and 64,000/16,000 pairs. The config should eventually
   be corrected or labeled explicitly as an exploratory reconstruction.

4. **Keep the current Chafee config separate from Marcio's paper run.**
   `src/latentdynamics/configs/chafee_infante.yaml` adds a 200-trajectory
   validation split. Marcio's archived production training used only the
   30,000-row training CSV and had no validation split.

5. **Treat two Brittany README claims as stale.** The Leslie3D counts claimed
   in `../archive/brittany/README.md` conflict with the actual 64,000/16,000
   CSVs. The README also calls coral sampling uniform, while the metadata and
   raw initial conditions agree with scrambled Sobol sampling.

6. **Mirror any eventual clarification in the older maintained source.**
   `../paper/main_KM.tex` contains a duplicate dataset table with the same
   values.

## Strongest local sources

- Active table: `../paper/main_KM2.tex:2830-2850`
- Pair-generation rule: `src/latentdynamics/sampling/trajectories.py:66-86`
- Leslie2Gen10D: `src/latentdynamics/configs/leslie_2gen_contraction.yaml`,
  `data/leslie_2gen_contraction/{train,val}_metadata.json`, and the associated
  CSV row counts
- Leslie3D Example1:
  `../archive/brittany/data/Leslie_3D_larger_domain_tail_only/28.9_29.8_22.0/{2train,2test}.csv`
- Leslie3D Example2: `../paper/main_KM2.tex:808-814`,
  `docs/figure_contracts/leslie3d_example2.md`, Patrick's archived scaler, and
  `legacy/main_scripts/scale_data.py:23-25`
- Chafee--Infante: `../archive/marcio/scripts/{generate_data,train_model,autoencoder_model}.py`
  and `../archive/marcio/computations/run_dataset_*/train_data.csv`
- Coral: `../archive/brittany/main_scripts/{make_data,train}.py` and
  `../archive/brittany/data/coral/*_metadata.json`

## New `T=40` Leslie3D Example2 sweep

The exploratory sweep requested after this audit intentionally retains the
paper's 8,000/2,000 initial-condition counts and changes only `T: 20 -> 40`.
Each of its five independently sampled training datasets therefore contains
320,000 training pairs. Its shared validation holdout contains 80,000 pairs.
These are exploratory code-local sizes and are not paper values unless a
specific `T=40` result is later selected for the manuscript.

## New `T=25`, `N=50,000` Leslie3D Example2 sweep

The second exploratory sweep interprets `N=50,000` in the paper table's
notation: total sampled initial conditions per dataset, split 40,000 for
training and 10,000 for the common validation holdout. With `T=25` and
`T0=0`, that gives 1,000,000 training pairs and 250,000 validation pairs per
dataset tree. The current paper table stays unchanged unless this exploratory
variant is ultimately selected for the manuscript; no paper source was edited
for the sweep.

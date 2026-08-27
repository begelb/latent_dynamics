# tab_sampled_residual_tolerance

Paper table `tab:sampled_residual_tolerance` (appendix): dense sampled
residual and tolerance estimates for every minimal node of the four learned
Morse graphs displayed in the manuscript. Fifteen rows across six example
blocks.

## Producing module

`latentdynamics.analysis.sampled_metrics`, driven by
`scripts/compute_sampled_residual_tolerance.py`. For each minimal Morse node
`q` with attracting block `N_q = |π⁻¹(q)|`:

- **Residual** `R̂_q = max ||g(E(x)) - E(f(x))||₂` over sampled states whose
  encodings lie in `N_q` (`sampled_metrics.residual_protocol`). Candidate
  states combine the stored training/validation transitions, fresh Sobol
  trajectory batches, and decoder-guided states inside the computational
  domain.
- **Tolerance** `τ̂_q = min dist₂(g(z), Z \ Int N_q)` over at least `2^23`
  latent samples per node (`sampled_metrics.tolerance_protocol`): boxwise
  scrambled Sobol from two independent seeds, every box corner and center,
  and local minimization around the smallest observed values.

Both are finite-sample estimates in plain floating point, not certified
bounds: `R̂_q` is a lower estimate of the residual supremum, `τ̂_q` an upper
estimate of the true tolerance. `R̂ ≥ τ̂` exhibits a numerical violation of
the sufficient inequality; `R̂ < τ̂` only means no sampled violation was
found. This matches the paper's own framing.

The Chafee--Infante d=1/d=3 rows run through the staged
`sampled_metrics.chafee_appendix` pipeline (their blocks are exact
hyperrectangles, validated before membership tests); the d=2 residual search
is a seed ensemble merged by `sampled_metrics.merge`.

## Published rows and frozen results

Frozen copies of every result JSON ship under
`artifacts/reference_results/sampled_residual_tolerance/`; new runs write
under `output/sampled_residual_tolerance/`. The frozen full-precision values
round exactly to every published table entry.

| Paper row | q | published \|S_q\| / R̂ / τ̂ | R̂<τ̂? | frozen full-precision (R̂; \|S_q\|; τ̂) | frozen result |
|---|---|---|---|---|---|
| 3D Leslie, fine | 0 | 5.68e5 / 1.07 / 4.25e-4 | No | 1.0681129693984985; 568,045; 4.2457133531570435e-4 | `leslie3d_example1/dense_sampling.json` |
| 3D Leslie, fine | 1 | 3.78e6 / 6.97e-1 / 4.06e-4 | No | 0.6969771385192871; 3,777,740; 4.063080996274948e-4 | same |
| 3D Leslie, fine | 4 | 2.84e4 / 2.31e-1 / 4.62e-4 | No | 0.2313537299633026; 28,363; 4.622591659426689e-4 | same |
| 3D Leslie, coarse (22,22,24) | 0 | 9.04e5 / 1.07 / 8.01e-4 | No | 1.0681129693984985; 904,399; 8.005722775124013e-4 | coarse `sampled_residual_tolerance.json` (see below) |
| 3D Leslie, coarse (22,22,24) | 1 | 3.07e6 / 6.97e-1 / 7.92e-4 | No | 0.6969771385192871; 3,074,979; 7.924759411253035e-4 | same |
| Extended Leslie (10D) | 0 | 2.24e6 / 6.80e-2 / 5.20e-5 | No | 0.06798265129327774; 2,240,516; 5.1976014219690114e-5 | `leslie_2gen_contraction/dense_sampling.json` |
| Extended Leslie (10D) | 1 | 1.31e5 / 5.31e-2 / 5.41e-5 | No | 0.05311565101146698; 131,462; 5.4065500080469064e-5 | same |
| Red coral | 0 | 1.94e6 / 5.40e-2 / 7.79e-3 | No | 0.053966641426086426; 1,940,312; 7.788083888590203e-3 | `coral_candidate_train500_seed16/dense_sampling.json` |
| Red coral | 1 | 7.16e5 / 2.48e-1 / 7.96e-3 | No | 0.24785225093364716; 716,059; 7.956178160384353e-3 | same |
| Chafee--Infante d=1 | 0 | 1.33e5 / 6.58 / 1.04e-1 | No | 6.583362102508545; 132,968; 0.103884458541871 | `chafee_latent_dimensions/dense_sampling.json` (dimensions.1) |
| Chafee--Infante d=1 | 1 | 1.31e5 / 6.11 / 6.58e-2 | No | 6.107765197753906; 131,246; 0.06583485603332484 | same |
| Chafee--Infante d=2 | 0 | 1.11e5 / 3.52e-2 / 3.95e-2 | **Yes** | 0.03520679101347923; 111,015; 0.039529092609882355 | `chafee_infante_current/dense_sampling.json` |
| Chafee--Infante d=2 | 1 | 1.25e5 / 1.60e-2 / 4.25e-2 | **Yes** | 0.016002139076590538; 125,123; 0.04251862317323685 | same |
| Chafee--Infante d=3 | 0 | 1.19e5 / 4.31e-3 / 2.34e-2 | **Yes** | 0.0043133459985256195; 119,378; 0.02338654361665249 | `chafee_latent_dimensions/dense_sampling.json` (dimensions.3) |
| Chafee--Infante d=3 | 1 | 1.18e5 / 4.73e-3 / 2.36e-2 | **Yes** | 0.00472954660654068; 118,473; 0.02355029061436653 | same |

Every tolerance search evaluated at least `2^23 = 8,388,608` latent points,
as the table caption states; the per-node counts are recorded in each frozen
JSON. All sampled `g`-images stayed in block interiors.

SHA-256 of the frozen `dense_sampling.json` files as shipped:

- `leslie3d_example1/`:
  `d038af408c86df4ad259d06916f77992532c34c677fa895365cc8b6977b3d294`
- `leslie_2gen_contraction/`:
  `e61fad9b15d3fa74c07075ec4763e66efb7cd9c6fe65395b535e23365e2e8f07`
- `coral_candidate_train500_seed16/`:
  `296ecffb8168e94d3ff3dae5d866342cbcff6f6a5271b78f8e1bd7c720fe8a3e`
- `chafee_infante_current/`:
  `1a900ce6215656ee5850c9b8075847cbada321d9ce62aac79a9d532e49902306`
- `chafee_latent_dimensions/`:
  `ce2b219749b5db5de1b317496ceb537095938b2516b8ec543a600fa27c382e7b`

The coarse (22,22,24) rows belong to the fixed22_vs_merged45 workflow (see
[`leslie3d_example1.md`](leslie3d_example1.md)); their frozen
`sampled_residual_tolerance.json`, `tolerance_sampling.json`, and the exact
forward-closure verification ship under
`artifacts/reference_results/leslie3d_example1/`.

## Required artifacts (fetched bundles)

| Example block | model / blocks | stored pair data |
|---|---|---|
| 3D Leslie fine and coarse | `replay_sources/leslie3d_example1/spurious_attractor_ex/` (checkpoint, `MG/morse_sets`); scaler `replay_sources/leslie3d_example1/28.9_29.8_22.0/scalers/scaler.gz`; coarse rows also need the coarse (22,22,24) blocks | `replay_sources/leslie3d_example1/data_pairs/{train,val}.csv` and the archived `2train`/`2test` CSVs |
| Extended Leslie (10D) | `replay_sources/leslie_2gen_contraction/` (model, `MG/morse_sets`, `scalers/train/scaler.gz`) | `replay_sources/leslie_2gen_contraction/data_pairs/{train,val}.csv` |
| Red coral | `replay_sources/coral/train_500/seed_16/` (model triplet, `MG/morse_sets`); scaler `replay_sources/coral/data/scalers/train_500/scaler.gz` | `replay_sources/coral/data/coral/{train_500,test}.csv` (`test.csv`, 62 MB, supplies most residual candidates including the node-0 witness) |
| Chafee d=2 | `replay_sources/chafee_infante/replay/` (converted author weights, blocks) | `replay_sources/chafee_infante/data/train.csv` (the mirrored `test.csv` is deliberately not double-counted) |
| Chafee d=1, d=3 | `replay_sources/chafee_infante/latent_dimension_study/latent_{1,3}d/seed_0/` (checkpoints and `MG_adaptive/morse_sets`, sha-pinned inside the frozen JSON) | `replay_sources/chafee_infante/data/train_data.csv` (30,000 one-step pairs) |

Sampling seeds are fixed in the protocol and recorded in
`artifacts/reference_results/sampled_residual_tolerance/run_plan.yaml` and
the frozen JSONs (tolerance Sobol seeds 20260725/20260726; residual fresh
trajectory seed base 20260727 with per-batch seeds 20260727-20260731;
decoder-guided seeds 20260732-20260735 for the Chafee ensemble, 20260827
otherwise).

## Reproduction commands

```bash
# Single-block examples (discrete maps):
python scripts/compute_sampled_residual_tolerance.py leslie3d_example1
python scripts/compute_sampled_residual_tolerance.py leslie_2gen_contraction
python scripts/compute_sampled_residual_tolerance.py coral_candidate_train500_seed16

# Chafee d=2 (tolerance once, residual per seed batch, then merge):
python scripts/compute_sampled_residual_tolerance.py chafee_infante_current \
  --stage residual --seed 20260728 --output-suffix seed20260728
# ... repeat for the remaining fresh/decoder seeds, then --merge-suffixes

# Chafee d=1/d=3 staged appendix pipeline:
python scripts/compute_sampled_residual_tolerance.py chafee_latent_dimensions \
  --stage tolerance --dimension 1
# ... then the stored/fresh/decoder stages and merge

# 3D Leslie coarse rows (coarse (22,22,24) workflow):
python scripts/leslie3d_example1_uniform_sampled_metrics.py --depth 22 --stage all
python scripts/leslie3d_example1_verify_closures.py
```

## Runtime notes (honest estimates)

Recorded wall-clock on the development machine (Apple M4 Pro, CPU):

- **Leslie fine / Extended Leslie / coral**: recorded per-phase timings are
  small (trajectory generation 2-11 s; tolerance passes seconds to tens of
  seconds per node), but no total wall-clock was logged for the 2026-07-25
  runs; expect minutes per example for the batched residual evaluation over
  millions of candidates. The Extended Leslie residual searches alone
  recorded ~1,308 s (node 0) and ~982 s (node 1) in the results summary.
- **Leslie coarse (22,22,24)**: fixed-depth graph build 27 s; metric stage
  ~113 s recorded; total well under 5 minutes.
- **Chafee d=2**: ~6,430 s (~1.8 h) total recorded elapsed across the timed
  phases; PDE trajectory generation dominates and the residual seed batches
  parallelize.
- **Chafee d=1 + d=3**: tolerance stages 3-6 s; residual batches 0.7-710 s
  each (~85 min CPU combined, ~13 min wall with concurrent batches).

Do not quote these as guarantees for other hardware or backends.

## Verification

Compare a fresh run's `dense_sampling.json` per-node
`residual.sampled_maximum`, `residual.accepted_samples`, and
`tolerance.sampled_minimum` against the frozen values above. Identical seeds
and identical dependency versions reproduce the frozen values; different
BLAS/torch builds may perturb the last digits without changing any table
entry at the published precision or any Yes/No verdict.

## Known limitations

- Finite sampling with plain floating-point arithmetic and no outward
  rounding: no row is a computer-assisted proof, and a "Yes" row is not a
  verification of the localized semiconjugacy inequality.
- The four 2026-07-25 single-block frozen JSONs do not embed script or
  checkpoint checksums (the Chafee d=1/d=3 and coarse-grid artifacts do);
  their provenance is fixed by `run_plan.yaml`, the co-located
  `RESULTS.md`, and the value-exact match to the published table.
- A quick per-run `metrics.json` tolerance diagnostic exists in some replay
  trees (`tau_bar`); it can differ from the dense value in the fourth digit
  (Extended Leslie node 1: 5.4075e-5 vs 5.4066e-5). The dense evaluation is
  the paper source.

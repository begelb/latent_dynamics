# RMSE-bounded CMGDB replay

This directory began as a byte-for-byte copy of
`leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1`. The copy was verified
over 392 files by relative path, byte size, and SHA-256 hash before any new
analysis was run.

The copied baseline was not overwritten:

- each cell's original `MG/`, `metrics.json`, and `mg_params_log.txt` still
  describe the saved fine-grid `25/28/29` computation;
- the copied top-level `summary/` and `sweep_summary.json` also describe that
  fine-grid baseline;
- each new result is isolated under the cell's `rmse_bounded_cmgdb/` folder;
- the new aggregate report is isolated under `summary_rmsegrid/`.

For each saved checkpoint, the replay uses

`target_rmse = min(sqrt(final train L3), sqrt(final holdout L3))`

and chooses the coarsest total subdivision depth at which both nominal 2-D
box side lengths are no larger than `target_rmse`. The adaptive ladder keeps
the original gaps: `(subdiv_init, subdiv_min, subdiv_max) = (s-3, s, s+1)`.
All other recorded CMGDB settings and the exact per-cell latent bounds are
held fixed.

Thus the bound applies coordinatewise to the nominal `subdiv_min` scale and
to every returned `MG/morse_sets` box. It does not bound Euclidean box
diameter, and it does not bound every internal query box: the preserved
`subdiv_init=s-3` initialization is coarser than RMSE. A separate strict
"every queried box" experiment would need to start at `subdiv_init=s`.

No model was trained and no dataset or scaler was regenerated. The analysis
loads the copied checkpoints and scalers together with the existing training
pair CSVs. `RMSE_GRID_CLONE_MANIFEST.json` and `RMSE_GRID_PROVENANCE.json`
record the copy and execution provenance.

Interpretation caveat: these RMSE-matched grids are much coarser than the
baseline. Disappearance or merging of fine-grid recurrent components is a
resolution-sensitivity observation, not a proof that those learned invariant
sets do not exist. A trivial Conley index at the coarse scale is likewise not
evidence for an attracting invariant set.

The post-run containment audit found that every fine-grid Morse box, across
every fine component, is contained in the sole coarse cover in all 15 cells.
The observed change is therefore finite-cover merging, not exclusion of the
fine components. Full code-tree and input-hash provenance is recorded in
`analysis_codex/leslie3d_3x5_rmsegrid_replay/provenance_supplement.json`.

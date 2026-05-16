# C1: Conditional adaptive 2D Morse-set plot limits

Date: 2026-05-16
Status: spec — ready for implementation planning

## Context

The 2D Morse-set renderer (`code/src/latentdynamics/viz/morse_plots.py`) currently
applies adaptive axis limits unconditionally: it shrinks the plotted box to the
occupied Morse-set region plus a margin, clipped to the CMGDB bounds. This was
added in an in-flight change documented in `ASAP_HANDOFF.md` and is exercised by
`tests/test_viz.py::TestMorseSetPlotting::test_2d_morse_set_plot_uses_adaptive_limits_clipped_to_cmgdb_bounds`.

For runs with uniform CMGDB subdivision (`subdiv_init == subdiv_min ==
subdiv_max`), every Morse box has the same width on each axis, and the Morse
sets typically fill the CMGDB box. Adaptive shrinking in that regime produces no
useful zoom and visually disagrees with paper conventions for uniform-subdivision
figures.

## Goal

Make the 2D Morse-set plot adaptive **iff** CMGDB subdivision was non-uniform.
Detect non-uniformity from the box-width distribution in the saved
`morse_sets` CSV, so no new caller-side plumbing is needed.

## Non-goals

- Per-stage timing instrumentation in `pipeline_summary*.json` / `run_manifest.json`.
  Tracked separately as C1b; out of scope here.
- 1D Morse-set plots (`_plot_morse_sets_1d`). Untouched.
- 3D+ Morse-set plots. The function is 2D-only by construction.
- Any change to CMGDB invocation, the `morse` stage, or saved artifacts.
- A YAML configurability knob for adaptivity (rejected: data-driven check is sufficient).

## Approach

### Where the change lives

Single private helper in `code/src/latentdynamics/viz/morse_plots.py`:

- Current name: `_adaptive_2d_morse_set_limits` (lines 393-422).
- New name: `_morse_set_plot_limits_2d` (no longer always adapts).
- Only caller: `_plot_morse_sets_2d` at line 382. Update the call site.

No public-API changes. No new function parameters on `plot_morse_sets_from_csv`,
`render_morse_sets_from_csv`, `render_morse_from_files`, or
`_plot_morse_sets_2d`.

### Behavior

Given box rows `(lx, ly, ux, uy)` and optional `bounds_lower`, `bounds_upper`:

1. Compute per-axis widths `wx = ux - lx`, `wy = uy - ly` for every row.
2. **Uniform-subdivision check.** Treat the run as uniform iff
   `wx.max() == wx.min()` **and** `wy.max() == wy.min()`, compared via
   Python `==` (float-exact equality, no tolerance). Justification: CMGDB
   output for `subdiv_init == subdiv_min == subdiv_max` produces boxes
   whose widths are all computed by the same arithmetic, so they are
   bitwise equal. If a future CMGDB knob produces uniform widths via a
   different path, the data-driven check still does the right thing.
3. **Uniform branch.** Let `occupied = (lx.min(), ly.min(), ux.max(), uy.max())`.
   - If `bounds_lower` and `bounds_upper` are both provided, return
     `((bounds_lower[0], bounds_upper[0]), (bounds_lower[1], bounds_upper[1]))`
     unchanged (no margin, no clipping — they are already the desired plot
     extent).
   - Else return the occupied extent padded by one box width on each side:
     `((lx.min() - wx[0], ux.max() + wx[0]), (ly.min() - wy[0], uy.max() + wy[0]))`.
     `wx[0]` is well-defined because every entry of `wx` is equal in this
     branch (and analogously for `wy`).
4. **Non-uniform branch.** Unchanged from current implementation:
   `margin = max(2 * median_box_width, 0.03 * occupied_span)` per axis,
   clipped to CMGDB bounds when provided.

### Tests (in `tests/test_viz.py::TestMorseSetPlotting`)

| Test name | Inputs | Expected limits |
|---|---|---|
| `test_non_uniform_subdivision_uses_adaptive_limits_clipped_to_cmgdb_bounds` (rewrite of existing test) | mixed widths (e.g. one 0.1×0.1 box, one 0.2×0.2 box), CMGDB bounds `[0,0]`-`[10,20]` | adaptive: occupied + max(2·median, 3% span), clipped |
| `test_uniform_subdivision_with_bounds_uses_full_cmgdb_bounds` (new) | all 0.1×0.1 boxes, CMGDB bounds `[0,0]`-`[10,20]` | `(0,10)` × `(0,20)` |
| `test_uniform_subdivision_without_bounds_falls_back_to_occupied` (new) | all 0.1×0.1 boxes, no CMGDB bounds | occupied ± one box width on each axis |

The existing focused test
(`test_2d_morse_set_plot_uses_adaptive_limits_clipped_to_cmgdb_bounds`) is
renamed and reseeded with non-uniform widths. Its current input boxes are both
0.1 × 0.1, which under the new rule trigger the uniform branch and yield full
CMGDB bounds; the rewrite makes it a true non-uniform case so the adaptive
margin rule is still pinned.

## Verification (Bar 2)

After implementation, before declaring done:

1. Focused suite:

   ```bash
   cd code
   ../.venv/bin/pytest tests/test_viz.py tests/test_experiments.py::TestRenderReplayRouting -q
   ```

   All tests must pass.

2. Re-render the two `leslie2d_to_2d_chafee_like` seeds without rerunning CMGDB:

   ```bash
   cd code
   ../.venv/bin/python pipeline.py \
     --config configs/leslie2d_to_2d_chafee_like.yaml \
     --stages render,metrics \
     --cell-index 0 \
     --expected-cells 2

   ../.venv/bin/python pipeline.py \
     --config configs/leslie2d_to_2d_chafee_like.yaml \
     --stages render,metrics \
     --cell-index 1 \
     --expected-cells 2
   ```

3. Inspect:

   - `code/output/leslie2d_to_2d_chafee_like/seed_0/MG/morse_sets.png`
   - `code/output/leslie2d_to_2d_chafee_like/seed_1/MG/morse_sets.png`

   This config uses `22/22/22` subdivision (uniform). Expected: axis limits =
   full CMGDB bounds (the pre-adaptive look). Confirm no shrinkage.

## Risks and mitigations

- **Risk: float-exact equality is too strict.** Mitigation: CMGDB widths under
  uniform subdivision come from identical arithmetic and are bitwise equal in
  current builds. If a future CMGDB version introduces rounding, switch to
  `numpy.allclose` with `rtol=0, atol=0` initially and loosen only on observed
  failure.
- **Risk: a CMGDB run produces all-equal widths despite non-uniform subdivision
  params.** Possible only if the dynamics happen to be subdivided to the same
  depth everywhere; in that case the box geometry truly is uniform and the
  full-bounds plot is the right answer.
- **Risk: existing renders downstream of `plot_morse_sets_from_csv` (e.g.,
  overlay scripts in `docs/figure_contracts/`) depend on the old adaptive
  default.** Mitigation: the public API of `plot_morse_sets_from_csv` is
  unchanged; only the limits are different in the uniform-subdivision case,
  and that matches paper convention for those figures.

## Out-of-scope follow-ups (do not bundle)

- **C1b — per-stage timing in `pipeline_summary*.json` / `run_manifest.json`.**
  Separate spec after C1 lands.
- **C2/C3 — figure-parity regressions in `leslie3d_success`,
  `leslie_contraction`, `chafee_infante`, and `coral_basic`.** Tracked in
  `code/docs/FIGURE_PARITY.md`; orthogonal to plot bounds.

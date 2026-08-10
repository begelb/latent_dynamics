# Chafee--Infante continuation data

This directory is a copy of the completed computation from Codex task
`019fd857-e1fb-7740-8644-9df10199f3f7`. The original working artifacts are in
`analysis_codex/chafee_10k_continuation/`.

The computation continued the 2,138 trajectories that had not met the original
convergence criterion by time 6. All 2,138 subsequently met the same criterion.
The completed set contains 10,000 labels, with 5,030 trajectories converging to
the negative equilibrium and 4,970 to the positive equilibrium. Independent
LSODA and BDF integrations agreed on every completed label.

The manuscript statistics are taken from `updated_paper_statistics.csv` and
`updated_paper_statistics.json`. These files rescore the unchanged CMGDB basin
assignments for all 45 learned models against the completed labels.

See `REPORT.md` and `summary.json` for the numerical protocol and provenance.
The original `run_continuation.py` and `run_config.json` are preserved with the
results. Their relative paths refer to the original analysis directory recorded
above.

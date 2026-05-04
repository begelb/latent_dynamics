# Legacy code archive

This directory holds the pre-restructure pipeline. The active paper pipeline
lives in `code/src/latentdynamics/` and is driven by `code/pipeline.py` /
`code/reproduce_paper.py`.

| Subdirectory | Status | Why it stays |
|---|---|---|
| `src/` | **Required at runtime.** | The 3-file checkpoints under `output/.../models/{encoder,dynamics,decoder}.pt` are pickled `nn.Module` instances whose qualified class name is `src.models.Encoder` etc. `latentdynamics.training.checkpoints.load_legacy_checkpoint` adds this directory's parent to `sys.path` so unpickling resolves. **Do not delete.** |
| `main_scripts/` | Historical. | Pre-restructure CLI (`make_data.py`, `train.py`, `morse_graph.py`, `scale_data.py`). Superseded by `latentdynamics.cli.*`. |
| `Leslie_analysis_scripts/` | Historical. | Pre-restructure figure scripts for the Leslie 3D / Leslie contraction cases. Superseded by `latentdynamics.cli.render` + `latentdynamics.viz`. |
| `coral_experiment_scripts/` | Historical. | Pre-restructure coral analysis scripts. Superseded by `latentdynamics.cli.metrics:_coral_metrics`. |
| `config_yaml/` | Historical. | Pre-restructure YAMLs (no defaults inheritance, no pydantic validation). Superseded by `code/configs/*.yaml`. |

`pyproject.toml` packages only `src/latentdynamics/`, so nothing in this
directory ships in the installed wheel.

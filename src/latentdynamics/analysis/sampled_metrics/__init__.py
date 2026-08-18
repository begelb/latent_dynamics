"""Sampled residual and tolerance estimates for latent attracting blocks.

For each minimal Morse node ``q`` with block ``N_q``, two dense sampled
estimates are computed in the stored latent coordinates:

* the residual ``R_hat(q) = max ||g(E(x)) - E(f(x))||_2`` over sampled states
  whose encodings lie in ``N_q`` (:mod:`.residual_protocol`); and
* the tolerance ``tau_hat(q) = min dist_2(g(z), Z \\ Int N_q)`` over at least
  ``2**23`` latent samples per node (:mod:`.tolerance_protocol`).

Both are finite-sample estimates, not bounds: ``R_hat >= tau_hat`` exhibits a
numerical violation of the sufficient inequality, while ``R_hat < tau_hat``
only means no sampled violation was found.

:mod:`.chafee_appendix` runs the same protocols for the Chafee--Infante
latent d=1 and d=3 models, whose blocks are exact hyperrectangles.
Frozen published results live under
``artifacts/reference_results/sampled_residual_tolerance/``; new runs write
under ``output/sampled_residual_tolerance/`` by default.  The command-line
driver is ``scripts/compute_sampled_residual_tolerance.py``.
"""

from . import chafee_appendix
from .chafee_appendix import HyperrectangleBlock
from .merge import merge_chafee_dense_runs
from .residual_protocol import run_dense_sampling
from .tolerance_protocol import (
    EXAMPLES,
    BlockGeometry,
    Example,
    default_output_root,
    reference_results_root,
    run_tolerance_evaluation,
)

__all__ = [
    "EXAMPLES",
    "BlockGeometry",
    "Example",
    "HyperrectangleBlock",
    "chafee_appendix",
    "default_output_root",
    "merge_chafee_dense_runs",
    "reference_results_root",
    "run_dense_sampling",
    "run_tolerance_evaluation",
]

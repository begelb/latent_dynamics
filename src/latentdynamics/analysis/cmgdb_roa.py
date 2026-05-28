"""Re-export of exact regions of attraction from the CMGDB fork.

The pure-CMGDB RoA computation has migrated upstream. This module is
preserved for backward compatibility during the transition.
"""

from CMGDB.cmgdb_roa import (  # noqa: F401
    BOUNDARY,
    ESCAPE,
    EXACT_ROA_FILENAME,
    MULTI,
    CellROA,
    collapse_roa_to_lca,
    compute_and_save_exact_roa,
    compute_exact_roa,
    load_exact_roa,
    save_exact_roa,
)

__all__ = [
    "BOUNDARY",
    "ESCAPE",
    "EXACT_ROA_FILENAME",
    "MULTI",
    "CellROA",
    "collapse_roa_to_lca",
    "compute_and_save_exact_roa",
    "compute_exact_roa",
    "load_exact_roa",
    "save_exact_roa",
]

"""Re-export of MorseGraph parser from the CMGDB fork.

The DOT-format parser and pure-CMGDB graph algorithms have migrated upstream.
This module is preserved for backward compatibility during the transition.
"""

from CMGDB.morse_graph_parser import MorseGraph  # noqa: F401

__all__ = ["MorseGraph"]

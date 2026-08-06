# Ives Lake Mývatn 3x5 replication summary

**Report status:** COMPLETE AND STRICTLY VERIFIED

Verified cells: 15/15; complete artifact sets: 15/15; machine passes: 1/15 evaluated.

A machine pass requires a directed graph isomorphic to the archived four-node branch-then-chain graph, a fixed point in exactly one sink, and at least 11 of 12 cycle phases uniquely in the other sink, with no phase in a conflicting sink. The sink-period set `{1, 12}` is reported only as a Conley-index diagnostic because the archived JSON did not retain indices.

| data seed | model seed | status | nodes | edges | sinks | fixed sink | cycle sink | cycle phases | conflicts | periods | pass |
|---:|---:|:---|---:|---:|---:|:---|:---|---:|---:|:---|:---:|
| 2158 | 0 | verified_fail | 2 | 1 | 1 | None | None | 0/12 | 4 | `[1]` | False |
| 2158 | 1 | verified_fail | 3 | 2 | 1 | None | None | 0/12 | 11 | `[12]` | False |
| 2158 | 2 | verified_pass | 4 | 3 | 2 | 2 | 0 | 12/12 | 0 | `[12,1]` | True |
| 4792 | 0 | verified_fail | 3 | 2 | 1 | None | None | 0/12 | 5 | `[1]` | False |
| 4792 | 1 | verified_fail | 3 | 2 | 1 | None | None | 0/12 | 4 | `[1]` | False |
| 4792 | 2 | verified_fail | 5 | 4 | 1 | 0 | None | 0/12 | 0 | `[1]` | False |
| 3174 | 0 | verified_fail | 4 | 3 | 1 | None | None | 0/12 | 4 | `[12]` | False |
| 3174 | 1 | verified_fail | 2 | 1 | 1 | None | None | 0/12 | 12 | `[1]` | False |
| 3174 | 2 | verified_fail | 4 | 3 | 1 | None | None | 0/12 | 12 | `[12]` | False |
| 688 | 0 | verified_fail | 2 | 1 | 1 | None | None | 0/12 | 11 | `[1]` | False |
| 688 | 1 | verified_fail | 3 | 2 | 1 | None | None | 0/12 | 10 | `[12]` | False |
| 688 | 2 | verified_fail | 4 | 3 | 1 | None | None | 0/12 | 12 | `[12]` | False |
| 5727 | 0 | verified_fail | 2 | 1 | 1 | None | None | 0/12 | 9 | `[1]` | False |
| 5727 | 1 | verified_fail | 3 | 2 | 1 | None | None | 0/12 | 4 | `[1]` | False |
| 5727 | 2 | verified_fail | 1 | 0 | 1 | 0 | None | 0/12 | 11 | `[1]` | False |

## Provenance and evidence

`cells.json` retains every DOT node id, exact Conley tuple and components, directed edge, degree, sink/minimal flag, inferred period, all 13 encoded reference points, and every per-point Morse-node and sink membership. `cells.csv` includes the compact classification evidence, selected training losses, and SHA-256 hashes for every required artifact.

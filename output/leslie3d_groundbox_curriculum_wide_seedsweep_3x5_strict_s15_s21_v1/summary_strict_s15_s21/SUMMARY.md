# Leslie3D strict CMGDB sweep: s=15,...,21

All 105 runs load saved checkpoints and use `(s,s,s+1)` with no training.

The dimensionally matched scale comparison is `D^2/2 <= L3`, equivalently 
`D <= sqrt(2*L3)`, because L3 is mean-reduced MSE over two latent coordinates.

| s | node-count distribution | minimal-count distribution | train D pass | holdout D pass | coordinate-side pass | limit flags | CMGDB s |
|---:|---|---|---:|---:|---:|---:|---:|
| 15 | `{1: 15}` | `{1: 15}` | 14/15 | 15/15 | 12/15 | 0 | 3.199 |
| 16 | `{1: 14, 2: 1}` | `{1: 15}` | 15/15 | 15/15 | 15/15 | 0 | 5.696 |
| 17 | `{1: 10, 2: 5}` | `{1: 15}` | 15/15 | 15/15 | 15/15 | 1 | 10.279 |
| 18 | `{1: 5, 2: 6, 3: 4}` | `{1: 12, 2: 3}` | 15/15 | 15/15 | 15/15 | 5 | 17.705 |
| 19 | `{1: 1, 2: 6, 3: 8}` | `{1: 10, 2: 5}` | 15/15 | 15/15 | 15/15 | 8 | 31.711 |
| 20 | `{2: 2, 3: 10, 4: 1, 5: 1, 7: 1}` | `{1: 4, 2: 11}` | 15/15 | 15/15 | 15/15 | 9 | 53.654 |
| 21 | `{2: 1, 3: 9, 4: 4, 5: 1}` | `{1: 3, 2: 12}` | 15/15 | 15/15 | 15/15 | 7 | 105.510 |

The raw MSE and box diameter have different physical units. The report retains both 
numbers but never treats `D <= MSE` as a meaningful inequality.

Because `padding=true` pads by the current box width, the outer approximations are 
not guaranteed to form a nested sequence as s changes.

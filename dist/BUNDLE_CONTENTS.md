# Replay artifact bundle contents

Built 2026-08-17 from code repo commit `1e1c6d5` (1e1c6d5616e77c3ce0239caf34d83470d45dc22e) plus the
workspace `output/` and `archive/` trees. All tarballs are deterministic
(members sorted, numeric owner 0/0, gzip -9 -n, source mtimes preserved)
and are rooted at a single top-level directory equal to the bundle key.
Per-member SHA-256 listings live next to this file as `<key>.sha256`
(format: `<sha256>  <member path>`).

| bundle | members | uncompressed | tar.gz | tarball sha256 |
|---|---:|---:|---:|---|
| `leslie_2gen_contraction` | 27 | 55.6 MiB | 15.4 MiB | `6118bb90d7d707f0fc6db3f85f5360d1b7a48faeb937afacfa150223ae4eb634` |
| `leslie3d_example1` | 111 | 90.1 MiB | 18.8 MiB | `74f033fedf9e55395953c296c43cc44103be66c1a13f41ebeef9fc449e5fc2b3` |
| `chafee_infante` | 1558 | 326.6 MiB | 84.0 MiB | `db2db555282f6947e044ea64b0d45081caa3728c4cfcc87406da1ae13d9e58c9` |
| `coral` | 12 | 62.4 MiB | 16.6 MiB | `d4443ad65c9280d585a756490a6fb9ae25020540e601337a23891986437fea2e` |
| `original_leslie_2d_reference` | 8 | 146.9 MiB | 13.3 MiB | `5c3923393b5fb1bf9f48720ef7bff293d34867e0beabb299814e9d489879c709` |
| `original_leslie3d_reference` | 47 | 152.1 MiB | 12.9 MiB | `ea26f90b2ed093bf3bf09841975f297aefa8fe869ad7af74d70054a724513af9` |
| `chafee_training_datasets` | 5 | 467.0 MiB | 105.3 MiB | `4f097c2f37ecd198696b22a2e6f185bc9ece137dbd38dfe3a678f82535a8405d` |

## leslie_2gen_contraction

- file: `replay_leslie_2gen_contraction.tar.gz`
- members: 27 files
- uncompressed size: 58,350,651 bytes (55.6 MiB)
- compressed size: 16,135,287 bytes (15.4 MiB)
- tarball sha256: `6118bb90d7d707f0fc6db3f85f5360d1b7a48faeb937afacfa150223ae4eb634`
- per-member sha256 listing: `dist/leslie_2gen_contraction.sha256`

## leslie3d_example1

- file: `replay_leslie3d_example1.tar.gz`
- members: 111 files
- uncompressed size: 94,503,920 bytes (90.1 MiB)
- compressed size: 19,732,676 bytes (18.8 MiB)
- tarball sha256: `74f033fedf9e55395953c296c43cc44103be66c1a13f41ebeef9fc449e5fc2b3`
- per-member sha256 listing: `dist/leslie3d_example1.sha256`

## chafee_infante

- file: `replay_chafee_infante.tar.gz`
- members: 1558 files (11 zero-byte, allowlisted)
- uncompressed size: 342,501,963 bytes (326.6 MiB)
- compressed size: 88,094,304 bytes (84.0 MiB)
- tarball sha256: `db2db555282f6947e044ea64b0d45081caa3728c4cfcc87406da1ae13d9e58c9`
- per-member sha256 listing: `dist/chafee_infante.sha256`

## coral

- file: `replay_coral.tar.gz`
- members: 12 files
- uncompressed size: 65,411,018 bytes (62.4 MiB)
- compressed size: 17,438,366 bytes (16.6 MiB)
- tarball sha256: `d4443ad65c9280d585a756490a6fb9ae25020540e601337a23891986437fea2e`
- per-member sha256 listing: `dist/coral.sha256`

## original_leslie_2d_reference

- file: `replay_original_leslie_2d_reference.tar.gz`
- members: 8 files
- uncompressed size: 154,066,870 bytes (146.9 MiB)
- compressed size: 13,958,693 bytes (13.3 MiB)
- tarball sha256: `5c3923393b5fb1bf9f48720ef7bff293d34867e0beabb299814e9d489879c709`
- per-member sha256 listing: `dist/original_leslie_2d_reference.sha256`

## original_leslie3d_reference

- file: `replay_original_leslie3d_reference.tar.gz`
- members: 47 files
- uncompressed size: 159,486,627 bytes (152.1 MiB)
- compressed size: 13,509,237 bytes (12.9 MiB)
- tarball sha256: `ea26f90b2ed093bf3bf09841975f297aefa8fe869ad7af74d70054a724513af9`
- per-member sha256 listing: `dist/original_leslie3d_reference.sha256`

## chafee_training_datasets

- file: `replay_chafee_training_datasets.tar.gz`
- members: 5 files
- uncompressed size: 489,640,675 bytes (467.0 MiB)
- compressed size: 110,386,630 bytes (105.3 MiB)
- tarball sha256: `4f097c2f37ecd198696b22a2e6f185bc9ece137dbd38dfe3a678f82535a8405d`
- per-member sha256 listing: `dist/chafee_training_datasets.sha256`

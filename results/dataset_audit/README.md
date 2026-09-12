# CubiCasa5K dataset audit

Generated from the official split files and the verified five-class mapping.

| Split | Official | Audited | Valid | Failures | Rooms | Walls | Doors | Windows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 4200 | 4200 | 4200 | 0 | 45002 | 110624 | 42004 | 36985 |
| val | 400 | 400 | 400 | 0 | 4189 | 10188 | 3853 | 3395 |
| test | 400 | 400 | 400 | 0 | 4351 | 10711 | 4088 | 3629 |

Test was audited only after the baseline configuration was frozen.

Detailed distributions, class frequencies, checksums, failures and selected example IDs are in the JSON reports.

# Dataset Card: TDR Transmission Line Fault & Load Identification

## Summary
- **Generation Timestamp**: 2026-11-06 08:40:59
- **Total Generation Time**: 164.95 seconds
- **Mode**: Full Mode

## Split Sizes and Generation Speed
| Split | Samples | Generation Time (s) | Time per 1k samples (s) |
|---|---|---|---|
| `train` | 60,000 | 116.50 | 1.94 |
| `val` | 10,000 | 19.43 | 1.94 |
| `test` | 10,000 | 19.28 | 1.93 |
| `ood` | 5,000 | 8.92 | 1.78 |

## Fault Class Distribution Across Splits
| Fault Class | Train | Val | Test | OOD |
|---|---|---|---|---|
| `none` | 8,622 | 1,455 | 1,413 | 521 |
| `short` | 8,497 | 1,451 | 1,408 | 746 |
| `open` | 8,602 | 1,363 | 1,413 | 748 |
| `shunt_R` | 8,469 | 1,424 | 1,435 | 735 |
| `shunt_C` | 8,507 | 1,477 | 1,434 | 782 |
| `series_R` | 8,794 | 1,416 | 1,416 | 757 |
| `z0_section` | 8,509 | 1,414 | 1,481 | 711 |

## Load Class Distribution Across Splits
| Load Class | Train | Val | Test | OOD |
|---|---|---|---|---|
| `open` | 9,910 | 1,644 | 1,671 | 841 |
| `short` | 9,971 | 1,672 | 1,678 | 826 |
| `matched` | 9,903 | 1,596 | 1,663 | 827 |
| `R` | 10,024 | 1,672 | 1,662 | 885 |
| `RL` | 10,166 | 1,737 | 1,645 | 832 |
| `RC` | 10,026 | 1,679 | 1,681 | 789 |

## Parameter Ranges
- **Cable Length**: 20 m – 200 m (OOD long_cable: 200 m – 300 m)
- **Characteristic Impedance Z0**: 48 Ω – 52 Ω (OOD z0_shift: 44–47 Ω, 53–56 Ω)
- **Velocity Factor vf**: 0.66 * [0.98, 1.02]
- **Loss Scale**: 0.7 – 1.3 (OOD high_loss: 1.5 – 2.0)
- **Rise Time**: 1.0 ns – 5.0 ns (OOD slow_rise: 5.0 ns – 8.0 ns)
- **Fault Position d/l**: 0.10 – 0.95 (OOD near_load: 0.3 m – 3.0 m from load)

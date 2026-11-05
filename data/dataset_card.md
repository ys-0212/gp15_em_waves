# Dataset Card: TDR Transmission Line Fault & Load Identification

## Summary
- **Generation Timestamp**: 2026-11-05 22:27:33
- **Total Generation Time**: 19.29 seconds
- **Mode**: Quick Mode

## Split Sizes and Generation Speed
| Split | Samples | Generation Time (s) | Time per 1k samples (s) |
|---|---|---|---|
| `train_quick` | 3,000 | 13.05 | 4.35 |
| `val_quick` | 500 | 1.68 | 3.37 |
| `test_quick` | 500 | 2.11 | 4.21 |
| `ood_quick` | 500 | 2.23 | 4.46 |

## Fault Class Distribution Across Splits
| Fault Class | Train | Val | Test | OOD |
|---|---|---|---|---|
| `none` | 442 | 72 | 73 | 50 |
| `short` | 423 | 82 | 66 | 65 |
| `open` | 441 | 55 | 76 | 76 |
| `shunt_R` | 399 | 79 | 74 | 79 |
| `shunt_C` | 436 | 68 | 68 | 82 |
| `series_R` | 413 | 67 | 82 | 65 |
| `z0_section` | 446 | 77 | 61 | 83 |

## Load Class Distribution Across Splits
| Load Class | Train | Val | Test | OOD |
|---|---|---|---|---|
| `open` | 497 | 83 | 89 | 98 |
| `short` | 493 | 79 | 91 | 75 |
| `matched` | 499 | 89 | 71 | 80 |
| `R` | 496 | 71 | 81 | 82 |
| `RL` | 517 | 86 | 87 | 84 |
| `RC` | 498 | 92 | 81 | 81 |

## Parameter Ranges
- **Cable Length**: 20 m – 200 m (OOD long_cable: 200 m – 300 m)
- **Characteristic Impedance Z0**: 48 Ω – 52 Ω (OOD z0_shift: 44–47 Ω, 53–56 Ω)
- **Velocity Factor vf**: 0.66 * [0.98, 1.02]
- **Loss Scale**: 0.7 – 1.3 (OOD high_loss: 1.5 – 2.0)
- **Rise Time**: 1.0 ns – 5.0 ns (OOD slow_rise: 5.0 ns – 8.0 ns)
- **Fault Position d/l**: 0.10 – 0.95 (OOD near_load: 0.3 m – 3.0 m from load)

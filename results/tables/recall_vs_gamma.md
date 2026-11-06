| method              | fault_class   | gamma_bin                 |   n |   detected_pct |   correct_pct |
|:--------------------|:--------------|:--------------------------|----:|---------------:|--------------:|
| Conventional TDR    | shunt_R       | [0.00, 0.05)              |  27 |        11.1111 |       3.7037  |
| Conventional TDR    | shunt_R       | [0.05, 0.10)              |   9 |        22.2222 |      22.2222  |
| Conventional TDR    | shunt_R       | [0.10, 0.20)              |   9 |        77.7778 |      77.7778  |
| Conventional TDR    | shunt_R       | [0.20, 0.40)              |   6 |       100      |     100       |
| Conventional TDR    | shunt_R       | [0.40, 1.00)              |  23 |       100      |      65.2174  |
| Conventional TDR    | series_R      | [0.00, 0.05)              |  23 |        13.0435 |       8.69565 |
| Conventional TDR    | series_R      | [0.05, 0.10)              |  13 |        15.3846 |      15.3846  |
| Conventional TDR    | series_R      | [0.10, 0.20)              |  14 |        85.7143 |      85.7143  |
| Conventional TDR    | series_R      | [0.20, 0.40)              |  17 |       100      |     100       |
| Conventional TDR    | series_R      | [0.40, 1.00)              |  15 |       100      |      93.3333  |
| Conventional TDR    | shunt_C       | all (frequency dependent) |  68 |        70.5882 |      35.2941  |
| Conventional TDR    | z0_section    | [0.00, 0.05)              |   3 |        33.3333 |       0       |
| Conventional TDR    | z0_section    | [0.05, 0.10)              |  26 |        26.9231 |      19.2308  |
| Conventional TDR    | z0_section    | [0.10, 0.20)              |  32 |        43.75   |      21.875   |
| LightGBM            | shunt_R       | [0.00, 0.05)              |  27 |        55.5556 |      22.2222  |
| LightGBM            | shunt_R       | [0.05, 0.10)              |   9 |        88.8889 |      77.7778  |
| LightGBM            | shunt_R       | [0.10, 0.20)              |   9 |       100      |     100       |
| LightGBM            | shunt_R       | [0.20, 0.40)              |   6 |       100      |     100       |
| LightGBM            | shunt_R       | [0.40, 1.00)              |  23 |       100      |      86.9565  |
| LightGBM            | series_R      | [0.00, 0.05)              |  23 |        52.1739 |       8.69565 |
| LightGBM            | series_R      | [0.05, 0.10)              |  13 |       100      |      84.6154  |
| LightGBM            | series_R      | [0.10, 0.20)              |  14 |       100      |      92.8571  |
| LightGBM            | series_R      | [0.20, 0.40)              |  17 |       100      |     100       |
| LightGBM            | series_R      | [0.40, 1.00)              |  15 |       100      |     100       |
| LightGBM            | shunt_C       | all (frequency dependent) |  68 |        86.7647 |      51.4706  |
| LightGBM            | z0_section    | [0.00, 0.05)              |   3 |        33.3333 |       0       |
| LightGBM            | z0_section    | [0.05, 0.10)              |  26 |        30.7692 |      26.9231  |
| LightGBM            | z0_section    | [0.10, 0.20)              |  32 |        71.875  |      50       |
| Multi-Task ResNet1D | shunt_R       | [0.00, 0.05)              |  27 |        74.0741 |      11.1111  |
| Multi-Task ResNet1D | shunt_R       | [0.05, 0.10)              |   9 |        55.5556 |      44.4444  |
| Multi-Task ResNet1D | shunt_R       | [0.10, 0.20)              |   9 |       100      |      88.8889  |
| Multi-Task ResNet1D | shunt_R       | [0.20, 0.40)              |   6 |       100      |     100       |
| Multi-Task ResNet1D | shunt_R       | [0.40, 1.00)              |  23 |       100      |      52.1739  |
| Multi-Task ResNet1D | series_R      | [0.00, 0.05)              |  23 |        56.5217 |      26.087   |
| Multi-Task ResNet1D | series_R      | [0.05, 0.10)              |  13 |       100      |      30.7692  |
| Multi-Task ResNet1D | series_R      | [0.10, 0.20)              |  14 |       100      |      71.4286  |
| Multi-Task ResNet1D | series_R      | [0.20, 0.40)              |  17 |        94.1176 |      82.3529  |
| Multi-Task ResNet1D | series_R      | [0.40, 1.00)              |  15 |       100      |     100       |
| Multi-Task ResNet1D | shunt_C       | all (frequency dependent) |  68 |        58.8235 |      22.0588  |
| Multi-Task ResNet1D | z0_section    | [0.00, 0.05)              |   3 |       100      |      66.6667  |
| Multi-Task ResNet1D | z0_section    | [0.05, 0.10)              |  26 |        65.3846 |      34.6154  |
| Multi-Task ResNet1D | z0_section    | [0.10, 0.20)              |  32 |        53.125  |      12.5     |

| Configuration                | Fault macro-F1   | Loc MAE (m)   | Load acc (%)   | Epochs run (best)   |
|:-----------------------------|:-----------------|:--------------|:---------------|:--------------------|
| Full ResNet1D (v, dv, coord) | 0.8578           | 5.71          | 97.22          | 40 (39)             |
| No coord channel (v, dv)     | 0.8369           | 7.65          | 95.67          | 15 (7)              |
| Voltage only (v)             | 0.8544           | 5.76          | 97.05          | 40 (39)             |
| Single-task: fault           | 0.8536           | --            | --             | 40 (40)             |
| Single-task: location        | --               | 4.48          | --             | 40 (40)             |
| Single-task: load            | --               | --            | 97.02          | 40 (38)             |

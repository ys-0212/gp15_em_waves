# Machine-Learning-Based Transmission Line Fault and Load Identification via Time-Domain Reflectometry (TDR)

**Course:** EE 305 Electromagnetic Waves, Indian Institute of Technology Indore (IIT Indore)  
**Theme 4:** Machine-Learning-Based Transmission Line Fault and Load Identification  
**Group 15:** Yogendra Singh (240002083), Dhananjay Dhumal (240051008), Gadgil Rucha Vinay (240002025), Karan Hitesh Bagthariya (240002028), Yash Brijesh Modi (240002082), Rohan Jain (240002060)  
**Deliverables:** Complete tested codebase and the project report ([`report/report.pdf`](report/report.pdf), LaTeX source in `report/`).

---

## 1. Problem Statement & Physics Background

Time-Domain Reflectometry (TDR) is the benchmark non-destructive diagnostic technique for transmission lines, coaxial cables, and distributed wiring networks. A high-speed voltage step is injected into a lossy coaxial line (RG-58-like) through source resistance $R_g$, and the input terminal voltage $V(0, t)$ is recorded over time.

Every impedance discontinuity along the line reflects a fraction of the propagating electromagnetic wave according to the local voltage reflection coefficient:
$$\Gamma = \frac{Z - Z_0}{Z + Z_0}$$
The reflected wave returns to the sending terminal at round-trip time:
$$t = \frac{2d}{u} = \frac{2d}{v_f c}$$
where $u = v_f c \approx 1.98 \times 10^8$ m/s ($v_f \approx 0.66$ in polyethylene). In lossy lines, high frequencies suffer from conductor skin effect ($R(f) = R_{\text{dc}} + k_{\text{skin}}\sqrt{f}$) and dielectric loss ($G(f) = 2\pi f C \tan\delta$), causing dispersion, attenuation, and smearing of sharp wave edges.

From a single raw TDR trace, this system simultaneously solves three coupled diagnostic tasks:
| Task | Nature | Output Description |
|---|---|---|
| **T1: Fault Classification** | 7-class | `none`, `short`, `open`, `shunt_R`, `shunt_C`, `series_R`, `z0_section` |
| **T2: Fault Localization** | Regression | Fault position as fraction of cable length $d/l \in [0, 1]$, reported in metres |
| **T3: Load Identification** | 6-class + Regression | Load class (`open`, `short`, `matched`, `R`, `RL`, `RC`), termination resistance $R_L$ ($\Omega$), and reactive parameters: $L$ ($\mu\text{H}$) for $RL$ or $C$ (nF) for $RC$ |

### Fundamental Physics Masks
- **$T_2$ Masking:** For pristine cables (`none`), no fault discontinuity exists; $T_2$ localization is physically undefined and masked during both training and evaluation.
- **$T_3$ Masking:** For hard faults (`short` or `open`), the discontinuity reflects 100% of the wave energy ($\Gamma = \pm 1$); downstream cable segments and the terminal load are completely unobservable and masked during both training and evaluation.

---

## 2. End-to-End Pipeline

![TDR Pipeline](figures/fig_pipeline.png)

1. **Exact Physics Simulator (`tdr/simulator.py`):** Solves the lossy telegrapher's equations in the frequency domain via analytical two-port cascade matrix transformations ($N_{\text{sim}} = 8192$ points), synthesized via inverse FFT without grid dispersion.
2. **Unified Preprocessing (`tdr/data.py`):** Normalizes voltage to the incident plateau ($\tilde{v} = v/v_{\text{inc}}$), computes the smoothed numerical first derivative ($d\tilde{v}/dt$) using a 1D Conv Savitzky-Golay FIR filter, and appends a dimensionless coordinate channel ($t/T_{\text{nom}}$), outputting a $[3, 1024]$ tensor.
3. **Multi-Model Diagnosis:**
   - **Conventional TDR (no ML):** robust noise estimate, edge detection with a median-difference step test, hard faults from the level after the edge relative to before it, soft faults from edge shape, load by DC de-embedding $Z_{in} = Z_0 L_{end}/(2-L_{end})$; all thresholds tuned on the validation split.
   - **LightGBM:** Four gradient boosted tree ensembles trained on 94 engineered physics features.
   - **Multi-Task 1D-ResNet:** convolutional stem, 4 residual stages (32, 64, 128, 128 channels), flattened $128\times64$ feature map (keeps time position for location) projected to a 256-d shared embedding, and 4 task heads trained with a physics-masked multi-task loss (AdamW, warm-up + cosine, mixed precision on GPU, fresh noise every batch).

---

## 3. Repository Map

```
EM_WAVES_PROJECT/
├── antigravity.md              # Project memory, status log, decisions log, and instructions
├── README.md                   # Complete repository documentation and guide
├── requirements.txt            # Python dependencies (CPU-only PyTorch, LightGBM, Streamlit, etc.)
├── config.yaml                 # Single source of truth for simulation and model parameters
├── .gitignore                  # Git exclusions (checkpoints, full datasets, venv)
├── tdr/                        # Core Python library
│   ├── __init__.py             # Package exports
│   ├── simulator.py            # Exact frequency-domain lossy transmission line cascade simulator
│   ├── data.py                 # Dataset generator, preprocessing, PyTorch dataset, and real data parser
│   ├── features.py             # Conventional TDR rule-based solver and 94-feature extractor
│   ├── models.py               # LightGBM suite and Multi-Task 1D-ResNet architecture
│   └── metrics.py              # Physics-masked evaluation metrics for T1, T2, and T3
├── 1_validate_simulator.py     # Physics and bounce-diagram simulator validation
├── 2_generate_data.py          # Synthetic dataset generator CLI
├── 3_train.py                  # Model training script (LightGBM and ResNet1D)
├── 4_evaluate.py               # Complete evaluation suite covering E1–E8 experiments
├── 5_make_figures.py           # Publication-grade figure generator (all 18 figures)
├── run_all.py                  # Master pipeline orchestrator (--mode quick/full, --device cpu/cuda)
├── app.py                      # Interactive Streamlit application (simulation and CSV upload)
├── tests.py                    # 15 physics, simulator, pipeline and baseline-acceptance tests
├── kaggle_run_final.ipynb      # Kaggle GPU notebook (with the outputs of the full run)
├── data/
│   ├── README.md               # Synthetic and real dataset documentation
│   ├── dataset_card.md         # Full dataset specifications, splits, and parameter bounds
│   └── real/wise_casing/       # Real NREL/LBNL laboratory coaxial cable TDR measurements
├── results/
│   ├── LOG.md                  # Execution and evaluation log with git hashes and metrics
│   ├── tables/                 # Generated tables (CSV, Markdown, and booktabs LaTeX)
│   ├── experiments/            # Raw JSON experiment outputs (E1–E8)
│   └── runs/                   # Model training configs, histories, and predictions
├── figures/                    # 16 publication figures (300-dpi PNG and vector PDF)
├── app/weights/                # Exported deployment model weights for the Streamlit app
└── report/                     # Project report (report.pdf + LaTeX source)
    ├── main.tex                # Root LaTeX document (single-column report)
    ├── refs.bib                # BibTeX references
    ├── README.md               # Overleaf compilation instructions
    ├── sections/               # Sections 1 to 7 (Theory, Methods, Results, Discussion)
    ├── figures/                # Pre-synchronized vector PDF figures
    └── tables/                 # Pre-synchronized booktabs LaTeX tables
```

---

## 4. Quick Start: Local Run, Tests & Streamlit App

### Prerequisites
- Windows OS with PowerShell
- Python 3.11

### 1. Setup Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Run the Unit Test Suite (15 Tests)
```powershell
pytest -q tests.py
```
Validates causality, DC limits, characteristic impedance, attenuation, bounce diagrams, reactive tails, $Z_0$ bump widths, dataset integrity, preprocessing determinism, and multi-task loss masking.

### 3. Run the End-to-End Pipeline Locally (Quick Mode)
```powershell
python run_all.py --mode quick --device cpu
```
This generates the quick dataset (4,500 samples), trains Conventional, LightGBM, and ResNet models, runs ablation and learning curve runs, evaluates E1–E8 experiments, generates all 18 figures, and exports deployment weights to `app/weights/`.

### 4. Launch the Interactive Streamlit App
```powershell
streamlit run app.py
```
- **Tab 1 (Interactive Simulation):** Configure cable length, $Z_0$, loss, rise time, fault type/location/value, load type/values, and SNR. Plots the simulated trace with a dual distance/time axis, compares all three models, and displays ResNet class probabilities.
- **Tab 2 (Upload CSV):** Upload a step-type TDR trace (`time_s,voltage_v` or `Time(us),Amplitude(mV)`) for automatic resampling and diagnosis. The models are trained on step responses; pulse-type records (such as WISE-CASING) are out of their domain.

---

## 5. Full Pipeline on Kaggle GPU

To train on the full 85,000-sample dataset across multiple seeds:
1. Push your local repository to a private or public GitHub repo:
   ```powershell
   git remote add origin https://github.com/ys-0212/gp15_em_waves.git
   git push -u origin master
   ```
2. Open [Kaggle](https://www.kaggle.com/), create a new Notebook, and import or paste `kaggle_run_final.ipynb`.
3. Set **Accelerator = GPU (T4 or P100)** and **Internet = On**.
4. Update `REPO_URL` in Cell 2 with your GitHub repository URL.
5. Run all cells:
   - Cell 1: Explains required settings.
   - Cell 2: Clones the clean repository.
   - Cell 3: Installs lightweight dependencies and verifies CUDA.
   - Cell 4: Runs `pytest -q tests.py`.
   - Cell 5: Executes `python run_all.py --mode full --device cuda --skip-existing`.
   - Cell 6: Compresses outputs into `/kaggle/working/outputs.zip` and prints key metrics.
6. Download `outputs.zip` from Kaggle and unzip it into your local repository root to restore full model weights, tables, and figures in place:
   ```
   tar -xf outputs.zip
   ```
   (PowerShell alternative: `Expand-Archive -Path outputs.zip -DestinationPath . -Force`)

**Measured full run on a Kaggle T4 GPU (`results/timings.json`), 135 min in total:**

| Stage | Wall-clock |
|---|---|
| Dataset generation (85,000 traces) | 3 min |
| Conventional TDR (threshold tuning on validation) | 4 min |
| LightGBM (3 seeds + 4 learning-curve runs) | 63 min |
| ResNet1D (3 seeds + 4 learning-curve + 5 ablation runs, ~9.4 s per epoch) | 51 min |
| Evaluation E1–E8 | 14 min |
| Figures | < 1 min |

---

## 6. Experimental Results Summary

All numbers below are from the **full** Kaggle run (60,000 training traces; test split of 10,000 traces at 30 dB SNR). LightGBM and ResNet1D are mean ± std over 3 training seeds; the conventional baseline is deterministic.

| Model | Fault acc (%) | Fault macro-F1 | Loc MAE (m) | Loc median (m) | Loc P90 (m) | Loc < 1 m (%) | Load acc (%) | Load macro-F1 | MAPE R (%) | MAPE L (%) | MAPE C (%) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Conventional TDR | 61.83 | 0.622 | 9.47 | **0.91** | 36.06 | **51.7** | 80.19 | 0.812 | 38.2 | 134.3 | 140.5 |
| LightGBM (94 features) | 78.27 ± 0.07 | 0.790 ± 0.001 | 7.67 ± 0.04 | 1.42 | 25.66 | 42.5 | 93.40 ± 0.11 | 0.935 | **12.1** | 22.6 | 24.3 |
| Multi-task ResNet1D | **84.38 ± 0.83** | **0.848 ± 0.007** | **6.06 ± 0.63** | 1.64 | **17.30** | 38.2 | **96.34 ± 0.83** | **0.964** | 16.9 | **18.6** | **17.2** |

Key findings (details and discussion in the report):

- **ResNet1D has the best fault F1, location MAE and load accuracy at every SNR** (`results/tables/noise_sweep.md`). At 60 dB it reaches fault F1 0.986, location MAE 1.2 m and load accuracy 99.2%; at 15 dB (below the 20–50 dB training range) it keeps 84% load accuracy while LightGBM drops to 63%.
- **The conventional detector is the most precise when it fires** (best median error, 0.91 m) but misses about one fault in three, which produces the heavy error tail (P90 36 m). Its detection limit is |Γ| ≈ 0.1–0.2 at 30 dB; the ML models detect 94–100% of resistive faults with |Γ| = 0.05–0.10 (`results/tables/recall_vs_gamma_wide.md`).
- **Location error is dominated by weak discontinuities:** ResNet MAE is 14.5 m for |Γ| < 0.2 but 1.0–1.7 m for |Γ| ≥ 0.2.
- **Out of distribution** (`results/tables/ood_metrics.md`): shifts in Z0 and rise time cost almost nothing, a 1.5–2× higher loss costs about 0.04 in F1, and cables longer than any seen in training cost more; **faults within 3 m of the load defeat all three methods** (fault F1 ≤ 0.28), because the fault and load echoes merge.
- **Data efficiency** (validation): ResNet trained on 5% of the data (3,000 traces) already beats LightGBM trained on 100%.
- **Ablations** (validation): the voltage channel alone matches the 3-channel input; multi-task training matches single-task networks on fault and load, but a single-task location network is better (4.48 m vs 5.71 m MAE).
- **Real hardware (WISE-CASING):** velocity factor 0.638 from the undamaged 80 m record; the 40 m shield cutaway is located at 38.5 m from echo timing (−1.5 m).

Two of the twelve ResNet runs (seed 2 and the (v, dv) ablation) were stopped early by the patience-8 rule because the validation loss is noisy (`figures/fig_training_curves.png`); this accounts for most of the ResNet seed spread. Seed 0, which is used for the noise sweep, OOD and error-breakdown experiments, trained for all 40 epochs.

---

## 7. Data Description, Reproducibility & Credits

### Synthetic Dataset
Full dataset specifications and parameter distributions are documented in [`data/dataset_card.md`](data/dataset_card.md). All random number generators (NumPy and PyTorch) are strictly seeded from `config.yaml`. Running the pipeline with identical seeds produces byte-identical traces and reproducible metrics.

### Real Laboratory Dataset: WISE-CASING (physics check)
- **Citation:** Y. Wu and J. Wang (Lawrence Berkeley National Laboratory), *WISE-CASING: Time Domain Reflectometry Data from Lab Experiment on Coaxial Cable*, Geothermal Data Repository, submission 1164, 2018. DOI [10.15121/1559012](https://doi.org/10.15121/1559012). Licence and terms: see <https://gdr.openei.org/submissions/1164>.
- **Documented setup:** 80 m RG-85 coax, shorted termination, undamaged and with a 120° shield cutaway at 40 m; plus records documented as 20 / 10 / 5 m cables (verbatim quotes in `data/real/wise_casing/labels.yaml`).
- **How it is used (E8):** the records are *pulse-type* TDR, not step responses, so the step-trained models are not applied. Instead `tdr.data.analyse_real_wise_casing` measures echo round-trip times, estimates the velocity factor from the undamaged 80 m record, and locates the cutaway by echo-time ratio. The full run gives vf = 0.638 and a cutaway estimate of 38.5 m (documented: 40 m). The 20 / 10 / 5 m records' echo times are inconsistent with their documented lengths and are flagged. See `data/README.md`, `results/tables/real_data.md` and `figures/fig_real_data.png`.

---

## 8. Project Report

The compiled report is [`report/report.pdf`](report/report.pdf) (20 pages: front page, index, and nine sections). Its LaTeX source is in `report/`:
- `report/main.tex`: Root document (front page, index, project summary).
- `report/sections/`: Introduction, Theory, Simulator and Dataset, Methods, Experimental Setup, Results, Discussion, Limitations, Conclusion.
- `report/figures/`: Pre-synchronized vector PDF figures.
- `report/tables/`: Pre-synchronized booktabs LaTeX tables.
- `report/refs.bib`: BibTeX references (verified against publisher records); `report/IEEEtran.bst`: bibliography style.

To compile on **Overleaf**, simply upload the `report/` folder and click **Recompile**.

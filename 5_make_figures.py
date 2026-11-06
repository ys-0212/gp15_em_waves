"""Publication-quality figure generator for all 18 research figures.

Adheres strictly to style rules:
- Consistent colorblind-safe palettes
- 300-dpi PNG and vector PDF outputs in figures/
- Synchronizes PDF figures to report/figures/ and LaTeX tables to report/tables/
- Gracefully skips figures if inputs are missing
"""

import json
import warnings
from pathlib import Path
import shutil
import h5py
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from tdr.simulator import C_CONST
import numpy as np
import pandas as pd
import yaml

from tdr.simulator import (
    LineParams,
    simulate_trace,
    analytic_bounce_diagram,
    analytic_reactive_tail,
    compute_tnom,
)
from tdr.data import preprocess_numpy, FAULT_CLASSES, LOAD_CLASSES


# Set matplotlib style
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Color-blind safe palette
CB_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]


def save_fig(fig: plt.Figure, base_name: str, fig_dir: Path, report_fig_dir: Path):
    """Save both PNG (300 dpi) and PDF, and copy PDF to report/figures/."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_fig_dir.mkdir(parents=True, exist_ok=True)

    png_path = fig_dir / f"{base_name}.png"
    pdf_path = fig_dir / f"{base_name}.pdf"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # gridspec + colorbar axes are not tight_layout-compatible
        fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    shutil.copy2(pdf_path, report_fig_dir / f"{base_name}.pdf")
    plt.close(fig)
    print(f"  Saved {base_name} (PNG & PDF)")


def sync_tables(results_dir: Path, report_dir: Path):
    """Copy all LaTeX tables from results/tables/ to report/tables/."""
    src_dir = results_dir / "tables"
    dst_dir = report_dir / "tables"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src_dir.exists():
        for tex_file in src_dir.glob("*.tex"):
            shutil.copy2(tex_file, dst_dir / tex_file.name)
        print("Synchronized LaTeX tables to report/tables/")


# =========================================================================
# Figure 1: Pipeline Block Diagram
# =========================================================================
def make_fig_pipeline(fig_dir: Path, rep_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Boxes
    boxes = [
        ("Physics Simulator\n(ABCD Lossy Cascade)", (0.3, 1.8), (1.8, 1.4), "#56B4E9"),
        ("TDR Traces\nClean Data + HDF5", (2.6, 1.8), (1.5, 1.4), "#E69F00"),
        ("Preprocessing\n(Noise, Norm, SG-1D)", (4.6, 1.8), (1.7, 1.4), "#009E73"),
    ]
    for text, (x, y), (w, h), color in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=color, ec="black", lw=1.2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", weight="bold", fontsize=9)

    # Models column
    model_boxes = [
        ("Conventional TDR\n(Rule-based solver)", (6.8, 3.2), (1.6, 0.9), "#CC79A7"),
        ("LightGBM Suite\n(Engineered Features)", (6.8, 2.0), (1.6, 0.9), "#F0E442"),
        ("Multi-Task ResNet\n(1D CNN Backbone)", (6.8, 0.8), (1.6, 0.9), "#0072B2"),
    ]
    for text, (x, y), (w, h), color in model_boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=color, ec="black", lw=1.2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", weight="bold", fontsize=8.5)

    # Tasks Output
    rect_out = patches.FancyBboxPatch((8.8, 1.5), 1.0, 2.0, boxstyle="round,pad=0.1", fc="#D55E00", ec="black", lw=1.2, alpha=0.85)
    ax.add_patch(rect_out)
    ax.text(9.3, 2.5, "Tasks:\nT1 Fault\nT2 Loc\nT3 Load", ha="center", va="center", weight="bold", color="white", fontsize=9)

    # Arrows
    arrows = [
        ((2.1, 2.5), (2.6, 2.5)),
        ((4.1, 2.5), (4.6, 2.5)),
        ((6.3, 2.5), (6.8, 3.65)),
        ((6.3, 2.5), (6.8, 2.45)),
        ((6.3, 2.5), (6.8, 1.25)),
        ((8.4, 3.65), (8.8, 2.7)),
        ((8.4, 2.45), (8.8, 2.5)),
        ((8.4, 1.25), (8.8, 2.3)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

    save_fig(fig, "fig_pipeline", fig_dir, rep_dir)


# =========================================================================
# Figure 2: Simulator Validation
# =========================================================================
def make_fig_sim_validation(fig_dir: Path, rep_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Bounce diagram
    loads = [("open", 1e9, "Open", "#D55E00"), ("short", 1e-6, "Short", "#0072B2"), ("150", 150.0, r"150 $\Omega$", "#009E73")]
    for name, rl, label, color in loads:
        p = LineParams(length_m=100.0, z0=50.0, vf=0.66, rg=25.0, v0=1.0, load_class="R" if name not in ("open", "short") else name, load_r=rl, rise_time_s=1.0e-9)
        t_norm, v_sim = simulate_trace(p, lossless=True)
        ax1.plot(t_norm, v_sim, label=f"Simulated ({label})", color=color, lw=1.5)

        # Plot analytic levels
        ana = analytic_bounce_diagram(1.0, 25.0, 50.0, rl, max_k=3)
        for k in range(3):
            t_mid = 0.05 + k + 0.5
            ax1.plot(t_mid, ana[f"k{k}"], "k+", markersize=8, mew=2, label="Analytic" if k == 0 and name == "open" else "")

    ax1.set_xlim(0, 2.5)
    ax1.set_xlabel("Normalized Time $t / T_{nom}$")
    ax1.set_ylabel("Input Voltage $V(0,t)$ (V)")
    ax1.set_title(r"(a) Lossless Bounce Diagram ($R_g=25\ \Omega, Z_0=50\ \Omega$)")
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: Reactive tails
    length = 100.0
    tnom = compute_tnom(length, 0.66)
    for l_cls, color, lbl in [("RL", "#CC79A7", "Series RL"), ("RC", "#56B4E9", "Parallel RC")]:
        p = LineParams(length_m=length, z0=50.0, rg=50.0, load_class=l_cls, load_r=75.0, load_tau_over_t=0.10, rise_time_s=1.0e-9)
        t_norm, v_sim = simulate_trace(p, lossless=True)
        ax2.plot(t_norm, v_sim, label=f"Simulated {lbl}", color=color, lw=1.5)

        # Analytic exponential
        t_sec = t_norm * tnom
        ana_tail = analytic_reactive_tail(1.0, 50.0, 75.0, 0.10 * tnom, l_cls, t_sec, 1.05 * tnom)
        mask = t_norm >= 1.05
        ax2.plot(t_norm[mask], ana_tail[mask], "k--", lw=1.2, alpha=0.7, label=f"Analytic {lbl}" if l_cls == "RL" else "")

    ax2.set_xlim(0.8, 2.0)
    ax2.set_xlabel("Normalized Time $t / T_{nom}$")
    ax2.set_ylabel("Voltage $V(0,t)$ (V)")
    ax2.set_title(r"(b) Lossless Reactive Load Tails ($R_g=Z_0=50\ \Omega$)")
    ax2.legend(loc="upper right", fontsize=8)

    save_fig(fig, "fig_sim_validation", fig_dir, rep_dir)


# =========================================================================
# Figure 3: Fault Gallery
# =========================================================================
def make_fig_fault_gallery(fig_dir: Path, rep_dir: Path):
    fault_configs = [
        ("none", 0.0, 0.0, "No Fault"),
        ("short", 0.0, 0.0, "Short ($Z_f=0$)"),
        ("open", 0.0, 0.0, r"Open ($Z_f=\infty$)"),
        ("shunt_R", 25.0, 0.0, r"Shunt R ($R_f=25\ \Omega$)"),
        ("shunt_C", 150e-12, 0.0, "Shunt C ($C_f=150$ pF)"),
        ("series_R", 20.0, 0.0, r"Series R ($R_f=20\ \Omega$)"),
        ("z0_section", 65.0, 3.0, r"Impedance Discontinuity ($Z_0'=65\ \Omega$)"),
    ]
    fig, axes = plt.subplots(len(fault_configs), 1, figsize=(8, 11), sharex=True)

    for i, (fc, val, w, label) in enumerate(fault_configs):
        p = LineParams(length_m=100.0, z0=50.0, vf=0.66, fault_class=fc, fault_pos_m=40.0, fault_val=val, fault_width_m=w, load_class="matched", load_r=50.0, rise_time_s=1.5e-9)
        t_norm, v = simulate_trace(p)
        axes[i].plot(t_norm, v, color=CB_COLORS[i % len(CB_COLORS)], lw=1.5)
        axes[i].set_ylabel("$V(0,t)$ (V)", fontsize=8)
        axes[i].set_title(label, fontsize=9.5, loc="left", pad=2)
        if fc != "none":
            axes[i].axvline(0.45, color="red", linestyle=":", alpha=0.7)

    axes[-1].set_xlabel("Normalized Time $t / T_{nom}$")
    save_fig(fig, "fig_fault_gallery", fig_dir, rep_dir)


# =========================================================================
# Figure 4: Load Gallery
# =========================================================================
def make_fig_load_gallery(fig_dir: Path, rep_dir: Path):
    load_configs = [
        ("matched", 50.0, 0.0, r"Matched Load ($R_L=50\ \Omega$)"),
        ("open", 1e9, 0.0, r"Open Load ($R_L=\infty$)"),
        ("short", 1e-6, 0.0, "Short Load ($R_L=0$)"),
        ("R", 150.0, 0.0, r"Resistive Mismatch ($R_L=150\ \Omega$)"),
        ("RL", 75.0, 0.10, r"Series RL ($\tau=0.1 T_{nom}$)"),
        ("RC", 75.0, 0.10, r"Parallel RC ($\tau=0.1 T_{nom}$)"),
    ]
    fig, axes = plt.subplots(len(load_configs), 1, figsize=(8, 10), sharex=True)

    for i, (lc, rl, tau, label) in enumerate(load_configs):
        p = LineParams(length_m=100.0, z0=50.0, vf=0.66, fault_class="none", load_class=lc, load_r=rl, load_tau_over_t=tau, rise_time_s=1.5e-9)
        t_norm, v = simulate_trace(p)
        axes[i].plot(t_norm, v, color=CB_COLORS[i % len(CB_COLORS)], lw=1.5)
        axes[i].set_ylabel("$V(0,t)$ (V)", fontsize=8)
        axes[i].set_title(label, fontsize=9.5, loc="left", pad=2)
        axes[i].axvline(1.05, color="green", linestyle="--", alpha=0.6)

    axes[-1].set_xlabel("Normalized Time $t / T_{nom}$")
    save_fig(fig, "fig_load_gallery", fig_dir, rep_dir)


# =========================================================================
# Figure 5: Loss Effect
# =========================================================================
def make_fig_loss_effect(fig_dir: Path, rep_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    lengths = [20.0, 100.0, 200.0]
    colors = ["#0072B2", "#009E73", "#D55E00"]

    for l_val, c in zip(lengths, colors):
        p = LineParams(length_m=l_val, z0=50.0, vf=0.66, fault_class="open", fault_pos_m=l_val * 0.4, load_class="matched", load_r=50.0, rise_time_s=1.5e-9)
        t_norm, v = simulate_trace(p, lossless=False)
        ax.plot(t_norm, v, label=f"Cable Length = {l_val:.0f} m", color=c, lw=1.5)

    ax.set_xlim(0.35, 0.70)
    ax.set_xlabel("Normalized Time $t / T_{nom}$")
    ax.set_ylabel("Voltage $V(0,t)$ (V)")
    ax.set_title("Frequency-Dependent Skin Effect & Dielectric Smearing vs. Cable Length")
    ax.legend(loc="lower right")
    save_fig(fig, "fig_loss_effect", fig_dir, rep_dir)


# =========================================================================
# Figure 6: Noise Examples
# =========================================================================
def make_fig_noise_examples(fig_dir: Path, rep_dir: Path):
    p = LineParams(length_m=100.0, z0=50.0, fault_class="shunt_R", fault_pos_m=40.0, fault_val=30.0, load_class="RC", load_r=75.0, load_tau_over_t=0.08)
    t_norm, v_clean = simulate_trace(p)

    snr_levels = [50.0, 30.0, 15.0]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    for i, snr in enumerate(snr_levels):
        preproc = preprocess_numpy(v_clean, v_inc=0.5, snr_db=snr, seed=42)
        v_noisy = preproc[0, 0]
        axes[i].plot(t_norm, v_noisy, color=CB_COLORS[i], lw=1.2)
        axes[i].set_ylabel(r"$\tilde{v}(t)$", fontsize=9)
        axes[i].set_title(f"SNR = {snr:.0f} dB", fontsize=10, loc="left")

    axes[-1].set_xlabel("Normalized Time $t / T_{nom}$")
    save_fig(fig, "fig_noise_examples", fig_dir, rep_dir)


# =========================================================================
# Figure 7: Hard Case (Weak Shunt R near RC Load)
# =========================================================================
def make_fig_hard_case(fig_dir: Path, rep_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    # Weak shunt R (e.g. 500 ohm) near RC load (d/l = 0.90)
    p_fault = LineParams(length_m=100.0, fault_class="shunt_R", fault_pos_m=90.0, fault_val=350.0, load_class="RC", load_r=60.0, load_tau_over_t=0.06)
    p_nofault = LineParams(length_m=100.0, fault_class="none", load_class="RC", load_r=60.0, load_tau_over_t=0.06)

    t_norm, v_clean = simulate_trace(p_fault)
    _, v_nofault = simulate_trace(p_nofault)
    preproc = preprocess_numpy(v_clean, v_inc=0.5, snr_db=25.0, seed=123)
    v_noisy = preproc[0, 0]

    ax.plot(t_norm, v_nofault / 0.5, label="No Fault (Pure RC Load)", color="gray", linestyle="--", lw=1.5)
    ax.plot(t_norm, v_clean / 0.5, label=r"Weak Shunt-R Clean ($d/l=0.90, R_f=350\ \Omega$)", color="navy", lw=1.5)
    ax.plot(t_norm, v_noisy, label="Weak Shunt-R Noisy (25 dB SNR)", color="crimson", alpha=0.65, lw=1.2)

    ax.set_xlim(0.85, 1.45)
    ax.set_xlabel("Normalized Time $t / T_{nom}$")
    ax.set_ylabel(r"Normalized Voltage $\tilde{v}(t)$")
    ax.set_title("Challenging TDR Scenario: High-Impedance Shunt Fault in Proximity to RC Load")
    ax.legend(loc="upper right")
    save_fig(fig, "fig_hard_case", fig_dir, rep_dir)


# =========================================================================
# Figure 8: Confusion Matrices (Fault and Load)
# =========================================================================
def make_fig_confusion_matrices(fig_dir: Path, rep_dir: Path):
    """Row-normalised TEST-set confusion matrices (30 dB, seed 0) for the three methods."""
    tm_path = Path("results/experiments/test_metrics_s0.json")
    if not tm_path.exists():
        print("test_metrics_s0.json not found (run 4_evaluate.py). Skipping confusion matrices.")
        return
    with open(tm_path, "r", encoding="utf-8") as f:
        tm = json.load(f)
    methods = [(n, k) for n, k in [("Conventional", "conventional"), ("LightGBM", "lgbm"), ("ResNet1D", "resnet")] if k in tm]

    for task, classes, cmap, base, ylab in [
        ("t1_fault", FAULT_CLASSES, "Blues", "fig_confusion_fault", "True fault"),
        ("t3_load", LOAD_CLASSES, "Greens", "fig_confusion_load", "True load"),
    ]:
        fig = plt.figure(figsize=(4.2 * len(methods) + 0.8, 4.6))
        gs = fig.add_gridspec(1, len(methods) + 1, width_ratios=[1] * len(methods) + [0.05], wspace=0.08)
        im = None
        for i, (name, key) in enumerate(methods):
            ax = fig.add_subplot(gs[0, i])
            m = tm[key][task]
            cm = np.array(m["confusion_matrix"], dtype=float)
            cm_norm = cm / np.maximum(1.0, cm.sum(axis=1, keepdims=True))
            im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
            for r in range(cm_norm.shape[0]):
                for c in range(cm_norm.shape[1]):
                    if cm_norm[r, c] >= 0.05:
                        ax.text(c, r, f"{cm_norm[r, c]:.2f}", ha="center", va="center", fontsize=6.5,
                                color="white" if cm_norm[r, c] > 0.55 else "black")
            ax.set_title(f"{name} (acc {m['accuracy'] * 100:.1f}%)", fontsize=10)
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7.5)
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels(classes if i == 0 else [], fontsize=7.5)
            if i == 0:
                ax.set_ylabel(ylab)
            ax.set_xlabel("Predicted")
        cax = fig.add_subplot(gs[0, -1])
        fig.colorbar(im, cax=cax, label="Recall (row-normalised)")
        save_fig(fig, base, fig_dir, rep_dir)


# =========================================================================
# Figure 9: Location Error Breakdown
# =========================================================================
def make_fig_loc_error_breakdowns(fig_dir: Path, rep_dir: Path):
    brk_path = Path("results/experiments/error_breakdowns.json")
    if not brk_path.exists():
        return
    with open(brk_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 9a: Loc error vs position
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    pos_data = data.get("loc_error_vs_position", [])
    x_pos = [0.5 * (d["bin_start"] + d["bin_end"]) for d in pos_data]
    y_err = [d["mae_m"] for d in pos_data]
    ax1.plot(x_pos, y_err, "o-", color="#0072B2", lw=1.8, markersize=6)
    ax1.set_xlabel("Fault Fractional Position $d/l$")
    ax1.set_ylabel("Location MAE (m)")
    ax1.set_title("ResNet1D location error vs. fault position (test, 30 dB)")
    save_fig(fig1, "fig_loc_error_vs_position", fig_dir, rep_dir)

    # 9b: Loc error vs gamma
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    gamma_data = data.get("loc_error_vs_gamma", [])
    x_g = [0.5 * (d["gamma_start"] + d["gamma_end"]) for d in gamma_data]
    y_g = [d["mae_m"] for d in gamma_data]
    ax2.plot(x_g, y_g, "s-", color="#D55E00", lw=1.8, markersize=6)
    ax2.set_xlabel(r"Reflection Coefficient Magnitude $|\Gamma_{nominal}|$")
    ax2.set_ylabel("Location MAE (m)")
    ax2.set_title(r"ResNet1D location error vs. $|\Gamma|$ of the fault (test, 30 dB)")
    save_fig(fig2, "fig_loc_error_vs_gamma", fig_dir, rep_dir)


# =========================================================================
# Figure 10: Noise Sweep (SNR Sensitivity)
# =========================================================================
def make_fig_noise_sweep(fig_dir: Path, rep_dir: Path):
    ns_path = Path("results/experiments/noise_sweep.json")
    if not ns_path.exists():
        return
    with open(ns_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    snrs = [float(s) for s in data["snrs"]]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.8))

    methods = [("Conventional", "conventional", "gray", "^"), ("LightGBM", "lgbm", "#D55E00", "s"), ("ResNet1D", "resnet", "#0072B2", "o")]
    for name, key, color, marker in methods:
        if key not in data:
            continue
        f1s = [data[key][str(int(s) if s.is_integer() else s)]["fault_f1"] for s in snrs]
        maes = [data[key][str(int(s) if s.is_integer() else s)]["loc_mae"] for s in snrs]
        accs = [data[key][str(int(s) if s.is_integer() else s)]["load_acc"] * 100 for s in snrs]

        ax1.plot(snrs, f1s, label=name, color=color, marker=marker, lw=1.5)
        ax2.plot(snrs, maes, label=name, color=color, marker=marker, lw=1.5)
        ax3.plot(snrs, accs, label=name, color=color, marker=marker, lw=1.5)

    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("Fault Macro-F1")
    ax1.set_title("(a) Fault Classification F1")
    ax1.legend(loc="lower right")

    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("Location MAE (m)")
    ax2.set_title("(b) Localization Error (m)")

    ax3.set_xlabel("SNR (dB)")
    ax3.set_ylabel("Load Accuracy (%)")
    ax3.set_title("(c) Load Classification Accuracy")

    save_fig(fig, "fig_noise_sweep", fig_dir, rep_dir)


# =========================================================================
# Figure 11: OOD Performance
# =========================================================================
def make_fig_ood(fig_dir: Path, rep_dir: Path):
    ood_path = Path("results/experiments/ood_results.json")
    if not ood_path.exists():
        return
    with open(ood_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    subsets = list(data.keys())
    x = np.arange(len(subsets))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    methods = [("Conventional", "conventional", "gray"), ("LightGBM", "lgbm", "#D55E00"), ("ResNet1D", "resnet", "#0072B2")]

    for i, (name, key, col) in enumerate(methods):
        f1s = [data[sub][key].get("fault_f1", 0.0) for sub in subsets]
        maes = [data[sub][key].get("loc_mae", 0.0) for sub in subsets]
        ax1.bar(x + (i - 1) * width, f1s, width, label=name, color=col, alpha=0.85)
        ax2.bar(x + (i - 1) * width, maes, width, label=name, color=col, alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(subsets, rotation=25, ha="right")
    ax1.set_ylabel("Fault Macro-F1")
    ax1.set_title("(a) OOD Fault Macro-F1")
    ax1.legend(loc="upper center", bbox_to_anchor=(1.1, -0.28), ncol=3, frameon=False)

    ax2.set_xticks(x)
    ax2.set_xticklabels(subsets, rotation=25, ha="right")
    ax2.set_ylabel("Location MAE (m)")
    ax2.set_title("(b) OOD Localization MAE (m)")

    save_fig(fig, "fig_ood", fig_dir, rep_dir)


# =========================================================================
# Figure 12: Learning Curves
# =========================================================================
def make_fig_learning_curves(fig_dir: Path, rep_dir: Path):
    lc_path = Path("results/experiments/learning_curves.json")
    if not lc_path.exists():
        return
    with open(lc_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fracs = [float(f) for f in data["fractions"]]
    fig, ax = plt.subplots(figsize=(6, 4))

    if "lgbm" in data and data["lgbm"]:
        f1_lg = [data["lgbm"][str(f)]["fault_f1"] for f in fracs if str(f) in data["lgbm"]]
        ax.plot([f for f in fracs if str(f) in data["lgbm"]], f1_lg, "s-", label="LightGBM", color="#D55E00", lw=1.8)

    if "resnet" in data and data["resnet"]:
        f1_rs = [data["resnet"][str(f)]["fault_f1"] for f in fracs if str(f) in data["resnet"]]
        ax.plot([f for f in fracs if str(f) in data["resnet"]], f1_rs, "o-", label="ResNet1D", color="#0072B2", lw=1.8)

    ax.set_xlabel("Training Set Fraction")
    ax.set_ylabel("Validation Fault Macro-F1")
    ax.set_title("Data Efficiency / Sample Complexity Learning Curve")
    ax.legend(loc="lower right")
    save_fig(fig, "fig_learning_curve", fig_dir, rep_dir)


# =========================================================================
# Figure 13: Ablations
# =========================================================================
def make_fig_ablations(fig_dir: Path, rep_dir: Path):
    """Validation-set ablations: each panel shows only the runs that train that task."""
    abl_path = Path("results/experiments/ablations.json")
    if not abl_path.exists():
        return
    with open(abl_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nice = {"resnet_s0": "Full\n(v, dv, coord)", "resnet_ch_vdv_s0": "No coord\n(v, dv)",
            "resnet_ch_v_s0": "Voltage only\n(v)", "resnet_single_fault_s0": "Single-task",
            "resnet_single_loc_s0": "Single-task", "resnet_single_load_s0": "Single-task"}
    shared = ["resnet_s0", "resnet_ch_vdv_s0", "resnet_ch_v_s0"]
    panels = [
        ("(a) Fault macro-F1", shared + ["resnet_single_fault_s0"], lambda v: v["t1_fault"]["macro_f1"], "{:.3f}"),
        ("(b) Location MAE (m)", shared + ["resnet_single_loc_s0"], lambda v: v["t2_loc"]["mae_m"], "{:.2f}"),
        ("(c) Load accuracy (%)", shared + ["resnet_single_load_s0"],
         lambda v: 100.0 * v["t3_load"]["accuracy"] if v["t3_load"]["accuracy"] <= 1.0 else v["t3_load"]["accuracy"], "{:.1f}"),
    ]
    colors = ["#0072B2", "#56B4E9", "#009E73", "#E69F00"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (title, keys, getter, fmt) in zip(axes, panels):
        keys = [k for k in keys if k in data]
        vals = [getter(data[k]) for k in keys]
        bars = ax.bar(np.arange(len(keys)), vals, color=colors[:len(keys)], alpha=0.9, width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels([nice.get(k, k) for k in keys], fontsize=8)
        ax.set_title(title)
        lo, hi = min(vals), max(vals)
        pad = 0.25 * (hi - lo + 1e-9)
        ax.set_ylim(max(0.0, lo - 3 * pad), hi + 2 * pad)
    fig.suptitle("ResNet1D ablations (validation split, 30 dB, seed 0)", fontsize=11)
    save_fig(fig, "fig_ablation", fig_dir, rep_dir)


# =========================================================================
# Figure 14: SHAP Top Features
# =========================================================================
def make_fig_shap(fig_dir: Path, rep_dir: Path):
    shap_path = Path("results/experiments/shap_summary.json")
    if not shap_path.exists():
        return
    with open(shap_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    importances = np.asarray(data.get("top_feature_importances", []), dtype=float)
    names = data.get("top_feature_names")
    if importances.ndim != 1 or len(importances) == 0 or not names:
        print("  shap_summary.json is in an old format (re-run 4_evaluate.py); skipping fig_shap")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(len(importances))[::-1]
    ax.barh(y, importances, color="#009E73", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel(r"Mean $|$SHAP$|$ (averaged over the 7 fault classes)")
    ax.set_title(f"LightGBM fault model: top {len(names)} features "
                 f"(test, 30 dB, n={data.get('n_samples', '?')})", fontsize=11)
    save_fig(fig, "fig_shap", fig_dir, rep_dir)


# =========================================================================
# Figure 15: Training Curves
# =========================================================================
def make_fig_training_curves(fig_dir: Path, rep_dir: Path):
    runs = sorted(Path("results/runs").glob("resnet_s[0-9]*/history.csv"))
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, hp in enumerate(runs):
        df_h = pd.read_csv(hp)
        seed = hp.parent.name.split("_s")[-1]
        col = CB_COLORS[i % len(CB_COLORS)]
        ax.plot(df_h["epoch"], df_h["train_loss"], color=col, lw=1.6, label=f"seed {seed} train")
        ax.plot(df_h["epoch"], df_h["val_loss"], color=col, lw=1.4, linestyle="--", label=f"seed {seed} val")
        best = df_h["val_loss"].idxmin()
        ax.plot(df_h["epoch"][best], df_h["val_loss"][best], marker="*", color=col, markersize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Multi-task loss")
    ax.set_title("ResNet1D training and validation loss (star = saved checkpoint)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    save_fig(fig, "fig_training_curves", fig_dir, rep_dir)


# =========================================================================
# Figure 16: Real Data Trace Predictions
# =========================================================================
def make_fig_real_data(fig_dir: Path, rep_dir: Path):
    """Real WISE-CASING pulse-TDR records with measured incident pulse and strongest echo."""
    real_dir = Path("data/real/wise_casing")
    if not (real_dir / "labels.yaml").exists():
        print("Real data labels not found. Skipping fig_real_data gracefully.")
        return
    try:
        from tdr.data import analyse_real_wise_casing
        real = analyse_real_wise_casing(real_dir)
        df = real["table"]
        vf_est = real.get("vf_est")
        n = len(df)
        fig, axes = plt.subplots(n, 1, figsize=(9, 2.0 * n), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (_, r) in zip(axes, df.iterrows()):
            t, v = real["raw"][r["file"]]
            vn = v / np.max(np.abs(v[t < 0.12]))
            ax.plot(t, vn, color="#0072B2", lw=1.1)
            ax.axvline(r["t0_us"], color="0.4", ls="--", lw=1.0, label="incident pulse peak")
            ax.axvline(r["t_echo_us"], color="#D55E00", ls="-", lw=1.4, label="strongest echo")
            if vf_est:
                t_exp = r["t0_us"] + 2.0 * r["documented_length_m"] / (vf_est * C_CONST) * 1e6
                ax.axvline(t_exp, color="#009E73", ls=":", lw=1.6, label="expected termination echo")
                if pd.notna(r["documented_damage_m"]):
                    t_d = r["t0_us"] + 2.0 * r["documented_damage_m"] / (vf_est * C_CONST) * 1e6
                    ax.axvline(t_d, color="#CC79A7", ls="-.", lw=1.6, label="expected damage echo (40 m)")
            dmg = "" if pd.isna(r["documented_damage_m"]) else f", damage at {r['documented_damage_m']:.0f} m"
            flag = r.get("consistent_with_vf_est")
            flag_txt = "" if flag is None or (isinstance(flag, float) and np.isnan(flag)) else ("" if flag else "  [echo inconsistent with stated length]")
            ax.set_title(f"{r['file']}: documented {r['documented_length_m']:.0f} m, short termination{dmg}{flag_txt}", fontsize=9)
            ax.set_ylabel("v / |v_inc|", fontsize=8)
            ax.set_xlim(0, 2.0)
            ax.grid(True, ls=":", alpha=0.6)
        axes[0].legend(fontsize=7, loc="upper right", ncol=2)
        axes[-1].set_xlabel(r"Time ($\mu$s)")
        if vf_est:
            fig.suptitle(f"WISE-CASING pulse-TDR records (velocity factor estimated from 80 m reference: {vf_est:.3f})", fontsize=10)
        save_fig(fig, "fig_real_data", fig_dir, rep_dir)
        print("Real-data figure generated.")
    except Exception as e:
        print(f"fig_real_data warning: {e}")


def make_all_figures():
    fig_dir = Path("figures")
    rep_fig_dir = Path("report/figures")
    results_dir = Path("results")
    report_dir = Path("report")

    print("\nGenerating all 18 research figures...")
    make_fig_pipeline(fig_dir, rep_fig_dir)
    make_fig_sim_validation(fig_dir, rep_fig_dir)
    make_fig_fault_gallery(fig_dir, rep_fig_dir)
    make_fig_load_gallery(fig_dir, rep_fig_dir)
    make_fig_loss_effect(fig_dir, rep_fig_dir)
    make_fig_noise_examples(fig_dir, rep_fig_dir)
    make_fig_hard_case(fig_dir, rep_fig_dir)
    make_fig_confusion_matrices(fig_dir, rep_fig_dir)
    make_fig_loc_error_breakdowns(fig_dir, rep_fig_dir)
    make_fig_noise_sweep(fig_dir, rep_fig_dir)
    make_fig_ood(fig_dir, rep_fig_dir)
    make_fig_learning_curves(fig_dir, rep_fig_dir)
    make_fig_ablations(fig_dir, rep_fig_dir)
    make_fig_shap(fig_dir, rep_fig_dir)
    make_fig_training_curves(fig_dir, rep_fig_dir)
    make_fig_real_data(fig_dir, rep_fig_dir)

    # Sync LaTeX tables
    sync_tables(results_dir, report_dir)
    print("All available figures generated successfully!")


if __name__ == "__main__":
    make_all_figures()


"""Evaluation suite covering E1–E8 experiments, LaTeX/CSV/Markdown table generation, and run logging."""

import argparse
import json
from pathlib import Path
import subprocess
import time
import h5py
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F
import yaml

from tdr.data import (
    FAULT_CLASSES,
    LOAD_CLASSES,
    FAULT_TO_ID,
    LOAD_TO_ID,
    TDRDataset,
    preprocess_numpy,
    preprocess,
    load_real_wise_casing,
    decode_load_r,
    decode_load_tau,
)
from tdr.features import ConventionalTDR, extract_features
from tdr.models import LGBMSuite, ResNet1D
from tdr.metrics import compute_all_metrics


def get_git_hash() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def export_table(df: pd.DataFrame, base_path: Path):
    """Save dataframe as CSV, LaTeX (booktabs), and Markdown."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    df.to_markdown(base_path.with_suffix(".md"), index=False)
    latex_str = df.to_latex(index=False, escape=True)
    with open(base_path.with_suffix(".tex"), "w") as f:
        f.write(latex_str)


def log_run(log_path: Path, run_id: str, git_hash: str, mode: str, metrics_summary: str):
    """Append one line to results/LOG.md."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Execution and Evaluation Log\n\n| Timestamp | Run ID | Git Hash | Mode | Key Metrics |\n|---|---|---|---|---|\n"
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write(header)

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"| {ts} | `{run_id}` | `{git_hash[:7]}` | {mode} | {metrics_summary} |\n")


def load_resnet_model(checkpoint_path: Path, in_channels: int = 3, device: torch.device = torch.device("cpu")) -> ResNet1D:
    """Load trained ResNet1D model."""
    model = ResNet1D(in_channels=in_channels, embed_dim=256)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_resnet_on_traces(
    model: ResNet1D,
    clean_traces: np.ndarray,
    meta_df: pd.DataFrame,
    norm_stats: dict,
    snr_db: float,
    seed: int,
    in_channels: int = 3,
    device: torch.device = torch.device("cpu"),
) -> pd.DataFrame:
    """Evaluate ResNet on given clean traces with specific noise."""
    b = len(clean_traces)
    mean_t = torch.tensor(norm_stats["mean"], device=device, dtype=torch.float32).view(1, 3, 1)
    std_t = torch.tensor(norm_stats["std"], device=device, dtype=torch.float32).view(1, 3, 1)

    # Preprocess
    preproc = preprocess_numpy(clean_traces, meta_df["v_inc"].values, snr_db=snr_db, seed=seed, add_noise=True)
    x_t = torch.from_numpy(preproc).to(device)
    x_norm = (x_t - mean_t) / std_t
    if in_channels == 1:
        x_norm = x_norm[:, [0], :]
    elif in_channels == 2:
        x_norm = x_norm[:, [0, 1], :]

    with torch.no_grad():
        preds = model(x_norm)

    fault_probs = F.softmax(preds["fault_logits"], dim=-1).cpu().numpy()
    fault_pred_id = np.argmax(fault_probs, axis=1)
    loc_pred = preds["loc_pred"].cpu().numpy()
    load_cls_probs = F.softmax(preds["load_cls_logits"], dim=-1).cpu().numpy()
    load_cls_pred_id = np.argmax(load_cls_probs, axis=1)
    load_reg_pred = preds["load_reg_pred"].cpu().numpy()

    r_pred = decode_load_r(np.clip(load_reg_pred[:, 0], np.log10(5.0), np.log10(500.0)))
    tau_pred = decode_load_tau(np.clip(load_reg_pred[:, 1], -3.0, 0.5))

    df_out = meta_df.copy()
    df_out["pred_fault_class"] = [FAULT_CLASSES[i] for i in fault_pred_id]
    df_out["pred_fault_pos_frac"] = loc_pred
    df_out["pred_load_class"] = [LOAD_CLASSES[i] for i in load_cls_pred_id]
    df_out["pred_load_R"] = r_pred
    df_out["pred_load_tau_over_T"] = tau_pred
    df_out["error_m"] = np.abs(df_out["pred_fault_pos_frac"] - df_out["fault_pos_frac"].fillna(0.0)) * df_out["length_m"]
    return df_out


def evaluate_lgbm_on_traces(
    suite: LGBMSuite,
    clean_traces: np.ndarray,
    meta_df: pd.DataFrame,
    snr_db: float,
    seed: int,
) -> pd.DataFrame:
    """Evaluate LGBMSuite on given clean traces with specific noise."""
    preproc = preprocess_numpy(clean_traces, meta_df["v_inc"].values, snr_db=snr_db, seed=seed, add_noise=True)
    feats = extract_features(preproc)
    preds = suite.predict(feats)

    df_out = meta_df.copy()
    df_out["pred_fault_class"] = preds["fault_class"]
    df_out["pred_fault_pos_frac"] = preds["fault_pos_frac"]
    df_out["pred_load_class"] = preds["load_class"]
    df_out["pred_load_R"] = preds["load_R"]
    df_out["pred_load_tau_over_T"] = preds["load_tau_over_T"]
    df_out["error_m"] = np.abs(df_out["pred_fault_pos_frac"] - df_out["fault_pos_frac"].fillna(0.0)) * df_out["length_m"]
    return df_out


def evaluate_conventional_on_traces(
    solver: ConventionalTDR,
    clean_traces: np.ndarray,
    meta_df: pd.DataFrame,
    snr_db: float,
    seed: int,
) -> pd.DataFrame:
    """Evaluate ConventionalTDR on given clean traces with specific noise."""
    preproc = preprocess_numpy(clean_traces, meta_df["v_inc"].values, snr_db=snr_db, seed=seed, add_noise=True)
    preds = solver.predict_batch(preproc)

    df_out = meta_df.copy()
    df_out["pred_fault_class"] = [p["fault_class"] for p in preds]
    df_out["pred_fault_pos_frac"] = [p["fault_pos_frac"] for p in preds]
    df_out["detected"] = [p["detected"] for p in preds]
    df_out["pred_load_class"] = [p["load_class"] for p in preds]
    df_out["pred_load_R"] = [p["load_R"] for p in preds]
    df_out["pred_load_tau_over_T"] = [p["load_tau_over_T"] for p in preds]
    df_out["error_m"] = np.abs(df_out["pred_fault_pos_frac"] - df_out["fault_pos_frac"].fillna(0.0)) * df_out["length_m"]
    return df_out


def run_all_evaluations(
    cfg_path: Path = Path("config.yaml"),
    quick: bool = False,
    device_name: str = "cpu",
):
    """Execute evaluation protocol covering E1-E8."""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data"])
    results_dir = Path(cfg["paths"]["results"])
    runs_dir = results_dir / "runs"
    tables_dir = results_dir / "tables"
    exp_dir = results_dir / "experiments"
    log_path = results_dir / "LOG.md"

    tables_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_quick" if quick else ""
    git_hash = get_git_hash()
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    eval_seed = cfg["noise"]["eval_noise_seed"]

    # Load test and OOD datasets
    test_h5 = data_dir / f"test{suffix}.h5"
    test_meta_path = data_dir / f"test{suffix}_meta.parquet"
    with h5py.File(test_h5, "r") as f:
        clean_test = np.array(f["traces"], dtype=np.float32)
    meta_test = pd.read_parquet(test_meta_path)

    ood_h5 = data_dir / f"ood{suffix}.h5"
    ood_meta_path = data_dir / f"ood{suffix}_meta.parquet"
    with h5py.File(ood_h5, "r") as f:
        clean_ood = np.array(f["traces"], dtype=np.float32)
    meta_ood = pd.read_parquet(ood_meta_path)

    # Load conventional solver
    conv_cfg_path = runs_dir / "conventional" / "conventional.json"
    if conv_cfg_path.exists():
        with open(conv_cfg_path, "r") as f:
            conv_cfg = json.load(f)
        conv_solver = ConventionalTDR(
            k_sigma=conv_cfg["best_k_sigma"],
            min_sep_samples=conv_cfg["min_sep_samples"],
            transient_return_samples=conv_cfg["transient_return_samples"],
            z0_nominal=conv_cfg["z0_nominal"],
        )
    else:
        conv_solver = ConventionalTDR(k_sigma=4.0)

    # Load main model suites (seeds)
    seeds = cfg["quick"]["seeds"] if quick else cfg["experiments"]["seeds"]

    lgbm_suites = {}
    for s in seeds:
        suite = LGBMSuite(cfg)
        suite_dir = runs_dir / f"lgbm_s{s}"
        if suite_dir.exists():
            suite.load(suite_dir)
            lgbm_suites[s] = suite

    resnet_models = {}
    resnet_norms = {}
    for s in seeds:
        res_dir = runs_dir / f"resnet_s{s}"
        ckpt = res_dir / "best_model.pt"
        norm_file = res_dir / "resnet_norm.json"
        if ckpt.exists() and norm_file.exists():
            resnet_models[s] = load_resnet_model(ckpt, in_channels=3, device=device)
            with open(norm_file, "r") as f:
                resnet_norms[s] = json.load(f)

    # =====================================================================
    # E1: Main Comparison on Test at 30 dB
    # =====================================================================
    print("\n[E1] Running Main Comparison on Test at 30 dB...")
    e1_records = []

    # 1. Conventional
    df_conv = evaluate_conventional_on_traces(conv_solver, clean_test, meta_test, snr_db=30.0, seed=eval_seed)
    m_conv = compute_all_metrics(
        pd.DataFrame({
            "true_fault_class": meta_test["fault_class"],
            "pred_fault_class": df_conv["pred_fault_class"],
            "true_fault_pos_frac": meta_test["fault_pos_frac"].fillna(0.0),
            "pred_fault_pos_frac": df_conv["pred_fault_pos_frac"],
            "length_m": meta_test["length_m"],
            "detected": df_conv["detected"],
            "true_load_class": meta_test["load_class"],
            "pred_load_class": df_conv["pred_load_class"],
            "true_load_R": meta_test["load_R"],
            "pred_load_R": df_conv["pred_load_R"],
            "true_load_tau_over_T": meta_test["load_tau_over_T"],
            "pred_load_tau_over_T": df_conv["pred_load_tau_over_T"],
            "mask_load": meta_test["mask_load"],
        })
    )
    e1_records.append({
        "Model": "Conventional TDR",
        "Fault Acc (%)": f"{m_conv['t1_fault']['accuracy']*100:.2f}",
        "Fault Macro-F1": f"{m_conv['t1_fault']['macro_f1']:.4f}",
        "Loc MAE (m)": f"{m_conv['t2_loc']['mae_m']:.2f}",
        "Loc Median (m)": f"{m_conv['t2_loc']['median_m']:.2f}",
        "Loc P90 (m)": f"{m_conv['t2_loc']['p90_m']:.2f}",
        "Loc <1m (%)": f"{m_conv['t2_loc']['pct_within_1_0m']:.1f}",
        "Load Acc (%)": f"{m_conv['t3_load']['accuracy']*100:.2f}",
        "Load Macro-F1": f"{m_conv['t3_load']['macro_f1']:.4f}",
        "MAPE R (%)": f"{m_conv['t3_load']['mape_r']:.1f}",
        "MAPE L (%)": f"{m_conv['t3_load']['mape_l']:.1f}",
        "MAPE C (%)": f"{m_conv['t3_load']['mape_c']:.1f}",
    })

    # 2. LightGBM (across seeds)
    lgbm_metrics = []
    for s, suite in lgbm_suites.items():
        df_l = evaluate_lgbm_on_traces(suite, clean_test, meta_test, snr_db=30.0, seed=eval_seed)
        m_l = compute_all_metrics(
            pd.DataFrame({
                "true_fault_class": meta_test["fault_class"],
                "pred_fault_class": df_l["pred_fault_class"],
                "true_fault_pos_frac": meta_test["fault_pos_frac"].fillna(0.0),
                "pred_fault_pos_frac": df_l["pred_fault_pos_frac"],
                "length_m": meta_test["length_m"],
                "true_load_class": meta_test["load_class"],
                "pred_load_class": df_l["pred_load_class"],
                "true_load_R": meta_test["load_R"],
                "pred_load_R": df_l["pred_load_R"],
                "true_load_tau_over_T": meta_test["load_tau_over_T"],
                "pred_load_tau_over_T": df_l["pred_load_tau_over_T"],
                "mask_load": meta_test["mask_load"],
            })
        )
        lgbm_metrics.append(m_l)

    if lgbm_metrics:
        f_acc = [m["t1_fault"]["accuracy"] * 100 for m in lgbm_metrics]
        f_f1 = [m["t1_fault"]["macro_f1"] for m in lgbm_metrics]
        l_mae = [m["t2_loc"]["mae_m"] for m in lgbm_metrics]
        l_med = [m["t2_loc"]["median_m"] for m in lgbm_metrics]
        l_p90 = [m["t2_loc"]["p90_m"] for m in lgbm_metrics]
        l_1m = [m["t2_loc"]["pct_within_1_0m"] for m in lgbm_metrics]
        ld_acc = [m["t3_load"]["accuracy"] * 100 for m in lgbm_metrics]
        ld_f1 = [m["t3_load"]["macro_f1"] for m in lgbm_metrics]
        m_r = [m["t3_load"]["mape_r"] for m in lgbm_metrics]
        m_l_val = [m["t3_load"]["mape_l"] for m in lgbm_metrics]
        m_c = [m["t3_load"]["mape_c"] for m in lgbm_metrics]

        e1_records.append({
            "Model": "LightGBM",
            "Fault Acc (%)": f"{np.mean(f_acc):.2f}" if len(seeds) == 1 else f"{np.mean(f_acc):.2f} ± {np.std(f_acc):.2f}",
            "Fault Macro-F1": f"{np.mean(f_f1):.4f}" if len(seeds) == 1 else f"{np.mean(f_f1):.4f} ± {np.std(f_f1):.4f}",
            "Loc MAE (m)": f"{np.mean(l_mae):.2f}" if len(seeds) == 1 else f"{np.mean(l_mae):.2f} ± {np.std(l_mae):.2f}",
            "Loc Median (m)": f"{np.mean(l_med):.2f}" if len(seeds) == 1 else f"{np.mean(l_med):.2f} ± {np.std(l_med):.2f}",
            "Loc P90 (m)": f"{np.mean(l_p90):.2f}" if len(seeds) == 1 else f"{np.mean(l_p90):.2f} ± {np.std(l_p90):.2f}",
            "Loc <1m (%)": f"{np.mean(l_1m):.1f}" if len(seeds) == 1 else f"{np.mean(l_1m):.1f} ± {np.std(l_1m):.1f}",
            "Load Acc (%)": f"{np.mean(ld_acc):.2f}" if len(seeds) == 1 else f"{np.mean(ld_acc):.2f} ± {np.std(ld_acc):.2f}",
            "Load Macro-F1": f"{np.mean(ld_f1):.4f}" if len(seeds) == 1 else f"{np.mean(ld_f1):.4f} ± {np.std(ld_f1):.4f}",
            "MAPE R (%)": f"{np.mean(m_r):.1f}" if len(seeds) == 1 else f"{np.mean(m_r):.1f} ± {np.std(m_r):.1f}",
            "MAPE L (%)": f"{np.mean(m_l_val):.1f}" if len(seeds) == 1 else f"{np.mean(m_l_val):.1f} ± {np.std(m_l_val):.1f}",
            "MAPE C (%)": f"{np.mean(m_c):.1f}" if len(seeds) == 1 else f"{np.mean(m_c):.1f} ± {np.std(m_c):.1f}",
        })

    # 3. ResNet (across seeds)
    resnet_metric_list = []
    for s, model in resnet_models.items():
        df_r = evaluate_resnet_on_traces(model, clean_test, meta_test, resnet_norms[s], snr_db=30.0, seed=eval_seed, device=device)
        m_r_eval = compute_all_metrics(
            pd.DataFrame({
                "true_fault_class": meta_test["fault_class"],
                "pred_fault_class": df_r["pred_fault_class"],
                "true_fault_pos_frac": meta_test["fault_pos_frac"].fillna(0.0),
                "pred_fault_pos_frac": df_r["pred_fault_pos_frac"],
                "length_m": meta_test["length_m"],
                "true_load_class": meta_test["load_class"],
                "pred_load_class": df_r["pred_load_class"],
                "true_load_R": meta_test["load_R"],
                "pred_load_R": df_r["pred_load_R"],
                "true_load_tau_over_T": meta_test["load_tau_over_T"],
                "pred_load_tau_over_T": df_r["pred_load_tau_over_T"],
                "mask_load": meta_test["mask_load"],
            })
        )
        resnet_metric_list.append(m_r_eval)

    if resnet_metric_list:
        f_acc = [m["t1_fault"]["accuracy"] * 100 for m in resnet_metric_list]
        f_f1 = [m["t1_fault"]["macro_f1"] for m in resnet_metric_list]
        l_mae = [m["t2_loc"]["mae_m"] for m in resnet_metric_list]
        l_med = [m["t2_loc"]["median_m"] for m in resnet_metric_list]
        l_p90 = [m["t2_loc"]["p90_m"] for m in resnet_metric_list]
        l_1m = [m["t2_loc"]["pct_within_1_0m"] for m in resnet_metric_list]
        ld_acc = [m["t3_load"]["accuracy"] * 100 for m in resnet_metric_list]
        ld_f1 = [m["t3_load"]["macro_f1"] for m in resnet_metric_list]
        m_r = [m["t3_load"]["mape_r"] for m in resnet_metric_list]
        m_l_val = [m["t3_load"]["mape_l"] for m in resnet_metric_list]
        m_c = [m["t3_load"]["mape_c"] for m in resnet_metric_list]

        e1_records.append({
            "Model": "Multi-Task ResNet1D",
            "Fault Acc (%)": f"{np.mean(f_acc):.2f}" if len(seeds) == 1 else f"{np.mean(f_acc):.2f} ± {np.std(f_acc):.2f}",
            "Fault Macro-F1": f"{np.mean(f_f1):.4f}" if len(seeds) == 1 else f"{np.mean(f_f1):.4f} ± {np.std(f_f1):.4f}",
            "Loc MAE (m)": f"{np.mean(l_mae):.2f}" if len(seeds) == 1 else f"{np.mean(l_mae):.2f} ± {np.std(l_mae):.2f}",
            "Loc Median (m)": f"{np.mean(l_med):.2f}" if len(seeds) == 1 else f"{np.mean(l_med):.2f} ± {np.std(l_med):.2f}",
            "Loc P90 (m)": f"{np.mean(l_p90):.2f}" if len(seeds) == 1 else f"{np.mean(l_p90):.2f} ± {np.std(l_p90):.2f}",
            "Loc <1m (%)": f"{np.mean(l_1m):.1f}" if len(seeds) == 1 else f"{np.mean(l_1m):.1f} ± {np.std(l_1m):.1f}",
            "Load Acc (%)": f"{np.mean(ld_acc):.2f}" if len(seeds) == 1 else f"{np.mean(ld_acc):.2f} ± {np.std(ld_acc):.2f}",
            "Load Macro-F1": f"{np.mean(ld_f1):.4f}" if len(seeds) == 1 else f"{np.mean(ld_f1):.4f} ± {np.std(ld_f1):.4f}",
            "MAPE R (%)": f"{np.mean(m_r):.1f}" if len(seeds) == 1 else f"{np.mean(m_r):.1f} ± {np.std(m_r):.1f}",
            "MAPE L (%)": f"{np.mean(m_l_val):.1f}" if len(seeds) == 1 else f"{np.mean(m_l_val):.1f} ± {np.std(m_l_val):.1f}",
            "MAPE C (%)": f"{np.mean(m_c):.1f}" if len(seeds) == 1 else f"{np.mean(m_c):.1f} ± {np.std(m_c):.1f}",
        })

    df_e1 = pd.DataFrame(e1_records)
    export_table(df_e1, tables_dir / "main_comparison")
    print(df_e1.to_string(index=False))

    # =====================================================================
    # E2: Noise Sweep at SNR {60, 50, 40, 30, 25, 20, 15} dB
    # =====================================================================
    print("\n[E2] Running Noise Sweep across SNR levels...")
    snrs = cfg["noise"]["sweep_snr_db"]
    e2_dict = {"snrs": snrs, "conventional": {}, "lgbm": {}, "resnet": {}}

    for snr in snrs:
        # Conv
        df_c = evaluate_conventional_on_traces(conv_solver, clean_test, meta_test, snr_db=snr, seed=eval_seed)
        mc = compute_all_metrics(pd.DataFrame({
            "true_fault_class": meta_test["fault_class"], "pred_fault_class": df_c["pred_fault_class"],
            "true_fault_pos_frac": meta_test["fault_pos_frac"].fillna(0.0), "pred_fault_pos_frac": df_c["pred_fault_pos_frac"],
            "length_m": meta_test["length_m"], "true_load_class": meta_test["load_class"], "pred_load_class": df_c["pred_load_class"],
            "true_load_R": meta_test["load_R"], "pred_load_R": df_c["pred_load_R"], "true_load_tau_over_T": meta_test["load_tau_over_T"],
            "pred_load_tau_over_T": df_c["pred_load_tau_over_T"], "mask_load": meta_test["mask_load"],
        }))
        e2_dict["conventional"][str(snr)] = {
            "fault_f1": mc["t1_fault"]["macro_f1"], "loc_mae": mc["t2_loc"]["mae_m"], "load_acc": mc["t3_load"]["accuracy"]
        }

        # LGBM s0
        if 0 in lgbm_suites:
            df_lg = evaluate_lgbm_on_traces(lgbm_suites[0], clean_test, meta_test, snr_db=snr, seed=eval_seed)
            mlg = compute_all_metrics(pd.DataFrame({
                "true_fault_class": meta_test["fault_class"], "pred_fault_class": df_lg["pred_fault_class"],
                "true_fault_pos_frac": meta_test["fault_pos_frac"].fillna(0.0), "pred_fault_pos_frac": df_lg["pred_fault_pos_frac"],
                "length_m": meta_test["length_m"], "true_load_class": meta_test["load_class"], "pred_load_class": df_lg["pred_load_class"],
                "true_load_R": meta_test["load_R"], "pred_load_R": df_lg["pred_load_R"], "true_load_tau_over_T": meta_test["load_tau_over_T"],
                "pred_load_tau_over_T": df_lg["pred_load_tau_over_T"], "mask_load": meta_test["mask_load"],
            }))
            e2_dict["lgbm"][str(snr)] = {
                "fault_f1": mlg["t1_fault"]["macro_f1"], "loc_mae": mlg["t2_loc"]["mae_m"], "load_acc": mlg["t3_load"]["accuracy"]
            }

        # ResNet s0
        if 0 in resnet_models:
            df_rs = evaluate_resnet_on_traces(resnet_models[0], clean_test, meta_test, resnet_norms[0], snr_db=snr, seed=eval_seed, device=device)
            mrs = compute_all_metrics(pd.DataFrame({
                "true_fault_class": meta_test["fault_class"], "pred_fault_class": df_rs["pred_fault_class"],
                "true_fault_pos_frac": meta_test["fault_pos_frac"].fillna(0.0), "pred_fault_pos_frac": df_rs["pred_fault_pos_frac"],
                "length_m": meta_test["length_m"], "true_load_class": meta_test["load_class"], "pred_load_class": df_rs["pred_load_class"],
                "true_load_R": meta_test["load_R"], "pred_load_R": df_rs["pred_load_R"], "true_load_tau_over_T": meta_test["load_tau_over_T"],
                "pred_load_tau_over_T": df_rs["pred_load_tau_over_T"], "mask_load": meta_test["mask_load"],
            }))
            e2_dict["resnet"][str(snr)] = {
                "fault_f1": mrs["t1_fault"]["macro_f1"], "loc_mae": mrs["t2_loc"]["mae_m"], "load_acc": mrs["t3_load"]["accuracy"]
            }

    with open(exp_dir / "noise_sweep.json", "w") as f:
        json.dump(e2_dict, f, indent=2)

    # Export noise sweep table
    e2_rows = []
    for snr in snrs:
        row = {"SNR (dB)": snr}
        for mod in ["conventional", "lgbm", "resnet"]:
            if str(snr) in e2_dict[mod]:
                row[f"{mod.capitalize()} F1"] = f"{e2_dict[mod][str(snr)]['fault_f1']:.4f}"
                row[f"{mod.capitalize()} MAE (m)"] = f"{e2_dict[mod][str(snr)]['loc_mae']:.2f}"
                row[f"{mod.capitalize()} Load Acc"] = f"{e2_dict[mod][str(snr)]['load_acc']*100:.1f}%"
        e2_rows.append(row)
    export_table(pd.DataFrame(e2_rows), tables_dir / "noise_sweep")

    # =====================================================================
    # E3: OOD Generalization Subsets
    # =====================================================================
    print("\n[E3] Running OOD Generalization Evaluations...")
    ood_subsets = ["long_cable", "high_loss", "z0_shift", "slow_rise", "near_load"]
    e3_dict = {}

    for subset in ood_subsets:
        m_sub = meta_ood[meta_ood["ood_subset"] == subset]
        indices = m_sub.index.values
        c_sub = clean_ood[indices]

        # Conv
        df_c = evaluate_conventional_on_traces(conv_solver, c_sub, m_sub, snr_db=30.0, seed=eval_seed)
        mc = compute_all_metrics(pd.DataFrame({
            "true_fault_class": m_sub["fault_class"], "pred_fault_class": df_c["pred_fault_class"],
            "true_fault_pos_frac": m_sub["fault_pos_frac"].fillna(0.0), "pred_fault_pos_frac": df_c["pred_fault_pos_frac"],
            "length_m": m_sub["length_m"], "true_load_class": m_sub["load_class"], "pred_load_class": df_c["pred_load_class"],
            "true_load_R": m_sub["load_R"], "pred_load_R": df_c["pred_load_R"], "true_load_tau_over_T": m_sub["load_tau_over_T"],
            "pred_load_tau_over_T": df_c["pred_load_tau_over_T"], "mask_load": m_sub["mask_load"],
        }))

        # LGBM s0
        df_lg = evaluate_lgbm_on_traces(lgbm_suites[0], c_sub, m_sub, snr_db=30.0, seed=eval_seed) if 0 in lgbm_suites else None
        mlg = compute_all_metrics(pd.DataFrame({
            "true_fault_class": m_sub["fault_class"], "pred_fault_class": df_lg["pred_fault_class"],
            "true_fault_pos_frac": m_sub["fault_pos_frac"].fillna(0.0), "pred_fault_pos_frac": df_lg["pred_fault_pos_frac"],
            "length_m": m_sub["length_m"], "true_load_class": m_sub["load_class"], "pred_load_class": df_lg["pred_load_class"],
            "true_load_R": m_sub["load_R"], "pred_load_R": df_lg["pred_load_R"], "true_load_tau_over_T": m_sub["load_tau_over_T"],
            "pred_load_tau_over_T": df_lg["pred_load_tau_over_T"], "mask_load": m_sub["mask_load"],
        })) if df_lg is not None else None

        # ResNet s0
        df_rs = evaluate_resnet_on_traces(resnet_models[0], c_sub, m_sub, resnet_norms[0], snr_db=30.0, seed=eval_seed, device=device) if 0 in resnet_models else None
        mrs = compute_all_metrics(pd.DataFrame({
            "true_fault_class": m_sub["fault_class"], "pred_fault_class": df_rs["pred_fault_class"],
            "true_fault_pos_frac": m_sub["fault_pos_frac"].fillna(0.0), "pred_fault_pos_frac": df_rs["pred_fault_pos_frac"],
            "length_m": m_sub["length_m"], "true_load_class": m_sub["load_class"], "pred_load_class": df_rs["pred_load_class"],
            "true_load_R": m_sub["load_R"], "pred_load_R": df_rs["pred_load_R"], "true_load_tau_over_T": m_sub["load_tau_over_T"],
            "pred_load_tau_over_T": df_rs["pred_load_tau_over_T"], "mask_load": m_sub["mask_load"],
        })) if df_rs is not None else None

        e3_dict[subset] = {
            "conventional": {"fault_f1": mc["t1_fault"]["macro_f1"], "loc_mae": mc["t2_loc"]["mae_m"], "load_acc": mc["t3_load"]["accuracy"]},
            "lgbm": {"fault_f1": mlg["t1_fault"]["macro_f1"], "loc_mae": mlg["t2_loc"]["mae_m"], "load_acc": mlg["t3_load"]["accuracy"]} if mlg else {},
            "resnet": {"fault_f1": mrs["t1_fault"]["macro_f1"], "loc_mae": mrs["t2_loc"]["mae_m"], "load_acc": mrs["t3_load"]["accuracy"]} if mrs else {},
        }

    with open(exp_dir / "ood_results.json", "w") as f:
        json.dump(e3_dict, f, indent=2)

    e3_rows = []
    for subset in ood_subsets:
        row = {"OOD Subset": subset}
        for mod in ["conventional", "lgbm", "resnet"]:
            if mod in e3_dict[subset] and e3_dict[subset][mod]:
                row[f"{mod.capitalize()} F1"] = f"{e3_dict[subset][mod]['fault_f1']:.4f}"
                row[f"{mod.capitalize()} MAE (m)"] = f"{e3_dict[subset][mod]['loc_mae']:.2f}"
                row[f"{mod.capitalize()} Load Acc"] = f"{e3_dict[subset][mod]['load_acc']*100:.1f}%"
        e3_rows.append(row)
    export_table(pd.DataFrame(e3_rows), tables_dir / "ood_metrics")

    # =====================================================================
    # E4: Error Breakdowns on Test at 30 dB
    # =====================================================================
    print("\n[E4] Computing Error Breakdowns on Test Set...")
    # Use LightGBM predictions for detailed error breakdowns
    if 0 in lgbm_suites:
        df_l4 = evaluate_lgbm_on_traces(lgbm_suites[0], clean_test, meta_test, snr_db=30.0, seed=eval_seed)
        # Location error vs d/l (10 bins)
        m_loc = df_l4["mask_loc"] == 1
        d_frac_valid = df_l4.loc[m_loc, "fault_pos_frac"].values
        err_m_valid = df_l4.loc[m_loc, "error_m"].values

        bins = np.linspace(0.10, 0.95, 11)
        loc_err_vs_pos = []
        for b_idx in range(10):
            in_bin = (d_frac_valid >= bins[b_idx]) & (d_frac_valid < bins[b_idx + 1])
            loc_err_vs_pos.append({
                "bin_start": float(bins[b_idx]),
                "bin_end": float(bins[b_idx + 1]),
                "mae_m": float(np.mean(err_m_valid[in_bin])) if np.sum(in_bin) > 0 else 0.0,
                "count": int(np.sum(in_bin)),
            })

        # Location error vs |Gamma_nominal|
        gamma_valid = np.abs(df_l4.loc[m_loc, "fault_gamma_nominal"].values)
        gamma_bins = np.linspace(0.0, 1.0, 6)
        loc_err_vs_gamma = []
        for g_idx in range(5):
            in_bin = (gamma_valid >= gamma_bins[g_idx]) & (gamma_valid < gamma_bins[g_idx + 1])
            loc_err_vs_gamma.append({
                "gamma_start": float(gamma_bins[g_idx]),
                "gamma_end": float(gamma_bins[g_idx + 1]),
                "mae_m": float(np.mean(err_m_valid[in_bin])) if np.sum(in_bin) > 0 else 0.0,
                "count": int(np.sum(in_bin)),
            })

        breakdowns = {
            "loc_error_vs_position": loc_err_vs_pos,
            "loc_error_vs_gamma": loc_err_vs_gamma,
        }
        with open(exp_dir / "error_breakdowns.json", "w") as f:
            json.dump(breakdowns, f, indent=2)

    # =====================================================================
    # E5: Learning Curves
    # =====================================================================
    print("\n[E5] Aggregating Learning Curves...")
    fracs = cfg["quick"]["learning_curve_fracs"] if quick else cfg["experiments"]["learning_curve_fracs"]
    e5_records = []
    e5_dict = {"fractions": fracs, "lgbm": {}, "resnet": {}}

    for f_val in fracs:
        # Check if run exists
        l_id = f"lgbm_s0" if f_val == 1.0 else f"lgbm_frac{f_val}_s0"
        l_m_path = runs_dir / l_id / "metrics.json"
        if l_m_path.exists():
            with open(l_m_path, "r") as f_m:
                m_data = json.load(f_m)
            e5_dict["lgbm"][str(f_val)] = {
                "fault_f1": m_data["t1_fault"]["macro_f1"],
                "loc_mae": m_data["t2_loc"]["mae_m"],
                "load_acc": m_data["t3_load"]["accuracy"],
            }

        r_id = f"resnet_s0" if f_val == 1.0 else f"resnet_frac{f_val}_s0"
        r_m_path = runs_dir / r_id / "metrics.json"
        if r_m_path.exists():
            with open(r_m_path, "r") as f_m:
                m_data = json.load(f_m)
            e5_dict["resnet"][str(f_val)] = {
                "fault_f1": m_data["t1_fault"]["macro_f1"],
                "loc_mae": m_data["t2_loc"]["mae_m"],
                "load_acc": m_data["t3_load"]["accuracy"],
            }

    with open(exp_dir / "learning_curves.json", "w") as f:
        json.dump(e5_dict, f, indent=2)

    for f_val in fracs:
        row = {"Train Fraction": f_val}
        if str(f_val) in e5_dict["lgbm"]:
            row["LGBM F1"] = f"{e5_dict['lgbm'][str(f_val)]['fault_f1']:.4f}"
            row["LGBM MAE (m)"] = f"{e5_dict['lgbm'][str(f_val)]['loc_mae']:.2f}"
        if str(f_val) in e5_dict["resnet"]:
            row["ResNet F1"] = f"{e5_dict['resnet'][str(f_val)]['fault_f1']:.4f}"
            row["ResNet MAE (m)"] = f"{e5_dict['resnet'][str(f_val)]['loc_mae']:.2f}"
        e5_records.append(row)
    export_table(pd.DataFrame(e5_records), tables_dir / "learning_curve")

    # =====================================================================
    # E6: Ablation Studies
    # =====================================================================
    print("\n[E6] Aggregating Ablation Studies...")
    ablation_runs = [
        ("Full ResNet1D (v, dv, coord)", "resnet_s0"),
        ("No Coord Channel (v, dv)", "resnet_ch_vdv_s0"),
        ("Voltage Only (v)", "resnet_ch_v_s0"),
        ("Single-Task: Fault", "resnet_single_fault_s0"),
        ("Single-Task: Location", "resnet_single_loc_s0"),
        ("Single-Task: Load", "resnet_single_load_s0"),
    ]
    e6_rows = []
    e6_dict = {}

    for label, r_id in ablation_runs:
        m_path = runs_dir / r_id / "metrics.json"
        if m_path.exists():
            with open(m_path, "r") as f_m:
                m_data = json.load(f_m)
            e6_dict[r_id] = m_data
            e6_rows.append({
                "Configuration": label,
                "Fault Macro-F1": f"{m_data['t1_fault']['macro_f1']:.4f}",
                "Loc MAE (m)": f"{m_data['t2_loc']['mae_m']:.2f}",
                "Load Acc (%)": f"{m_data['t3_load']['accuracy']*100:.2f}",
            })
        elif r_id == "resnet_s0" and 0 in resnet_models:
            # Main model baseline
            e6_dict[r_id] = resnet_metric_list[0] if resnet_metric_list else {}
            if resnet_metric_list:
                m_data = resnet_metric_list[0]
                e6_rows.append({
                    "Configuration": label,
                    "Fault Macro-F1": f"{m_data['t1_fault']['macro_f1']:.4f}",
                    "Loc MAE (m)": f"{m_data['t2_loc']['mae_m']:.2f}",
                    "Load Acc (%)": f"{m_data['t3_load']['accuracy']*100:.2f}",
                })

    with open(exp_dir / "ablations.json", "w") as f:
        json.dump(e6_dict, f, indent=2)
    export_table(pd.DataFrame(e6_rows), tables_dir / "ablations")

    # =====================================================================
    # E7: SHAP Feature Importance (TreeExplainer)
    # =====================================================================
    print("\n[E7] Computing SHAP Values for LightGBM Fault Model...")
    if 0 in lgbm_suites:
        try:
            n_shap = min(len(clean_test), cfg.get("experiments", {}).get("shap_samples", 500))
            preproc_shap = preprocess_numpy(clean_test[:n_shap], meta_test["v_inc"].iloc[:n_shap].values, snr_db=30.0, seed=eval_seed)
            X_shap = extract_features(preproc_shap)

            explainer = shap.TreeExplainer(lgbm_suites[0].model_fault)
            shap_vals = explainer.shap_values(X_shap)
            # Compute mean absolute SHAP value per feature across classes
            if isinstance(shap_vals, list):
                # Multiclass: list of [N, F]
                mean_abs_shap = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_vals], axis=0)
            else:
                mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

            shap_summary = {
                "top_feature_indices": np.argsort(mean_abs_shap)[::-1][:20].tolist(),
                "top_feature_importances": np.sort(mean_abs_shap)[::-1][:20].tolist(),
            }
            with open(exp_dir / "shap_summary.json", "w") as f:
                json.dump(shap_summary, f, indent=2)
            print("SHAP values computed and saved to shap_summary.json")
        except Exception as e:
            print(f"SHAP computation warning: {e}")

    # =====================================================================
    # E8: Real Data (WISE-CASING)
    # =====================================================================
    print("\n[E8] Evaluating on Real WISE-CASING Lab TDR Data...")
    real_meta_path = data_dir / "real" / "wise_casing" / "labels.yaml"
    if real_meta_path.exists():
        try:
            real_data = load_real_wise_casing(data_dir / "real" / "wise_casing")
            real_traces = real_data["traces"]
            real_inputs = real_data["inputs"]
            real_meta = real_data["meta"]

            # 1. Conventional
            conv_preds = pd.DataFrame(conv_solver.predict_batch(real_inputs.numpy()))

            # 2. LGBM
            lgbm_preds = None
            if 0 in lgbm_suites:
                feats = extract_features(real_inputs.numpy())
                lgbm_preds = lgbm_suites[0].predict(feats)

            # 3. ResNet
            resnet_preds = None
            if 0 in resnet_models:
                resnet = resnet_models[0].to(device)
                resnet.eval()
                mu = torch.tensor(resnet_norms[0]["mean"]).view(1, 3, 1).to(device)
                sig = torch.tensor(resnet_norms[0]["std"]).view(1, 3, 1).to(device)
                x_norm = (real_inputs.to(device) - mu) / sig
                with torch.no_grad():
                    out = resnet(x_norm)
                    pred_f_cls = torch.argmax(out["fault_logits"], dim=1).cpu().numpy()
                    pred_l_cls = torch.argmax(out["load_cls_logits"], dim=1).cpu().numpy()
                    pred_loc_frac = out["loc_pred"].cpu().numpy()

                resnet_preds = {
                    "fault_class": [FAULT_CLASSES[i] for i in pred_f_cls],
                    "load_class": [LOAD_CLASSES[i] for i in pred_l_cls],
                    "fault_pos_frac": pred_loc_frac,
                }

            real_rows = []
            for i, r in real_meta.iterrows():
                row = {
                    "Trace": r["filename"],
                    "Cable (m)": f"{r['cable_length_m']:.0f}",
                    "True Fault": r["fault_type"],
                    "True Loc (m)": f"{r['fault_pos_m']:.1f}" if pd.notna(r["fault_pos_m"]) else "None",
                    "True Load": r["load_type"],
                    "Conv Fault": conv_preds["fault_class"].iloc[i],
                    "Conv Load": conv_preds["load_class"].iloc[i],
                }
                if lgbm_preds is not None:
                    row["LGBM Fault"] = lgbm_preds["fault_class"][i]
                    row["LGBM Load"] = lgbm_preds["load_class"][i]
                if resnet_preds is not None:
                    row["ResNet Fault"] = resnet_preds["fault_class"][i]
                    row["ResNet Load"] = resnet_preds["load_class"][i]
                    res_loc_m = resnet_preds["fault_pos_frac"][i] * r["cable_length_m"]
                    row["ResNet Loc (m)"] = f"{res_loc_m:.1f}"
                real_rows.append(row)

            df_real = pd.DataFrame(real_rows)
            export_table(df_real, tables_dir / "real_data")

            with open(exp_dir / "real_data_eval.json", "w") as f:
                json.dump({
                    "meta": real_meta.to_dict(orient="records"),
                    "eval_table": real_rows
                }, f, indent=2)
            print("Real data evaluation completed and saved to results/tables/real_data.md")
        except Exception as e:
            print(f"Real data evaluation warning: {e}")
    else:
        print("Real data not yet loaded. Skipping E8 gracefully (logged in report/antigravity.md).")

    # Append to LOG.md
    summary_str = f"LGBM F1: {df_e1.iloc[1]['Fault Macro-F1']}, ResNet F1: {df_e1.iloc[2]['Fault Macro-F1']}" if len(df_e1) > 2 else "Eval done"
    log_run(log_path, "evaluate_all", git_hash, "quick" if quick else "full", summary_str)
    print("\nEvaluation suite completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_all_evaluations(cfg_path=args.config, quick=args.quick, device_name=args.device)

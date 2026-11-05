"""Metrics evaluation for T1 (fault classification), T2 (fault location), and T3 (load ID).

Adheres strictly to physics masks:
- T2 evaluated on true_fault != 'none'
- T3 evaluated on mask_load == 1 (true_fault not in short, open)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from tdr.simulator import compute_tnom


def evaluate_t1_fault(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    class_names: List[str] = ["none", "short", "open", "shunt_R", "shunt_C", "series_R", "z0_section"],
) -> Dict[str, Union[float, Dict[str, float], List[List[int]]]]:
    """Evaluate T1 Fault Classification."""
    acc = float(accuracy_score(y_true_cls, y_pred_cls))
    macro_f1 = float(f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0))
    per_class_f1_arr = f1_score(y_true_cls, y_pred_cls, average=None, labels=class_names, zero_division=0)
    per_class_f1 = {cls: float(per_class_f1_arr[i]) for i, cls in enumerate(class_names)}
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=class_names).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }


def evaluate_t2_loc(
    y_true_frac: np.ndarray,
    y_pred_frac: np.ndarray,
    cable_lengths_m: np.ndarray,
    is_fault: np.ndarray,
    detected_flags: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate T2 Fault Location.

    Computed on all samples whose TRUE class != none, regardless of predicted class.
    Error in metres = |pred - true| * l.
    """
    valid = np.asarray(is_fault, dtype=bool)
    if np.sum(valid) == 0:
        return {
            "mae_m": 0.0,
            "median_m": 0.0,
            "p90_m": 0.0,
            "pct_within_0_5m": 0.0,
            "pct_within_1_0m": 0.0,
            "pct_within_1pct_length": 0.0,
            "detection_rate": 1.0,
        }

    y_t = np.asarray(y_true_frac)[valid]
    y_p = np.asarray(y_pred_frac)[valid]
    l_m = np.asarray(cable_lengths_m)[valid]

    err_frac = np.abs(y_p - y_t)
    err_m = err_frac * l_m

    mae_m = float(np.mean(err_m))
    median_m = float(np.median(err_m))
    p90_m = float(np.percentile(err_m, 90))
    pct_0_5m = float(np.mean(err_m <= 0.5) * 100.0)
    pct_1_0m = float(np.mean(err_m <= 1.0) * 100.0)
    pct_1pct_l = float(np.mean(err_frac <= 0.01) * 100.0)

    res = {
        "mae_m": mae_m,
        "median_m": median_m,
        "p90_m": p90_m,
        "pct_within_0_5m": pct_0_5m,
        "pct_within_1_0m": pct_1_0m,
        "pct_within_1pct_length": pct_1pct_l,
    }

    if detected_flags is not None:
        det = np.asarray(detected_flags)[valid]
        res["detection_rate"] = float(np.mean(det) * 100.0)

    return res


def evaluate_t3_load(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    y_true_r: np.ndarray,
    y_pred_r: np.ndarray,
    y_true_tau_over_t: np.ndarray,
    y_pred_tau_over_t: np.ndarray,
    cable_lengths_m: np.ndarray,
    mask_load: np.ndarray,
    class_names: List[str] = ["open", "short", "matched", "R", "RL", "RC"],
    z0_nominal: float = 50.0,
) -> Dict[str, Union[float, Dict[str, float], List[List[int]]]]:
    """Evaluate T3 Load Identification.

    Computed on rows where mask_load == 1.
    """
    valid = np.asarray(mask_load, dtype=bool)
    if np.sum(valid) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "confusion_matrix": [],
            "mape_r": 0.0,
            "mape_l": 0.0,
            "mape_c": 0.0,
        }

    t_cls = np.asarray(y_true_cls)[valid]
    p_cls = np.asarray(y_pred_cls)[valid]
    t_r = np.asarray(y_true_r)[valid]
    p_r = np.asarray(y_pred_r)[valid]
    t_tau = np.asarray(y_true_tau_over_t)[valid]
    p_tau = np.asarray(y_pred_tau_over_t)[valid]
    l_m = np.asarray(cable_lengths_m)[valid]

    acc = float(accuracy_score(t_cls, p_cls))
    macro_f1 = float(f1_score(t_cls, p_cls, average="macro", zero_division=0))
    cm = confusion_matrix(t_cls, p_cls, labels=class_names).tolist()

    # MAPE R for classes R, RL, RC
    mask_r = np.isin(t_cls, ["R", "RL", "RC"]) & (~np.isnan(t_r)) & (t_r > 0)
    if np.sum(mask_r) > 0:
        mape_r = float(np.mean(np.abs(p_r[mask_r] - t_r[mask_r]) / t_r[mask_r]) * 100.0)
    else:
        mape_r = 0.0

    # MAPE L for class RL
    # Convert tau/T to L using nominal Z0=50, Tnom, and predicted R_L
    mask_rl = (t_cls == "RL") & (~np.isnan(t_tau)) & (t_tau > 0)
    if np.sum(mask_rl) > 0:
        tnom_rl = compute_tnom(l_m[mask_rl], 0.66)
        true_tau_s = t_tau[mask_rl] * tnom_rl
        true_l = true_tau_s * (t_r[mask_rl] + z0_nominal)

        pred_tau_s = np.clip(p_tau[mask_rl], 1e-4, 1.0) * tnom_rl
        pred_l = pred_tau_s * (np.clip(p_r[mask_rl], 1.0, 1000.0) + z0_nominal)
        mape_l = float(np.mean(np.abs(pred_l - true_l) / true_l) * 100.0)
    else:
        mape_l = 0.0

    # MAPE C for class RC
    mask_rc = (t_cls == "RC") & (~np.isnan(t_tau)) & (t_tau > 0)
    if np.sum(mask_rc) > 0:
        tnom_rc = compute_tnom(l_m[mask_rc], 0.66)
        true_tau_s = t_tau[mask_rc] * tnom_rc
        true_c = true_tau_s * (t_r[mask_rc] + z0_nominal) / (t_r[mask_rc] * z0_nominal)

        pred_tau_s = np.clip(p_tau[mask_rc], 1e-4, 1.0) * tnom_rc
        pred_r_c = np.clip(p_r[mask_rc], 1.0, 1000.0)
        pred_c = pred_tau_s * (pred_r_c + z0_nominal) / (pred_r_c * z0_nominal)
        mape_c = float(np.mean(np.abs(pred_c - true_c) / true_c) * 100.0)
    else:
        mape_c = 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "mape_r": mape_r,
        "mape_l": mape_l,
        "mape_c": mape_c,
    }


def compute_all_metrics(
    df_eval: pd.DataFrame,
    z0_nominal: float = 50.0,
) -> Dict[str, dict]:
    """Compute complete metric bundle across T1, T2, and T3 from predictions dataframe."""
    is_fault = (df_eval["true_fault_class"] != "none").values
    detected_flags = df_eval["detected"].values if "detected" in df_eval.columns else None

    t1_metrics = evaluate_t1_fault(
        df_eval["true_fault_class"].values,
        df_eval["pred_fault_class"].values,
    )

    t2_metrics = evaluate_t2_loc(
        df_eval["true_fault_pos_frac"].values,
        df_eval["pred_fault_pos_frac"].values,
        df_eval["length_m"].values,
        is_fault=is_fault,
        detected_flags=detected_flags,
    )

    t3_metrics = evaluate_t3_load(
        df_eval["true_load_class"].values,
        df_eval["pred_load_class"].values,
        df_eval["true_load_R"].values,
        df_eval["pred_load_R"].values,
        df_eval["true_load_tau_over_T"].values,
        df_eval["pred_load_tau_over_T"].values,
        df_eval["length_m"].values,
        mask_load=df_eval["mask_load"].values,
        z0_nominal=z0_nominal,
    )

    return {
        "t1_fault": t1_metrics,
        "t2_loc": t2_metrics,
        "t3_load": t3_metrics,
    }

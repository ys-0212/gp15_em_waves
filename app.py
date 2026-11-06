"""Interactive Streamlit application for TDR fault and load identification.

Supports:
1. Synthetic interactive simulation with customizable line, fault, load, and SNR parameters.
2. Real experimental CSV trace upload with automatic preprocessing and dual-axis display.
3. Multi-model inference comparing Conventional TDR, LightGBM, and Multi-Task ResNet1D.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn.functional as F

from tdr.simulator import (
    LineParams,
    simulate_trace,
    compute_tnom,
    C_CONST,
)
from tdr.data import (
    FAULT_CLASSES,
    LOAD_CLASSES,
    FAULT_TO_ID,
    LOAD_TO_ID,
    preprocess_numpy,
    preprocess,
)
from tdr.features import (
    ConventionalTDR,
    extract_features,
)
from tdr.models import (
    LGBMSuite,
    ResNet1D,
)


# =========================================================================
# Model Loading & Standalone Prediction Engine
# =========================================================================

def load_deployment_models(weights_dir: Union[str, Path] = "app/weights") -> Dict[str, object]:
    """Loads trained models from deployment directory app/weights/."""
    weights_dir = Path(weights_dir)
    models = {}

    # 1. Weights metadata
    meta_path = weights_dir / "weights_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            models["meta"] = json.load(f)
    else:
        models["meta"] = {"mode": "unknown"}

    # 2. Conventional baseline
    conv_path = weights_dir / "conventional.json"
    if conv_path.exists():
        with open(conv_path, "r", encoding="utf-8") as f:
            cfg_c = json.load(f)
        models["conventional"] = ConventionalTDR.from_dict(cfg_c)
    else:
        models["conventional"] = ConventionalTDR()

    # 3. LightGBM
    lgbm = LGBMSuite()
    try:
        lgbm.load(weights_dir)
        models["lgbm"] = lgbm
    except Exception as e:
        models["lgbm"] = None
        models["lgbm_err"] = str(e)

    # 4. ResNet1D
    resnet_path = weights_dir / "resnet.pt"
    norm_path = weights_dir / "resnet_norm.json"
    if resnet_path.exists():
        try:
            resnet = ResNet1D(in_channels=3)
            state = torch.load(resnet_path, map_location=torch.device("cpu"))
            resnet.load_state_dict(state)
            resnet.eval()
            models["resnet"] = resnet

            if norm_path.exists():
                with open(norm_path, "r", encoding="utf-8") as f:
                    models["resnet_norm"] = json.load(f)
            else:
                models["resnet_norm"] = {"mean": [0.5, 0.0, 0.5], "std": [0.5, 0.1, 0.5]}
        except Exception as e:
            models["resnet"] = None
            models["resnet_err"] = str(e)
    else:
        models["resnet"] = None

    return models


def predict_trace(
    v_clean: np.ndarray,
    v_inc: float,
    length_m: float,
    vf: float = 0.66,
    snr_db: Optional[float] = None,
    models: Optional[Dict[str, object]] = None,
    seed: int = 42,
) -> Dict[str, object]:
    """Pure prediction function for a single TDR trace.

    Args:
        v_clean: 1D numpy array of length 1024 (dimensionless normalized time axis).
        v_inc: Incident plateau voltage step.
        length_m: Physical cable length in metres.
        vf: Velocity factor.
        snr_db: Optional noise level in dB (if None, no noise added).
        models: Preloaded models dict from load_deployment_models(). If None, loads from app/weights/.
        seed: Random seed for noise generation.

    Returns:
        Dictionary with predictions from all 3 methods, plus preprocessed inputs.
    """
    if models is None:
        models = load_deployment_models()

    # Preprocess trace to shape [1, 3, 1024]
    v_arr = np.expand_dims(v_clean, axis=0)
    v_inc_arr = np.array([v_inc], dtype=np.float32)

    preprocessed_np = preprocess_numpy(
        v_arr,
        v_inc_arr,
        snr_db=snr_db,
        add_noise=(snr_db is not None),
        seed=seed,
    )  # [1, 3, 1024]

    u = vf * C_CONST
    t_nom = 2.0 * length_m / u
    z0 = 50.0

    results = {
        "preprocessed": preprocessed_np[0],
        "length_m": length_m,
        "t_nom": t_nom,
        "predictions": {},
    }

    # 1. Conventional TDR
    conv = models.get("conventional")
    if conv is not None:
        raw_pred = conv.predict_single(preprocessed_np[0], length_m=length_m)
        pred_d_m = raw_pred["fault_pos_frac"] * length_m if raw_pred["detected"] else None
        # Derive L or C
        tau_s = raw_pred["load_tau_over_T"] * t_nom
        r_val = raw_pred["load_R"]
        l_derived = tau_s * (r_val + z0) if raw_pred["load_class"] == "RL" else None
        c_derived = tau_s * (r_val + z0) / (r_val * z0) if raw_pred["load_class"] == "RC" and r_val > 0 else None

        results["predictions"]["conventional"] = {
            "fault_class": raw_pred["fault_class"],
            "fault_pos_frac": raw_pred["fault_pos_frac"],
            "fault_pos_m": pred_d_m,
            "detected": raw_pred["detected"],
            "load_class": raw_pred["load_class"],
            "load_R": r_val,
            "load_tau_over_T": raw_pred["load_tau_over_T"],
            "derived_L": l_derived,
            "derived_C": c_derived,
        }

    # 2. LightGBM
    lgbm = models.get("lgbm")
    if lgbm is not None:
        feats = extract_features(preprocessed_np)
        lg_pred = lgbm.predict(feats)
        f_cls = lg_pred["fault_class"][0]
        pos_frac = float(lg_pred["fault_pos_frac"][0])
        pos_m = pos_frac * length_m if f_cls != "none" else None
        l_cls = lg_pred["load_class"][0]
        r_val = float(lg_pred["load_R"][0])
        tau_val = float(lg_pred["load_tau_over_T"][0])
        tau_s = tau_val * t_nom
        l_derived = tau_s * (r_val + z0) if l_cls == "RL" else None
        c_derived = tau_s * (r_val + z0) / (r_val * z0) if l_cls == "RC" and r_val > 0 else None

        results["predictions"]["lgbm"] = {
            "fault_class": f_cls,
            "fault_pos_frac": pos_frac,
            "fault_pos_m": pos_m,
            "load_class": l_cls,
            "load_R": r_val,
            "load_tau_over_T": tau_val,
            "derived_L": l_derived,
            "derived_C": c_derived,
        }

    # 3. ResNet1D
    resnet = models.get("resnet")
    if resnet is not None:
        norm_info = models.get("resnet_norm", {"mean": [0.5, 0.0, 0.5], "std": [0.5, 0.1, 0.5]})
        mu = torch.tensor(norm_info["mean"]).view(1, 3, 1)
        sig = torch.tensor(norm_info["std"]).view(1, 3, 1)

        x_t = torch.from_numpy(preprocessed_np).float()
        x_norm = (x_t - mu) / sig

        with torch.no_grad():
            out = resnet(x_norm)
            f_probs = F.softmax(out["fault_logits"], dim=1)[0].numpy()
            f_cls_id = int(np.argmax(f_probs))
            f_cls = FAULT_CLASSES[f_cls_id]

            pos_frac = float(out["loc_pred"][0].item())
            pos_m = pos_frac * length_m if f_cls != "none" else None

            l_probs = F.softmax(out["load_cls_logits"], dim=1)[0].numpy()
            l_cls_id = int(np.argmax(l_probs))
            l_cls = LOAD_CLASSES[l_cls_id]

            reg = out["load_reg_pred"][0].numpy()
            r_val = float(np.power(10.0, np.clip(reg[0], 0.0, 4.0)))
            tau_val = float(np.power(10.0, np.clip(reg[1], -3.0, 0.5)))
            tau_s = tau_val * t_nom

            l_derived = tau_s * (r_val + z0) if l_cls == "RL" else None
            c_derived = tau_s * (r_val + z0) / (r_val * z0) if l_cls == "RC" and r_val > 0 else None

        results["predictions"]["resnet"] = {
            "fault_class": f_cls,
            "fault_class_probs": f_probs,
            "fault_pos_frac": pos_frac,
            "fault_pos_m": pos_m,
            "load_class": l_cls,
            "load_class_probs": l_probs,
            "load_R": r_val,
            "load_tau_over_T": tau_val,
            "derived_L": l_derived,
            "derived_C": c_derived,
        }

    return results


# =========================================================================
# Streamlit UI Rendering
# =========================================================================

def render_ui():
    st.set_page_config(page_title="TDR Fault & Load Identification", layout="wide")
    st.title("⚡ Transmission Line Fault & Load Identification (TDR + ML)")
    st.markdown(
        "**EE 305 Electromagnetic Waves Project — Theme 4**  \n"
        "Time Domain Reflectometry (TDR) diagnostics using Conventional Edge Detection, "
        "LightGBM with 94 Physics Features, and Deep Multi-Task 1D-ResNet."
    )

    models = load_deployment_models()
    meta = models.get("meta", {})
    if meta.get("mode") == "quick":
        st.warning(
            "⚠️ **Quick-mode models active:** The currently loaded models were trained in quick "
            "smoke-test mode on CPU (small dataset, few epochs). To load production weights, "
            "run the full training pipeline on Kaggle GPU via `kaggle_run.ipynb` and unzip the outputs."
        )

    # Sidebar parameters
    st.sidebar.header("Transmission Line Parameters")
    length_m = st.sidebar.number_input("Cable Length $l$ (m)", min_value=5.0, max_value=500.0, value=100.0, step=5.0)
    z0 = st.sidebar.number_input("Nominal $Z_0$ (Ω)", min_value=20.0, max_value=150.0, value=50.0, step=1.0)
    vf = st.sidebar.slider("Velocity Factor ($v_f$)", min_value=0.50, max_value=0.99, value=0.66, step=0.01)
    loss_scale = st.sidebar.slider("Loss Attenuation Scale", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    rise_time_ns = st.sidebar.number_input("Generator Rise Time (ns)", min_value=0.2, max_value=20.0, value=2.0, step=0.2)
    rg = st.sidebar.number_input("Generator Resistance $R_g$ (Ω)", min_value=5.0, max_value=200.0, value=50.0, step=5.0)

    st.sidebar.markdown("---")
    st.sidebar.header("Discontinuity / Fault")
    fault_cls = st.sidebar.selectbox("Fault Type", FAULT_CLASSES, index=0)
    fault_pos_m = 0.0
    fault_val = 0.0
    fault_width_m = 0.0

    if fault_cls != "none":
        fault_pos_m = st.sidebar.slider("Fault Position $d$ (m)", min_value=1.0, max_value=float(length_m - 1.0), value=float(min(40.0, length_m * 0.4)), step=0.5)
        if fault_cls in ["shunt_R", "series_R"]:
            fault_val = st.sidebar.number_input(f"Fault Resistance $R_f$ (Ω)", min_value=1.0, max_value=1000.0, value=25.0 if fault_cls == "shunt_R" else 50.0)
        elif fault_cls == "shunt_C":
            c_pf = st.sidebar.number_input("Shunt Capacitance $C_f$ (pF)", min_value=5.0, max_value=5000.0, value=200.0)
            fault_val = c_pf * 1e-12
        elif fault_cls == "z0_section":
            fault_val = st.sidebar.number_input("Section $Z_0'$ (Ω)", min_value=10.0, max_value=200.0, value=75.0)
            fault_width_m = st.sidebar.number_input("Section Width $w$ (m)", min_value=0.2, max_value=10.0, value=2.0, step=0.5)

    st.sidebar.markdown("---")
    st.sidebar.header("Load Termination")
    load_cls = st.sidebar.selectbox("Load Type", LOAD_CLASSES, index=2)  # default matched
    load_r = 50.0
    load_l = 0.0
    load_c = 0.0

    if load_cls in ["R", "RL", "RC"]:
        load_r = st.sidebar.number_input("Load Resistance $R_L$ (Ω)", min_value=1.0, max_value=2000.0, value=150.0)
    if load_cls == "RL":
        l_uh = st.sidebar.number_input("Load Inductance $L$ (μH)", min_value=0.01, max_value=100.0, value=2.5)
        load_l = l_uh * 1e-6
    elif load_cls == "RC":
        c_nf = st.sidebar.number_input("Load Capacitance $C$ (nF)", min_value=0.01, max_value=100.0, value=1.0)
        load_c = c_nf * 1e-9

    st.sidebar.markdown("---")
    st.sidebar.header("Measurement Noise")
    snr_db = st.sidebar.slider("Signal-to-Noise Ratio (dB)", min_value=10, max_value=60, value=30, step=5)

    tab1, tab2 = st.tabs(["Interactive Simulation", "Upload Experimental CSV"])

    with tab1:
        if st.button("🚀 Simulate & Identify Discontinuities", type="primary"):
            with st.spinner("Simulating lossy cascade wave and running neural inference..."):
                u = vf * C_CONST
                t_nom = 2.0 * length_m / u

                # Calculate nominal reactive tau/T
                tau_val = 0.0
                if load_cls == "RL" and (load_r + z0) > 0:
                    tau_val = (load_l / (load_r + z0)) / t_nom
                elif load_cls == "RC":
                    tau_val = (load_c * (load_r * z0) / (load_r + z0)) / t_nom if (load_r + z0) > 0 else 0.0

                params = LineParams(
                    length_m=length_m,
                    z0=z0,
                    vf=vf,
                    loss_scale=loss_scale,
                    rise_time_s=rise_time_ns * 1e-9,
                    rg=rg,
                    v0=1.0,
                    fault_class=fault_cls,
                    fault_pos_m=fault_pos_m,
                    fault_val=fault_val,
                    fault_width_m=fault_width_m,
                    load_class=load_cls,
                    load_r=load_r,
                    load_l=load_l,
                    load_c=load_c,
                    load_tau_over_t=tau_val,
                )

                t_grid, v_clean = simulate_trace(params, n_out=1024)
                v_inc = params.v0 * params.z0 / (params.z0 + params.rg)
                pred_bundle = predict_trace(
                    v_clean,
                    v_inc,
                    length_m,
                    vf=vf,
                    snr_db=float(snr_db),
                    models=models,
                )

                # Plot
                v_preproc = pred_bundle["preprocessed"][0]
                t_norm = pred_bundle["preprocessed"][2]

                fig, ax1 = plt.subplots(figsize=(10, 4.5))
                ax1.plot(t_norm, v_preproc, color="#0072B2", lw=1.8, label="Input Terminal Voltage $\\tilde{v}(0, t)$")
                ax1.axvline(0.05, color="#555555", linestyle="--", alpha=0.7, label="Incident Edge Launch")
                ax1.axvspan(0.98, 1.45, color="#F0E442", alpha=0.25, label="Expected Load Echo Region")

                # Ground truth fault marker
                if fault_cls != "none":
                    true_norm_pos = 0.05 + fault_pos_m / length_m
                    ax1.axvline(true_norm_pos, color="#009E73", lw=2.0, linestyle="-", label=f"True Fault ({fault_pos_m:.1f}m)")

                # Predicted fault markers
                preds = pred_bundle["predictions"]
                if "resnet" in preds and preds["resnet"]["fault_class"] != "none" and preds["resnet"]["fault_pos_m"] is not None:
                    res_norm_pos = 0.05 + preds["resnet"]["fault_pos_frac"]
                    ax1.axvline(res_norm_pos, color="#D55E00", lw=2.0, linestyle=":", label=f"ResNet Loc ({preds['resnet']['fault_pos_m']:.1f}m)")

                if "lgbm" in preds and preds["lgbm"]["fault_class"] != "none" and preds["lgbm"]["fault_pos_m"] is not None:
                    lg_norm_pos = 0.05 + preds["lgbm"]["fault_pos_frac"]
                    ax1.axvline(lg_norm_pos, color="#CC79A7", lw=1.5, linestyle="-.", label=f"LGBM Loc ({preds['lgbm']['fault_pos_m']:.1f}m)")

                ax1.set_xlabel("Dimensionless Normalized Time ($t / T_{\\mathrm{nom}}$)", fontsize=11)
                ax1.set_ylabel("Normalized Voltage ($\\tilde{v} = v / v_{\\mathrm{inc}}$)", fontsize=11)
                ax1.set_xlim(0.0, 2.5)
                ax1.set_ylim(-0.3, max(2.3, np.max(v_preproc) + 0.3))
                ax1.grid(True, linestyle=":", alpha=0.6)
                ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

                # Secondary Distance Axis
                ax2 = ax1.twiny()
                ax2.set_xlim(ax1.get_xlim())
                t_ticks = np.array([0.05, 0.55, 1.05, 1.55, 2.05])
                d_ticks = (t_ticks - 0.05) * length_m
                ax2.set_xticks(t_ticks)
                ax2.set_xticklabels([f"{d:.0f}m" for d in d_ticks])
                ax2.set_xlabel("Apparent Single-Trip Physical Distance ($d$ in metres)", fontsize=10)

                st.pyplot(fig)
                plt.close(fig)

                # Comparison Table
                st.subheader("Model Diagnostic Comparison")
                rows = [
                    {
                        "Diagnostic Task": "T1 Fault Classification",
                        "Ground Truth": fault_cls,
                        "Conventional TDR": preds.get("conventional", {}).get("fault_class", "N/A"),
                        "LightGBM": preds.get("lgbm", {}).get("fault_class", "N/A"),
                        "Multi-Task ResNet1D": preds.get("resnet", {}).get("fault_class", "N/A"),
                    },
                    {
                        "Diagnostic Task": "T2 Fault Location (m)",
                        "Ground Truth": f"{fault_pos_m:.1f} m ({fault_pos_m/length_m*100:.1f}%)" if fault_cls != "none" else "None",
                        "Conventional TDR": f"{preds.get('conventional', {}).get('fault_pos_m'):.1f} m" if preds.get("conventional", {}).get("fault_pos_m") is not None else "None",
                        "LightGBM": f"{preds.get('lgbm', {}).get('fault_pos_m'):.1f} m" if preds.get("lgbm", {}).get("fault_pos_m") is not None else "None",
                        "Multi-Task ResNet1D": f"{preds.get('resnet', {}).get('fault_pos_m'):.1f} m" if preds.get("resnet", {}).get("fault_pos_m") is not None else "None",
                    },
                    {
                        "Diagnostic Task": "T3 Load Classification",
                        "Ground Truth": load_cls if fault_cls not in ["short", "open"] else f"{load_cls} (Masked)",
                        "Conventional TDR": preds.get("conventional", {}).get("load_class", "N/A"),
                        "LightGBM": preds.get("lgbm", {}).get("load_class", "N/A"),
                        "Multi-Task ResNet1D": preds.get("resnet", {}).get("load_class", "N/A"),
                    },
                    {
                        "Diagnostic Task": "T3 Load Resistance $R_L$",
                        "Ground Truth": f"{load_r:.1f} Ω" if load_cls in ["R", "RL", "RC"] and fault_cls not in ["short", "open"] else "N/A",
                        "Conventional TDR": f"{preds.get('conventional', {}).get('load_R', 0):.1f} Ω",
                        "LightGBM": f"{preds.get('lgbm', {}).get('load_R', 0):.1f} Ω",
                        "Multi-Task ResNet1D": f"{preds.get('resnet', {}).get('load_R', 0):.1f} Ω",
                    },
                    {
                        "Diagnostic Task": "T3 Derived Reactive Value",
                        "Ground Truth": f"{load_l*1e6:.2f} μH" if load_cls == "RL" else (f"{load_c*1e9:.2f} nF" if load_cls == "RC" else "N/A"),
                        "Conventional TDR": f"{preds.get('conventional', {}).get('derived_L', 0)*1e6:.2f} μH" if preds.get('conventional', {}).get('derived_L') is not None else (f"{preds.get('conventional', {}).get('derived_C', 0)*1e9:.2f} nF" if preds.get('conventional', {}).get('derived_C') is not None else "N/A"),
                        "LightGBM": f"{preds.get('lgbm', {}).get('derived_L', 0)*1e6:.2f} μH" if preds.get('lgbm', {}).get('derived_L') is not None else (f"{preds.get('lgbm', {}).get('derived_C', 0)*1e9:.2f} nF" if preds.get('lgbm', {}).get('derived_C') is not None else "N/A"),
                        "Multi-Task ResNet1D": f"{preds.get('resnet', {}).get('derived_L', 0)*1e6:.2f} μH" if preds.get('resnet', {}).get('derived_L') is not None else (f"{preds.get('resnet', {}).get('derived_C', 0)*1e9:.2f} nF" if preds.get('resnet', {}).get('derived_C') is not None else "N/A"),
                    },
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                # ResNet Probabilities
                if "resnet" in preds:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("ResNet1D Fault Class Probabilities")
                        df_fp = pd.DataFrame({
                            "Fault Class": FAULT_CLASSES,
                            "Probability": preds["resnet"]["fault_class_probs"],
                        })
                        st.bar_chart(df_fp.set_index("Fault Class"))
                    with c2:
                        st.caption("ResNet1D Load Class Probabilities")
                        df_lp = pd.DataFrame({
                            "Load Class": LOAD_CLASSES,
                            "Probability": preds["resnet"]["load_class_probs"],
                        })
                        st.bar_chart(df_lp.set_index("Load Class"))

    with tab2:
        st.subheader("Upload Real / Field TDR Trace")
        st.markdown(
            "Upload a comma-separated CSV with columns `time_s,voltage_v` (or `Time(us),Amplitude(mV)`).  \n"
            "The physical cable length ($l$) and velocity factor ($v_f$) configured in the sidebar will be used to map to the normalized coordinates."
        )
        uploaded = st.file_uploader("Select TDR CSV File", type=["csv"])
        if uploaded is not None:
            df_up = pd.read_csv(uploaded)
            col_t = [c for c in df_up.columns if "time" in c.lower()][0]
            col_v = [c for c in df_up.columns if "volt" in c.lower() or "amp" in c.lower()][0]

            t_raw = df_up[col_t].values.astype(np.float64)
            v_raw = df_up[col_v].values.astype(np.float64)

            # Convert units if in us or mV
            if "us" in col_t.lower():
                t_raw = t_raw * 1e-6
            if "mv" in col_v.lower():
                v_raw = v_raw * 1e-3

            u = vf * C_CONST
            t_nom = 2.0 * length_m / u

            # Incident edge detection in real trace
            dv = np.diff(v_raw)
            idx_edge = np.argmax(np.abs(dv))
            t_inc = t_raw[idx_edge]

            # Resample to 1024 points over t/T in [0, 2.5) with edge at 0.05
            t_norm_grid = np.linspace(0.0, 2.5 * 1023.0 / 1024.0, 1024)
            t_target = t_inc + (t_norm_grid - 0.05) * t_nom

            # Baseline & plateau normalization
            pre_m = t_raw < (t_inc - 0.01 * t_nom)
            v_base = np.median(v_raw[pre_m]) if np.any(pre_m) else v_raw[0]

            post_m = (t_raw > t_inc + 0.01 * t_nom) & (t_raw < t_inc + 0.3 * t_nom)
            v_plat = np.median(v_raw[post_m]) if np.any(post_m) else v_raw[-1]

            delta = v_plat - v_base
            if abs(delta) < 1e-3:
                v_norm = v_raw - v_base
            elif delta > 0:
                v_norm = (v_raw - v_base) / delta
            else:
                v_norm = (v_base - v_raw) / abs(delta)

            v_resamp = np.interp(t_target, t_raw, v_norm).astype(np.float32)

            pred_bundle = predict_trace(
                v_resamp,
                v_inc=1.0,
                length_m=length_m,
                vf=vf,
                snr_db=None,
                models=models,
            )

            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(t_norm_grid, v_resamp, color="#0072B2", lw=1.8, label="Uploaded Resampled Trace")
            ax.axvline(0.05, color="#555555", linestyle="--", label="Incident Edge")
            ax.axvspan(0.98, 1.45, color="#F0E442", alpha=0.25, label="Expected Load Echo Region")

            preds = pred_bundle["predictions"]
            if "resnet" in preds and preds["resnet"]["fault_class"] != "none" and preds["resnet"]["fault_pos_m"] is not None:
                res_norm = 0.05 + preds["resnet"]["fault_pos_frac"]
                ax.axvline(res_norm, color="#D55E00", lw=2.0, linestyle=":", label=f"ResNet Loc ({preds['resnet']['fault_pos_m']:.1f}m)")

            ax.set_xlabel("Normalized Time ($t / T_{\\mathrm{nom}}$)", fontsize=11)
            ax.set_ylabel("Normalized Voltage $\\tilde{v}$", fontsize=11)
            ax.set_xlim(0.0, 2.5)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right")
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Trace Diagnostics (No Ground Truth)")
            rows_up = [
                {
                    "Model": "Conventional TDR",
                    "Fault": preds.get("conventional", {}).get("fault_class", "N/A"),
                    "Location (m)": f"{preds.get('conventional', {}).get('fault_pos_m', 0):.1f} m" if preds.get('conventional', {}).get('fault_pos_m') is not None else "None",
                    "Load": preds.get("conventional", {}).get("load_class", "N/A"),
                    "Load R": f"{preds.get('conventional', {}).get('load_R', 0):.1f} Ω",
                },
                {
                    "Model": "LightGBM",
                    "Fault": preds.get("lgbm", {}).get("fault_class", "N/A"),
                    "Location (m)": f"{preds.get('lgbm', {}).get('fault_pos_m', 0):.1f} m" if preds.get('lgbm', {}).get('fault_pos_m') is not None else "None",
                    "Load": preds.get("lgbm", {}).get("load_class", "N/A"),
                    "Load R": f"{preds.get('lgbm', {}).get('load_R', 0):.1f} Ω",
                },
                {
                    "Model": "Multi-Task ResNet1D",
                    "Fault": preds.get("resnet", {}).get("fault_class", "N/A"),
                    "Location (m)": f"{preds.get('resnet', {}).get('fault_pos_m', 0):.1f} m" if preds.get('resnet', {}).get('fault_pos_m') is not None else "None",
                    "Load": preds.get("resnet", {}).get("load_class", "N/A"),
                    "Load R": f"{preds.get('resnet', {}).get('load_R', 0):.1f} Ω",
                },
            ]
            st.dataframe(pd.DataFrame(rows_up), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    render_ui()

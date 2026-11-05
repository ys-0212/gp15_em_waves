"""Validate TDR simulator against exact analytic solutions.

Produces numeric comparison table and writes results to results/sim_validation.json.
"""

import json
from pathlib import Path
import numpy as np
import yaml

from tdr.simulator import (
    LineParams,
    simulate_trace,
    analytic_bounce_diagram,
    analytic_reactive_tail,
    compute_tnom,
    compute_fault_gamma_nominal,
)


def run_validation(config_path: Path = Path("config.yaml"), out_path: Path = Path("results/sim_validation.json")):
    """Run analytic validation checks and export numeric comparisons to JSON."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    results = {}
    print("=" * 75)
    print("TDR SIMULATOR NUMERICAL VALIDATION AGAINST ANALYTIC BENCHMARKS")
    print("=" * 75)

    # 1. Bounce Diagram Validation
    rg = 25.0
    z0 = 50.0
    v0 = 1.0
    length = 100.0
    test_loads = [("open", 1e9), ("short", 1e-6), ("150_ohm", 150.0), ("10_ohm", 10.0)]
    bounce_results = {}

    print(f"\n1. Bounce Diagram (Lossless, Rg={rg} ohm, Z0={z0} ohm, length={length} m):")
    print(f"{'Load':<10} {'Plateau':<10} {'Simulated (V)':<16} {'Analytic (V)':<16} {'Abs Diff (V)':<14} {'Passed?'}")
    print("-" * 75)

    all_bounce_pass = True
    for load_name, rl in test_loads:
        load_cls = "open" if load_name == "open" else ("short" if load_name == "short" else "R")
        p = LineParams(
            length_m=length, z0=z0, vf=0.66, rg=rg, v0=v0, load_class=load_cls, load_r=rl, rise_time_s=1.0e-9
        )
        t_norm, v_sim = simulate_trace(p, lossless=True)
        analytic = analytic_bounce_diagram(v0, rg, z0, rl, t_launch_t=0.05, max_k=3)

        bounce_results[load_name] = {}
        for k in range(3):
            t_mid = 0.05 + (k + 0.5)
            idx = int(np.argmin(np.abs(t_norm - t_mid)))
            sim_val = float(v_sim[idx])
            true_val = float(analytic[f"k{k}"])
            diff = abs(sim_val - true_val)
            passed = diff < 2.0e-3
            all_bounce_pass = all_bounce_pass and passed
            print(f"{load_name:<10} k={k:<8} {sim_val:<16.5f} {true_val:<16.5f} {diff:<14.2e} {str(passed)}")
            bounce_results[load_name][f"k{k}"] = {
                "simulated": sim_val,
                "analytic": true_val,
                "abs_diff": diff,
                "passed": passed,
            }
    results["bounce_diagram"] = bounce_results

    # 2. Reactive Load Tails Validation
    rl = 75.0
    tnom = compute_tnom(length, 0.66)
    tau_over_t = 0.10
    tau_analytic = tau_over_t * tnom
    reactive_results = {}

    print(f"\n2. Reactive Load Tails (Lossless, Matched Rg=Z0=50 ohm, RL={rl} ohm, tau={tau_analytic*1e9:.2f} ns):")
    print(f"{'Type':<10} {'Fitted tau (ns)':<18} {'Analytic tau (ns)':<18} {'Rel Error (%)':<14} {'Passed?'}")
    print("-" * 75)

    for load_cls in ["RL", "RC"]:
        p = LineParams(
            length_m=length, z0=50.0, rg=50.0, load_class=load_cls, load_r=rl, load_tau_over_t=tau_over_t, rise_time_s=1.0e-9
        )
        t_norm, v_sim = simulate_trace(p, lossless=True)
        t_sec = t_norm * tnom
        mask_tail = (t_norm >= 1.15) & (t_norm <= 1.45)
        t_fit = t_sec[mask_tail]
        v_fit = v_sim[mask_tail]
        v_inc = 0.5
        gamma_inf = (rl - 50.0) / (rl + 50.0)
        v_inf = v_inc * (1.0 + gamma_inf)
        y = np.abs(v_fit - v_inf)
        valid = y > 1e-4
        slope, _ = np.polyfit(t_fit[valid], np.log(y[valid]), 1)
        tau_fitted = float(-1.0 / slope)
        rel_err = float(abs(tau_fitted - tau_analytic) / tau_analytic)
        passed = rel_err < 0.05
        print(f"{load_cls:<10} {tau_fitted*1e9:<18.3f} {tau_analytic*1e9:<18.3f} {rel_err*100:<14.2f} {str(passed)}")
        reactive_results[load_cls] = {
            "fitted_tau_ns": tau_fitted * 1e9,
            "analytic_tau_ns": tau_analytic * 1e9,
            "relative_error": rel_err,
            "passed": passed,
        }
    results["reactive_tails"] = reactive_results

    # 3. Fault Nominal Gamma Validation
    fault_cases = [
        ("short", 1e-6, -1.0),
        ("open", 1e9, 1.0),
        ("shunt_R", 25.0, -50.0 / (50.0 + 50.0)),
        ("series_R", 20.0, 20.0 / (20.0 + 100.0)),
        ("z0_section", 65.0, (65.0 - 50.0) / (65.0 + 50.0)),
    ]
    fault_results = {}
    print(f"\n3. Fault Nominal Reflections:")
    print(f"{'Fault':<15} {'Measured Gamma':<18} {'Nominal Gamma':<18} {'Abs Diff':<14} {'Passed?'}")
    print("-" * 75)
    for fc, val, expected_gamma in fault_cases:
        p = LineParams(
            length_m=length, z0=50.0, rg=50.0, v0=1.0, fault_class=fc, fault_pos_m=40.0, fault_val=val, fault_width_m=3.0, load_class="matched", load_r=50.0, rise_time_s=1.0e-9
        )
        t_norm, v_sim = simulate_trace(p, lossless=True)
        mask_inc = (t_norm >= 0.09) & (t_norm <= 0.11)
        v_inc = np.median(v_sim[mask_inc])
        v_tilde = v_sim / v_inc

        if fc == "z0_section":
            window_mask = (t_norm >= 0.42) & (t_norm <= 0.55)
            meas_gamma = float(np.max(v_tilde[window_mask]) - 1.0)
            diff = abs(meas_gamma - expected_gamma)
            passed = diff < 0.05 * abs(expected_gamma)
        else:
            mask_plateau = (t_norm >= 0.60) & (t_norm <= 0.90)
            meas_gamma = float(np.median(v_tilde[mask_plateau]) - 1.0)
            diff = abs(meas_gamma - expected_gamma)
            passed = diff < 3.0e-3

        print(f"{fc:<15} {meas_gamma:<18.5f} {expected_gamma:<18.5f} {diff:<14.2e} {str(passed)}")
        fault_results[fc] = {
            "measured_gamma": meas_gamma,
            "nominal_gamma": expected_gamma,
            "diff": diff,
            "passed": passed,
        }
    results["fault_reflections"] = fault_results

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nValidation metrics saved to {out_path}")


if __name__ == "__main__":
    run_validation()


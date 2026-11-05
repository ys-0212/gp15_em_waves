"""Comprehensive test suite for TDR physics, simulator, preprocessing, models, and metrics."""

import math
from pathlib import Path
import numpy as np
import pytest
import scipy.signal

from tdr.simulator import (
    LineParams,
    simulate_trace,
    simulate_batch,
    analytic_bounce_diagram,
    analytic_reactive_tail,
    compute_tnom,
    compute_fault_gamma_nominal,
    transform_impedance,
    C_CONST,
)


# =========================================================================
# Test 1: Bounce diagram
# =========================================================================
def test_bounce_diagram():
    """Lossless, Rg = 25 ohm, Z0 = 50 ohm, loads {open, short, 150 ohm, 10 ohm}.

    Simulated V(0,t) at mid-plateau times t_launch + (k + 0.5)T for k = 0, 1, 2
    must match analytic formula within 2e-3.
    """
    rg = 25.0
    z0 = 50.0
    length = 100.0
    v0 = 1.0
    vf = 0.66
    t_launch_t = 0.05
    test_loads = [("open", 1e9), ("short", 1e-6), ("150ohm", 150.0), ("10ohm", 10.0)]

    for load_name, r_val in test_loads:
        load_cls = "open" if load_name == "open" else ("short" if load_name == "short" else "R")
        p = LineParams(
            length_m=length,
            z0=z0,
            vf=vf,
            rg=rg,
            v0=v0,
            load_class=load_cls,
            load_r=r_val,
            rise_time_s=1.0e-9,
        )
        t_norm, v_sim = simulate_trace(p, lossless=True)
        analytic = analytic_bounce_diagram(v0, rg, z0, r_val, t_launch_t=t_launch_t, max_k=3)

        for k in range(3):
            t_mid = t_launch_t + (k + 0.5)
            # Find closest index on normalized time grid
            idx = int(np.argmin(np.abs(t_norm - t_mid)))
            sim_val = v_sim[idx]
            true_val = analytic[f"k{k}"]
            diff = abs(sim_val - true_val)
            assert diff < 2.0e-3, (
                f"Bounce diagram mismatch for load {load_name} at k={k}: "
                f"sim={sim_val:.5f}, analytic={true_val:.5f}, diff={diff:.5e}"
            )


# =========================================================================
# Test 2: Matched source, resistive loads
# =========================================================================
def test_matched_source_resistive_loads():
    """Lossless, Rg = Z0.

    For R_L in {0, 25, 50, 100, inf} ohm, the level after the load echo
    equals 1 + Gamma_L within 2e-3, and stays flat for all t > T_launch + T_nom.
    """
    z0 = 50.0
    rg = 50.0
    length = 100.0
    v0 = 1.0
    # Values: 0 (short), 25, 50, 100, inf (open)
    test_resistors = [("short", 1.0e-6, -1.0),
                      ("R25", 25.0, (25.0 - 50.0) / (25.0 + 50.0)),
                      ("matched", 50.0, 0.0),
                      ("R100", 100.0, (100.0 - 50.0) / (100.0 + 50.0)),
                      ("open", 1.0e9, 1.0)]

    for name, rl, gamma_l in test_resistors:
        load_cls = "short" if name == "short" else ("open" if name == "open" else "R")
        p = LineParams(
            length_m=length,
            z0=z0,
            rg=rg,
            v0=v0,
            load_class=load_cls,
            load_r=rl,
            rise_time_s=1.0e-9,
        )
        t_norm, v_sim = simulate_trace(p, lossless=True)

        # Normalize by incident plateau over [0.09, 0.11]
        mask_inc = (t_norm >= 0.09) & (t_norm <= 0.11)
        v_inc = np.median(v_sim[mask_inc])
        v_tilde = v_sim / v_inc

        # Expected level after load echo is 1 + Gamma_L
        expected = 1.0 + gamma_l

        # Level after load echo (t/T in [1.2, 1.8])
        mask_after = (t_norm >= 1.20) & (t_norm <= 1.80)
        v_after = np.median(v_tilde[mask_after])
        diff = abs(v_after - expected)
        assert diff < 2.0e-3, (
            f"Resistive load mismatch for {name} (R_L={rl}): "
            f"measured={v_after:.5f}, expected={expected:.5f}, diff={diff:.5e}"
        )

        # Stays flat for all t > t_launch + t_nom (with t_nom = 1.0 and margin for rise time)
        mask_flat = (t_norm >= 1.15) & (t_norm <= 2.45)
        flat_diff = np.max(np.abs(v_tilde[mask_flat] - expected))
        assert flat_diff < 2.0e-3, (
            f"Trace does not stay flat for {name}: max diff={flat_diff:.5e} > 2e-3"
        )


# =========================================================================
# Test 3: shunt_R and series_R faults
# =========================================================================
def test_shunt_and_series_faults():
    """Matched load, lossless.

    v_tilde after fault echo must equal 1 + Gamma_nominal within 3e-3.
    """
    z0 = 50.0
    rg = 50.0
    length = 100.0
    v0 = 1.0
    d_m = 40.0  # d/l = 0.4, fault echo at 0.05 + 0.4 = 0.45

    # Test cases: (fault_class, val)
    cases = [
        ("shunt_R", 25.0),
        ("shunt_R", 100.0),
        ("series_R", 15.0),
        ("series_R", 50.0),
    ]

    for fc, val in cases:
        p = LineParams(
            length_m=length,
            z0=z0,
            rg=rg,
            v0=v0,
            fault_class=fc,
            fault_pos_m=d_m,
            fault_val=val,
            load_class="matched",
            load_r=z0,
            rise_time_s=1.0e-9,
        )
        t_norm, v_sim = simulate_trace(p, lossless=True)

        mask_inc = (t_norm >= 0.09) & (t_norm <= 0.11)
        v_inc = np.median(v_sim[mask_inc])
        v_tilde = v_sim / v_inc

        # Fault echo is at 0.45. Observe between fault echo and load echo (e.g. [0.6, 0.9])
        mask_plateau = (t_norm >= 0.60) & (t_norm <= 0.90)
        v_plateau = np.median(v_tilde[mask_plateau])

        gamma_nom = compute_fault_gamma_nominal(fc, val, z0)
        expected = 1.0 + gamma_nom
        diff = abs(v_plateau - expected)
        assert diff < 3.0e-3, (
            f"Fault reflection mismatch for {fc} val={val}: "
            f"v_plateau={v_plateau:.5f}, expected={expected:.5f}, diff={diff:.5e}"
        )


# =========================================================================
# Test 4: z0_section
# =========================================================================
def test_z0_section():
    """Lossless, fast rise (0.2 ns), w = 3 m.

    First edge amplitude must be within 5% of Gamma and bump width within 10%
    of round-trip time across section, 2w/u.
    """
    z0 = 50.0
    z0_prime = 65.0
    w = 3.0
    length = 100.0
    d_m = 40.0
    vf = 0.66
    u = vf * C_CONST
    tnom = compute_tnom(length, vf_nominal=0.66)
    delta_t_section_norm = (2.0 * w / u) / tnom

    p = LineParams(
        length_m=length,
        z0=z0,
        vf=vf,
        rg=z0,
        v0=1.0,
        fault_class="z0_section",
        fault_pos_m=d_m,
        fault_val=z0_prime,
        fault_width_m=w,
        load_class="matched",
        load_r=z0,
        rise_time_s=0.2e-9,
    )
    t_norm, v_sim = simulate_trace(p, lossless=True)
    mask_inc = (t_norm >= 0.09) & (t_norm <= 0.11)
    v_inc = np.median(v_sim[mask_inc])
    v_tilde = v_sim / v_inc

    # Fault appears near t/T = 0.05 + 0.40 = 0.45
    gamma_nom = compute_fault_gamma_nominal("z0_section", z0_prime, z0)
    # Expected peak value: 1.0 + gamma_nom
    # Find peak in window [0.42, 0.55]
    window_mask = (t_norm >= 0.42) & (t_norm <= 0.55)
    t_win = t_norm[window_mask]
    v_win = v_tilde[window_mask]

    peak_val = np.max(v_win)
    measured_gamma = peak_val - 1.0
    rel_amp_err = abs(measured_gamma - gamma_nom) / abs(gamma_nom)
    assert rel_amp_err < 0.05, f"z0_section amp error: measured={measured_gamma:.4f}, expected={gamma_nom:.4f}"

    # Check bump width at half height using interpolated 50% crossings
    half_height = 1.0 + 0.5 * gamma_nom
    peak_idx = int(np.argmax(v_win))
    t_rise = float(np.interp(half_height, v_win[:peak_idx], t_win[:peak_idx]))
    t_fall = float(np.interp(half_height, v_win[peak_idx:][::-1], t_win[peak_idx:][::-1]))
    interp_width = t_fall - t_rise
    rel_width_err = abs(interp_width - delta_t_section_norm) / delta_t_section_norm
    assert rel_width_err < 0.10, (
        f"z0_section bump width error: measured={interp_width:.5f}, "
        f"expected={delta_t_section_norm:.5f}, err={rel_width_err:.3f}"
    )


# =========================================================================
# Test 5: Causality
# =========================================================================
def test_causality():
    """For each fault class, trace must equal no-fault trace within 1e-4

    for t < t_launch + (d/l)*T*(vf_nom/vf) - 5*sigma.
    """
    length = 100.0
    z0 = 50.0
    vf = 0.66
    d_m = 50.0
    d_frac = d_m / length
    tnom = compute_tnom(length, 0.66)
    t_r = 2.0e-9
    delta_t_out = 2.5 * tnom / 1024.0
    t_r_eff = math.sqrt(t_r**2 + (2.0 * delta_t_out)**2)
    sigma = t_r_eff / 2.563
    sigma_norm = sigma / tnom
    t_causal_bound = 0.05 + d_frac - 5.0 * sigma_norm

    # Baseline no-fault
    p_none = LineParams(
        length_m=length, z0=z0, vf=vf, fault_class="none", load_class="matched", load_r=z0
    )
    t_norm, v_none = simulate_trace(p_none, lossless=True)

    fault_configs = [
        ("short", 0.0, 0.0),
        ("open", 0.0, 0.0),
        ("shunt_R", 30.0, 0.0),
        ("shunt_C", 100.0e-12, 0.0),
        ("series_R", 20.0, 0.0),
        ("z0_section", 65.0, 2.0),
    ]

    for fc, val, w in fault_configs:
        p_fault = LineParams(
            length_m=length,
            z0=z0,
            vf=vf,
            fault_class=fc,
            fault_pos_m=d_m,
            fault_val=val,
            fault_width_m=w,
            load_class="matched",
            load_r=z0,
        )
        _, v_fault = simulate_trace(p_fault, lossless=True)
        mask = (t_norm >= 0.0) & (t_norm < t_causal_bound)
        max_diff = np.max(np.abs(v_fault[mask] - v_none[mask]))
        assert max_diff < 1.0e-4, f"Causality violated for fault {fc}: max diff = {max_diff:.3e}"


# =========================================================================
# Test 6: RL and RC tails
# =========================================================================
def test_reactive_tails():
    """Lossless and matched source.

    tau fitted to simulated tail must be within 5% of analytic value.
    Tests both:
    1. Series RL: tau = L / (R_L + Z_0) => L_L = tau * (R_L + Z_0)
    2. Parallel RC: tau = C * R_L * Z_0 / (R_L + Z_0) => C_L = tau * (R_L + Z_0) / (R_L * Z_0)
    Both with parameterised tau/T and with explicit L_L / C_L.
    """
    length = 100.0
    z0 = 50.0
    rg = 50.0
    rl = 75.0
    tnom = compute_tnom(length, 0.66)
    tau_over_t = 0.10
    tau_analytic = tau_over_t * tnom

    for load_cls in ["RL", "RC"]:
        # Test A: parameterized via load_tau_over_t
        p_tau = LineParams(
            length_m=length,
            z0=z0,
            rg=rg,
            load_class=load_cls,
            load_r=rl,
            load_tau_over_t=tau_over_t,
            rise_time_s=1.0e-9,
        )
        # Test B: explicit L_L or C_L computed using exact formulas
        l_explicit = tau_analytic * (rl + z0) if load_cls == "RL" else 0.0
        c_explicit = tau_analytic * (rl + z0) / (rl * z0) if load_cls == "RC" else 0.0
        p_explicit = LineParams(
            length_m=length,
            z0=z0,
            rg=rg,
            load_class=load_cls,
            load_r=rl,
            load_l=l_explicit,
            load_c=c_explicit,
            load_tau_over_t=0.0,
            rise_time_s=1.0e-9,
        )

        for p, mode in [(p_tau, "tau_param"), (p_explicit, "explicit_L_or_C")]:
            t_norm, v_sim = simulate_trace(p, lossless=True)
            # Load echo starts at t/T = 1.05
            # Observe tail in [1.15, 1.45]
            t_sec = t_norm * tnom
            mask_tail = (t_norm >= 1.15) & (t_norm <= 1.45)
            t_fit = t_sec[mask_tail]
            v_fit = v_sim[mask_tail]

            # Final asymptote v_inf
            v_inc = 0.5  # v0=1, z0=50, rg=50 -> 0.5
            gamma_inf = (rl - z0) / (rl + z0)
            v_inf = v_inc * (1.0 + gamma_inf)

            y = np.abs(v_fit - v_inf)
            valid = y > 1e-4
            t_sub = t_fit[valid]
            y_sub = y[valid]

            # Linear fit on log(y) = -t / tau + const
            slope, intercept = np.polyfit(t_sub, np.log(y_sub), 1)
            tau_fitted = -1.0 / slope

            rel_err = abs(tau_fitted - tau_analytic) / tau_analytic
            assert rel_err < 0.05, f"{load_cls} ({mode}) tail error: fitted={tau_fitted:.4e}, true={tau_analytic:.4e}, err={rel_err:.3f}"


# =========================================================================
# Test 7: Lossy DC limit
# =========================================================================
def test_lossy_dc_limit():
    """With loss enabled and a short load, |H(f_min)| must be within 1e-3

    of R_dc*l / (R_dc*l + R_g).
    With an open load, final value at 7.5T must be within 2% of V0.
    """
    length = 100.0
    r_dc = 0.04
    rg = 50.0
    z0 = 50.0
    v0 = 1.0
    tnom = compute_tnom(length, 0.66)
    n_sim = 8192
    dt = tnom / 1024.0
    freqs = scipy.fft.rfftfreq(n_sim, dt)
    freqs[0] = freqs[1] * 1.0e-3
    omega = 2.0 * np.pi * freqs
    s = 1j * omega

    u = 0.66 * C_CONST
    l_per_m = z0 / u
    c_per_m = 1.0 / (z0 * u)
    loss_scale = 1.0
    k_skin = 1.5e-4
    tan_delta = 2.0e-4

    z_s = r_dc + k_skin * loss_scale * np.sqrt(freqs) * (1.0 + 1j) + s * l_per_m
    y_p = 2.0 * np.pi * freqs * c_per_m * tan_delta * loss_scale + s * c_per_m
    z_c = np.sqrt(z_s / y_p)
    gamma = np.sqrt(z_s * y_p)

    # 1. Short load
    f_min = 1.0  # Hz
    omega_min = 2.0 * np.pi * f_min
    s_min = 1j * omega_min
    z_s_min = r_dc + k_skin * loss_scale * np.sqrt(f_min) * (1.0 + 1j) + s_min * l_per_m
    y_p_min = 2.0 * np.pi * f_min * c_per_m * tan_delta * loss_scale + s_min * c_per_m
    z_c_min = np.sqrt(z_s_min / y_p_min)
    gamma_min = np.sqrt(z_s_min * y_p_min)

    z_short = np.array([1e-6], dtype=np.complex128)
    z_in_short = transform_impedance(z_short, np.array([z_c_min]), np.array([gamma_min]), length)[0]
    h_short = z_in_short / (z_in_short + rg)
    dc_h_short = float(np.abs(h_short))

    expected_dc = (r_dc * length) / (r_dc * length + rg)
    diff_dc = abs(dc_h_short - expected_dc)
    assert diff_dc < 1.0e-3, f"Lossy DC limit short mismatch: {dc_h_short:.5f} vs {expected_dc:.5f}"

    # 2. Open load: simulate on extended duration up to 8*Tnom
    # In time domain, synthesis gives v(t) = V0 * cumsum(irfft(H*P))
    # At t = 7.5 Tnom (towards end of 8 Tnom grid), multiple reflections bring voltage to V0.
    p_open = LineParams(
        length_m=length, z0=z0, rg=rg, v0=v0, load_class="open", load_r=1e9
    )
    # Check full synthesis up to 8*Tnom
    h_open = transform_impedance(np.full_like(freqs, 1e9, dtype=np.complex128), z_c, gamma, length)
    h_open = h_open / (h_open + rg)
    delta_t_out = 2.5 * tnom / 1024.0
    t_r_eff = math.sqrt((2.0e-9)**2 + (2.0 * delta_t_out)**2)
    sigma = t_r_eff / 2.563
    p_f = np.exp(-0.5 * (2.0 * np.pi * freqs * sigma)**2) * np.exp(-1j * 2.0 * np.pi * freqs * 0.05 * tnom)
    step_resp = np.cumsum(scipy.fft.irfft(h_open * p_f, n=n_sim)) * v0
    idx_7p5 = int(7.5 * 1024)
    v_final = step_resp[idx_7p5]
    rel_diff_open = abs(v_final - v0) / v0
    assert rel_diff_open < 0.02, f"Open final DC voltage error: v={v_final:.4f}, expected={v0:.4f}"


# =========================================================================
# Test 8: Lossy sanity
# =========================================================================
def test_lossy_sanity():
    """With loss enabled, open-load echo edge must get slower (10-90% rise time increases)

    as length goes from 20 to 200 m.
    """
    rise_times_measured = []
    lengths = [20.0, 200.0]

    for l_val in lengths:
        p = LineParams(
            length_m=l_val,
            z0=50.0,
            rg=50.0,
            load_class="open",
            load_r=1e9,
            rise_time_s=1.0e-9,
        )
        t_norm, v_sim = simulate_trace(p, lossless=False)
        tnom = compute_tnom(l_val, 0.66)
        t_sec = t_norm * tnom

        # Load echo edge around t/T = 1.05
        mask_edge = (t_norm >= 1.00) & (t_norm <= 1.25)
        t_e = t_sec[mask_edge]
        v_e = v_sim[mask_edge]

        v_base = v_e[0]
        v_top = np.max(v_e)
        v10 = v_base + 0.10 * (v_top - v_base)
        v90 = v_base + 0.90 * (v_top - v_base)

        t10 = t_e[np.where(v_e >= v10)[0][0]]
        t90 = t_e[np.where(v_e >= v90)[0][0]]
        tr_meas = t90 - t10
        rise_times_measured.append(tr_meas)

    assert rise_times_measured[1] > rise_times_measured[0], (
        f"Lossy edge smearing sanity failed: 20m tr={rise_times_measured[0]:.3e}s, "
        f"200m tr={rise_times_measured[1]:.3e}s"
    )


# =========================================================================
# Test 9: Determinism
# =========================================================================
def test_determinism():
    """Generating 200 samples twice with the same seed must give byte-identical

    traces and metadata.
    """
    import yaml
    from tdr.data import sample_batch_params
    from tdr.simulator import simulate_batch

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    seed = 42
    n_samples = 200

    # Run 1
    rng1 = np.random.default_rng(seed)
    params1, meta1 = sample_batch_params(n_samples, rng1, cfg)
    _, traces1 = simulate_batch(params1, n_sim=cfg["signal"]["n_sim"], n_out=cfg["signal"]["n_out"])

    # Run 2
    rng2 = np.random.default_rng(seed)
    params2, meta2 = sample_batch_params(n_samples, rng2, cfg)
    _, traces2 = simulate_batch(params2, n_sim=cfg["signal"]["n_sim"], n_out=cfg["signal"]["n_out"])

    # Verify byte-identical traces
    assert np.array_equal(traces1, traces2), "Traces are not byte-identical across identical seeds!"
    assert traces1.tobytes() == traces2.tobytes(), "Trace raw byte representation differs!"

    # Verify identical metadata
    for col in meta1.columns:
        if meta1[col].dtype == float:
            np.testing.assert_array_equal(meta1[col].fillna(-999).values, meta2[col].fillna(-999).values)
        else:
            assert (meta1[col] == meta2[col]).all(), f"Column {col} differs in metadata!"


# =========================================================================
# Test 10: Preprocessing
# =========================================================================
def test_preprocessing():
    """NumPy and PyTorch paths must give identical outputs (within 1e-6).

    After normalisation, incident plateau must be 1 within noise level.
    """
    import torch
    from tdr.data import preprocess, preprocess_numpy

    b, n = 8, 1024
    # Clean step traces: incident level = 0.5, rising around idx=51 (0.05 T)
    clean = np.zeros((b, n), dtype=np.float32)
    t_grid = np.linspace(0.0, 2.5, n, endpoint=False)
    for i in range(b):
        clean[i, t_grid >= 0.05] = 0.5 + 0.05 * i  # different levels

    v_inc = np.array([0.5 + 0.05 * i for i in range(b)], dtype=np.float32)
    seed = 1234
    snr_db = 40.0

    # NumPy path
    out_np = preprocess_numpy(clean, v_inc, snr_db=snr_db, seed=seed, add_noise=True)

    # Torch path with matching generator
    clean_t = torch.from_numpy(clean)
    v_inc_t = torch.from_numpy(v_inc)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out_torch = preprocess(clean_t, v_inc_t, snr_db=snr_db, generator=gen, add_noise=True, device=torch.device("cpu")).numpy()

    # Verify outputs match within 1e-6
    max_diff = np.max(np.abs(out_np - out_torch))
    assert max_diff < 1.0e-6, f"Preprocessing torch/numpy mismatch: max diff = {max_diff:.3e}"

    # Verify incident plateau is 1.0 within noise level (or exact without noise)
    # 1. Clean (exact 1.0 within numerical precision)
    out_clean = preprocess(clean_t, v_inc_t, add_noise=False, device=torch.device("cpu")).numpy()
    mask_plateau = (t_grid >= 0.09) & (t_grid <= 0.11)
    for i in range(b):
        med_clean = np.median(out_clean[i, 0, mask_plateau])
        assert abs(med_clean - 1.0) < 1.0e-6, f"Clean normalized plateau differs from 1.0: {med_clean}"

    # 2. Noisy (within noise level 3 * sigma)
    noise_sigma = 10.0 ** (-snr_db / 20.0)
    for i in range(b):
        plateau_v = out_torch[i, 0, mask_plateau]
        med_plateau = np.median(plateau_v)
        assert abs(med_plateau - 1.0) < 3.0 * noise_sigma, (
            f"Normalized plateau differs from 1.0 beyond noise level: {med_plateau:.5f}"
        )


# =========================================================================
# Test 11: Masks and Multi-Task Loss
# =========================================================================
def test_masks_and_loss():
    """Multi-task loss must ignore masked rows.

    A batch where every mask is off must give zero loss for that head with no NaNs.
    """
    import torch
    from tdr.models import MaskedMultiTaskLoss

    b = 4
    preds = {
        "fault_logits": torch.randn(b, 7, requires_grad=True),
        "loc_pred": torch.sigmoid(torch.randn(b)),
        "load_cls_logits": torch.randn(b, 6, requires_grad=True),
        "load_reg_pred": torch.randn(b, 2, requires_grad=True),
    }

    # Case A: all masks OFF (0.0)
    targets_all_masked = {
        "fault_cls": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "fault_pos": torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32),
        "load_cls": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "load_reg": torch.zeros(b, 2, dtype=torch.float32),
        "mask_loc": torch.zeros(b, dtype=torch.float32),  # all loc OFF
        "mask_load": torch.zeros(b, dtype=torch.float32), # all load OFF
    }

    criterion = MaskedMultiTaskLoss()
    total_loss, loss_dict = criterion(preds, targets_all_masked)

    assert not torch.isnan(total_loss), "Loss contains NaNs when masks are off!"
    assert loss_dict["loss_loc"] == 0.0, f"Expected 0.0 loc loss when mask_loc is all 0, got {loss_dict['loss_loc']}"
    assert loss_dict["loss_load_cls"] == 0.0, f"Expected 0.0 load_cls loss when mask_load is all 0, got {loss_dict['loss_load_cls']}"
    assert loss_dict["loss_load_reg"] == 0.0, f"Expected 0.0 load_reg loss when mask_load is all 0, got {loss_dict['loss_load_reg']}"
    assert loss_dict["loss_fault"] > 0.0, "Fault loss should be active for all samples."

    # Case B: partially masked rows
    # Modifying masked target value must NOT change the loss
    targets_partial1 = {
        "fault_cls": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "fault_pos": torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32),
        "load_cls": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "load_reg": torch.zeros(b, 2, dtype=torch.float32),
        "mask_loc": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32), # only row 0 active
        "mask_load": torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32),
    }
    targets_partial2 = {
        "fault_cls": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        # change row 1, 2, 3 fault_pos completely
        "fault_pos": torch.tensor([0.2, 0.99, 0.01, 0.50], dtype=torch.float32),
        "load_cls": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "load_reg": torch.zeros(b, 2, dtype=torch.float32),
        "mask_loc": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        "mask_load": torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32),
    }

    _, loss_dict1 = criterion(preds, targets_partial1)
    _, loss_dict2 = criterion(preds, targets_partial2)
    assert abs(loss_dict1["loss_loc"] - loss_dict2["loss_loc"]) < 1e-7, "Masked values affected location loss!"


# =========================================================================
# Test 12: ResNet Shapes
# =========================================================================
def test_resnet_shapes():
    """ResNet forward pass on [4, 3, 1024] input must give head shapes

    (4, 7), (4, 1) or (4,), (4, 6) and (4, 2).
    """
    import torch
    from tdr.models import ResNet1D

    b = 4
    x = torch.randn(b, 3, 1024)
    model = ResNet1D(in_channels=3, num_fault_classes=7, num_load_classes=6, embed_dim=256)
    model.eval()

    with torch.no_grad():
        out = model(x)

    assert out["fault_logits"].shape == (b, 7), f"Expected (4, 7), got {out['fault_logits'].shape}"
    assert out["loc_pred"].shape == (b,), f"Expected (4,), got {out['loc_pred'].shape}"
    assert out["load_cls_logits"].shape == (b, 6), f"Expected (4, 6), got {out['load_cls_logits'].shape}"
    assert out["load_reg_pred"].shape == (b, 2), f"Expected (4, 2), got {out['load_reg_pred'].shape}"


# =========================================================================
# Test 13: Metrics on Toy Inputs
# =========================================================================
def test_metrics_toy_inputs():
    """Metric functions must return correct values on hand-made toy inputs."""
    import pandas as pd
    from tdr.metrics import evaluate_t1_fault, evaluate_t2_loc, evaluate_t3_load, compute_all_metrics

    # 1. T1
    y_true_t1 = np.array(["short", "open", "none", "shunt_R"])
    y_pred_t1 = np.array(["short", "open", "none", "series_R"])  # 3/4 correct = 75%
    t1_res = evaluate_t1_fault(y_true_t1, y_pred_t1)
    assert abs(t1_res["accuracy"] - 0.75) < 1e-5, f"T1 accuracy error: {t1_res['accuracy']}"

    # 2. T2: errors in metres = |pred - true| * length
    # Only evaluated on true != 'none'
    y_true_pos = np.array([0.5, 0.4, 0.0, 0.8])  # 3rd is none
    y_pred_pos = np.array([0.52, 0.40, 0.3, 0.75])
    lengths = np.array([100.0, 50.0, 100.0, 200.0])
    is_fault = np.array([True, True, False, True])
    # Expected errors:
    # 0: |0.52 - 0.50| * 100 = 2.0 m
    # 1: |0.40 - 0.40| * 50 = 0.0 m
    # 3: |0.75 - 0.80| * 200 = 10.0 m
    # MAE = (2.0 + 0.0 + 10.0) / 3 = 4.0 m
    t2_res = evaluate_t2_loc(y_true_pos, y_pred_pos, lengths, is_fault)
    assert abs(t2_res["mae_m"] - 4.0) < 1e-5, f"T2 MAE error: {t2_res['mae_m']}"
    assert abs(t2_res["median_m"] - 2.0) < 1e-5, f"T2 Median error: {t2_res['median_m']}"

    # 3. T3: evaluated on mask_load == 1
    y_true_load = np.array(["matched", "R", "open", "RL"])
    y_pred_load = np.array(["matched", "R", "open", "RC"])  # 3/4 = 75%
    true_r = np.array([50.0, 100.0, 1e9, 75.0])
    pred_r = np.array([50.0, 110.0, 1e9, 75.0])  # R error: |110-100|/100 = 10%
    true_tau = np.array([0.0, 0.0, 0.0, 0.10])
    pred_tau = np.array([0.0, 0.0, 0.0, 0.10])
    mask_load = np.array([1, 1, 1, 1])
    t3_res = evaluate_t3_load(y_true_load, y_pred_load, true_r, pred_r, true_tau, pred_tau, lengths, mask_load)
    assert abs(t3_res["accuracy"] - 0.75) < 1e-5
    # R classes: matched is not in [R, RL, RC], so only R and RL are included.
    # For R: 10% error. For RL: 0% error. Mean = 5.0%
    assert abs(t3_res["mape_r"] - 5.0) < 1e-4, f"T3 MAPE R error: {t3_res['mape_r']}"


# =========================================================================
# Test 14: Encode -> Decode Round-Trip Precision
# =========================================================================
def test_encode_decode_path():
    """Verify encode -> decode path for load parameters achieves MAPE < 1e-6."""
    from tdr.data import encode_load_r, decode_load_r, encode_load_tau, decode_load_tau

    rng = np.random.default_rng(42)
    # R_L in [5, 500] Ohm
    true_r = np.exp(rng.uniform(np.log(5.0), np.log(500.0), size=1000))
    enc_r = encode_load_r(true_r)
    dec_r = decode_load_r(enc_r)
    mape_r = np.mean(np.abs(dec_r - true_r) / true_r) * 100.0
    assert mape_r < 1e-6, f"MAPE on R_L encode/decode failed: {mape_r:.2e}%"

    # tau/T in [0.02, 0.30]
    true_tau = np.exp(rng.uniform(np.log(0.02), np.log(0.30), size=1000))
    enc_tau = encode_load_tau(true_tau)
    dec_tau = decode_load_tau(enc_tau)
    mape_tau = np.mean(np.abs(dec_tau - true_tau) / true_tau) * 100.0
    assert mape_tau < 1e-6, f"MAPE on tau encode/decode failed: {mape_tau:.2e}%"

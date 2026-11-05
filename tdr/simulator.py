"""TDR Transmission Line Physics Simulator and Analytic Reference Models.

Implements exact frequency-domain solution of lossy, uniform transmission lines
with cascade transformations across faults and loads, plus analytic bounce diagram
and reactive tail references.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.fft import rfftfreq, irfft


C_CONST = 299792458.0


@dataclass
class LineParams:
    """Parameters for a transmission line simulation sample."""
    length_m: float
    z0: float = 50.0
    vf: float = 0.66
    r_dc: float = 0.04
    k_skin: float = 1.5e-4
    tan_delta: float = 2.0e-4
    loss_scale: float = 1.0
    rise_time_s: float = 2.0e-9
    rg: float = 50.0
    v0: float = 1.0
    fault_class: str = "none"
    fault_pos_m: float = 0.0
    fault_val: float = 0.0  # Rf (shunt/series), Cf (shunt_C), or Z0_prime (z0_section)
    fault_width_m: float = 0.0
    load_class: str = "matched"
    load_r: float = 50.0
    load_l: float = 0.0
    load_c: float = 0.0
    load_tau_over_t: float = 0.0
    open_ohm: float = 1.0e9
    short_ohm: float = 1.0e-6


def compute_tnom(length_m: Union[float, np.ndarray], vf_nominal: float = 0.66) -> Union[float, np.ndarray]:
    """Nominal round-trip time T_nom = 2 * length / (vf_nominal * c)."""
    return 2.0 * length_m / (vf_nominal * C_CONST)


def transform_impedance(z_load: np.ndarray, z_c: np.ndarray, gamma: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Transform impedance Z through a line section of length x with characteristic Z_c and gamma.

    Z_in = Z_c * (Z + Z_c * tanh(gamma * x)) / (Z_c + Z * tanh(gamma * x))
    """
    gamma_x = gamma * x
    # Clip real part of gamma_x to avoid overflow in tanh
    real_part = np.clip(np.real(gamma_x), -50.0, 50.0)
    imag_part = np.imag(gamma_x)
    gamma_x_clipped = real_part + 1j * imag_part
    th = np.tanh(gamma_x_clipped)
    num = z_load + z_c * th
    den = z_c + z_load * th
    # Avoid zero division
    den = np.where(np.abs(den) == 0, 1e-12 + 0j, den)
    return z_c * (num / den)


def compute_fault_gamma_nominal(fault_class: str, fault_val: float, z0: float = 50.0) -> float:
    """Compute nominal reflection coefficient for a fault."""
    if fault_class == "short":
        return -1.0
    elif fault_class == "open":
        return 1.0
    elif fault_class == "shunt_R":
        rf = fault_val
        return -z0 / (z0 + 2.0 * rf)
    elif fault_class == "series_R":
        rf = fault_val
        return rf / (rf + 2.0 * z0)
    elif fault_class == "z0_section":
        z0_p = fault_val
        return (z0_p - z0) / (z0_p + z0)
    elif fault_class == "none":
        return 0.0
    elif fault_class == "shunt_C":
        return float("nan")
    return float("nan")


def simulate_trace(
    params: LineParams,
    n_sim: int = 8192,
    sim_dt_per_t: int = 1024,
    n_out: int = 1024,
    window_t: float = 2.5,
    t_launch_t: float = 0.05,
    lossless: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a single clean TDR trace.

    Returns:
        (t_norm, v_out): normalized time grid [0, 2.5) of length n_out and clean voltage trace.
    """
    batch = [params]
    t_norm, traces = simulate_batch(
        batch,
        n_sim=n_sim,
        sim_dt_per_t=sim_dt_per_t,
        n_out=n_out,
        window_t=window_t,
        t_launch_t=t_launch_t,
        lossless=lossless,
    )
    return t_norm, traces[0]


def simulate_batch(
    params_list: List[LineParams],
    n_sim: int = 8192,
    sim_dt_per_t: int = 1024,
    n_out: int = 1024,
    window_t: float = 2.5,
    t_launch_t: float = 0.05,
    lossless: bool = False,
    chunk_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a batch of TDR traces in chunks.

    Returns:
        (t_out_norm, traces_arr): output normalized grid [N_out] and voltage array [B, N_out].
    """
    total_samples = len(params_list)
    t_out_norm = np.linspace(0.0, window_t, n_out, endpoint=False, dtype=np.float32)
    all_traces = np.empty((total_samples, n_out), dtype=np.float32)

    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        chunk = params_list[start_idx:end_idx]
        traces_chunk = _simulate_chunk(
            chunk,
            n_sim=n_sim,
            sim_dt_per_t=sim_dt_per_t,
            n_out=n_out,
            window_t=window_t,
            t_launch_t=t_launch_t,
            lossless=lossless,
        )
        all_traces[start_idx:end_idx] = traces_chunk

    return t_out_norm, all_traces


def _simulate_chunk(
    chunk: List[LineParams],
    n_sim: int = 8192,
    sim_dt_per_t: int = 1024,
    n_out: int = 1024,
    window_t: float = 2.5,
    t_launch_t: float = 0.05,
    lossless: bool = False,
) -> np.ndarray:
    """Internal helper to simulate a single batch chunk vectorized over frequency."""
    b = len(chunk)

    # Extract scalar arrays: shape [B, 1]
    lengths = np.array([p.length_m for p in chunk], dtype=np.float64)[:, None]
    z0s = np.array([p.z0 for p in chunk], dtype=np.float64)[:, None]
    vfs = np.array([p.vf for p in chunk], dtype=np.float64)[:, None]
    r_dcs = np.array([0.0 if lossless else p.r_dc for p in chunk], dtype=np.float64)[:, None]
    k_skins = np.array([0.0 if lossless else p.k_skin for p in chunk], dtype=np.float64)[:, None]
    tan_deltas = np.array([0.0 if lossless else p.tan_delta for p in chunk], dtype=np.float64)[:, None]
    loss_scales = np.array([1.0 if lossless else p.loss_scale for p in chunk], dtype=np.float64)[:, None]
    rise_times = np.array([p.rise_time_s for p in chunk], dtype=np.float64)[:, None]
    rgs = np.array([p.rg for p in chunk], dtype=np.float64)[:, None]
    v0s = np.array([p.v0 for p in chunk], dtype=np.float64)[:, None]

    tnoms = compute_tnom(lengths, vf_nominal=0.66)  # [B, 1]
    dts = tnoms / sim_dt_per_t  # [B, 1]

    # Output array for resampled traces
    traces = np.empty((b, n_out), dtype=np.float32)
    t_out_norm = np.linspace(0.0, window_t, n_out, endpoint=False, dtype=np.float64)

    for i in range(b):
        p = chunk[i]
        length = lengths[i, 0]
        z0 = z0s[i, 0]
        vf = vfs[i, 0]
        u = vf * C_CONST
        l_per_m = z0 / u
        c_per_m = 1.0 / (z0 * u)
        tnom = tnoms[i, 0]
        dt = dts[i, 0]
        rg = rgs[i, 0]
        v0 = v0s[i, 0]

        # Frequencies
        freqs = rfftfreq(n_sim, dt)
        if len(freqs) > 1:
            freqs[0] = freqs[1] * 1.0e-3
        omega = 2.0 * np.pi * freqs
        s = 1j * omega

        if lossless:
            z_c = np.full_like(freqs, z0, dtype=np.complex128)
            gamma = s / u
        else:
            r_dc = r_dcs[i, 0]
            k_skin = k_skins[i, 0]
            tan_delta = tan_deltas[i, 0]
            loss_scale = loss_scales[i, 0]

            # Series impedance and shunt admittance
            z_s = r_dc + k_skin * loss_scale * np.sqrt(freqs) * (1.0 + 1j) + s * l_per_m
            y_p = 2.0 * np.pi * freqs * c_per_m * tan_delta * loss_scale + s * c_per_m
            z_c = np.sqrt(z_s / y_p)
            gamma = np.sqrt(z_s * y_p)

        # 1. Load impedance Z_L(f)
        load_cls = p.load_class
        open_ohm = p.open_ohm
        short_ohm = p.short_ohm
        rl = p.load_r

        if load_cls == "open":
            z_load = np.full_like(freqs, open_ohm, dtype=np.complex128)
        elif load_cls == "short":
            z_load = np.full_like(freqs, short_ohm, dtype=np.complex128)
        elif load_cls in ("matched", "R"):
            z_load = np.full_like(freqs, rl, dtype=np.complex128)
        elif load_cls == "RL":
            # Series RL
            tau = p.load_tau_over_t * tnom
            l_l = p.load_l if p.load_l > 0 else tau * (rl + z0)
            z_load = rl + s * l_l
        elif load_cls == "RC":
            # Parallel RC
            tau = p.load_tau_over_t * tnom
            c_l = p.load_c if p.load_c > 0 else tau * (rl + z0) / (rl * z0)
            z_load = 1.0 / (1.0 / rl + s * c_l)
        else:
            z_load = np.full_like(freqs, rl, dtype=np.complex128)

        # 2. Line cascade back to source
        fault_cls = p.fault_class
        d = p.fault_pos_m
        w = p.fault_width_m

        if fault_cls == "none":
            # Direct transform through full length
            z_in = transform_impedance(z_load, z_c, gamma, length)
        else:
            # Distance from load to fault
            if fault_cls == "z0_section":
                x_post = max(0.0, length - d - w)
            else:
                x_post = max(0.0, length - d)

            z_at_fault = transform_impedance(z_load, z_c, gamma, x_post)

            # Apply fault
            val = p.fault_val
            if fault_cls == "short":
                z_fault = np.full_like(freqs, short_ohm, dtype=np.complex128)
            elif fault_cls == "open":
                z_fault = np.full_like(freqs, open_ohm, dtype=np.complex128)
            elif fault_cls == "shunt_R":
                rf = val
                z_fault = (z_at_fault * rf) / (z_at_fault + rf)
            elif fault_cls == "shunt_C":
                cf = val
                z_fault = 1.0 / (1.0 / z_at_fault + s * cf)
            elif fault_cls == "series_R":
                rf = val
                z_fault = z_at_fault + rf
            elif fault_cls == "z0_section":
                z0_prime = val
                z_c_prime = z_c * (z0_prime / z0)
                # Same gamma
                z_fault = transform_impedance(z_at_fault, z_c_prime, gamma, w)
            else:
                z_fault = z_at_fault

            # Transform from fault back to source through distance d
            z_in = transform_impedance(z_fault, z_c, gamma, d)

        # 3. Transfer function H(f) = Z_in / (Z_in + R_g)
        h = z_in / (z_in + rg)

        # 4. Pulse spectrum P(f)
        delta_t_out = 2.5 * tnom / 1024.0
        t_r = rise_times[i, 0]
        t_r_eff = np.sqrt(t_r**2 + (2.0 * delta_t_out)**2)
        sigma = t_r_eff / 2.563
        t_launch = t_launch_t * tnom

        p_f = np.exp(-0.5 * (2.0 * np.pi * freqs * sigma)**2) * np.exp(-1j * 2.0 * np.pi * freqs * t_launch)

        # 5. Time-domain synthesis
        # v(t) = V0 * cumsum(irfft(H * P, N_sim))
        step_response = np.cumsum(irfft(h * p_f, n=n_sim)) * v0

        # Resample onto output grid
        # Sim time grid: t_sim = np.arange(n_sim) * dt / tnom
        t_sim_norm = np.arange(n_sim, dtype=np.float64) * dt / tnom
        traces[i] = np.interp(t_out_norm, t_sim_norm, step_response).astype(np.float32)

    return traces


# =========================================================================
# Analytic reference models for testing and verification
# =========================================================================

def analytic_bounce_diagram(
    v0: float,
    rg: float,
    z0: float,
    rl: float,
    t_launch_t: float = 0.05,
    max_k: int = 3,
) -> Dict[str, float]:
    """Calculate analytic mid-plateau voltages for lossless line with resistive load.

    Returns dict mapping 'k0', 'k1', 'k2' ... to analytic V(0, t).
    """
    v_inc = v0 * z0 / (z0 + rg)
    gamma_s = (rg - z0) / (rg + z0)
    gamma_l = (rl - z0) / (rl + z0)

    plateaus = {}
    for k in range(max_k):
        if k == 0:
            v_k = v_inc
        else:
            sum_terms = sum((gamma_l**n) * (gamma_s**(n - 1)) for n in range(1, k + 1))
            v_k = v_inc * (1.0 + (1.0 + gamma_s) * sum_terms)
        plateaus[f"k{k}"] = v_k
    return plateaus


def analytic_reactive_tail(
    v0: float,
    z0: float,
    rl: float,
    tau: float,
    load_type: str,
    t_eval: np.ndarray,
    t_echo: float,
) -> np.ndarray:
    """Calculate analytic tail for lossless matched line (R_g = Z_0).

    V(0,t) = v_inc * [1 + gamma_inf + (gamma_0 - gamma_inf) * exp(-(t - t_echo)/tau)]
    """
    v_inc = v0 * z0 / (z0 + z0)  # Rg = Z0
    gamma_inf = (rl - z0) / (rl + z0)
    if load_type == "RL":
        gamma_0 = 1.0
    elif load_type == "RC":
        gamma_0 = -1.0
    else:
        raise ValueError(f"Unknown reactive load type: {load_type}")

    dt = np.maximum(0.0, t_eval - t_echo)
    return v_inc * (1.0 + gamma_inf + (gamma_0 - gamma_inf) * np.exp(-dt / tau))

"""Data generation, preprocessing, noise models, and PyTorch dataset definitions.

Single source of truth for:
- Preprocessing function (PyTorch implementation with NumPy wrapper)
- Feature grids and normalization
- Data generation from config parameters
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml

from tdr.simulator import (
    LineParams,
    compute_tnom,
    compute_fault_gamma_nominal,
    simulate_batch,
    C_CONST,
)


FAULT_CLASSES = ["none", "short", "open", "shunt_R", "shunt_C", "series_R", "z0_section"]
LOAD_CLASSES = ["open", "short", "matched", "R", "RL", "RC"]

FAULT_TO_ID = {cls: idx for idx, cls in enumerate(FAULT_CLASSES)}
ID_TO_FAULT = {idx: cls for idx, cls in enumerate(FAULT_CLASSES)}

LOAD_TO_ID = {cls: idx for idx, cls in enumerate(LOAD_CLASSES)}
ID_TO_LOAD = {idx: cls for idx, cls in enumerate(LOAD_CLASSES)}


def encode_load_r(r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Encode load resistance R_L (Ohm) to log10(R_L)."""
    return np.log10(r)


def decode_load_r(log_r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Decode log10(R_L) back to resistance R_L in Ohms."""
    return np.power(10.0, log_r)


def encode_load_tau(tau_over_t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Encode load tau/T to log10(tau/T)."""
    return np.log10(tau_over_t)


def decode_load_tau(log_tau: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Decode log10(tau/T) back to tau/T."""
    return np.power(10.0, log_tau)


# =========================================================================
# Preprocessing (Single Source of Truth)
# =========================================================================

# Savitzky-Golay derivative filter kernel: window=11, poly=2, deriv=1
_SAVGOL_WINDOW = 11
_SAVGOL_POLY = 2
# Note: savgol_coeffs calculates convolution kernel for y[n] = sum c_k x[n+k].
# In PyTorch F.conv1d, input is correlated with weight.
# SciPy savgol_filter computes sum_{k=-m}^m c_k x_{n+k}.
# By flipping coefficients, conv1d matches savgol_filter.
_SG_COEFFS = savgol_coeffs(_SAVGOL_WINDOW, _SAVGOL_POLY, deriv=1)
# PyTorch conv1d weight shape: [out_channels, in_channels, kernel_size]
_SG_KERNEL_TORCH = torch.tensor(_SG_COEFFS[::-1].copy(), dtype=torch.float32).view(1, 1, _SAVGOL_WINDOW)


def preprocess(
    clean_batch: torch.Tensor,
    v_inc: torch.Tensor,
    snr_db: Optional[Union[float, torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    add_noise: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Preprocess a batch of clean TDR traces into a 3-channel representation [B, 3, 1024].

    Channels:
        0: v_tilde = normalized noisy voltage trace
        1: dv_tilde = Savitzky-Golay 1st derivative with respect to sample index
        2: coord = normalized time grid t/T in [0, 2.5)

    Args:
        clean_batch: Tensor of shape [B, 1024] or [1024], clean voltage traces.
        v_inc: Tensor of shape [B] or scalar, incident step amplitude V0 * Z0 / (Z0 + Rg).
        snr_db: SNR in dB. Can be scalar or [B] tensor. If None and add_noise=True, draws U[20, 50].
        generator: Optional torch.Generator for deterministic random noise.
        add_noise: Whether to add Gaussian noise.
        device: Device to place computations on (defaults to clean_batch.device).

    Returns:
        Tensor of shape [B, 3, 1024], float32.
    """
    if clean_batch.ndim == 1:
        clean_batch = clean_batch.unsqueeze(0)
    b, n = clean_batch.shape
    if device is None:
        device = clean_batch.device

    clean = clean_batch.to(device=device, dtype=torch.float32)

    if not isinstance(v_inc, torch.Tensor):
        v_inc_t = torch.full((b, 1), float(v_inc), device=device, dtype=torch.float32)
    else:
        v_inc_t = v_inc.to(device=device, dtype=torch.float32).view(b, 1)

    # 1. Additive Gaussian noise
    if add_noise:
        if snr_db is None:
            # Uniform draw between 20 and 50 dB
            snr = torch.empty((b, 1), device=device, dtype=torch.float32).uniform_(20.0, 50.0, generator=generator)
        elif isinstance(snr_db, (int, float)) or (isinstance(snr_db, torch.Tensor) and snr_db.numel() == 1):
            snr = torch.full((b, 1), float(snr_db), device=device, dtype=torch.float32)
        else:
            snr = snr_db.to(device=device, dtype=torch.float32).view(b, 1)

        sigma = v_inc_t * torch.pow(10.0, -snr / 20.0)
        noise = torch.randn(clean.shape, generator=generator, device=device, dtype=torch.float32) * sigma
        v_noisy = clean + noise
    else:
        v_noisy = clean

    # 2. Normalisation: v_tilde = v / median(v[t/T in [0.09, 0.11]])
    # For grid [0, 2.5) with 1024 points, dt/T = 2.5 / 1024 = 0.0024414
    # t/T in [0.09, 0.11] corresponds to indices:
    idx_start = int(np.round(0.09 / 2.5 * n))
    idx_end = int(np.round(0.11 / 2.5 * n))
    # Median along time dimension for each sample in batch
    norm_window = v_noisy[:, idx_start:idx_end]
    scale = torch.median(norm_window, dim=1, keepdim=True).values
    # Avoid zero division
    scale = torch.where(scale.abs() < 1e-6, torch.ones_like(scale), scale)
    v_tilde = v_noisy / scale  # [B, 1024]

    # 3. Savitzky-Golay derivative channel using Conv1D
    # Pad symmetrically or reflectively with (window - 1) // 2 = 5
    pad_len = (_SAVGOL_WINDOW - 1) // 2
    v_in = v_tilde.unsqueeze(1)  # [B, 1, 1024]
    v_padded = F.pad(v_in, (pad_len, pad_len), mode="replicate")
    kernel = _SG_KERNEL_TORCH.to(device=device, dtype=torch.float32)
    dv_tilde = F.conv1d(v_padded, kernel)  # [B, 1, 1024]
    dv_tilde = dv_tilde.squeeze(1)  # [B, 1024]

    # 4. Coordinate channel: t/T grid in [0, 2.5) with endpoint=False
    # dt = 2.5 / n, end = 2.5 - dt
    end_val = 2.5 * (n - 1) / n
    t_coord = torch.linspace(0.0, end_val, steps=n, device=device, dtype=torch.float32)
    coord = t_coord.unsqueeze(0).expand(b, -1)  # [B, 1024]

    # Stack into [B, 3, 1024]
    out = torch.stack([v_tilde, dv_tilde, coord], dim=1)
    return out


def preprocess_numpy(
    clean_batch: np.ndarray,
    v_inc: Union[float, np.ndarray],
    snr_db: Optional[Union[float, np.ndarray]] = None,
    seed: Optional[int] = None,
    add_noise: bool = True,
) -> np.ndarray:
    """NumPy wrapper for preprocess calling CPU PyTorch function."""
    clean_t = torch.from_numpy(np.array(clean_batch, copy=True))
    if isinstance(v_inc, np.ndarray):
        v_inc_t = torch.from_numpy(np.array(v_inc, copy=True))
    else:
        v_inc_t = torch.tensor(v_inc)

    if snr_db is not None:
        if isinstance(snr_db, np.ndarray):
            snr_t = torch.from_numpy(np.array(snr_db, copy=True))
        else:
            snr_t = torch.tensor(snr_db)
    else:
        snr_t = None

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    out_t = preprocess(
        clean_batch=clean_t,
        v_inc=v_inc_t,
        snr_db=snr_t,
        generator=generator,
        add_noise=add_noise,
        device=torch.device("cpu"),
    )
    return out_t.cpu().numpy()


# =========================================================================
# Synthetic Dataset Generation
# =========================================================================

def sample_batch_params(
    n_samples: int,
    rng: np.random.Generator,
    cfg: dict,
    ood_type: Optional[str] = None,
) -> Tuple[List[LineParams], pd.DataFrame]:
    """Sample line parameters according to Section 2.3 specification."""
    sampling_cfg = cfg["sampling"]

    # Base parameters
    length_m = rng.uniform(*sampling_cfg["length_m"], size=n_samples)
    z0 = rng.uniform(*sampling_cfg["z0"], size=n_samples)
    vf = 0.66 * rng.uniform(*sampling_cfg["vf_rel"], size=n_samples)
    loss_scale = rng.uniform(*sampling_cfg["loss_scale"], size=n_samples)
    rise_time_s = rng.uniform(*sampling_cfg["rise_time_s"], size=n_samples)
    rg = rng.uniform(*sampling_cfg["rg"], size=n_samples)
    v0 = np.ones(n_samples, dtype=np.float64)

    # Apply OOD perturbations if specified
    if ood_type == "long_cable":
        length_m = rng.uniform(200.0, 300.0, size=n_samples)
    elif ood_type == "high_loss":
        loss_scale = rng.uniform(1.5, 2.0, size=n_samples)
    elif ood_type == "z0_shift":
        # U[44, 47] or U[53, 56]
        choice = rng.uniform(0.0, 1.0, size=n_samples) < 0.5
        z0 = np.where(choice, rng.uniform(44.0, 47.0, size=n_samples), rng.uniform(53.0, 56.0, size=n_samples))
    elif ood_type == "slow_rise":
        rise_time_s = rng.uniform(5.0e-9, 8.0e-9, size=n_samples)

    # Fault class
    if ood_type == "near_load":
        # Exclude 'none'
        non_none_faults = [f for f in FAULT_CLASSES if f != "none"]
        fault_classes = rng.choice(non_none_faults, size=n_samples)
    else:
        fault_classes = rng.choice(FAULT_CLASSES, size=n_samples)

    # Sample faults
    fault_pos_frac = np.full(n_samples, np.nan, dtype=np.float64)
    fault_pos_m = np.full(n_samples, np.nan, dtype=np.float64)
    fault_values = np.full(n_samples, np.nan, dtype=np.float64)
    fault_widths_m = np.full(n_samples, np.nan, dtype=np.float64)
    fault_gamma_nom = np.full(n_samples, np.nan, dtype=np.float64)

    for i in range(n_samples):
        fc = fault_classes[i]
        l_i = length_m[i]

        if fc == "none":
            fault_gamma_nom[i] = 0.0
            continue

        if ood_type == "near_load":
            # Fault placed 0.3 - 3.0 m from load
            dist_to_load = rng.uniform(0.3, 3.0)
            if fc == "z0_section":
                w = rng.uniform(*sampling_cfg["z0_section_width_m"])
                d = max(1.0, l_i - dist_to_load - w)
                fault_widths_m[i] = w
            else:
                d = max(1.0, l_i - dist_to_load)
            fault_pos_m[i] = d
            fault_pos_frac[i] = d / l_i
        else:
            if fc == "z0_section":
                w = rng.uniform(*sampling_cfg["z0_section_width_m"])
                # enforce (d + w)/l <= 0.95
                max_d_frac = min(0.95, 0.95 - w / l_i)
                d_frac = rng.uniform(0.10, max(0.11, max_d_frac))
                fault_widths_m[i] = w
                fault_pos_frac[i] = d_frac
                fault_pos_m[i] = d_frac * l_i
            else:
                d_frac = rng.uniform(*sampling_cfg["fault_pos_frac"])
                fault_pos_frac[i] = d_frac
                fault_pos_m[i] = d_frac * l_i

        # Fault values
        if fc == "shunt_R":
            # log-uniform [5, 2000]
            val = np.exp(rng.uniform(np.log(5.0), np.log(2000.0)))
            fault_values[i] = val
        elif fc == "shunt_C":
            # log-uniform [20pF, 500pF]
            val = np.exp(rng.uniform(np.log(2.0e-11), np.log(5.0e-10)))
            fault_values[i] = val
        elif fc == "series_R":
            # log-uniform [1, 200]
            val = np.exp(rng.uniform(np.log(1.0), np.log(200.0)))
            fault_values[i] = val
        elif fc == "z0_section":
            # U[35, 45] or U[55, 70]
            if rng.uniform() < 0.5:
                z0_p = rng.uniform(35.0, 45.0)
            else:
                z0_p = rng.uniform(55.0, 70.0)
            fault_values[i] = z0_p
        elif fc == "short":
            fault_values[i] = 1.0e-6
        elif fc == "open":
            fault_values[i] = 1.0e9

        fault_gamma_nom[i] = compute_fault_gamma_nominal(fc, fault_values[i], z0[i])

    # Sample load classes independently
    load_classes = rng.choice(LOAD_CLASSES, size=n_samples)
    load_r = np.full(n_samples, np.nan, dtype=np.float64)
    load_l = np.full(n_samples, np.nan, dtype=np.float64)
    load_c = np.full(n_samples, np.nan, dtype=np.float64)
    load_tau_over_t = np.full(n_samples, np.nan, dtype=np.float64)

    for i in range(n_samples):
        lc = load_classes[i]
        l_i = length_m[i]
        z0_i = z0[i]
        tnom_i = compute_tnom(l_i, vf_nominal=0.66)

        if lc == "open":
            load_r[i] = 1.0e9
        elif lc == "short":
            load_r[i] = 1.0e-6
        elif lc == "matched":
            load_r[i] = rng.uniform(*sampling_cfg["matched_ohm"])
        elif lc == "R":
            # log[5, 500] rejecting [40, 62]
            while True:
                r_cand = np.exp(rng.uniform(np.log(5.0), np.log(500.0)))
                if not (40.0 <= r_cand <= 62.0):
                    break
            load_r[i] = r_cand
        elif lc == "RL":
            load_r[i] = np.exp(rng.uniform(np.log(5.0), np.log(500.0)))
            tau_over_t = np.exp(rng.uniform(np.log(0.02), np.log(0.3)))
            load_tau_over_t[i] = tau_over_t
            tau = tau_over_t * tnom_i
            load_l[i] = tau * (load_r[i] + z0_i)
        elif lc == "RC":
            load_r[i] = np.exp(rng.uniform(np.log(5.0), np.log(500.0)))
            tau_over_t = np.exp(rng.uniform(np.log(0.02), np.log(0.3)))
            load_tau_over_t[i] = tau_over_t
            tau = tau_over_t * tnom_i
            load_c[i] = tau * (load_r[i] + z0_i) / (load_r[i] * z0_i)

    # Incident voltage v_inc
    v_inc = v0 * z0 / (z0 + rg)

    # Masks
    mask_loc = (fault_classes != "none").astype(np.int32)
    mask_load = (~np.isin(fault_classes, ["short", "open"])).astype(np.int32)

    # Build LineParams objects
    params_list = []
    for i in range(n_samples):
        p = LineParams(
            length_m=float(length_m[i]),
            z0=float(z0[i]),
            vf=float(vf[i]),
            r_dc=float(cfg["physics"]["r_dc_ohm_per_m"]),
            k_skin=float(cfg["physics"]["k_skin"]),
            tan_delta=float(cfg["physics"]["tan_delta"]),
            loss_scale=float(loss_scale[i]),
            rise_time_s=float(rise_time_s[i]),
            rg=float(rg[i]),
            v0=float(v0[i]),
            fault_class=str(fault_classes[i]),
            fault_pos_m=float(0.0 if np.isnan(fault_pos_m[i]) else fault_pos_m[i]),
            fault_val=float(0.0 if np.isnan(fault_values[i]) else fault_values[i]),
            fault_width_m=float(0.0 if np.isnan(fault_widths_m[i]) else fault_widths_m[i]),
            load_class=str(load_classes[i]),
            load_r=float(50.0 if np.isnan(load_r[i]) else load_r[i]),
            load_l=float(0.0 if np.isnan(load_l[i]) else load_l[i]),
            load_c=float(0.0 if np.isnan(load_c[i]) else load_c[i]),
            load_tau_over_t=float(0.0 if np.isnan(load_tau_over_t[i]) else load_tau_over_t[i]),
            open_ohm=float(cfg["physics"]["open_ohm"]),
            short_ohm=float(cfg["physics"]["short_ohm"]),
        )
        params_list.append(p)

    # Build DataFrame
    df = pd.DataFrame({
        "sample_id": np.arange(n_samples),
        "split": "",  # will be assigned by caller
        "ood_subset": ood_type if ood_type is not None else "none",
        "length_m": length_m,
        "z0": z0,
        "vf": vf,
        "loss_scale": loss_scale,
        "rise_time_s": rise_time_s,
        "rg": rg,
        "v_inc": v_inc,
        "fault_class": fault_classes,
        "fault_class_id": [FAULT_TO_ID[f] for f in fault_classes],
        "fault_pos_m": fault_pos_m,
        "fault_pos_frac": fault_pos_frac,
        "fault_value": fault_values,
        "fault_width_m": fault_widths_m,
        "fault_gamma_nominal": fault_gamma_nom,
        "load_class": load_classes,
        "load_class_id": [LOAD_TO_ID[lc] for lc in load_classes],
        "load_R": load_r,
        "load_L": load_l,
        "load_C": load_c,
        "load_tau_over_T": load_tau_over_t,
        "mask_loc": mask_loc,
        "mask_load": mask_load,
    })

    return params_list, df


def generate_dataset_split(
    split_name: str,
    n_samples: int,
    seed: int,
    cfg: dict,
    out_dir: Path,
    ood_type: Optional[str] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Generate clean traces and metadata for a single split."""
    rng = np.random.default_rng(seed)
    params_list, meta_df = sample_batch_params(n_samples, rng, cfg, ood_type=ood_type)
    meta_df["split"] = split_name

    # Simulate traces
    t_out_norm, traces = simulate_batch(
        params_list,
        n_sim=cfg["signal"]["n_sim"],
        sim_dt_per_t=cfg["signal"]["sim_dt_per_T"],
        n_out=cfg["signal"]["n_out"],
        window_t=cfg["signal"]["window_T"],
        t_launch_t=cfg["signal"]["t_launch_T"],
        lossless=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / f"{split_name}.h5"
    meta_path = out_dir / f"{split_name}_meta.parquet"

    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset("traces", data=traces, compression="gzip", compression_opts=4)
        ds.attrs["t_grid"] = t_out_norm

    meta_df.to_parquet(meta_path, index=False)
    return traces, meta_df


# =========================================================================
# PyTorch Dataset and DataLoader
# =========================================================================

class TDRDataset(Dataset):
    """PyTorch Dataset for TDR traces with on-the-fly or cached noise."""

    def __init__(
        self,
        h5_path: Path,
        meta_path: Path,
        snr_db: Optional[float] = None,
        cached_noise: bool = False,
        seed: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.meta_df = pd.read_parquet(meta_path)
        self.snr_db = snr_db
        self.cached_noise = cached_noise
        self.seed = seed
        self.device = device

        with h5py.File(self.h5_path, "r") as f:
            self.clean_traces = np.array(f["traces"], dtype=np.float32)

        self.v_inc = self.meta_df["v_inc"].values.astype(np.float32)
        self.fault_cls = self.meta_df["fault_class_id"].values.astype(np.int64)
        self.fault_pos = self.meta_df["fault_pos_frac"].fillna(0.0).values.astype(np.float32)
        self.load_cls = self.meta_df["load_class_id"].values.astype(np.int64)

        # Regression targets for load: log10(R) and log10(tau/T)
        log_r = encode_load_r(np.clip(self.meta_df["load_R"].fillna(50.0).values, 1.0, 1000.0)).astype(np.float32)
        log_tau = encode_load_tau(np.clip(self.meta_df["load_tau_over_T"].fillna(0.1).values, 1e-4, 10.0)).astype(np.float32)
        self.load_reg = np.stack([log_r, log_tau], axis=1)

        self.mask_loc = self.meta_df["mask_loc"].values.astype(np.float32)
        self.mask_load = self.meta_df["mask_load"].values.astype(np.float32)

        # Precompute cached inputs if requested (e.g. for eval)
        if self.cached_noise:
            gen = torch.Generator().manual_seed(seed if seed is not None else 1234)
            clean_t = torch.from_numpy(self.clean_traces)
            v_inc_t = torch.from_numpy(self.v_inc)
            self.preprocessed_data = preprocess(
                clean_t,
                v_inc_t,
                snr_db=snr_db if snr_db is not None else 30.0,
                generator=gen,
                add_noise=True,
            )
        else:
            self.preprocessed_data = None

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.preprocessed_data is not None:
            x = self.preprocessed_data[idx]
        else:
            clean = torch.from_numpy(self.clean_traces[idx:idx+1])
            v_inc = torch.tensor([self.v_inc[idx]], dtype=torch.float32)
            # On-the-fly random noise
            x = preprocess(clean, v_inc, snr_db=self.snr_db, add_noise=True).squeeze(0)

        return {
            "x": x,  # [3, 1024]
            "fault_cls": torch.tensor(self.fault_cls[idx], dtype=torch.long),
            "fault_pos": torch.tensor(self.fault_pos[idx], dtype=torch.float32),
            "load_cls": torch.tensor(self.load_cls[idx], dtype=torch.long),
            "load_reg": torch.tensor(self.load_reg[idx], dtype=torch.float32),
            "mask_loc": torch.tensor(self.mask_loc[idx], dtype=torch.float32),
            "mask_load": torch.tensor(self.mask_load[idx], dtype=torch.float32),
            "length_m": torch.tensor(self.meta_df["length_m"].iloc[idx], dtype=torch.float32),
        }


class TensorTDRData:
    """Whole split held as tensors on one device; batches are preprocessed on that device.

    Replaces per-sample preprocessing inside a DataLoader (CPU-bound) for training.
    Uses exactly the same target encoding as TDRDataset and the same preprocess().
    """

    TARGET_KEYS = ("fault_cls", "fault_pos", "load_cls", "load_reg", "mask_loc", "mask_load")

    def __init__(
        self,
        h5_path: Path,
        meta_path: Path,
        device: torch.device,
        indices: Optional[np.ndarray] = None,
    ):
        base = TDRDataset(h5_path, meta_path, cached_noise=False)
        sel = np.arange(len(base)) if indices is None else np.asarray(indices)
        self.device = device
        self.clean = torch.from_numpy(base.clean_traces[sel]).to(device)
        self.v_inc = torch.from_numpy(base.v_inc[sel]).to(device)
        self.targets = {
            "fault_cls": torch.from_numpy(base.fault_cls[sel]).to(device),
            "fault_pos": torch.from_numpy(base.fault_pos[sel]).to(device),
            "load_cls": torch.from_numpy(base.load_cls[sel]).to(device),
            "load_reg": torch.from_numpy(base.load_reg[sel]).to(device),
            "mask_loc": torch.from_numpy(base.mask_loc[sel]).to(device),
            "mask_load": torch.from_numpy(base.mask_load[sel]).to(device),
        }
        self.meta_df = base.meta_df.iloc[sel].reset_index(drop=True)
        self.n = len(sel)
        self.fixed_inputs: Optional[torch.Tensor] = None

    def fix_noise(self, snr_db: float, seed: int) -> None:
        """Freeze one noisy copy (for validation) using the numpy reference path."""
        x = preprocess_numpy(self.clean.cpu().numpy(), self.v_inc.cpu().numpy(),
                             snr_db=snr_db, seed=seed, add_noise=True)
        self.fixed_inputs = torch.from_numpy(x).to(self.device)

    def batches(
        self,
        batch_size: int,
        shuffle: bool,
        perm_generator: Optional[torch.Generator] = None,
        noise_generator: Optional[torch.Generator] = None,
    ):
        """Yield (x [B,3,1024], targets dict). Fresh U[20,50] dB noise unless fix_noise() was called."""
        if shuffle:
            order = torch.randperm(self.n, generator=perm_generator).to(self.device)
        else:
            order = torch.arange(self.n, device=self.device)
        for start in range(0, self.n, batch_size):
            idx = order[start:start + batch_size]
            if self.fixed_inputs is not None:
                x = self.fixed_inputs[idx]
            else:
                x = preprocess(self.clean[idx], self.v_inc[idx], snr_db=None,
                               generator=noise_generator, add_noise=True, device=self.device)
            yield x, {k: v[idx] for k, v in self.targets.items()}


# =========================================================================
# Real Data Loader: WISE-CASING (NREL / LBNL)
# =========================================================================

def load_real_wise_casing(data_dir: Union[str, Path] = "data/real/wise_casing") -> Dict[str, Any]:
    """Load the raw WISE-CASING pulse-TDR records and their documentation (labels.yaml).

    Returns {"labels": dict, "traces": {filename: (t_us, v_mV)}}. No resampling or
    normalisation: these are pulse-type records, not step responses (see labels.yaml).
    """
    data_dir = Path(data_dir)
    with open(data_dir / "labels.yaml", "r", encoding="utf-8") as f:
        labels = yaml.safe_load(f)
    traces = {}
    for fname in labels.get("traces", {}):
        path = data_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            traces[fname] = (df.iloc[:, 0].to_numpy(np.float64), df.iloc[:, 1].to_numpy(np.float64))
    return {"labels": labels, "traces": traces}


def analyse_real_wise_casing(
    data_dir: Union[str, Path] = "data/real/wise_casing",
    vf_nominal: float = 0.66,
    min_gap_us: float = 0.08,
) -> Dict[str, Any]:
    """Physics consistency check on the real pulse-TDR records (no ML).

    For each record: t0 = incident pulse peak (largest positive sample before 0.12 us);
    echo = largest |v| at t > t0 + min_gap_us; dt = t_echo - t0 (round trip).
      * implied vf from the documented length: vf = 2 L / (c dt)
      * vf_est from the undamaged 80 m reference trace
      * damage location by echo-time ratio (velocity-free): d = L * dt_damage / dt_ref
      * a record is "consistent" if its implied vf is within 10% of vf_est
    Echo polarity gives the sign of the reflection coefficient (short -> negative).
    """
    raw = load_real_wise_casing(data_dir)
    labels, traces = raw["labels"], raw["traces"]
    ref = labels.get("reference_trace")
    rows = []
    for fname, info in labels["traces"].items():
        if fname not in traces:
            continue
        t, v = traces[fname]
        early = t < 0.12
        i0 = int(np.argmax(np.where(early, v, -np.inf)))
        t0 = float(t[i0])
        late = t > t0 + min_gap_us
        k = int(np.argmax(np.where(late, np.abs(v), -np.inf)))
        dt_us = float(t[k] - t0)
        L = float(info["cable_length_m"])
        rows.append({
            "file": fname,
            "resource": info.get("resource_name", ""),
            "documented_length_m": L,
            "documented_damage_m": (info.get("damage") or {}).get("location_m"),
            "t0_us": t0,
            "t_echo_us": float(t[k]),
            "echo_polarity": "negative" if v[k] < 0 else "positive",
            "echo_amplitude_rel": float(v[k] / max(v[i0], 1e-9)),
            "round_trip_us": dt_us,
            "implied_vf": 2.0 * L / (C_CONST * dt_us * 1e-6),
        })
    df = pd.DataFrame(rows)
    out: Dict[str, Any] = {"table": df, "raw": traces, "labels": labels}
    if ref in set(df["file"]):
        r = df[df["file"] == ref].iloc[0]
        vf_est = float(r["implied_vf"])
        df["distance_at_vf_est_m"] = vf_est * C_CONST * df["round_trip_us"] * 1e-6 / 2.0
        cons = (np.abs(df["implied_vf"] - vf_est) / vf_est) < 0.10
        # For a damaged record the strongest echo is the damage, not the termination,
        # so the length-based consistency test does not apply.
        df["consistent_with_vf_est"] = np.where(df["documented_damage_m"].notna(), None, cons)
        dmg = df[df["documented_damage_m"].notna()]
        loc = []
        for _, d in dmg.iterrows():
            d_ratio = float(r["documented_length_m"]) * d["round_trip_us"] / float(r["round_trip_us"])
            loc.append({
                "file": d["file"],
                "documented_damage_m": float(d["documented_damage_m"]),
                "estimated_by_time_ratio_m": d_ratio,
                "estimated_at_vf_nominal_m": vf_nominal * C_CONST * d["round_trip_us"] * 1e-6 / 2.0,
                "error_m": d_ratio - float(d["documented_damage_m"]),
                "echo_polarity": d["echo_polarity"],
            })
        out.update({"vf_est": vf_est, "damage_localisation": loc})
    out["table"] = df
    return out



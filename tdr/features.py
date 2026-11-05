"""Feature engineering and Conventional (rule-based) TDR solver.

Implements:
1. ConventionalTDR: baseline rule-based fault detection, location estimation,
   and load characterization with validation threshold tuning.
2. extract_features: ~94-100 engineered physics features from preprocessed
   (v_tilde, dv_tilde, coord) channels for LightGBM.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.signal import find_peaks, savgol_filter


def compute_mad(arr: np.ndarray) -> float:
    """Median Absolute Deviation (MAD) of 1D array."""
    med = np.median(arr)
    return float(np.median(np.abs(arr - med))) * 1.4826


class ConventionalTDR:
    """Rule-based Conventional TDR baseline."""

    def __init__(
        self,
        k_sigma: float = 4.0,
        min_sep_samples: int = 6,
        transient_return_samples: int = 25,
        z0_nominal: float = 50.0,
    ):
        self.k_sigma = k_sigma
        self.min_sep_samples = min_sep_samples
        self.transient_return_samples = transient_return_samples
        self.z0_nominal = z0_nominal

    def tune_on_val(
        self,
        val_traces: np.ndarray,
        val_meta: "pd.DataFrame",
        k_grid: List[float] = [3.0, 4.0, 5.0, 6.0],
    ) -> float:
        """Grid search over k_sigma on validation data to maximize fault F1 / accuracy."""
        best_k = self.k_sigma
        best_score = -1.0

        for k in k_grid:
            self.k_sigma = k
            preds = self.predict_batch(val_traces)
            # Score on fault accuracy
            true_faults = val_meta["fault_class"].values
            pred_faults = [p["fault_class"] for p in preds]
            acc = float(np.mean(np.array(true_faults) == np.array(pred_faults)))
            if acc > best_score:
                best_score = acc
                best_k = k

        self.k_sigma = best_k
        return best_k

    def predict_single(self, preprocessed_trace: np.ndarray) -> Dict[str, Union[str, float, bool]]:
        """Predict fault and load for a single sample [3, 1024].

        Channels: 0: v_tilde, 1: dv_tilde, 2: coord (t/T)
        """
        if preprocessed_trace.ndim == 1:
            v = preprocessed_trace
            dv = np.gradient(v)
            t = np.linspace(0.0, 2.5 * (len(v) - 1) / len(v), len(v), dtype=np.float32)
        else:
            v = preprocessed_trace[0]
            dv = preprocessed_trace[1]
            t = preprocessed_trace[2]
        n = len(v)

        # 1. Noise estimate on incident plateau [0.09, 0.11]
        idx_09 = int(np.round(0.09 / 2.5 * n))
        idx_11 = int(np.round(0.11 / 2.5 * n))
        sigma_hat = max(compute_mad(dv[idx_09:idx_11]), 1e-5)
        sigma_v = max(compute_mad(v[idx_09:idx_11]), 1e-4)

        # Savitzky-Golay smoothed trace for robust plateau / level measurement
        v_smooth = savgol_filter(v, 11, 2)

        idx_load_window = int(np.round(1.02 / 2.5 * n))

        # 2. Peak / edge detection for t/T in [0.11, 1.02]
        height_thresh = max(self.k_sigma * sigma_hat, 0.005)
        peaks, _ = find_peaks(np.abs(dv[idx_11:idx_load_window]), height=height_thresh, distance=self.min_sep_samples)
        peaks = peaks + idx_11

        valid_fault_peaks = []
        for p in peaks:
            b_idx = max(0, p - 8)
            a_idx = min(n, p + 8)
            step = float(np.median(v_smooth[p:a_idx]) - np.median(v_smooth[b_idx:p]))
            if abs(step) >= max(0.04, 3.0 * sigma_v):
                valid_fault_peaks.append((p, step))

        detected = len(valid_fault_peaks) > 0

        if not detected:
            fault_class = "none"
            pred_d_frac = 0.5
            pred_gamma_f = 0.0
        else:
            # Check for dominant hard fault (|step| >= 0.25)
            hard_steps = [p_step for p_step in valid_fault_peaks if abs(p_step[1]) >= 0.25]
            if len(hard_steps) > 0:
                chosen_p, chosen_step = hard_steps[0]
            else:
                chosen_p, chosen_step = valid_fault_peaks[0]

            b_start = max(0, chosen_p - 10)
            a_end = min(n, chosen_p + 15)
            v_before = float(np.median(v_smooth[b_start:chosen_p]))
            v_after = float(np.median(v_smooth[chosen_p:a_end]))
            pred_gamma_f = chosen_step

            # Half-amplitude crossing for location
            half_val = 0.5 * (v_before + v_after)
            w_start = max(0, chosen_p - 6)
            w_end = min(n, chosen_p + 7)
            local_sub = v[w_start:w_end]
            t_sub = t[w_start:w_end]
            if len(local_sub) > 1 and (local_sub[-1] != local_sub[0]):
                t_edge = float(np.interp(half_val, local_sub if local_sub[-1] > local_sub[0] else local_sub[::-1],
                                         t_sub if local_sub[-1] > local_sub[0] else t_sub[::-1]))
            else:
                t_edge = float(t[chosen_p])
            pred_d_frac = max(0.0, min(1.0, float(t_edge - 0.05)))

            # Transient return check within 25 samples
            ret_idx = min(n, chosen_p + self.transient_return_samples)
            v_ret = float(np.median(v_smooth[ret_idx-5:ret_idx]))
            returns_to_baseline = abs(v_ret - v_before) < 0.40 * max(abs(pred_gamma_f), 0.05)

            # Check for bipolar bump (opposite sign edge within 25 samples)
            window_peaks = [p for p, s in valid_fault_peaks if chosen_p < p <= chosen_p + self.transient_return_samples]
            has_opposite_edge = any(np.sign(dv[p]) != np.sign(dv[chosen_p]) for p in window_peaks)

            # DC level at end of trace
            idx_end_start = int(np.round(2.2 / 2.5 * n))
            v_trace_end = float(np.median(v_smooth[idx_end_start:]))

            # Fault classification rules
            if pred_gamma_f <= -0.25 or v_after < 0.45 or v_trace_end < 0.38:
                fault_class = "short"
            elif pred_gamma_f >= 0.25 or v_after > 1.35 or v_trace_end > 1.60:
                fault_class = "open"
            elif returns_to_baseline and has_opposite_edge:
                fault_class = "z0_section"
            elif returns_to_baseline and (pred_gamma_f < 0 or np.min(v_smooth[chosen_p:ret_idx]) < v_before - 0.05):
                fault_class = "shunt_C"
            elif pred_gamma_f < -0.04:
                fault_class = "shunt_R"
            elif pred_gamma_f > 0.04:
                fault_class = "series_R"
            else:
                fault_class = "none"

        # 3. Load identification (masked when fault is short or open)
        # Plateau after load echo: window t/T in [1.35, 1.80]
        idx_load_plat_start = int(np.round(1.35 / 2.5 * n))
        idx_load_plat_end = int(np.round(1.80 / 2.5 * n))
        v_load_plat = float(np.median(v_smooth[idx_load_plat_start:idx_load_plat_end]))

        # Level before load echo [0.92, 0.98]
        idx_pre_load_start = int(np.round(0.92 / 2.5 * n))
        idx_pre_load_end = int(np.round(0.98 / 2.5 * n))
        v_pre_load = float(np.median(v_smooth[idx_pre_load_start:idx_pre_load_end]))

        step_load = v_load_plat - v_pre_load
        gamma_l_hat = (v_load_plat - 1.0)

        # Transient right after load arrival [1.05, 1.25]
        idx_trans_start = int(np.round(1.05 / 2.5 * n))
        idx_trans_end = int(np.round(1.25 / 2.5 * n))
        v_trans = v_smooth[idx_trans_start:idx_trans_end]

        peak_trans = float(np.max(v_trans))
        trough_trans = float(np.min(v_trans))
        overshoot = peak_trans - max(v_load_plat, v_pre_load)
        undershoot = min(v_load_plat, v_pre_load) - trough_trans

        # Short / Open / Matched Classification
        if step_load < -0.40 or v_load_plat < 0.45:
            load_class = "short"
        elif step_load > 0.40 or v_load_plat > 1.45:
            load_class = "open"
        elif abs(step_load) < 0.18 and overshoot < 0.12 and undershoot < 0.12:
            load_class = "matched"
        elif overshoot > 0.10 and overshoot > undershoot * 1.3:
            load_class = "RL"
        elif undershoot > 0.10 and undershoot > overshoot * 1.3:
            load_class = "RC"
        else:
            load_class = "R"

        # Fit tau/T if RL or RC
        tau_over_t_hat = 0.05
        if load_class in ("RL", "RC"):
            y_tail = np.abs(v[idx_trans_start:idx_trans_end] - v_load_plat)
            valid = y_tail > max(0.02, 2.0 * sigma_v)
            if np.sum(valid) >= 5:
                t_fit = t[idx_trans_start:idx_trans_end][valid]
                try:
                    slope, _ = np.polyfit(t_fit, np.log(y_tail[valid]), 1)
                    if slope < -1e-4:
                        tau_over_t_hat = float(np.clip(-1.0 / slope, 0.01, 1.0))
                except Exception:
                    pass

        # Estimate R_L
        g_clamped = max(-0.95, min(0.95, gamma_l_hat))
        r_l_hat = float(np.clip(self.z0_nominal * (1.0 + g_clamped) / max(1e-4, (1.0 - g_clamped)), 2.0, 2000.0))

        return {
            "fault_class": fault_class,
            "fault_pos_frac": pred_d_frac,
            "detected": detected,
            "gamma_f": pred_gamma_f,
            "load_class": load_class,
            "load_R": r_l_hat,
            "load_tau_over_T": tau_over_t_hat,
            "sigma_hat": sigma_hat,
        }

    def predict_batch(self, preprocessed_batch: np.ndarray) -> List[Dict[str, Union[str, float, bool]]]:
        """Predict on batch [B, 3, 1024]."""
        return [self.predict_single(preprocessed_batch[i]) for i in range(len(preprocessed_batch))]


# =========================================================================
# Feature Extraction for LightGBM
# =========================================================================

def extract_features_single(preprocessed_trace: np.ndarray) -> np.ndarray:
    """Extract ~94-100 engineered features from single sample [3, 1024].

    Returns 1D array of float32 features.
    """
    v = preprocessed_trace[0]
    dv = preprocessed_trace[1]
    t = preprocessed_trace[2]
    n = len(v)

    feats = []

    # 1. 64 bin-averages of v_tilde
    bin_size = n // 64  # 16 samples per bin
    for b in range(64):
        feats.append(float(np.mean(v[b * bin_size : (b + 1) * bin_size])))

    # 2. Final level, min, max, argmin, argmax of v over t/T > 0.11
    idx_11 = int(np.round(0.11 / 2.5 * n))
    idx_23 = int(np.round(2.3 / 2.5 * n))
    v_final = float(np.median(v[idx_23:]))
    v_after_11 = v[idx_11:]
    t_after_11 = t[idx_11:]

    v_min = float(np.min(v_after_11))
    v_max = float(np.max(v_after_11))
    t_argmin = float(t_after_11[np.argmin(v_after_11)])
    t_argmax = float(t_after_11[np.argmax(v_after_11)])
    feats.extend([v_final, v_min, v_max, t_argmin, t_argmax])

    # 3. Mean of v over [0.11, 1.0] and [1.0, 2.5]
    idx_10 = int(np.round(1.0 / 2.5 * n))
    feats.append(float(np.mean(v[idx_11:idx_10])))
    feats.append(float(np.mean(v[idx_10:])))

    # 4. Top 4 |dv_tilde| peaks: time, signed height, local step size, half-max width (16 features)
    idx_09 = int(np.round(0.09 / 2.5 * n))
    sigma_hat = max(compute_mad(dv[idx_09:idx_11]), 1e-5)
    peaks, _ = find_peaks(np.abs(dv[idx_11:]), distance=6)
    peaks = peaks + idx_11

    peak_records = []
    for p in peaks:
        h = float(dv[p])
        abs_h = abs(h)
        # local step size
        b_s = max(0, p - 8)
        a_s = min(n, p + 8)
        step = float(np.median(v[p:a_s]) - np.median(v[b_s:p]))
        # half-max width
        half_h = 0.5 * abs_h
        left = p
        while left > 0 and abs(dv[left]) >= half_h:
            left -= 1
        right = p
        while right < n - 1 and abs(dv[right]) >= half_h:
            right += 1
        width = float(t[right] - t[left])
        peak_records.append((abs_h, float(t[p]), h, step, width))

    # Sort descending by |height|
    peak_records.sort(key=lambda x: x[0], reverse=True)
    top_4 = peak_records[:4]

    for i in range(4):
        if i < len(top_4):
            _, t_p, h_p, step_p, w_p = top_4[i]
            feats.extend([t_p, h_p, step_p, w_p])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

    # 5. First edge Gamma_hat and Z/Z0
    if len(peaks) > 0:
        first_p = peaks[0]
        b_s = max(0, first_p - 8)
        a_s = min(n, first_p + 8)
        gamma_1 = float(np.median(v[first_p:a_s]) - np.median(v[b_s:first_p]))
        # z / z0 = (1 + gamma) / (1 - gamma)
        g_clamped = max(-0.99, min(0.99, gamma_1))
        z_over_z0 = float((1.0 + g_clamped) / (1.0 - g_clamped))
    else:
        gamma_1 = 0.0
        z_over_z0 = 1.0
    feats.extend([gamma_1, z_over_z0])

    # 6. Load region features: level over [0.90, 0.97], overshoot, undershoot, fitted tau/T
    idx_90 = int(np.round(0.90 / 2.5 * n))
    idx_97 = int(np.round(0.97 / 2.5 * n))
    level_90_97 = float(np.median(v[idx_90:idx_97]))

    idx_load_window = int(np.round(1.02 / 2.5 * n))
    idx_load_end = int(np.round(1.50 / 2.5 * n))
    v_post_load = v[idx_load_window:idx_load_end]
    overshoot = float(np.max(v_post_load) - v_final)
    undershoot = float(v_final - np.min(v_post_load))

    # fitted tau/T
    y_tail = np.abs(v_post_load - v_final)
    valid = y_tail > max(0.01, 2.0 * sigma_hat)
    tau_hat = 0.05
    if np.sum(valid) >= 5:
        try:
            t_fit = t[idx_load_window:idx_load_end][valid]
            slope, _ = np.polyfit(t_fit, np.log(y_tail[valid]), 1)
            if slope < -1e-4:
                tau_hat = float(np.clip(-1.0 / slope, 0.01, 1.0))
        except Exception:
            pass
    feats.extend([level_90_97, overshoot, undershoot, tau_hat])

    # 7. sigma_hat
    feats.append(sigma_hat)

    return np.array(feats, dtype=np.float32)


def extract_features(preprocessed_batch: np.ndarray) -> np.ndarray:
    """Extract engineered features for a batch [B, 3, 1024] -> [B, N_feats]."""
    b = len(preprocessed_batch)
    # Extract first sample to determine feature dimension
    sample_feat = extract_features_single(preprocessed_batch[0])
    n_feats = len(sample_feat)

    feats_all = np.empty((b, n_feats), dtype=np.float32)
    feats_all[0] = sample_feat
    for i in range(1, b):
        feats_all[i] = extract_features_single(preprocessed_batch[i])

    return feats_all

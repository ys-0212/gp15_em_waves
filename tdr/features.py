"""Feature engineering and Conventional (rule-based) TDR solver.

Implements:
1. ConventionalTDR: baseline rule-based fault detection, location estimation,
   and load characterization with validation threshold tuning.
2. extract_features: ~94-100 engineered physics features from preprocessed
   (v_tilde, dv_tilde, coord) channels for LightGBM.
"""

import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import find_peaks, savgol_filter


def compute_mad(arr: np.ndarray) -> float:
    """Median Absolute Deviation (MAD) of 1D array."""
    med = np.median(arr)
    return float(np.median(np.abs(arr - med))) * 1.4826


DEFAULT_CONV_PARAMS: Dict[str, float] = {
    # Defaults = values selected by tune_on_val() on the quick validation set at 30 dB.
    # The pipeline always re-tunes on the validation split of the run it is part of.
    # Stage 1: edge detection
    "k_sigma": 3.0,          # |dv| peak threshold in units of derivative noise
    "c_step": 1.0,           # accepted step >= max(min_step, c_step * sigma_v)
    "min_step": 0.02,
    # Stage 2: fault decision
    "a_short": 0.45,         # L_after / L_before below this -> short
    "b_open": 1.55,          # L_after / L_before above this -> open
    "return_ratio": 0.5,     # |net| < return_ratio * |excursion| -> transient
    "rho_bipolar": 0.8,      # rising/falling dv ratio separating z0_section from shunt_C
    # Stage 3: load decision
    "g_matched": 0.12,       # |Gamma_dc| below this (and no transient) -> matched
    "s_transient": 0.05,     # over/undershoot above this -> RL / RC
    "r_open": 500.0,         # R_hat above this -> open
    "r_short": 8.0,          # R_hat below this -> short (includes un-settled skin-effect creep)
}

_FAULTS = np.array(["none", "short", "open", "shunt_R", "shunt_C", "series_R", "z0_section"])
_LOADS = np.array(["open", "short", "matched", "R", "RL", "RC"])


def _idx(t_over_T: float, n: int) -> int:
    """Index on the [0, 2.5) grid with n samples."""
    return int(np.clip(np.round(t_over_T / 2.5 * n), 0, n - 1))


class ConventionalTDR:
    """Rule-based conventional TDR baseline (no machine learning).

    Physics used:
      * edge time -> location: d/l = t_edge/T - 0.05
      * level after the fault edge, relative to the level before it, decides hard
        faults (short -> ~0, open -> ~2; lossy smearing keeps far shorts above 0);
        the end-of-trace level is never used for the fault decision, because it
        reflects the load, not the fault
      * soft faults by edge shape: persistent step down/up -> shunt_R/series_R,
        transient bump -> z0_section, sharp dip with slow recovery -> shunt_C
      * load from the DC-settled end level: Z_in = Z0 * L_end / (2 - L_end),
        de-embedded through the estimated soft fault and the line's r_dc * l
      * load reactance from over/undershoot shape and an exponential fit of the tail

    Prediction is split into analyse() (per-trace descriptors, depends only on
    the detection parameters) and classify() (vectorised rules), so the
    thresholds can be tuned jointly on the validation set cheaply.
    """

    def __init__(
        self,
        k_sigma: float = 3.0,
        min_sep_samples: int = 6,
        transient_return_samples: int = 25,
        z0_nominal: float = 50.0,
        r_dc_ohm_per_m: float = 0.04,
        params: Optional[Dict[str, float]] = None,
    ):
        self.params = dict(DEFAULT_CONV_PARAMS)
        if params:
            self.params.update({k: float(v) for k, v in params.items() if k in DEFAULT_CONV_PARAMS})
        else:
            self.params["k_sigma"] = float(k_sigma)
        self.min_sep_samples = int(min_sep_samples)
        self.transient_return_samples = int(transient_return_samples)
        self.z0_nominal = float(z0_nominal)
        self.r_dc = float(r_dc_ohm_per_m)

    # ------------------------------------------------------------------ io
    @property
    def k_sigma(self) -> float:
        return self.params["k_sigma"]

    def to_dict(self) -> Dict[str, float]:
        return {
            "params": dict(self.params),
            "best_k_sigma": self.params["k_sigma"],
            "min_sep_samples": self.min_sep_samples,
            "transient_return_samples": self.transient_return_samples,
            "z0_nominal": self.z0_nominal,
            "r_dc_ohm_per_m": self.r_dc,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ConventionalTDR":
        params = d.get("params")
        if params is None:  # legacy json with only k_sigma
            params = {"k_sigma": d.get("best_k_sigma", d.get("k_sigma", 4.0))}
        return cls(
            min_sep_samples=d.get("min_sep_samples", 6),
            transient_return_samples=d.get("transient_return_samples", 25),
            z0_nominal=d.get("z0_nominal", 50.0),
            r_dc_ohm_per_m=d.get("r_dc_ohm_per_m", 0.04),
            params=params,
        )

    # ------------------------------------------------------------- stage 1
    def analyse_single(self, x: np.ndarray, length_m: Optional[float] = None) -> Dict[str, float]:
        """Compute rule descriptors for one preprocessed trace [3, 1024] (or [1024])."""
        if x.ndim == 1:
            v = x.astype(np.float64)
            dv = savgol_filter(v, 11, 2, deriv=1)
        else:
            v = x[0].astype(np.float64)
            dv = x[1].astype(np.float64)
        n = len(v)
        k_sigma = self.params["k_sigma"]
        c_step = self.params["c_step"]
        min_step = self.params["min_step"]

        i011, i100, i245 = _idx(0.11, n), _idx(1.00, n), _idx(2.45, n)
        diffs = np.diff(v[i011:i245])
        sigma_v = max(1.4826 * float(np.median(np.abs(diffs - np.median(diffs)))) / np.sqrt(2.0), 1e-4)
        dseg = dv[i011:i245]
        sigma_dv = max(1.4826 * float(np.median(np.abs(dseg - np.median(dseg)))), 1e-6)
        v_s = savgol_filter(v, 11, 2)

        def med(a: int, b: int) -> float:
            a, b = max(0, a), min(n, b)
            return float(np.median(v_s[a:b])) if b > a else float(v_s[min(a, n - 1)])

        out: Dict[str, float] = {"sigma_v": sigma_v, "has_cand": 0.0}

        # ---- fault edge candidates in t/T in [0.11, 1.00]
        peaks, _ = find_peaks(np.abs(dv[i011:i100]), height=k_sigma * sigma_dv,
                              distance=self.min_sep_samples)
        peaks = peaks + i011
        thr = max(min_step, c_step * sigma_v)
        cand = None
        for p in peaks:
            delta = med(p + 3, p + 13) - med(p - 13, p - 3)
            if abs(delta) >= thr:
                cand = int(p)
                break

        # width of the window in which a short section / capacitor transient returns
        w_frac = (3.0 / length_m) if length_m else 0.15
        w_samp = int(np.ceil(w_frac / 2.5 * n)) + 3 + self.transient_return_samples // 5

        if cand is not None:
            p = cand
            l_before = med(p - 13, p - 3)
            # half-amplitude crossing for the edge time
            a0, a1 = max(0, p - 13), min(n, p + 13)
            l_near = med(p + 3, p + 13)
            half = 0.5 * (l_before + l_near)
            seg = v_s[a0:a1] - half
            sgn = np.sign(seg)
            cross = np.where(np.diff(sgn) != 0)[0]
            if len(cross):
                j = cross[np.argmin(np.abs(cross + a0 - p))]
                y0, y1 = seg[j], seg[j + 1]
                frac = 0.0 if y1 == y0 else float(-y0 / (y1 - y0))
                t_edge = (a0 + j + frac) * 2.5 / n
            else:
                t_edge = p * 2.5 / n
            # settled level after the edge (before the load echo)
            s0 = _idx(t_edge + 0.05, n)
            s1 = _idx(min(t_edge + 0.25, 1.00), n)
            l_after = med(s0, s1) if s1 - s0 >= 5 else l_near
            # transient descriptors
            win = v_s[p:min(n, p + w_samp)]
            up_exc = float(np.max(win) - l_before)
            dn_exc = float(np.min(win) - l_before)
            exc = up_exc if abs(up_exc) >= abs(dn_exc) else dn_exc
            dwin = dv[p:min(n, p + w_samp + 6)]
            pos_pk = float(np.max(dwin)) if len(dwin) else 0.0
            neg_pk = float(-np.min(dwin)) if len(dwin) else 0.0
            out.update({
                "has_cand": 1.0, "t_edge": t_edge, "l_before": l_before,
                "l_after": l_after, "net": l_after - l_before, "exc": exc,
                "pos_pk": pos_pk, "neg_pk": neg_pk, "gamma_hat": l_near - l_before,
            })
        else:
            out.update({"t_edge": np.nan, "l_before": 1.0, "l_after": 1.0, "net": 0.0,
                        "exc": 0.0, "pos_pk": 0.0, "neg_pk": 0.0, "gamma_hat": 0.0})

        # ---- load descriptors
        l_end = med(_idx(2.30, n), _idx(2.45, n))
        l_pre = med(_idx(0.92, n), _idx(0.98, n))
        w0, w1 = _idx(1.03, n), _idx(1.40, n)
        wl = v_s[w0:w1]
        over = float(np.max(wl) - max(l_pre, l_end))
        under = float(min(l_pre, l_end) - np.min(wl))
        # exponential tail fit from the extremum onward
        tau = np.nan
        j_ext = int(np.argmax(np.abs(wl - l_end)))
        y = np.abs(v_s[w0 + j_ext:w1] - l_end)
        ok = y > max(0.02, 3.0 * sigma_v)
        if ok.sum() >= 5:
            tt = (np.arange(w0 + j_ext, w1) * 2.5 / n)[ok]
            slope = np.polyfit(tt, np.log(y[ok]), 1)[0]
            if slope < -1e-3:
                tau = float(np.clip(-1.0 / slope, 0.005, 1.0))
        out.update({"l_end": l_end, "l_pre": l_pre, "over": over, "under": under,
                    "tau_hat": tau, "length_m": float(length_m) if length_m else np.nan})
        return out

    def _analyse_chunk(self, X: np.ndarray, lengths: Optional[np.ndarray]) -> List[Dict[str, float]]:
        return [self.analyse_single(X[i], None if lengths is None else float(lengths[i]))
                for i in range(len(X))]

    def analyse_batch(self, X: np.ndarray, lengths: Optional[np.ndarray] = None,
                      n_jobs: int = -1) -> Dict[str, np.ndarray]:
        """Descriptors for a batch; parallel over CPU cores for large batches."""
        if len(X) >= 2000 and n_jobs != 1:
            chunks = range(0, len(X), 1000)
            parts = Parallel(n_jobs=n_jobs)(
                delayed(self._analyse_chunk)(X[i:i + 1000], None if lengths is None else lengths[i:i + 1000])
                for i in chunks)
            rows = [r for part in parts for r in part]
        else:
            rows = self._analyse_chunk(X, lengths)
        return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}

    # ------------------------------------------------------------- stage 2
    def classify(self, D: Dict[str, np.ndarray], params: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """Vectorised rules on descriptors from analyse_batch()."""
        P = dict(self.params)
        if params:
            P.update(params)
        z0 = self.z0_nominal
        n = len(D["has_cand"])
        cand = D["has_cand"] > 0.5
        la = D["l_after"] / np.maximum(D["l_before"], 1e-3)   # relative level after the edge
        net, exc = D["net"], D["exc"]

        fault = np.full(n, "none", dtype=object)
        short = cand & (la < P["a_short"])
        opn = cand & ~short & (la > P["b_open"])
        soft = cand & ~short & ~opn
        returns = np.abs(net) < P["return_ratio"] * np.maximum(np.abs(exc), 1e-9)
        bip_ratio = np.minimum(D["pos_pk"], D["neg_pk"]) / np.maximum(np.maximum(D["pos_pk"], D["neg_pk"]), 1e-9)
        fault[short] = "short"
        fault[opn] = "open"
        tr = soft & returns
        fault[tr & (exc > 0)] = "z0_section"
        fault[tr & (exc <= 0) & (bip_ratio >= P["rho_bipolar"])] = "z0_section"
        fault[tr & (exc <= 0) & (bip_ratio < P["rho_bipolar"])] = "shunt_C"
        fault[soft & ~returns & (net < 0)] = "shunt_R"
        fault[soft & ~returns & (net >= 0)] = "series_R"

        d_frac = np.where(cand, np.clip(D["t_edge"] - 0.05, 0.0, 1.0), 0.5)

        # ---- load: DC de-embedding of the input impedance
        l_end = np.clip(D["l_end"], 1e-3, 1.999)
        z_in = z0 * l_end / (2.0 - l_end)
        g = np.clip(D["gamma_hat"], -0.95, 0.95)
        z_f = z0 * (1 + g) / (1 - g)                      # impedance seen at the fault
        r_series = np.maximum(z_f - z0, 0.0)             # series_R: Z = Z0 + R_f
        r_shunt = np.where(z0 - z_f > 1e-6, z0 * z_f / np.maximum(z0 - z_f, 1e-6), np.inf)  # shunt: Z0 || R_f
        z_load = z_in.copy()
        m = fault == "series_R"
        z_load[m] = z_in[m] - r_series[m]
        m = fault == "shunt_R"
        inv = 1.0 / np.maximum(z_in[m], 1e-6) - 1.0 / r_shunt[m]
        z_load[m] = np.where(inv > 1e-9, 1.0 / np.maximum(inv, 1e-9), 1e6)
        lengths = np.nan_to_num(D["length_m"], nan=0.0)
        r_hat = z_load - self.r_dc * lengths
        gamma_dc = (r_hat - z0) / (r_hat + z0)

        load = np.full(n, "R", dtype=object)
        trans_rl = (D["over"] > P["s_transient"]) & (D["over"] > 1.3 * D["under"])
        trans_rc = (D["under"] > P["s_transient"]) & (D["under"] > 1.3 * D["over"])
        load[trans_rl] = "RL"
        load[trans_rc] = "RC"
        quiet = ~trans_rl & ~trans_rc
        load[quiet & (np.abs(gamma_dc) < P["g_matched"])] = "matched"
        load[r_hat > P["r_open"]] = "open"
        load[(r_hat < P["r_short"]) & ~trans_rl] = "short"
        # Behind a predicted hard fault the load is physically unobservable; the rule
        # output is still returned (and scored only where the true load is observable),
        # consistent with how the ML models are evaluated.

        tau = np.where(np.isfinite(D["tau_hat"]), D["tau_hat"], 0.05)
        return {
            "fault_class": fault.astype(str),
            "fault_pos_frac": d_frac,
            "detected": cand,
            "gamma_f": D["gamma_hat"],
            "load_class": load.astype(str),
            "load_R": np.clip(np.where(np.isfinite(r_hat), r_hat, 2000.0), 2.0, 2000.0),
            "load_tau_over_T": np.clip(tau, 0.005, 1.0),
            "sigma_hat": D["sigma_v"],
        }

    # --------------------------------------------------------------- api
    def predict_batch(self, X: np.ndarray, lengths: Optional[np.ndarray] = None) -> List[Dict[str, Union[str, float, bool]]]:
        """Predict on a preprocessed batch [B, 3, 1024]; lengths (m) improve the load estimate."""
        out = self.classify(self.analyse_batch(X, lengths))
        keys = list(out)
        return [{k: (out[k][i].item() if hasattr(out[k][i], "item") else out[k][i]) for k in keys}
                for i in range(len(X))]

    def predict_single(self, x: np.ndarray, length_m: Optional[float] = None) -> Dict[str, Union[str, float, bool]]:
        return self.predict_batch(x[None] if x.ndim == 2 else x[None, :],
                                  None if length_m is None else np.array([length_m]))[0]

    def tune_on_val(
        self,
        val_X: np.ndarray,
        val_meta: "pd.DataFrame",
        k_grid: List[float] = (3.0, 4.0, 5.0, 6.0),
        c_grid: List[float] = (1.0, 1.5, 2.0, 3.0),
    ) -> Dict[str, float]:
        """Grid-search all thresholds on validation data (never on test).

        Objective: fault macro-F1 + load macro-F1 (load scored where the load is
        observable, i.e. mask_load). Detection parameters are searched in an
        outer loop; decision thresholds in a cheap inner loop on cached descriptors.
        """
        from itertools import product
        from sklearn.metrics import f1_score

        y_f = val_meta["fault_class"].astype(str).values
        y_l = val_meta["load_class"].astype(str).values
        m_l = val_meta["mask_load"].astype(bool).values
        lengths = val_meta["length_m"].values
        fault_grid = list(product([0.35, 0.40, 0.45, 0.50, 0.55], [1.35, 1.45, 1.55, 1.65],
                                  [0.3, 0.4, 0.5], [0.4, 0.6, 0.8]))
        load_grid = list(product([0.05, 0.08, 0.10, 0.12], [0.03, 0.05, 0.10], [500.0, 700.0, 1000.0], [4.0, 6.0, 8.0, 10.0]))

        best = (-1.0, None)
        for k, c in product(k_grid, c_grid):
            self.params.update({"k_sigma": float(k), "c_step": float(c)})
            D = self.analyse_batch(val_X, lengths)
            best_f = (-1.0, None)
            for a, b, r, rho in fault_grid:
                p = {"a_short": a, "b_open": b, "return_ratio": r, "rho_bipolar": rho}
                pred = self.classify(D, p)["fault_class"]
                s = f1_score(y_f, pred, average="macro")
                if s > best_f[0]:
                    best_f = (s, p)
            best_l = (-1.0, None)
            for g, s_t, ro, rs in load_grid:
                p = dict(best_f[1], g_matched=g, s_transient=s_t, r_open=ro, r_short=rs)
                pred = self.classify(D, p)["load_class"]
                s = f1_score(y_l[m_l], pred[m_l], average="macro")
                if s > best_l[0]:
                    best_l = (s, p)
            score = best_f[0] + best_l[0]
            if score > best[0]:
                best = (score, dict(best_l[1], k_sigma=float(k), c_step=float(c)))
        self.params.update(best[1])
        return dict(self.params)


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


def _feature_names() -> list:
    """Human-readable names, in the exact order produced by extract_features_single()."""
    names = [f"v_bin{b:02d} (t/T={b * 2.5 / 64:.2f}-{(b + 1) * 2.5 / 64:.2f})" for b in range(64)]
    names += ["v_final", "v_min", "v_max", "t_argmin", "t_argmax", "v_mean[0.11,1.0]", "v_mean[1.0,2.5]"]
    for i in range(1, 5):
        names += [f"edge{i}_time", f"edge{i}_dv_height", f"edge{i}_step", f"edge{i}_width"]
    names += ["first_edge_gamma", "first_edge_Z/Z0", "level[0.90,0.97]", "load_overshoot",
              "load_undershoot", "load_tau_fit", "noise_sigma"]
    return names


FEATURE_NAMES = _feature_names()


def _extract_chunk(chunk: np.ndarray) -> np.ndarray:
    return np.stack([extract_features_single(chunk[i]) for i in range(len(chunk))]).astype(np.float32)


def extract_features(preprocessed_batch: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    """Extract engineered features for a batch [B, 3, 1024] -> [B, N_feats].

    Large batches are processed in parallel chunks (joblib); results are identical
    to the serial path because each sample is processed independently.
    """
    b = len(preprocessed_batch)
    if b >= 2000 and n_jobs != 1:
        parts = Parallel(n_jobs=n_jobs)(
            delayed(_extract_chunk)(preprocessed_batch[i:i + 1000]) for i in range(0, b, 1000))
        return np.concatenate(parts, axis=0)
    return _extract_chunk(preprocessed_batch)


FEATURE_VERSION = "v2"


def array_fingerprint(*arrays: np.ndarray) -> str:
    """Short content hash used to key feature caches (changes if the data changes)."""
    h = hashlib.md5()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:12]


def cached_features(
    cache_dir: Union[str, Path],
    clean_traces: np.ndarray,
    v_inc: np.ndarray,
    snr_db: Optional[float],
    noise_seed: int,
    preprocess_fn: Callable[..., np.ndarray],
) -> np.ndarray:
    """Features for (clean traces, noise draw), cached on disk under data/cache/.

    The key contains the data fingerprint, SNR, noise seed and FEATURE_VERSION, so a
    cache entry can never be reused for different data or a different noise draw.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snr_tag = "rand" if snr_db is None else f"{float(snr_db):g}"
    key = f"feat_{FEATURE_VERSION}_{array_fingerprint(clean_traces, v_inc)}_snr{snr_tag}_seed{noise_seed}.npy"
    path = cache_dir / key
    if path.exists():
        return np.load(path)
    x = preprocess_fn(clean_traces, v_inc, snr_db=snr_db, seed=noise_seed, add_noise=True)
    feats = extract_features(x)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, feats)
    os.replace(tmp, path)
    return feats

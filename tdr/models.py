"""Model architectures: LightGBM Suite and Multi-task 1D-ResNet.

Adheres strictly to Section 2.5 and 2.6 specifications.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tdr.features import extract_features
from tdr.data import (
    FAULT_CLASSES, LOAD_CLASSES, FAULT_TO_ID, LOAD_TO_ID,
    encode_load_r, decode_load_r, encode_load_tau, decode_load_tau
)


# =========================================================================
# LightGBM Multi-Model Suite
# =========================================================================

class LGBMSuite:
    """Suite of 4 LightGBM models solving T1, T2, and T3."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.lgbm_params = self.cfg.get("lgbm", {})
        self.model_fault = None
        self.model_loc = None
        self.model_load_cls = None
        self.model_load_r = None
        self.model_load_tau = None
        self.feature_names = None

    def fit(
        self,
        X_train: np.ndarray,
        y_fault_train: np.ndarray,
        y_loc_train: np.ndarray,
        y_load_cls_train: np.ndarray,
        y_load_r_train: np.ndarray,
        y_load_tau_train: np.ndarray,
        mask_loc_train: np.ndarray,
        mask_load_train: np.ndarray,
        X_val: np.ndarray,
        y_fault_val: np.ndarray,
        y_loc_val: np.ndarray,
        y_load_cls_val: np.ndarray,
        y_load_r_val: np.ndarray,
        y_load_tau_val: np.ndarray,
        mask_loc_val: np.ndarray,
        mask_load_val: np.ndarray,
        n_estimators: Optional[int] = None,
    ):
        """Train all 4 LightGBM models with early stopping on validation set."""
        n_est = n_estimators if n_estimators is not None else self.lgbm_params.get("n_estimators", 1000)
        lr = self.lgbm_params.get("learning_rate", 0.05)
        num_leaves = self.lgbm_params.get("num_leaves", 63)
        min_child = self.lgbm_params.get("min_child_samples", 20)
        early_stop = self.lgbm_params.get("early_stopping_rounds", 50)
        seed = 42

        # 1. Fault Classifier (7 classes)
        print("Training LightGBM Fault Classifier...")
        self.model_fault = lgb.LGBMClassifier(
            n_estimators=n_est,
            learning_rate=lr,
            num_leaves=num_leaves,
            min_child_samples=min_child,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        self.model_fault.fit(
            X_train,
            y_fault_train,
            eval_set=[(X_val, y_fault_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stop, verbose=False)],
        )

        # 2. Location Regressor (on mask_loc == 1)
        print("Training LightGBM Location Regressor...")
        m_loc_tr = mask_loc_train.astype(bool)
        m_loc_va = mask_loc_val.astype(bool)
        self.model_loc = lgb.LGBMRegressor(
            n_estimators=n_est,
            learning_rate=lr,
            num_leaves=num_leaves,
            min_child_samples=min_child,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        self.model_loc.fit(
            X_train[m_loc_tr],
            y_loc_train[m_loc_tr],
            eval_set=[(X_val[m_loc_va], y_loc_val[m_loc_va])],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stop, verbose=False)],
        )

        # 3. Load Classifier (on mask_load == 1)
        print("Training LightGBM Load Classifier...")
        m_load_tr = mask_load_train.astype(bool)
        m_load_va = mask_load_val.astype(bool)
        self.model_load_cls = lgb.LGBMClassifier(
            n_estimators=n_est,
            learning_rate=lr,
            num_leaves=num_leaves,
            min_child_samples=min_child,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        self.model_load_cls.fit(
            X_train[m_load_tr],
            y_load_cls_train[m_load_tr],
            eval_set=[(X_val[m_load_va], y_load_cls_val[m_load_va])],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stop, verbose=False)],
        )

        # 4. Load Regression (log10(R) and log10(tau/T))
        print("Training LightGBM Load Regressors...")
        # R_L regressor is trained on resistive/reactive/matched loads with finite resistance
        finite_load_ids = [LOAD_TO_ID["matched"], LOAD_TO_ID["R"], LOAD_TO_ID["RL"], LOAD_TO_ID["RC"]]
        m_r_tr = m_load_tr & np.isin(y_load_cls_train, finite_load_ids) & (~np.isnan(y_load_r_train)) & (y_load_r_train > 0)
        m_r_va = m_load_va & np.isin(y_load_cls_val, finite_load_ids) & (~np.isnan(y_load_r_val)) & (y_load_r_val > 0)
        self.model_load_r = lgb.LGBMRegressor(
            n_estimators=n_est,
            learning_rate=lr,
            num_leaves=num_leaves,
            min_child_samples=min_child,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        self.model_load_r.fit(
            X_train[m_r_tr],
            encode_load_r(y_load_r_train[m_r_tr]),
            eval_set=[(X_val[m_r_va], encode_load_r(y_load_r_val[m_r_va]))],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stop, verbose=False)],
        )

        m_tau_tr = m_load_tr & (~np.isnan(y_load_tau_train)) & (y_load_tau_train > 0)
        m_tau_va = m_load_va & (~np.isnan(y_load_tau_val)) & (y_load_tau_val > 0)
        self.model_load_tau = lgb.LGBMRegressor(
            n_estimators=n_est,
            learning_rate=lr,
            num_leaves=num_leaves,
            min_child_samples=min_child,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        self.model_load_tau.fit(
            X_train[m_tau_tr],
            encode_load_tau(y_load_tau_train[m_tau_tr]),
            eval_set=[(X_val[m_tau_va], encode_load_tau(y_load_tau_val[m_tau_va]))],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stop, verbose=False)],
        )

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict all tasks on feature matrix X [B, N_feats]."""
        if hasattr(self.model_fault, "predict_proba"):
            fault_probs = self.model_fault.predict_proba(X)
        else:
            fault_probs = self.model_fault.predict(X)
        fault_pred_id = np.argmax(fault_probs, axis=1)

        loc_pred = np.clip(self.model_loc.predict(X), 0.0, 1.0)

        if hasattr(self.model_load_cls, "predict_proba"):
            load_cls_probs = self.model_load_cls.predict_proba(X)
        else:
            load_cls_probs = self.model_load_cls.predict(X)
        load_cls_pred_id = np.argmax(load_cls_probs, axis=1)

        log_r_pred = self.model_load_r.predict(X)
        r_pred = decode_load_r(np.clip(log_r_pred, np.log10(5.0), np.log10(500.0)))
        # For predicted open or short, assign exact physical asymptote values
        r_pred = np.where(load_cls_pred_id == LOAD_TO_ID["open"], 1.0e9, r_pred)
        r_pred = np.where(load_cls_pred_id == LOAD_TO_ID["short"], 1.0e-6, r_pred)

        log_tau_pred = self.model_load_tau.predict(X)
        tau_pred = decode_load_tau(np.clip(log_tau_pred, -3.0, 0.5))

        return {
            "fault_class_id": fault_pred_id,
            "fault_class": np.array([FAULT_CLASSES[i] for i in fault_pred_id]),
            "fault_probs": fault_probs,
            "fault_pos_frac": loc_pred,
            "load_class_id": load_cls_pred_id,
            "load_class": np.array([LOAD_CLASSES[i] for i in load_cls_pred_id]),
            "load_cls_probs": load_cls_probs,
            "load_R": r_pred,
            "load_tau_over_T": tau_pred,
        }

    def save(self, out_dir: Path):
        """Save models to text files."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.model_fault is not None:
            self.model_fault.booster_.save_model(str(out_dir / "lgbm_fault.txt"))
        if self.model_loc is not None:
            self.model_loc.booster_.save_model(str(out_dir / "lgbm_loc.txt"))
        if self.model_load_cls is not None:
            self.model_load_cls.booster_.save_model(str(out_dir / "lgbm_load_cls.txt"))
        if self.model_load_r is not None:
            self.model_load_r.booster_.save_model(str(out_dir / "lgbm_load_r.txt"))
        if self.model_load_tau is not None:
            self.model_load_tau.booster_.save_model(str(out_dir / "lgbm_load_tau.txt"))

    def load(self, in_dir: Path):
        """Load models from text files."""
        in_dir = Path(in_dir)
        self.model_fault = lgb.Booster(model_file=str(in_dir / "lgbm_fault.txt"))
        self.model_loc = lgb.Booster(model_file=str(in_dir / "lgbm_loc.txt"))
        self.model_load_cls = lgb.Booster(model_file=str(in_dir / "lgbm_load_cls.txt"))
        self.model_load_r = lgb.Booster(model_file=str(in_dir / "lgbm_load_r.txt"))
        self.model_load_tau = lgb.Booster(model_file=str(in_dir / "lgbm_load_tau.txt"))


# =========================================================================
# Multi-Task 1D-ResNet
# =========================================================================

class BasicBlock1D(nn.Module):
    """Basic residual block with two 1D convolutions."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + res)
        return out


class ResNet1D(nn.Module):
    """Multi-task 1D ResNet for TDR fault classification, localization, and load ID."""

    def __init__(
        self,
        in_channels: int = 3,
        num_fault_classes: int = 7,
        num_load_classes: int = 6,
        embed_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Stem: Conv1d(in_channels->32, kernel 7, stride 2) -> BN -> ReLU (out length 512)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # 4 stages with channels [32, 64, 128, 128] and strides [1, 2, 2, 2]
        # Each stage has 2 BasicBlocks
        self.stage1 = nn.Sequential(
            BasicBlock1D(32, 32, stride=1),
            BasicBlock1D(32, 32, stride=1),
        )
        self.stage2 = nn.Sequential(
            BasicBlock1D(32, 64, stride=2),
            BasicBlock1D(64, 64, stride=1),
        )
        self.stage3 = nn.Sequential(
            BasicBlock1D(64, 128, stride=2),
            BasicBlock1D(128, 128, stride=1),
        )
        self.stage4 = nn.Sequential(
            BasicBlock1D(128, 128, stride=2),
            BasicBlock1D(128, 128, stride=1),
        )

        # Output shape of stage4: [B, 128, 64] -> flatten to 8192
        self.flatten = nn.Flatten()
        self.fc_shared = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128 * 64, embed_dim),
            nn.ReLU(inplace=True),
        )

        # Task Heads
        self.head_fault = nn.Linear(embed_dim, num_fault_classes)
        self.head_loc = nn.Linear(embed_dim, 1)  # followed by sigmoid
        self.head_load_cls = nn.Linear(embed_dim, num_load_classes)
        self.head_load_reg = nn.Linear(embed_dim, 2)  # [log10(R), log10(tau/T)]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Input: [B, in_channels, 1024]
        Returns dict with keys: 'fault_logits', 'loc_pred', 'load_cls_logits', 'load_reg_pred'
        """
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        feat = self.flatten(out)
        embed = self.fc_shared(feat)

        fault_logits = self.head_fault(embed)
        loc_pred = torch.sigmoid(self.head_loc(embed)).squeeze(-1)
        load_cls_logits = self.head_load_cls(embed)
        load_reg_pred = self.head_load_reg(embed)

        return {
            "fault_logits": fault_logits,      # [B, 7]
            "loc_pred": loc_pred,              # [B]
            "load_cls_logits": load_cls_logits,# [B, 6]
            "load_reg_pred": load_reg_pred,    # [B, 2]
        }


# =========================================================================
# Masked Multi-Task Loss
# =========================================================================

class MaskedMultiTaskLoss(nn.Module):
    """Loss function with physics masks for T1, T2, and T3."""

    def __init__(
        self,
        weight_fault: float = 1.0,
        weight_loc: float = 10.0,
        weight_load_cls: float = 1.0,
        weight_load_reg: float = 1.0,
        huber_delta_loc: float = 0.02,
        active_heads: Optional[List[str]] = None,
    ):
        super().__init__()
        self.w_fault = weight_fault
        self.w_loc = weight_loc
        self.w_load_cls = weight_load_cls
        self.w_load_reg = weight_load_reg
        self.huber_loc = nn.HuberLoss(delta=huber_delta_loc, reduction="none")
        self.huber_load_reg = nn.HuberLoss(delta=0.1, reduction="none")
        self.active_heads = active_heads or ["fault", "loc", "load_cls", "load_reg"]

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute masked loss."""
        device = preds["fault_logits"].device
        zero = torch.tensor(0.0, device=device)

        # 1. Fault classification loss (all samples)
        if "fault" in self.active_heads:
            l_fault = F.cross_entropy(preds["fault_logits"], targets["fault_cls"])
        else:
            l_fault = zero

        # 2. Location regression loss (masked on mask_loc == 1)
        if "loc" in self.active_heads:
            mask_loc = targets["mask_loc"].bool()
            if mask_loc.sum() > 0:
                loc_errors = self.huber_loc(preds["loc_pred"][mask_loc], targets["fault_pos"][mask_loc])
                l_loc = loc_errors.mean()
            else:
                l_loc = zero
        else:
            l_loc = zero

        # 3. Load classification loss (masked on mask_load == 1)
        if "load_cls" in self.active_heads:
            mask_load = targets["mask_load"].bool()
            if mask_load.sum() > 0:
                l_load_cls = F.cross_entropy(
                    preds["load_cls_logits"][mask_load], targets["load_cls"][mask_load]
                )
            else:
                l_load_cls = zero
        else:
            l_load_cls = zero

        # 4. Load regression loss (masked on continuous terminations)
        if "load_reg" in self.active_heads:
            mask_load = targets["mask_load"].bool()
            load_cls_ids = targets["load_cls"]
            # R_L regression on matched, R, RL, RC (indices 2, 3, 4, 5)
            mask_r = mask_load & (load_cls_ids >= 2)
            # tau/T regression on RL, RC (indices 4, 5)
            mask_tau = mask_load & (load_cls_ids >= 4)

            l_r = self.huber_load_reg(preds["load_reg_pred"][mask_r, 0:1], targets["load_reg"][mask_r, 0:1]).mean() if mask_r.sum() > 0 else zero
            l_tau = self.huber_load_reg(preds["load_reg_pred"][mask_tau, 1:2], targets["load_reg"][mask_tau, 1:2]).mean() if mask_tau.sum() > 0 else zero
            l_load_reg = l_r + l_tau
        else:
            l_load_reg = zero

        total_loss = (
            self.w_fault * l_fault
            + self.w_loc * l_loc
            + self.w_load_cls * l_load_cls
            + self.w_load_reg * l_load_reg
        )

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "loss_fault": float(l_fault.item()),
            "loss_loc": float(l_loc.item()),
            "loss_load_cls": float(l_load_cls.item()),
            "loss_load_reg": float(l_load_reg.item()),
        }
        return total_loss, loss_dict

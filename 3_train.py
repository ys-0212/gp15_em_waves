"""Training script for LightGBM Suite and Multi-task 1D-ResNet.

Supports:
- --model lgbm|resnet
- --quick
- --seed int
- --channels v|v,dv|v,dv,coord
- --single-task fault|loc|load
- --train-frac float (0.05, 0.1, 0.25, 0.5, 1.0)
- --device auto|cpu|cuda
"""

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from tdr.data import (
    FAULT_CLASSES,
    LOAD_CLASSES,
    FAULT_TO_ID,
    LOAD_TO_ID,
    TDRDataset,
    TensorTDRData,
    preprocess_numpy,
    preprocess,
    decode_load_r,
    decode_load_tau,
)
from tdr.features import extract_features, cached_features
from tdr.models import LGBMSuite, ResNet1D, MaskedMultiTaskLoss
from tdr.metrics import compute_all_metrics


def get_git_hash() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def train_conventional(
    cfg: dict,
    quick: bool = False,
    out_dir: Path = Path("results/runs"),
):
    """Run Conventional TDR baseline: tune on val, evaluate on test, save outputs."""
    from tdr.features import ConventionalTDR
    data_dir = Path(cfg["paths"]["data"])
    suffix = "_quick" if quick else ""
    run_id = "conventional"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Running Conventional TDR Baseline [Run ID: {run_id}] ---")

    # Load val set for tuning
    with h5py.File(data_dir / f"val{suffix}.h5", "r") as f:
        clean_va = np.array(f["traces"], dtype=np.float32)
    meta_va = pd.read_parquet(data_dir / f"val{suffix}_meta.parquet")
    preproc_va = preprocess_numpy(
        clean_va, meta_va["v_inc"].values, snr_db=cfg["noise"]["eval_snr_db"], seed=cfg["noise"]["eval_noise_seed"], add_noise=True
    )

    solver = ConventionalTDR(
        min_sep_samples=cfg["conventional"]["min_sep_samples"],
        transient_return_samples=cfg["conventional"]["transient_return_samples"],
        z0_nominal=cfg["physics"]["z0_nominal"],
        r_dc_ohm_per_m=cfg["physics"]["r_dc_ohm_per_m"],
    )

    print("Tuning all conventional thresholds on the validation set (never on test)...")
    best_params = solver.tune_on_val(preproc_va, meta_va, k_grid=cfg["conventional"]["k_sigma_grid"])
    best_k = best_params["k_sigma"]
    print(f"Selected thresholds: {best_params}")

    # Load test set for evaluation
    with h5py.File(data_dir / f"test{suffix}.h5", "r") as f:
        clean_te = np.array(f["traces"], dtype=np.float32)
    meta_te = pd.read_parquet(data_dir / f"test{suffix}_meta.parquet")
    preproc_te = preprocess_numpy(
        clean_te, meta_te["v_inc"].values, snr_db=cfg["noise"]["eval_snr_db"], seed=cfg["noise"]["eval_noise_seed"], add_noise=True
    )

    print(f"Evaluating Conventional TDR on test set ({len(clean_te)} samples)...")
    preds = solver.predict_batch(preproc_te, meta_te["length_m"].values)

    pred_fault_cls = [p["fault_class"] for p in preds]
    pred_fault_pos = [p["fault_pos_frac"] for p in preds]
    detected = [p["detected"] for p in preds]
    pred_load_cls = [p["load_class"] for p in preds]
    pred_load_r = [p["load_R"] for p in preds]
    pred_load_tau = [p["load_tau_over_T"] for p in preds]

    df_pred = meta_te.copy()
    df_pred["pred_fault_class"] = pred_fault_cls
    df_pred["pred_fault_pos_frac"] = pred_fault_pos
    df_pred["detected"] = detected
    df_pred["pred_load_class"] = pred_load_cls
    df_pred["pred_load_R"] = pred_load_r
    df_pred["pred_load_tau_over_T"] = pred_load_tau
    df_pred["error_m"] = np.abs(df_pred["pred_fault_pos_frac"] - df_pred["fault_pos_frac"].fillna(0.0)) * df_pred["length_m"]

    eval_df = pd.DataFrame({
        "true_fault_class": meta_te["fault_class"],
        "pred_fault_class": pred_fault_cls,
        "true_fault_pos_frac": meta_te["fault_pos_frac"].fillna(0.0),
        "pred_fault_pos_frac": pred_fault_pos,
        "length_m": meta_te["length_m"],
        "detected": detected,
        "true_load_class": meta_te["load_class"],
        "pred_load_class": pred_load_cls,
        "true_load_R": meta_te["load_R"],
        "pred_load_R": pred_load_r,
        "true_load_tau_over_T": meta_te["load_tau_over_T"],
        "pred_load_tau_over_T": pred_load_tau,
        "mask_load": meta_te["mask_load"],
    })
    metrics = compute_all_metrics(eval_df)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    df_pred.to_parquet(run_dir / "predictions.parquet", index=False)

    # Save tuned parameters for app deployment
    conventional_params = solver.to_dict()
    with open(run_dir / "conventional.json", "w", encoding="utf-8") as f:
        json.dump(conventional_params, f, indent=2)

    config_log = {
        "run_id": run_id,
        "model": "conventional",
        "quick": quick,
        "best_k_sigma": best_k,
        "tuned_params": best_params,
        "git_hash": get_git_hash(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_log, f, indent=2)

    print(f"Conventional Test Fault Accuracy: {metrics['t1_fault']['accuracy']*100:.2f}% (Chance: 14.3%)")
    print(f"Conventional Test Fault Macro-F1: {metrics['t1_fault']['macro_f1']:.4f}")
    print(f"Conventional Test Location MAE: {metrics['t2_loc']['mae_m']:.4f} m (Detection rate: {metrics['t2_loc']['detection_rate']:.1f}%)")
    print(f"Conventional Test Load Accuracy: {metrics['t3_load']['accuracy']*100:.2f}% (Chance: 16.7%)")
    print(f"Outputs written to {run_dir}")


def train_lgbm(
    cfg: dict,
    quick: bool = False,
    seed: int = 0,
    train_frac: float = 1.0,
    out_dir: Path = Path("results/runs"),
):
    """Train LightGBM Suite."""
    data_dir = Path(cfg["paths"]["data"])
    suffix = "_quick" if quick else ""
    run_id = f"lgbm_frac{train_frac}_s{seed}" if train_frac < 1.0 else f"lgbm_s{seed}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Training LightGBM Suite [Run ID: {run_id}] ---")

    # Load data
    with h5py.File(data_dir / f"train{suffix}.h5", "r") as f:
        clean_tr = np.array(f["traces"], dtype=np.float32)
    meta_tr = pd.read_parquet(data_dir / f"train{suffix}_meta.parquet")

    with h5py.File(data_dir / f"val{suffix}.h5", "r") as f:
        clean_va = np.array(f["traces"], dtype=np.float32)
    meta_va = pd.read_parquet(data_dir / f"val{suffix}_meta.parquet")

    # Learning curve subsampling if train_frac < 1.0
    n_train = len(clean_tr)
    if train_frac < 1.0:
        n_sub = max(100, int(n_train * train_frac))
        rng_sub = np.random.default_rng(seed)
        sub_indices = rng_sub.choice(n_train, size=n_sub, replace=False)
        clean_tr = clean_tr[sub_indices]
        meta_tr = meta_tr.iloc[sub_indices].reset_index(drop=True)
        print(f"Subsampled train set to {len(clean_tr)} samples (frac={train_frac})")

    # Noise copies for training: 2 independent noise copies
    n_copies = cfg.get("lgbm", {}).get("noise_copies", 2)
    X_tr_list = []
    v_inc_tr = meta_tr["v_inc"].values

    print(f"Extracting features from {n_copies} noisy copies of training set...")
    cache_dir = data_dir / "cache"
    for copy_idx in range(n_copies):
        copy_seed = seed * 100 + copy_idx
        X_tr_list.append(cached_features(cache_dir, clean_tr, v_inc_tr, None, copy_seed, preprocess_numpy))

    X_train = np.concatenate(X_tr_list, axis=0)
    meta_tr_rep = pd.concat([meta_tr] * n_copies, ignore_index=True)

    # Validation: fixed noise copy (eval_noise_seed)
    print("Extracting features from validation set...")
    eval_seed = cfg["noise"]["eval_noise_seed"]
    v_inc_va = meta_va["v_inc"].values
    X_val = cached_features(cache_dir, clean_va, v_inc_va, cfg["noise"]["eval_snr_db"], eval_seed, preprocess_numpy)

    # Initialize and train LGBMSuite
    suite = LGBMSuite(cfg)
    n_est = cfg["quick"]["lgbm_n_estimators"] if quick else cfg["lgbm"]["n_estimators"]
    suite.fit(
        X_train=X_train,
        y_fault_train=meta_tr_rep["fault_class_id"].values,
        y_loc_train=meta_tr_rep["fault_pos_frac"].fillna(0.0).values,
        y_load_cls_train=meta_tr_rep["load_class_id"].values,
        y_load_r_train=meta_tr_rep["load_R"].values,
        y_load_tau_train=meta_tr_rep["load_tau_over_T"].values,
        mask_loc_train=meta_tr_rep["mask_loc"].values,
        mask_load_train=meta_tr_rep["mask_load"].values,
        X_val=X_val,
        y_fault_val=meta_va["fault_class_id"].values,
        y_loc_val=meta_va["fault_pos_frac"].fillna(0.0).values,
        y_load_cls_val=meta_va["load_class_id"].values,
        y_load_r_val=meta_va["load_R"].values,
        y_load_tau_val=meta_va["load_tau_over_T"].values,
        mask_loc_val=meta_va["mask_loc"].values,
        mask_load_val=meta_va["mask_load"].values,
        n_estimators=n_est,
    )

    # Save models
    suite.save(run_dir)

    # Evaluate on val
    preds_val = suite.predict(X_val)
    df_pred = meta_va.copy()
    df_pred["pred_fault_class"] = preds_val["fault_class"]
    df_pred["pred_fault_pos_frac"] = preds_val["fault_pos_frac"]
    df_pred["pred_load_class"] = preds_val["load_class"]
    df_pred["pred_load_R"] = preds_val["load_R"]
    df_pred["pred_load_tau_over_T"] = preds_val["load_tau_over_T"]
    df_pred["error_m"] = np.abs(df_pred["pred_fault_pos_frac"] - df_pred["fault_pos_frac"].fillna(0.0)) * df_pred["length_m"]

    # Compute metrics
    eval_df = pd.DataFrame({
        "true_fault_class": meta_va["fault_class"],
        "pred_fault_class": preds_val["fault_class"],
        "true_fault_pos_frac": meta_va["fault_pos_frac"].fillna(0.0),
        "pred_fault_pos_frac": preds_val["fault_pos_frac"],
        "length_m": meta_va["length_m"],
        "true_load_class": meta_va["load_class"],
        "pred_load_class": preds_val["load_class"],
        "true_load_R": meta_va["load_R"],
        "pred_load_R": preds_val["load_R"],
        "true_load_tau_over_T": meta_va["load_tau_over_T"],
        "pred_load_tau_over_T": preds_val["load_tau_over_T"],
        "mask_load": meta_va["mask_load"],
    })
    metrics = compute_all_metrics(eval_df)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    df_pred.to_parquet(run_dir / "predictions.parquet", index=False)

    config_log = {
        "run_id": run_id,
        "model": "lgbm",
        "quick": quick,
        "seed": seed,
        "train_frac": train_frac,
        "git_hash": get_git_hash(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_log, f, indent=2)

    print(f"Validation Fault Macro-F1: {metrics['t1_fault']['macro_f1']:.4f}")
    print(f"Validation Location MAE: {metrics['t2_loc']['mae_m']:.4f} m")
    print(f"Validation Load Accuracy: {metrics['t3_load']['accuracy']:.4f}")
    print(f"Outputs written to {run_dir}")


def train_resnet(
    cfg: dict,
    quick: bool = False,
    seed: int = 0,
    channels: str = "v,dv,coord",
    single_task: Optional[str] = None,
    train_frac: float = 1.0,
    device_name: str = "auto",
    out_dir: Path = Path("results/runs"),
):
    """Train Multi-task 1D ResNet."""
    data_dir = Path(cfg["paths"]["data"])
    suffix = "_quick" if quick else ""

    # Construct run_id
    if single_task:
        run_id = f"resnet_single_{single_task}_s{seed}"
    elif channels != "v,dv,coord":
        ch_str = channels.replace(",", "")
        run_id = f"resnet_ch_{ch_str}_s{seed}"
    elif train_frac < 1.0:
        run_id = f"resnet_frac{train_frac}_s{seed}"
    else:
        run_id = f"resnet_s{seed}"

    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Training ResNet1D [Run ID: {run_id}] ---")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Using compute device: {device}")

    # Determine in_channels and channel slicing
    if channels == "v":
        in_channels = 1
        ch_indices = [0]
    elif channels == "v,dv":
        in_channels = 2
        ch_indices = [0, 1]
    else:
        in_channels = 3
        ch_indices = [0, 1, 2]

    # Precompute channel-wise mean and std on preprocessed train set
    train_h5 = data_dir / f"train{suffix}.h5"
    train_meta_path = data_dir / f"train{suffix}_meta.parquet"
    with h5py.File(train_h5, "r") as f:
        clean_samples = np.array(f["traces"][:min(1000, len(f["traces"]))], dtype=np.float32)
    meta_sub = pd.read_parquet(train_meta_path).iloc[:len(clean_samples)]
    preproc_norm_batch = preprocess_numpy(clean_samples, meta_sub["v_inc"].values, seed=seed)
    # channel mean and std: shape [3]
    ch_mean = np.mean(preproc_norm_batch, axis=(0, 2)).tolist()
    ch_std = np.std(preproc_norm_batch, axis=(0, 2)).tolist()
    # Ensure std not 0
    ch_std = [s if s > 1e-4 else 1.0 for s in ch_std]

    norm_stats = {"mean": ch_mean, "std": ch_std}
    with open(run_dir / "resnet_norm.json", "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)

    mean_t = torch.tensor(ch_mean, device=device, dtype=torch.float32).view(1, 3, 1)
    std_t = torch.tensor(ch_std, device=device, dtype=torch.float32).view(1, 3, 1)

    # Data held as tensors on the device; noise + preprocessing run per batch on the device
    val_h5 = data_dir / f"val{suffix}.h5"
    val_meta_path = data_dir / f"val{suffix}_meta.parquet"
    n_train_total = len(pd.read_parquet(train_meta_path))
    train_idx = None
    if train_frac < 1.0:
        n_sub = max(100, int(n_train_total * train_frac))
        train_idx = np.random.default_rng(seed).choice(n_train_total, size=n_sub, replace=False)
        print(f"Subsampled train set to {n_sub} samples (frac={train_frac})")
    train_data = TensorTDRData(train_h5, train_meta_path, device, indices=train_idx)
    val_data = TensorTDRData(val_h5, val_meta_path, device)
    val_data.fix_noise(cfg["noise"]["eval_snr_db"], cfg["noise"]["eval_noise_seed"])
    perm_gen = torch.Generator().manual_seed(seed)
    noise_gen = torch.Generator(device=device).manual_seed(seed + 10_000)

    batch_size = cfg["train"]["batch_size"]
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Model and loss
    model = ResNet1D(in_channels=in_channels, embed_dim=cfg["resnet"]["embed_dim"], dropout=cfg["resnet"]["dropout"]).to(device)

    # Active heads for single task vs multi-task
    if single_task == "fault":
        active_heads = ["fault"]
    elif single_task == "loc":
        active_heads = ["loc"]
    elif single_task == "load":
        active_heads = ["load_cls", "load_reg"]
    else:
        active_heads = ["fault", "loc", "load_cls", "load_reg"]

    criterion = MaskedMultiTaskLoss(
        weight_fault=cfg["train"]["loss_weights"]["fault"],
        weight_loc=cfg["train"]["loss_weights"]["loc"],
        weight_load_cls=cfg["train"]["loss_weights"]["load_cls"],
        weight_load_reg=cfg["train"]["loss_weights"]["load_reg"],
        huber_delta_loc=cfg["train"]["huber_delta_loc"],
        active_heads=active_heads,
    )

    epochs = cfg["quick"]["epochs"] if quick else cfg["train"]["epochs"]
    lr = float(cfg["train"]["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg["train"]["weight_decay"]))
    steps_per_epoch = max(1, int(np.ceil(train_data.n / batch_size)))
    total_steps = epochs * steps_per_epoch
    warmup_steps = min(int(cfg["train"].get("warmup_epochs", 1)) * steps_per_epoch, max(total_steps - 1, 1))

    def lr_lambda(step: int) -> float:
        """Linear warm-up for warmup_epochs, then cosine decay to 0 (per optimisation step)."""
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    patience = cfg["train"]["patience"]
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    print(f"Beginning training for {epochs} epochs ({steps_per_epoch} steps/epoch, AMP={use_amp})...")
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        train_loss_total = 0.0
        n_batches = 0

        for x, targets in train_data.batches(batch_size, shuffle=True, perm_generator=perm_gen, noise_generator=noise_gen):
            x = (x - mean_t) / std_t
            if in_channels < 3:
                x = x[:, ch_indices, :]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                preds = model(x)
            preds = {k: v.float() for k, v in preds.items()}
            loss, _ = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_total += loss.item()
            n_batches += 1

        train_loss_avg = train_loss_total / max(1, n_batches)

        # Validation loop (fixed noise copy)
        model.eval()
        val_loss_total = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x, targets in val_data.batches(1024, shuffle=False):
                x = (x - mean_t) / std_t
                if in_channels < 3:
                    x = x[:, ch_indices, :]
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    preds = model(x)
                preds = {k: v.float() for k, v in preds.items()}
                loss, _ = criterion(preds, targets)
                val_loss_total += loss.item()
                n_val_batches += 1

        val_loss_avg = val_loss_total / max(1, n_val_batches)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "val_loss": val_loss_avg,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": time.time() - t_epoch,
        })
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | {time.time() - t_epoch:.1f}s")

        # Checkpointing
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Save training history to CSV
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # Load best checkpoint for evaluation
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()

    # Generate predictions on val set
    val_preds_list = []
    with torch.no_grad():
        for x, _ in val_data.batches(1024, shuffle=False):
            x = (x - mean_t) / std_t
            if in_channels < 3:
                x = x[:, ch_indices, :]
            preds = model(x)
            val_preds_list.append({
                "fault_probs": F.softmax(preds["fault_logits"], dim=-1).cpu().numpy(),
                "loc_pred": preds["loc_pred"].cpu().numpy(),
                "load_cls_probs": F.softmax(preds["load_cls_logits"], dim=-1).cpu().numpy(),
                "load_reg_pred": preds["load_reg_pred"].cpu().numpy(),
            })

    # Unpack validation predictions
    fault_probs = np.concatenate([p["fault_probs"] for p in val_preds_list], axis=0)
    fault_pred_id = np.argmax(fault_probs, axis=1)
    loc_pred = np.concatenate([p["loc_pred"] for p in val_preds_list], axis=0)
    load_cls_probs = np.concatenate([p["load_cls_probs"] for p in val_preds_list], axis=0)
    load_cls_pred_id = np.argmax(load_cls_probs, axis=1)
    load_reg_pred = np.concatenate([p["load_reg_pred"] for p in val_preds_list], axis=0)
    pred_r = decode_load_r(np.clip(load_reg_pred[:, 0], np.log10(5.0), np.log10(500.0)))
    pred_tau = decode_load_tau(np.clip(load_reg_pred[:, 1], -3.0, 0.5))

    val_meta = pd.read_parquet(val_meta_path)
    df_pred = val_meta.copy()
    df_pred["pred_fault_class"] = [FAULT_CLASSES[i] for i in fault_pred_id]
    df_pred["pred_fault_pos_frac"] = loc_pred
    df_pred["pred_load_class"] = [LOAD_CLASSES[i] for i in load_cls_pred_id]
    df_pred["pred_load_R"] = pred_r
    df_pred["pred_load_tau_over_T"] = pred_tau
    df_pred["error_m"] = np.abs(df_pred["pred_fault_pos_frac"] - df_pred["fault_pos_frac"].fillna(0.0)) * df_pred["length_m"]

    eval_df = pd.DataFrame({
        "true_fault_class": val_meta["fault_class"],
        "pred_fault_class": df_pred["pred_fault_class"],
        "true_fault_pos_frac": val_meta["fault_pos_frac"].fillna(0.0),
        "pred_fault_pos_frac": df_pred["pred_fault_pos_frac"],
        "length_m": val_meta["length_m"],
        "true_load_class": val_meta["load_class"],
        "pred_load_class": df_pred["pred_load_class"],
        "true_load_R": val_meta["load_R"],
        "pred_load_R": df_pred["pred_load_R"],
        "true_load_tau_over_T": val_meta["load_tau_over_T"],
        "pred_load_tau_over_T": df_pred["pred_load_tau_over_T"],
        "mask_load": val_meta["mask_load"],
    })
    metrics = compute_all_metrics(eval_df)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    df_pred.to_parquet(run_dir / "predictions.parquet", index=False)

    config_log = {
        "run_id": run_id,
        "model": "resnet",
        "quick": quick,
        "seed": seed,
        "channels": channels,
        "single_task": single_task,
        "train_frac": train_frac,
        "git_hash": get_git_hash(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_log, f, indent=2)

    print(f"Validation Fault Macro-F1: {metrics['t1_fault']['macro_f1']:.4f}")
    print(f"Validation Location MAE: {metrics['t2_loc']['mae_m']:.4f} m")
    print(f"Validation Load Accuracy: {metrics['t3_load']['accuracy']:.4f}")
    print(f"Outputs written to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["conventional", "lgbm", "resnet"], required=True)
    parser.add_argument("--config", default="config.yaml", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", default="v,dv,coord")
    parser.add_argument("--single-task", choices=["fault", "loc", "load"], default=None)
    parser.add_argument("--train-frac", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.model == "conventional":
        train_conventional(cfg, quick=args.quick)
    elif args.model == "lgbm":
        train_lgbm(cfg, quick=args.quick, seed=args.seed, train_frac=args.train_frac)
    elif args.model == "resnet":
        train_resnet(
            cfg,
            quick=args.quick,
            seed=args.seed,
            channels=args.channels,
            single_task=args.single_task,
            train_frac=args.train_frac,
            device_name=args.device,
        )


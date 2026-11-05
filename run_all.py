"""Master pipeline runner for TDR Fault and Load Identification.

Orchestrates all phases:
- gen: dataset generation (2_generate_data.py)
- validate: simulator validation (1_validate_simulator.py)
- conventional: baseline rule-based evaluation (3_train.py --model conventional)
- lgbm: main, learning-curve and ablation runs for LightGBM (3_train.py --model lgbm)
- resnet: main, learning-curve and ablation runs for ResNet1D (3_train.py --model resnet)
- evaluate: comprehensive E1-E8 experiments, tables and logs (4_evaluate.py)
- figures: publication-quality figures (5_make_figures.py)

Supports:
- --mode quick|full
- --device auto|cpu|cuda
- --skip-existing
- --stages comma-separated list
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml


def get_git_hash() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


ALL_STAGES = ["gen", "validate", "conventional", "lgbm", "resnet", "evaluate", "figures"]


def run_cmd(cmd_list: list):
    """Run command with current python interpreter and check success."""
    print(f"\n>>> Running: {' '.join(str(c) for c in cmd_list)}")
    t0 = time.time()
    res = subprocess.run(cmd_list, check=True)
    print(f">>> Finished in {time.time() - t0:.2f}s")


def export_app_weights(weights_dir: Path, runs_dir: Path, mode: str, git_hash: str):
    """Export best seed-0 models and metadata to app/weights/ for Streamlit app."""
    weights_dir = Path(weights_dir)
    runs_dir = Path(runs_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 1. ResNet weights
    resnet_s0 = runs_dir / "resnet_s0"
    if (resnet_s0 / "best_model.pt").exists():
        shutil.copy2(resnet_s0 / "best_model.pt", weights_dir / "resnet.pt")
    if (resnet_s0 / "resnet_norm.json").exists():
        shutil.copy2(resnet_s0 / "resnet_norm.json", weights_dir / "resnet_norm.json")

    # 2. LightGBM models
    lgbm_s0 = runs_dir / "lgbm_s0"
    for lgbm_file in ["lgbm_fault.txt", "lgbm_loc.txt", "lgbm_load_cls.txt", "lgbm_load_r.txt", "lgbm_load_tau.txt"]:
        src = lgbm_s0 / lgbm_file
        if src.exists():
            shutil.copy2(src, weights_dir / lgbm_file)

    # 3. Conventional parameters
    conv_dir = runs_dir / "conventional"
    if (conv_dir / "conventional.json").exists():
        shutil.copy2(conv_dir / "conventional.json", weights_dir / "conventional.json")

    # 4. Weights metadata
    meta = {
        "run_id": "seed0_main",
        "mode": mode,
        "git_hash": git_hash,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": ["resnet", "lgbm", "conventional"],
    }
    with open(weights_dir / "weights_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Deployment weights and weights_meta.json exported to {weights_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stages", default=",".join(ALL_STAGES))
    parser.add_argument("--config", default="config.yaml", type=Path)
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    py_exe = sys.executable
    quick_flag = ["--quick"] if args.mode == "quick" else []
    is_quick = args.mode == "quick"
    git_hash = get_git_hash()
    runs_dir = Path("results/runs")
    weights_dir = Path(cfg["paths"]["weights"])

    print("=" * 75)
    print(f"MASTER TDR PIPELINE: MODE={args.mode.upper()}, DEVICE={args.device}")
    print(f"STAGES: {stages}")
    print("=" * 75)

    # Stage: gen
    if "gen" in stages:
        suffix = "_quick" if is_quick else ""
        train_h5 = Path("data") / f"train{suffix}.h5"
        if args.skip_existing and train_h5.exists():
            print("Skipping 'gen' (datasets already exist)")
        else:
            run_cmd([py_exe, "2_generate_data.py"] + quick_flag)

    # Stage: validate
    if "validate" in stages:
        val_json = Path("results/sim_validation.json")
        if args.skip_existing and val_json.exists():
            print("Skipping 'validate' (sim_validation.json already exists)")
        else:
            run_cmd([py_exe, "1_validate_simulator.py"])

    # Stage: conventional
    if "conventional" in stages:
        conv_dir = runs_dir / "conventional"
        if args.skip_existing and (conv_dir / "metrics.json").exists():
            print("Skipping 'conventional' (already evaluated)")
        else:
            run_cmd([py_exe, "3_train.py", "--model", "conventional"] + quick_flag)

    # Stage: lgbm
    if "lgbm" in stages:
        seeds = cfg["quick"]["seeds"] if is_quick else cfg["experiments"]["seeds"]
        fracs = cfg["quick"]["learning_curve_fracs"] if is_quick else cfg["experiments"]["learning_curve_fracs"]

        # Main seeds
        for s in seeds:
            target_dir = runs_dir / f"lgbm_s{s}"
            if args.skip_existing and (target_dir / "metrics.json").exists():
                print(f"Skipping lgbm_s{s} (already trained)")
            else:
                run_cmd([py_exe, "3_train.py", "--model", "lgbm", "--seed", str(s)] + quick_flag)

        # Learning curve fractions (fraction < 1.0)
        for f_val in fracs:
            if f_val < 1.0:
                target_dir = runs_dir / f"lgbm_frac{f_val}_s0"
                if args.skip_existing and (target_dir / "metrics.json").exists():
                    print(f"Skipping lgbm_frac{f_val}_s0 (already trained)")
                else:
                    run_cmd([py_exe, "3_train.py", "--model", "lgbm", "--seed", "0", "--train-frac", str(f_val)] + quick_flag)

    # Stage: resnet
    if "resnet" in stages:
        seeds = cfg["quick"]["seeds"] if is_quick else cfg["experiments"]["seeds"]
        fracs = cfg["quick"]["learning_curve_fracs"] if is_quick else cfg["experiments"]["learning_curve_fracs"]

        # Main seeds
        for s in seeds:
            target_dir = runs_dir / f"resnet_s{s}"
            if args.skip_existing and (target_dir / "metrics.json").exists():
                print(f"Skipping resnet_s{s} (already trained)")
            else:
                run_cmd([py_exe, "3_train.py", "--model", "resnet", "--seed", str(s), "--device", args.device] + quick_flag)

        # Learning curve fractions
        for f_val in fracs:
            if f_val < 1.0:
                target_dir = runs_dir / f"resnet_frac{f_val}_s0"
                if args.skip_existing and (target_dir / "metrics.json").exists():
                    print(f"Skipping resnet_frac{f_val}_s0 (already trained)")
                else:
                    run_cmd([py_exe, "3_train.py", "--model", "resnet", "--seed", "0", "--train-frac", str(f_val), "--device", args.device] + quick_flag)

        # Channel ablations
        for ch in ["v", "v,dv"]:
            ch_str = ch.replace(",", "")
            target_dir = runs_dir / f"resnet_ch_{ch_str}_s0"
            if args.skip_existing and (target_dir / "metrics.json").exists():
                print(f"Skipping {target_dir.name} (already trained)")
            else:
                run_cmd([py_exe, "3_train.py", "--model", "resnet", "--seed", "0", "--channels", ch, "--device", args.device] + quick_flag)

        # Single-task ablations
        for task in ["fault", "loc", "load"]:
            target_dir = runs_dir / f"resnet_single_{task}_s0"
            if args.skip_existing and (target_dir / "metrics.json").exists():
                print(f"Skipping {target_dir.name} (already trained)")
            else:
                run_cmd([py_exe, "3_train.py", "--model", "resnet", "--seed", "0", "--single-task", task, "--device", args.device] + quick_flag)

    # Stage: evaluate
    if "evaluate" in stages:
        run_cmd([py_exe, "4_evaluate.py", "--device", args.device] + quick_flag)
        # Export seed-0 weights for app
        export_app_weights(weights_dir, runs_dir, args.mode, git_hash)

    # Stage: figures
    if "figures" in stages:
        if Path("5_make_figures.py").exists():
            run_cmd([py_exe, "5_make_figures.py"])

    print("\n" + "=" * 75)
    print("MASTER PIPELINE EXECUTION COMPLETE!")
    print("=" * 75)


if __name__ == "__main__":
    main()

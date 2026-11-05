"""Generate synthetic TDR datasets for train, val, test, and OOD splits.

Supports --quick mode, profile timing, dataset card generation,
and verification debug plots.
"""

import argparse
from pathlib import Path
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from tdr.data import generate_dataset_split, sample_batch_params
from tdr.simulator import simulate_batch


def generate_all_splits(
    cfg_path: Path = Path("config.yaml"),
    quick: bool = False,
    out_dir: Path = Path("data"),
    debug_dir: Path = Path("results/debug"),
):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(out_dir)
    debug_dir = Path(debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_quick" if quick else ""
    split_counts = {}
    if quick:
        split_counts["train"] = cfg["quick"]["n"]["train"]
        split_counts["val"] = cfg["quick"]["n"]["val"]
        split_counts["test"] = cfg["quick"]["n"]["test"]
        ood_n = cfg["quick"]["n"]["ood_per_subset"]
    else:
        split_counts["train"] = cfg["splits"]["train"]["n"]
        split_counts["val"] = cfg["splits"]["val"]["n"]
        split_counts["test"] = cfg["splits"]["test"]["n"]
        ood_n = cfg["splits"]["ood"]["n_per_subset"]

    seeds = {
        "train": cfg["splits"]["train"]["seed"],
        "val": cfg["splits"]["val"]["seed"],
        "test": cfg["splits"]["test"]["seed"],
        "ood": cfg["splits"]["ood"]["seed"],
    }

    ood_subsets = ["long_cable", "high_loss", "z0_shift", "slow_rise", "near_load"]

    t0_all = time.time()
    generation_stats = {}

    print(f"Generating datasets (mode: {'QUICK' if quick else 'FULL'})...")

    # Standard splits
    for split in ["train", "val", "test"]:
        n = split_counts[split]
        seed = seeds[split]
        split_file_name = f"{split}{suffix}"
        print(f"Generating {split_file_name}: N={n}, seed={seed}...")
        t_start = time.time()
        traces, meta = generate_dataset_split(
            split_name=split_file_name,
            n_samples=n,
            seed=seed,
            cfg=cfg,
            out_dir=out_dir,
            ood_type=None,
        )
        elapsed = time.time() - t_start
        generation_stats[split_file_name] = {
            "n_samples": n,
            "elapsed_sec": elapsed,
            "sec_per_1k": (elapsed / n) * 1000.0,
        }
        print(f"  Done in {elapsed:.2f}s ({(elapsed / n) * 1000.0:.2f}s per 1k)")

    # OOD split
    print(f"Generating OOD subsets: {len(ood_subsets)} subsets x {ood_n} samples...")
    ood_rng = np.random.default_rng(seeds["ood"])
    ood_params_all = []
    ood_dfs = []

    t_start_ood = time.time()
    for subset in ood_subsets:
        sub_seed = int(ood_rng.integers(1, 1000000))
        sub_rng = np.random.default_rng(sub_seed)
        sub_params, sub_df = sample_batch_params(ood_n, sub_rng, cfg, ood_type=subset)
        sub_df["ood_subset"] = subset
        ood_params_all.extend(sub_params)
        ood_dfs.append(sub_df)

    ood_meta_df = pd.concat(ood_dfs, ignore_index=True)
    ood_file_name = f"ood{suffix}"
    ood_meta_df["split"] = ood_file_name

    t_grid, ood_traces = simulate_batch(
        ood_params_all,
        n_sim=cfg["signal"]["n_sim"],
        sim_dt_per_t=cfg["signal"]["sim_dt_per_T"],
        n_out=cfg["signal"]["n_out"],
        window_t=cfg["signal"]["window_T"],
        t_launch_t=cfg["signal"]["t_launch_T"],
        lossless=False,
    )
    elapsed_ood = time.time() - t_start_ood

    ood_h5_path = out_dir / f"{ood_file_name}.h5"
    ood_meta_path = out_dir / f"{ood_file_name}_meta.parquet"

    with h5py.File(ood_h5_path, "w") as f:
        ds = f.create_dataset("traces", data=ood_traces, compression="gzip", compression_opts=4)
        ds.attrs["t_grid"] = t_grid

    ood_meta_df.to_parquet(ood_meta_path, index=False)
    generation_stats[ood_file_name] = {
        "n_samples": len(ood_params_all),
        "elapsed_sec": elapsed_ood,
        "sec_per_1k": (elapsed_ood / len(ood_params_all)) * 1000.0,
    }
    print(f"  OOD Done in {elapsed_ood:.2f}s ({(elapsed_ood / len(ood_params_all)) * 1000.0:.2f}s per 1k)")

    total_time = time.time() - t0_all
    print(f"\nAll datasets generated in {total_time:.2f}s.")

    # Write dataset card
    write_dataset_card(out_dir, suffix, generation_stats, total_time)

    # Generate debug plots: one trace per fault class
    create_debug_plot(out_dir, suffix, debug_dir)


def write_dataset_card(
    data_dir: Path,
    suffix: str,
    generation_stats: dict,
    total_time: float,
):
    """Write data/dataset_card.md with counts, distributions, and timing."""
    card_path = data_dir / "dataset_card.md"
    train_meta = pd.read_parquet(data_dir / f"train{suffix}_meta.parquet")
    val_meta = pd.read_parquet(data_dir / f"val{suffix}_meta.parquet")
    test_meta = pd.read_parquet(data_dir / f"test{suffix}_meta.parquet")
    ood_meta = pd.read_parquet(data_dir / f"ood{suffix}_meta.parquet")

    splits_data = [
        ("Train", train_meta),
        ("Val", val_meta),
        ("Test", test_meta),
        ("OOD", ood_meta),
    ]

    lines = [
        "# Dataset Card: TDR Transmission Line Fault & Load Identification",
        "",
        "## Summary",
        f"- **Generation Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Total Generation Time**: {total_time:.2f} seconds",
        f"- **Mode**: {'Quick Mode' if suffix else 'Full Mode'}",
        "",
        "## Split Sizes and Generation Speed",
        "| Split | Samples | Generation Time (s) | Time per 1k samples (s) |",
        "|---|---|---|---|",
    ]

    for split_key, stats in generation_stats.items():
        lines.append(f"| `{split_key}` | {stats['n_samples']:,} | {stats['elapsed_sec']:.2f} | {stats['sec_per_1k']:.2f} |")

    lines.extend([
        "",
        "## Fault Class Distribution Across Splits",
        "| Fault Class | Train | Val | Test | OOD |",
        "|---|---|---|---|---|",
    ])

    fault_classes = ["none", "short", "open", "shunt_R", "shunt_C", "series_R", "z0_section"]
    for fc in fault_classes:
        counts = [f"{(df['fault_class'] == fc).sum():,}" for _, df in splits_data]
        lines.append(f"| `{fc}` | {counts[0]} | {counts[1]} | {counts[2]} | {counts[3]} |")

    lines.extend([
        "",
        "## Load Class Distribution Across Splits",
        "| Load Class | Train | Val | Test | OOD |",
        "|---|---|---|---|---|",
    ])

    load_classes = ["open", "short", "matched", "R", "RL", "RC"]
    for lc in load_classes:
        counts = [f"{(df['load_class'] == lc).sum():,}" for _, df in splits_data]
        lines.append(f"| `{lc}` | {counts[0]} | {counts[1]} | {counts[2]} | {counts[3]} |")

    lines.extend([
        "",
        "## Parameter Ranges",
        "- **Cable Length**: 20 m – 200 m (OOD long_cable: 200 m – 300 m)",
        "- **Characteristic Impedance Z0**: 48 Ω – 52 Ω (OOD z0_shift: 44–47 Ω, 53–56 Ω)",
        "- **Velocity Factor vf**: 0.66 * [0.98, 1.02]",
        "- **Loss Scale**: 0.7 – 1.3 (OOD high_loss: 1.5 – 2.0)",
        "- **Rise Time**: 1.0 ns – 5.0 ns (OOD slow_rise: 5.0 ns – 8.0 ns)",
        "- **Fault Position d/l**: 0.10 – 0.95 (OOD near_load: 0.3 m – 3.0 m from load)",
    ])

    with open(card_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Dataset card written to {card_path}")


def create_debug_plot(data_dir: Path, suffix: str, debug_dir: Path):
    """Save quick verification plot showing one trace per fault class with echo positions."""
    train_h5 = data_dir / f"train{suffix}.h5"
    train_meta = pd.read_parquet(data_dir / f"train{suffix}_meta.parquet")

    with h5py.File(train_h5, "r") as f:
        traces = np.array(f["traces"][:500])
        t_grid = np.array(f["traces"].attrs["t_grid"])

    fault_classes = ["none", "short", "open", "shunt_R", "shunt_C", "series_R", "z0_section"]
    fig, axes = plt.subplots(len(fault_classes), 1, figsize=(9, 14), sharex=True)

    for i, fc in enumerate(fault_classes):
        ax = axes[i]
        # Pick first sample with this fault class
        sub_df = train_meta[train_meta["fault_class"] == fc]
        idx = int(sub_df.index[0])
        row = sub_df.iloc[0]
        v = traces[idx]

        ax.plot(t_grid, v, label=f"v(0,t) [{fc}]", color="navy", lw=1.5)
        # Mark incident launch
        ax.axvline(0.05, color="gray", linestyle="--", alpha=0.7, label="Launch (0.05 T)")

        # Mark fault position if exists
        if fc != "none":
            d_frac = row["fault_pos_frac"]
            vf_rel = row["vf"] / 0.66
            t_fault = 0.05 + d_frac / vf_rel
            ax.axvline(t_fault, color="crimson", linestyle=":", lw=1.8, label=f"Fault ({d_frac:.2f} l)")

        # Mark load echo
        t_load = 0.05 + 1.0 / (row["vf"] / 0.66)
        ax.axvline(t_load, color="forestgreen", linestyle="--", alpha=0.7, label=f"Load ({row['load_class']})")

        ax.set_ylabel("Voltage (V)", fontsize=9)
        ax.set_title(f"Fault: {fc} | Load: {row['load_class']} | L={row['length_m']:.1f}m", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Normalized Time t / T_nom", fontsize=10)
    fig.tight_layout()
    plot_path = debug_dir / "debug_fault_gallery.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Debug verification plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=Path)
    parser.add_argument("--quick", action="store_true", help="Generate small quick mode datasets")
    parser.add_argument("--out-dir", default="data", type=Path)
    args = parser.parse_args()

    generate_all_splits(cfg_path=args.config, quick=args.quick, out_dir=args.out_dir)


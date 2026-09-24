#!/usr/bin/env python3
"""Plot MONZA CC recall/FPR summaries from MONZA CSV/H5 outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _fpr_frr_io import load_fpr_frr as _load_fpr_frr_csv

# Global matplotlib aesthetic defaults for large, high-visibility plots
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.titlesize': 22,
})

SELECTED_CCS = ["2", "3", "5", "7", "8", "5_clean"]
COMPARISON_CCS = [2, 3, 5, 7, 8, "5_clean"]
ATTACK_TYPES = [
    # "malicious_alie",  # Commented out for unquarantined baseline testing
    "malicious_label",
    "malicious_random",
    "malicious_shuffle",
    "malicious_zeros",
]
DEFENSE_LABELS = {
    "2": "zPROBE",
    "2_noquarantine": "zPROBE - NoQuar",
    "3": "MONZA",
    "3_noquarantine": "MONZA - NoQuar",
    "5": "Without defense",
    "5_clean": "Default FL",
    "7": "DODGE",
    "7_noquarantine": "DODGE - NoQuar",
    "8": "FedSIGN",
    "8_noquarantine": "FedSIGN - NoQuar",
    2: "zPROBE",
    3: "MONZA",
    5: "Without defense",
    "5_clean": "Default FL",
    7: "DODGE",
    8: "FedSIGN",
}

COLORS = {
    "zPROBE": "#2ca02c",             # Green (matching paper image)
    "zPROBE - NoQuar": "#98df8a",
    "MONZA": "#1f77b4",              # Blue (matching paper image)
    "MONZA - NoQuar": "#aec7e8",
    "Without defense": "#ff7f0e",    # Orange (matching paper image)
    "Default FL": "#d62728",         # Red (matching paper image)
    "DODGE": "#9467bd",              # Purple (DODGE needs a new color since red is Default FL)
    "DODGE - NoQuar": "#c5b0d5",
    "FedSIGN": "#8c564b",            # Brown
    "FedSIGN - NoQuar": "#c49c94",
}

LINESTYLES = {
    "zPROBE": "-",
    "zPROBE - NoQuar": "--",
    "MONZA": "-",
    "MONZA - NoQuar": "--",
    "Without defense": "-",
    "Default FL": "-",
    "DODGE": "-",
    "DODGE - NoQuar": "--",
    "FedSIGN": "-",
    "FedSIGN - NoQuar": "--",
}

INDIVIDUAL_FPR_FRR = {
    "2": "fpr_frr_results_2.csv",
    "2_noquarantine": "fpr_frr_results_2_noquarantine.csv",
    "3": "fpr_frr_results_3.csv",
    "3_noquarantine": "fpr_frr_results_3_noquarantine.csv",
    "7": "fpr_frr_results_7.csv",
    "7_noquarantine": "fpr_frr_results_7_noquarantine.csv",
    "8": "fpr_frr_results_8.csv",
    "8_noquarantine": "fpr_frr_results_8_noquarantine.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system-dir",
        type=Path,
        nargs="+",
        default=[Path("PFLlibMonza/system")],
        help="Directory containing cc_type_results_*.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/runs/mnist/manual/analysis"),
        help="Directory where plots and summary CSV are written.",
    )
    parser.add_argument(
        "--tail-rounds",
        type=int,
        default=30,
        help="Number of final rounds used for the summary.",
    )
    parser.add_argument(
        "--dataset",
        default="MNIST",
        help="Dataset prefix used to locate result H5 files, e.g. MNIST or Cifar10.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        nargs="+",
        default=[Path("PFLlibMonza/results")],
        help="Directory containing *_FedAvg_*_test_*.h5 result files.",
    )
    parser.add_argument(
        "--num-malicious",
        type=int,
        default=30,
        help="Malicious-client count used to select comparison H5 files.",
    )
    return parser.parse_args()


def load_cc_type(system_dirs: list[Path], min_rounds: int, selected_ccs: list) -> pd.DataFrame:
    frames = []
    for system_dir in system_dirs:
        for cc in selected_ccs:
            path = system_dir / f"cc_type_results_{cc}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            # Load all runs instead of only the latest run
            for col in ["Round", "Total", "Removed"]:
                if col in df.columns:
                    df[col] = df[col].astype(int)
            df["Rate"] = df["Rate"].astype(float)
            df["CC_Key"] = str(cc)
            df["Defense"] = DEFENSE_LABELS.get(str(cc), f"cc={cc}")
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"Nenhum cc_type_results_*.csv encontrado.")
    return pd.concat(frames, ignore_index=True)


def latest_run(df: pd.DataFrame, min_rounds: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "RunID" in df.columns:
        run_ids = list(df["RunID"].drop_duplicates())
        for run_id in reversed(run_ids):
            run = df[df["RunID"] == run_id].copy()
            if run["Round"].astype(int).nunique() >= min_rounds:
                return run
        return df[df["RunID"] == run_ids[-1]].copy()
    rounds = df["Round"].astype(int)
    starts = df.index[rounds < rounds.shift(fill_value=rounds.iloc[0])].tolist()
    start = starts[-1] if starts else 0
    return df.loc[start:].copy()


def load_fpr_frr(system_dirs: list[Path], min_rounds: int, selected_ccs: list) -> dict[str, pd.DataFrame]:
    raw_frames: dict[str, list[pd.DataFrame]] = {}
    for system_dir in system_dirs:
        for cc in selected_ccs:
            path = system_dir / f"fpr_frr_results_{cc}.csv"
            if not path.exists():
                continue
            df = _load_fpr_frr_csv(path, min_rounds=0)
            if "QuarantineFPR" not in df.columns:
                continue
            label = DEFENSE_LABELS.get(str(cc), f"cc={cc}")
            raw_frames.setdefault(label, []).append(df)
        
    frames: dict[str, pd.DataFrame] = {}
    for label, dfs in raw_frames.items():
        frames[label] = pd.concat(dfs, ignore_index=True)
    return frames


def find_all_h5(
    system_dir: Path,
    results_dir: Path,
    dataset: str,
    cc: str | int,
    max_rounds: int | None = None,
) -> list[Path]:
    search_dirs = [system_dir, results_dir]
    cc_str = str(cc)
    cc_clean = cc_str.split("_")[0]
    is_noquar = "noquarantine" in cc_str
    
    valid_paths = []

    for d in search_dirs:
        if not d.exists():
            continue
        pattern = f"*{dataset}*{cc_clean}*.h5"
        raw_candidates = list(d.glob(pattern))
        matching = [
            p for p in raw_candidates
            if ("noquarantine" in p.name.lower()) == is_noquar
        ]
        if "clean" in cc_str:
            matching = [p for p in matching if "_0_test_" in p.name]
        elif cc_clean == "5":
            matching = [p for p in matching if "_0_test_" not in p.name]
        
        for p in matching:
            if max_rounds is not None:
                try:
                    with h5py.File(p, "r") as h5:
                        if "rs_test_acc" in h5:
                            # Allow some padding for runs that go slightly over
                            if len(h5["rs_test_acc"]) <= max_rounds + 5:
                                valid_paths.append(p)
                except Exception:
                    pass
            else:
                valid_paths.append(p)
                
    return list(set(valid_paths))


def load_accuracy(
    system_dirs: list[Path],
    results_dirs: list[Path],
    dataset: str,
    selected_ccs: list,
    max_rounds: int | None = None,
) -> pd.DataFrame:
    rows = []
    for cc in selected_ccs:
        for system_dir, results_dir in zip(system_dirs, results_dirs):
            paths = find_all_h5(system_dir, results_dir, dataset, cc, max_rounds=max_rounds)
            for path in paths:
                with h5py.File(path, "r") as h5:
                    if "rs_test_acc" not in h5:
                        continue
                    acc = np.asarray(h5["rs_test_acc"], dtype=float)
                    if max_rounds is not None and len(acc) > max_rounds + 1:
                        acc = acc[: max_rounds + 1]
                label = DEFENSE_LABELS.get(str(cc), f"cc={cc}")
                for round_idx, value in enumerate(acc):
                    rows.append(
                        {
                            "Round": round_idx,
                            "Accuracy": float(value),
                            "Defense": label,
                        }
                    )
    df = pd.DataFrame(rows)
    return df


def load_loss(
    system_dirs: list[Path],
    results_dirs: list[Path],
    dataset: str,
    selected_ccs: list,
    max_rounds: int | None = None,
) -> pd.DataFrame:
    rows = []
    for cc in selected_ccs:
        for system_dir, results_dir in zip(system_dirs, results_dirs):
            paths = find_all_h5(system_dir, results_dir, dataset, cc, max_rounds=max_rounds)
            for path in paths:
                with h5py.File(path, "r") as h5:
                    if "rs_train_loss" not in h5:
                        continue
                    loss = np.asarray(h5["rs_train_loss"], dtype=float)
                    if max_rounds is not None and len(loss) > max_rounds + 1:
                        loss = loss[: max_rounds + 1]
                label = DEFENSE_LABELS.get(str(cc), f"cc={cc}")
                for round_idx, value in enumerate(loss):
                    rows.append(
                        {
                            "Round": round_idx,
                            "Loss": float(value),
                            "Defense": label,
                        }
                    )
    df = pd.DataFrame(rows)
    return df


def summarize_tail(df: pd.DataFrame, tail_rounds: int) -> pd.DataFrame:
    rows = []
    group_col = "CC_Key" if "CC_Key" in df.columns else "Defense"
    for cc_key, cc_group in df.groupby(group_col, sort=True):
        tail_round_values = sorted(cc_group["Round"].astype(int).unique())[-tail_rounds:]
        tail_cc = cc_group[cc_group["Round"].isin(tail_round_values)]
        defense = DEFENSE_LABELS.get(str(cc_key), str(cc_key))
        for attack_type, group in tail_cc.groupby("AttackType", sort=True):
            total = group["Total"].sum()
            removed = group["Removed"].sum()
            rows.append(
                {
                    "CC": str(cc_key),
                    "CC_Key": str(cc_key),
                    "Defense": defense,
                    "AttackType": attack_type,
                    "Total": int(total),
                    "Removed": int(removed),
                    "Rate": float(removed / total) if total else 0.0,
                    "Metric": "FPR" if attack_type == "benign" else "recall",
                }
            )
    return pd.DataFrame(rows)


def plot_fpr(dfs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    if not dfs:
        return
    min_val = min((df["QuarantineFPR"].min() for df in dfs.values() if not df.empty), default=0.0)
    max_val = max((df["QuarantineFPR"].max() for df in dfs.values() if not df.empty), default=1.0)
    margin = max((max_val - min_val) * 0.1, 0.05)

    fig, ax = plt.subplots(figsize=(20, 10))
    for name, df in dfs.items():
        color = COLORS.get(name, "#333333")
        linestyle = LINESTYLES.get(name, "-")
        grouped = df.groupby("Round")["QuarantineFPR"]
        mean_line = grouped.mean()
        std = grouped.std().fillna(0)
        ax.plot(mean_line.index, mean_line * 100, label=name, color=color, linestyle=linestyle, linewidth=4.0)
        ax.fill_between(mean_line.index, (mean_line - std).clip(lower=0) * 100, (mean_line + std).clip(upper=1.0) * 100, color=color, alpha=0.3)
    ax.set_xlabel("Rounds", fontsize=40, fontweight="bold", labelpad=10)
    ax.set_ylabel("False Positive Rate (%)", fontsize=40, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=40, length=7, width=2)
    ax.grid(True, alpha=0.3, linewidth=1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=40, frameon=True, edgecolor="black", fancybox=True)
    ax.set_ylim(max(0.0, (min_val - margin) * 100), min(102.0, (max_val + margin) * 100))
    fig.tight_layout()
    fig.savefig(out_dir / "plot_fpr.pdf", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_frr(dfs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    if not dfs:
        return
    min_val = min((df["QuarantineFRR"].min() for df in dfs.values() if not df.empty), default=0.0)
    max_val = max((df["QuarantineFRR"].max() for df in dfs.values() if not df.empty), default=1.0)
    margin = max((max_val - min_val) * 0.1, 0.05)

    fig, ax = plt.subplots(figsize=(20, 10))
    for name, df in dfs.items():
        color = COLORS.get(name, "#333333")
        linestyle = LINESTYLES.get(name, "-")
        grouped = df.groupby("Round")["QuarantineFRR"]
        mean_line = grouped.mean()
        std = grouped.std().fillna(0)
        ax.plot(mean_line.index, mean_line * 100, label=name, color=color, linestyle=linestyle, linewidth=4.0)
        ax.fill_between(mean_line.index, (mean_line - std).clip(lower=0) * 100, (mean_line + std).clip(upper=1.0) * 100, color=color, alpha=0.3)
    ax.set_xlabel("Rounds", fontsize=40, fontweight="bold", labelpad=10)
    ax.set_ylabel("False Rejection Rate (%)", fontsize=40, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=40, length=7, width=2)
    ax.grid(True, alpha=0.3, linewidth=1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=40, frameon=True, edgecolor="black", fancybox=True)
    ax.set_ylim(max(0.0, (min_val - margin) * 100), min(102.0, (max_val + margin) * 100))
    fig.tight_layout()
    fig.savefig(out_dir / "plot_frr.pdf", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(accuracy_df: pd.DataFrame, out_dir: Path) -> None:
    if accuracy_df.empty:
        return
    fig, ax = plt.subplots(figsize=(20, 10))
    for name, group in accuracy_df.groupby("Defense", sort=False):
        color = COLORS.get(name, "#333333")
        linestyle = LINESTYLES.get(name, "-")
        grouped = group.groupby("Round")["Accuracy"]
        mean_line = grouped.mean().rolling(window=5, min_periods=1).mean()
        std = grouped.std().fillna(0).rolling(window=5, min_periods=1).mean()
        ax.plot(mean_line.index, mean_line * 100, label=name, color=color, linestyle=linestyle, linewidth=4.0)
        ax.fill_between(mean_line.index, (mean_line - std).clip(lower=0) * 100, (mean_line + std).clip(upper=1.02) * 100, color=color, alpha=0.3)
    ax.set_xlabel("Rounds", fontsize=40, fontweight="bold", labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=40, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=40, length=7, width=2)
    min_val = accuracy_df["Accuracy"].min()
    max_val = accuracy_df["Accuracy"].max()
    margin = max((max_val - min_val) * 0.1, 0.05)
    ax.set_ylim(max(0.0, (min_val - margin) * 100), min(102.0, (max_val + margin) * 100))
    ax.grid(True, alpha=0.3, linewidth=1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=40, frameon=True, edgecolor="black", fancybox=True)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_accuracy.pdf", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_loss(loss_df: pd.DataFrame, out_dir: Path) -> None:
    if loss_df.empty:
        return
    fig, ax = plt.subplots(figsize=(20, 10))
    for name, group in loss_df.groupby("Defense", sort=False):
        color = COLORS.get(name, "#333333")
        linestyle = LINESTYLES.get(name, "-")
        grouped = group.groupby("Round")["Loss"]
        mean_line = grouped.mean()
        # Remove standard deviation shading and smoothing to match paper aesthetic
        ax.plot(mean_line.index, mean_line, label=name, color=color, linestyle=linestyle, linewidth=3.0)
    ax.set_xlabel("Rounds", fontsize=40, fontweight="bold", labelpad=10)
    ax.set_ylabel("Loss", fontsize=40, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=40, length=7, width=2)
    ax.set_ylim(0.0, 3.0)
    ax.grid(True, alpha=0.3, linewidth=1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=40, frameon=True, edgecolor="black", fancybox=True)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_loss.pdf", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_recall(df: pd.DataFrame, out_dir: Path) -> None:
    group_col = "CC_Key" if "CC_Key" in df.columns else "CC"
    recall_df = df[
        (df["AttackType"].isin(ATTACK_TYPES))
        & (df["Metric"].str.lower() == "recall")
    ].copy()
    if recall_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(20, 10))
    for cc_key in SELECTED_CCS:
        group = recall_df[recall_df[group_col].astype(str) == str(cc_key)].sort_values("Round")
        if group.empty:
            continue
        label = DEFENSE_LABELS.get(str(cc_key), f"cc={cc_key}")
        color = COLORS.get(label, "#333333")
        linestyle = LINESTYLES.get(label, "-")
        
        grouped = group.groupby("Round")["Rate"]
        mean_line = grouped.mean().rolling(window=5, min_periods=1).mean()
        std = grouped.std().fillna(0).rolling(window=5, min_periods=1).mean()

        ax.plot(
            mean_line.index,
            mean_line * 100,
            marker="o",
            markersize=10,
            linewidth=4.0,
            linestyle=linestyle,
            color=color,
            label=label,
        )
        ax.fill_between(mean_line.index, (mean_line - std).clip(lower=0) * 100, (mean_line + std).clip(upper=1.0) * 100, color=color, alpha=0.3)
        
    ax.set_xlabel("Rounds", fontsize=40, fontweight="bold", labelpad=10)
    ax.set_ylabel("Recall (%)", fontsize=40, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=40, length=7, width=2)
    ax.set_ylim(-2.0, 105.0)
    ax.grid(True, alpha=0.3, linewidth=1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=40, frameon=True, edgecolor="black", fancybox=True)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_recall.pdf", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_cc_type(args.system_dir, args.tail_rounds, SELECTED_CCS)
    max_rounds = int(df["Round"].max()) if not df.empty and "Round" in df.columns else None

    fpr_frr = load_fpr_frr(args.system_dir, args.tail_rounds, SELECTED_CCS)
    accuracy = load_accuracy(args.system_dir, args.results_dir, args.dataset, SELECTED_CCS, max_rounds=max_rounds)
    loss = load_loss(args.system_dir, args.results_dir, args.dataset, SELECTED_CCS, max_rounds=max_rounds)
    summary = summarize_tail(df, args.tail_rounds)
    summary.to_csv(args.out_dir / "cc_attack_type_summary.csv", index=False)

    plot_fpr(fpr_frr, args.out_dir)
    plot_frr(fpr_frr, args.out_dir)
    plot_accuracy(accuracy, args.out_dir)
    plot_loss(loss, args.out_dir)
    plot_recall(df, args.out_dir)
    print(summary.sort_values(["CC", "AttackType"]).to_string(index=False))
    print(f"\nArquivos salvos em {args.out_dir}")


if __name__ == "__main__":
    main()

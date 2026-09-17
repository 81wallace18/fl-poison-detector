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

SELECTED_CCS = ["2", "2_noquarantine", "3", "3_noquarantine", "7", "7_noquarantine", "8", "8_noquarantine"]
COMPARISON_CCS = [2, 3, 5, 7, 8]
ATTACK_TYPES = [
    # "malicious_alie",  # Commented out for unquarantined baseline testing
    "malicious_label",
    "malicious_random",
    "malicious_shuffle",
    "malicious_zeros",
]
DEFENSE_LABELS = {
    "2": "CC2 (zPROBE + Quar)",
    "2_noquarantine": "CC2 (zPROBE - NoQuar)",
    "3": "CC3 (MONZA + Quar)",
    "3_noquarantine": "CC3 (MONZA - NoQuar)",
    "5": "CC5 (No Defense)",
    "7": "CC7 (MLP + Quar)",
    "7_noquarantine": "CC7 (MLP - NoQuar)",
    "8": "CC8 (FedSIGN + Quar)",
    "8_noquarantine": "CC8 (FedSIGN - NoQuar)",
    2: "CC2 (zPROBE + Quar)",
    3: "CC3 (MONZA + Quar)",
    5: "CC5 (No Defense)",
    7: "CC7 (MLP + Quar)",
    8: "CC8 (FedSIGN + Quar)",
}

COLORS = {
    "CC2 (zPROBE + Quar)": "#1f77b4",        # Blue
    "CC2 (zPROBE - NoQuar)": "#aec7e8",      # Light Blue
    "CC3 (MONZA + Quar)": "#ff7f0e",         # Solid Orange
    "CC3 (MONZA - NoQuar)": "#e6550d",       # Dark Orange
    "CC5 (No Defense)": "#7f7f7f",           # Gray
    "CC7 (MLP + Quar)": "#d62728",           # Solid Red
    "CC7 (MLP - NoQuar)": "#9467bd",         # Purple
    "CC8 (FedSIGN + Quar)": "#2ca02c",       # Green
    "CC8 (FedSIGN - NoQuar)": "#98df8a",     # Light Green
}

LINESTYLES = {
    "CC2 (zPROBE + Quar)": "-",
    "CC2 (zPROBE - NoQuar)": "--",
    "CC3 (MONZA + Quar)": "-",
    "CC3 (MONZA - NoQuar)": "--",
    "CC5 (No Defense)": ":",
    "CC7 (MLP + Quar)": "-",
    "CC7 (MLP - NoQuar)": "--",
    "CC8 (FedSIGN + Quar)": "-",
    "CC8 (FedSIGN - NoQuar)": "--",
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
        default=Path("PFLlibMonza/system"),
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
        default=Path("PFLlibMonza/results"),
        help="Directory containing *_FedAvg_*_test_*.h5 result files.",
    )
    parser.add_argument(
        "--num-malicious",
        type=int,
        default=30,
        help="Malicious-client count used to select comparison H5 files.",
    )
    return parser.parse_args()


def load_cc_type(system_dir: Path, min_rounds: int, selected_ccs: list) -> pd.DataFrame:
    frames = []
    for cc in selected_ccs:
        path = system_dir / f"cc_type_results_{cc}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = latest_run(df, min_rounds=min_rounds)
        for col in ["Round", "Total", "Removed"]:
            if col in df.columns:
                df[col] = df[col].astype(int)
        df["Rate"] = df["Rate"].astype(float)
        df["CC_Key"] = str(cc)
        df["Defense"] = DEFENSE_LABELS.get(str(cc), f"cc={cc}")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"Nenhum cc_type_results_*.csv encontrado em {system_dir}")
    out = pd.concat(frames, ignore_index=True)
    return out


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


def load_fpr_frr(system_dir: Path, min_rounds: int, selected_ccs: list) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for cc in selected_ccs:
        path = system_dir / f"fpr_frr_results_{cc}.csv"
        if not path.exists():
            continue
        df = _load_fpr_frr_csv(path, min_rounds=min_rounds)
        if "DetectionFPR" not in df.columns:
            raise ValueError(f"{path.name} sem coluna de deteccao (Detection/Upload FPR)")
        label = DEFENSE_LABELS.get(str(cc), f"cc={cc}")
        frames[label] = df
    return frames


def find_latest_h5(
    system_dir: Path,
    results_dir: Path,
    dataset: str,
    cc: str | int,
    max_rounds: int | None = None,
) -> Path | None:
    search_dirs = [system_dir, results_dir]
    cc_str = str(cc)
    cc_clean = cc_str.split("_")[0]
    is_noquar = "noquarantine" in cc_str

    for d in search_dirs:
        if not d.exists():
            continue
        pattern = f"*{dataset}*{cc_clean}*.h5"
        raw_candidates = list(d.glob(pattern))
        matching = [
            p for p in raw_candidates
            if ("noquarantine" in p.name.lower()) == is_noquar
        ]
        candidates = sorted(matching if matching else raw_candidates, key=lambda p: p.stat().st_mtime)
        if not candidates:
            continue
        if max_rounds is not None:
            for p in reversed(candidates):
                try:
                    with h5py.File(p, "r") as h5:
                        if "rs_test_acc" in h5:
                            if len(h5["rs_test_acc"]) <= max_rounds + 5:
                                return p
                except Exception:
                    pass
        return candidates[-1]
    return None


def load_accuracy(
    system_dir: Path,
    results_dir: Path,
    dataset: str,
    selected_ccs: list,
    max_rounds: int | None = None,
) -> pd.DataFrame:
    rows = []
    for cc in selected_ccs:
        path = find_latest_h5(system_dir, results_dir, dataset, cc, max_rounds=max_rounds)
        if path is None:
            continue
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
    return pd.DataFrame(rows)


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


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    order = ["benign", "malicious_label", "malicious_zeros", "malicious_random", "malicious_shuffle"]
    pivot = summary.pivot_table(index="AttackType", columns="Defense", values="Rate", aggfunc="mean")
    pivot = pivot.reindex([x for x in order if x in pivot.index] + [x for x in pivot.index if x not in order])

    fig, ax = plt.subplots(figsize=(14, 7))
    pivot.plot.bar(ax=ax, width=0.78, color=[COLORS.get(col, "#333333") for col in pivot.columns])
    ax.axhline(0.05, color="gray", linestyle=":", linewidth=2.0, label="FPR target 5%")
    ax.set_title("FPR on Benign and Recall by Attack Type across Defenses", fontsize=20, fontweight="bold", pad=14)
    ax.set_ylabel("Rate", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_xlabel("", fontsize=18, fontweight="bold")
    ax.set_ylim(0, max(1.0, float(pivot.max().max()) * 1.15))
    ax.set_xticklabels(pivot.index, rotation=20, ha="right", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16, length=7, width=2)
    ax.grid(axis="y", alpha=0.3, linewidth=1.0)
    ax.legend(fontsize=14, frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_cc_recall_by_attack_type.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_label(summary: pd.DataFrame, out_dir: Path) -> None:
    label = summary[summary["AttackType"] == "malicious_label"].sort_values("Defense")
    if label.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [COLORS.get(d, "#8c564b") for d in label["Defense"]]
    ax.bar(label["Defense"], label["Rate"], color=colors, width=0.6)
    ax.set_title("Recall in malicious_label Attack", fontsize=20, fontweight="bold", pad=14)
    ax.set_ylabel("Recall", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylim(0, max(1.0, float(label["Rate"].max()) * 1.2))
    ax.tick_params(axis="both", which="major", labelsize=16, length=7, width=2)
    ax.grid(axis="y", alpha=0.3, linewidth=1.0)
    for idx, value in enumerate(label["Rate"]):
        ax.text(idx, value + 0.02, f"{value:.2%}", ha="center", va="bottom", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_cc_malicious_label_recall.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_fpr_frr_by_round(dfs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    if not dfs:
        return

    # 1. Standalone plot: Detection FPR per Round
    fig, ax = plt.subplots(figsize=(14, 6.5))
    for name, df in dfs.items():
        color = COLORS.get(name, "#333333")
        linestyle = LINESTYLES.get(name, "-")
        ax.plot(df["Round"], df["DetectionFPR"], label=name, color=color, linestyle=linestyle, linewidth=3.0)
        if "QuarantineFPR" in df.columns and "NoQuar" not in name:
            ax.plot(df["Round"], df["QuarantineFPR"], color=color, linewidth=1.5, linestyle=":", alpha=0.6)
    ax.set_title("Detection FPR per Round", fontsize=20, fontweight="bold", pad=14)
    ax.set_xlabel("Round", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylabel("Detection FPR", fontsize=18, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=16, length=7, width=2)
    ax.grid(True, alpha=0.3, linewidth=1.0)
    ax.legend(loc="upper left", fontsize=13, frameon=True, framealpha=0.9)
    ax.set_ylim(-0.01, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_detection_fpr_by_round.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 2. Standalone plot: Detection FRR per Round
    fig, ax = plt.subplots(figsize=(14, 6.5))
    for name, df in dfs.items():
        color = COLORS.get(name, "#333333")
        linestyle = LINESTYLES.get(name, "-")
        ax.plot(df["Round"], df["DetectionFRR"], label=name, color=color, linestyle=linestyle, linewidth=3.0)
        if "QuarantineFRR" in df.columns and "NoQuar" not in name:
            ax.plot(df["Round"], df["QuarantineFRR"], color=color, linewidth=1.5, linestyle=":", alpha=0.6)
    ax.set_title("Detection FRR per Round", fontsize=20, fontweight="bold", pad=14)
    ax.set_xlabel("Round", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylabel("Detection FRR", fontsize=18, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=16, length=7, width=2)
    ax.grid(True, alpha=0.3, linewidth=1.0)
    ax.legend(loc="upper left", fontsize=13, frameon=True, framealpha=0.9)
    ax.set_ylim(-0.01, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_detection_frr_by_round.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 3. Two-panel combined plot (for backwards compatibility)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), sharex=True)
    for name, df in dfs.items():
        color = COLORS.get(name, "#333333")
        linestyle = LINESTYLES.get(name, "-")
        axes[0].plot(df["Round"], df["DetectionFPR"], label=name, color=color, linestyle=linestyle, linewidth=2.8)
        axes[1].plot(df["Round"], df["DetectionFRR"], label=name, color=color, linestyle=linestyle, linewidth=2.8)
        if "QuarantineFPR" in df.columns and "NoQuar" not in name:
            axes[0].plot(df["Round"], df["QuarantineFPR"], color=color, linewidth=1.3, linestyle=":", alpha=0.6)
            axes[1].plot(df["Round"], df["QuarantineFRR"], color=color, linewidth=1.3, linestyle=":", alpha=0.6)

    axes[0].set_title("Detection FPR per Round", fontsize=18, fontweight="bold", pad=12)
    axes[0].set_xlabel("Round", fontsize=16, fontweight="bold", labelpad=8)
    axes[0].set_ylabel("Detection FPR", fontsize=16, fontweight="bold", labelpad=8)
    axes[0].tick_params(axis="both", which="major", labelsize=14)
    axes[0].grid(True, alpha=0.3, linewidth=0.8)
    axes[0].legend(loc="upper left", fontsize=12, frameon=True)
    axes[0].set_ylim(-0.01, 1.02)

    axes[1].set_title("Detection FRR per Round", fontsize=18, fontweight="bold", pad=12)
    axes[1].set_xlabel("Round", fontsize=16, fontweight="bold", labelpad=8)
    axes[1].set_ylabel("Detection FRR", fontsize=16, fontweight="bold", labelpad=8)
    axes[1].tick_params(axis="both", which="major", labelsize=14)
    axes[1].grid(True, alpha=0.3, linewidth=0.8)
    axes[1].legend(loc="upper left", fontsize=12, frameon=True)
    axes[1].set_ylim(-0.01, 1.02)

    fig.tight_layout()
    fig.savefig(out_dir / "plot_fpr_frr_by_round.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(accuracy_df: pd.DataFrame, out_dir: Path) -> None:
    if accuracy_df.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 6.5))
    for name, group in accuracy_df.groupby("Defense", sort=False):
        linestyle = LINESTYLES.get(name, "-")
        ax.plot(
            group["Round"],
            group["Accuracy"],
            label=name,
            color=COLORS.get(name, "#333333"),
            linestyle=linestyle,
            linewidth=3.0,
        )
    ax.set_title("Global Model Test Accuracy per Round", fontsize=20, fontweight="bold", pad=14)
    ax.set_xlabel("Round", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylabel("Test Accuracy", fontsize=18, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=16, length=7, width=2)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3, linewidth=1.0)
    ax.legend(loc="lower right", fontsize=13, frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_global_accuracy_by_round.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_recall_by_round(df: pd.DataFrame, out_dir: Path) -> None:
    group_col = "CC_Key" if "CC_Key" in df.columns else "CC"
    recall_round = df[
        (df["AttackType"].isin(ATTACK_TYPES))
        & (df["Metric"].str.lower() == "recall")
    ].copy()
    if recall_round.empty:
        return

    # 1. Standalone single-panel plots for EACH individual attack type
    for attack_type in ATTACK_TYPES:
        sub = recall_round[recall_round["AttackType"] == attack_type]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(14, 6.5))
        for cc_key in SELECTED_CCS:
            group = sub[sub[group_col].astype(str) == str(cc_key)].sort_values("Round")
            if group.empty:
                continue
            label = DEFENSE_LABELS.get(str(cc_key), f"cc={cc_key}")
            linestyle = LINESTYLES.get(label, "-")
            ax.plot(
                group["Round"],
                group["Rate"],
                marker="o",
                markersize=6,
                linewidth=3.0,
                linestyle=linestyle,
                color=COLORS.get(label, "#333333"),
                label=label,
            )
        ax.set_title(f"Recall per Round: {attack_type}", fontsize=20, fontweight="bold", pad=14)
        ax.set_xlabel("Round", fontsize=18, fontweight="bold", labelpad=10)
        ax.set_ylabel("Recall", fontsize=18, fontweight="bold", labelpad=10)
        ax.tick_params(axis="both", which="major", labelsize=16, length=7, width=2)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.legend(fontsize=13, loc="lower right", frameon=True, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out_dir / f"plot_cc_recall_{attack_type}_by_round.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # 2. Combined 2x2 grid plot (for backwards compatibility)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, attack_type in zip(axes, ATTACK_TYPES):
        sub = recall_round[recall_round["AttackType"] == attack_type]
        for cc_key in SELECTED_CCS:
            group = sub[sub[group_col].astype(str) == str(cc_key)].sort_values("Round")
            if group.empty:
                continue
            label = DEFENSE_LABELS.get(str(cc_key), f"cc={cc_key}")
            linestyle = LINESTYLES.get(label, "-")
            ax.plot(
                group["Round"],
                group["Rate"],
                marker="o",
                markersize=4,
                linewidth=2.2,
                linestyle=linestyle,
                color=COLORS.get(label, "#333333"),
                label=label,
            )
        ax.set_title(f"Attack: {attack_type}", fontsize=16, fontweight="bold", pad=10)
        ax.set_xlabel("Round", fontsize=14, fontweight="bold", labelpad=6)
        ax.set_ylabel("Recall", fontsize=14, fontweight="bold", labelpad=6)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Recall by Attack Type Over Rounds", y=1.06, fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_cc_recall_by_attack_over_rounds.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_cc_type(args.system_dir, args.tail_rounds, SELECTED_CCS)
    max_rounds = int(df["Round"].max()) if not df.empty and "Round" in df.columns else None

    fpr_frr = load_fpr_frr(args.system_dir, args.tail_rounds, SELECTED_CCS)
    accuracy = load_accuracy(args.system_dir, args.results_dir, args.dataset, SELECTED_CCS, max_rounds=max_rounds)
    summary = summarize_tail(df, args.tail_rounds)
    summary.to_csv(args.out_dir / "cc_attack_type_summary.csv", index=False)

    plot_fpr_frr_by_round(fpr_frr, args.out_dir)
    plot_accuracy(accuracy, args.out_dir)
    plot_summary(summary, args.out_dir)
    plot_label(summary, args.out_dir)
    plot_recall_by_round(df, args.out_dir)
    print(summary.sort_values(["CC", "AttackType"]).to_string(index=False))
    print(f"\nArquivos salvos em {args.out_dir}")


if __name__ == "__main__":
    main()

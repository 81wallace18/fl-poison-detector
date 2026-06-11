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

SELECTED_CCS = [3, 6, 7]
ATTACK_TYPES = ["malicious_label", "malicious_random", "malicious_shuffle", "malicious_zeros"]
DEFENSE_LABELS = {
    3: "cc=3 (cosseno+score)",
    6: "cc=6 (NLP DistilBERT)",
    7: "cc=7 (MLP+features)",
}
COLORS = {
    "cc=3 (cosseno+score)": "#ff7f0e",
    "cc=6 (NLP DistilBERT)": "#1f77b4",
    "cc=7 (MLP+features)": "#d62728",
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
        default=Path("analysis_outputs"),
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
    return parser.parse_args()


def load_cc_type(system_dir: Path, min_rounds: int, selected_ccs: list[int]) -> pd.DataFrame:
    frames = []
    for cc in selected_ccs:
        path = system_dir / f"cc_type_results_{cc}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = latest_run(df, min_rounds=min_rounds)
        for col in ["Round", "CC", "Total", "Removed"]:
            df[col] = df[col].astype(int)
        df["Rate"] = df["Rate"].astype(float)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"Nenhum cc_type_results_*.csv encontrado em {system_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["Defense"] = out["CC"].map(DEFENSE_LABELS).fillna("cc=" + out["CC"].astype(str))
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


def load_fpr_frr(system_dir: Path, min_rounds: int, selected_ccs: list[int]) -> dict[str, pd.DataFrame]:
    # Columns are normalized to DetectionFPR/DetectionFRR (per-round, paper Eq 14/15)
    # and QuarantineFPR/QuarantineFRR (quarantine-occupancy diagnostic). Legacy CSVs
    # with UploadFPR/FPR are mapped automatically by _fpr_frr_io.load_fpr_frr.
    frames: dict[str, pd.DataFrame] = {}
    for cc in selected_ccs:
        path = system_dir / f"fpr_frr_results_{cc}.csv"
        if not path.exists():
            continue
        df = _load_fpr_frr_csv(path, min_rounds=min_rounds)
        if "DetectionFPR" not in df.columns:
            raise ValueError(f"{path.name} sem coluna de deteccao (Detection/Upload FPR)")
        frames[DEFENSE_LABELS.get(cc, f"cc={cc}")] = df
    return frames


def find_latest_h5(results_dir: Path, dataset: str, cc: int) -> Path | None:
    if not results_dir.exists():
        return None
    candidates = sorted(
        results_dir.glob(f"{dataset}_FedAvg_{cc}_*_test_*.h5"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def load_accuracy(results_dir: Path, dataset: str, selected_ccs: list[int]) -> pd.DataFrame:
    rows = []
    for cc in selected_ccs:
        path = find_latest_h5(results_dir, dataset, cc)
        if path is None:
            continue
        with h5py.File(path, "r") as h5:
            if "rs_test_acc" not in h5:
                continue
            acc = np.asarray(h5["rs_test_acc"], dtype=float)
        for round_idx, value in enumerate(acc):
            rows.append(
                {
                    "Round": round_idx,
                    "Accuracy": float(value),
                    "Defense": DEFENSE_LABELS.get(cc, f"cc={cc}"),
                }
            )
    return pd.DataFrame(rows)


def summarize_tail(df: pd.DataFrame, tail_rounds: int) -> pd.DataFrame:
    rows = []
    for cc, cc_group in df.groupby("CC", sort=True):
        tail_round_values = sorted(cc_group["Round"].astype(int).unique())[-tail_rounds:]
        tail_cc = cc_group[cc_group["Round"].isin(tail_round_values)]
        for attack_type, group in tail_cc.groupby("AttackType", sort=True):
            total = group["Total"].sum()
            removed = group["Removed"].sum()
            rows.append(
                {
                    "CC": cc,
                    "Defense": DEFENSE_LABELS.get(int(cc), f"cc={cc}"),
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

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot.bar(ax=ax, width=0.78)
    ax.axhline(0.05, color="gray", linestyle=":", linewidth=1.3, label="FPR target 5%")
    ax.set_title("FPR em benignos e recall por tipo de ataque nos CCs")
    ax.set_ylabel("Taxa")
    ax.set_xlabel("")
    ax.set_ylim(0, max(1.0, float(pivot.max().max()) * 1.15))
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_cc_recall_by_attack_type.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_label(summary: pd.DataFrame, out_dir: Path) -> None:
    label = summary[summary["AttackType"] == "malicious_label"].sort_values("CC")
    if label.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(label["Defense"], label["Rate"], color="#8c564b")
    ax.set_title("Recall do CC em malicious_label")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, max(1.0, float(label["Rate"].max()) * 1.2))
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(label["Rate"]):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_cc_malicious_label_recall.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_fpr_frr_by_round(dfs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    if not dfs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
    for name, df in dfs.items():
        color = COLORS.get(name, "#333333")
        # Headline: per-round detection rate (paper Eq 14/15).
        axes[0].plot(df["Round"], df["DetectionFPR"], label=name, color=color, linewidth=2)
        axes[1].plot(df["Round"], df["DetectionFRR"], label=name, color=color, linewidth=2)
        # Diagnostic: quarantine-occupancy snapshot (dashed, thinner).
        if "QuarantineFPR" in df.columns:
            axes[0].plot(df["Round"], df["QuarantineFPR"], color=color, linewidth=1,
                         linestyle="--", alpha=0.5)
            axes[1].plot(df["Round"], df["QuarantineFRR"], color=color, linewidth=1,
                         linestyle="--", alpha=0.5)
    axes[0].set_title("Detection FPR por round (tracejado = ocupacao de quarentena)")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("DetectionFPR")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_ylim(-0.01, max(0.30, axes[0].get_ylim()[1]))
    axes[1].set_title("Detection FRR por round (tracejado = ocupacao de quarentena)")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("DetectionFRR")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].set_ylim(-0.01, max(0.60, axes[1].get_ylim()[1]))
    fig.tight_layout()
    fig.savefig(out_dir / "plot_fpr_frr_by_round.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(accuracy_df: pd.DataFrame, out_dir: Path) -> None:
    if accuracy_df.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, group in accuracy_df.groupby("Defense", sort=False):
        ax.plot(
            group["Round"],
            group["Accuracy"],
            label=name,
            color=COLORS.get(name, "#333333"),
            linewidth=2,
        )
    ax.set_title("Acuracia global federada por round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Acuracia de teste")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_global_accuracy_by_round.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_recall_by_round(df: pd.DataFrame, out_dir: Path) -> None:
    recall_round = df[
        (df["CC"].isin(SELECTED_CCS))
        & (df["AttackType"].isin(ATTACK_TYPES))
        & (df["Metric"].str.lower() == "recall")
    ].copy()
    if recall_round.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, attack_type in zip(axes, ATTACK_TYPES):
        sub = recall_round[recall_round["AttackType"] == attack_type]
        for cc in SELECTED_CCS:
            group = sub[sub["CC"] == cc].sort_values("Round")
            if group.empty:
                continue
            label = DEFENSE_LABELS.get(cc, f"cc={cc}")
            ax.plot(
                group["Round"],
                group["Rate"],
                marker="o",
                markersize=3,
                linewidth=1.8,
                color=COLORS.get(label),
                label=label,
            )
        ax.set_title(attack_type)
        ax.set_xlabel("Round")
        ax.set_ylabel("Recall")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
    fig.suptitle("Recall por tipo de ataque ao longo dos rounds", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_cc_recall_by_attack_over_rounds.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    label_round = recall_round[recall_round["AttackType"] == "malicious_label"]
    for cc in SELECTED_CCS:
        group = label_round[label_round["CC"] == cc].sort_values("Round")
        if group.empty:
            continue
        label = DEFENSE_LABELS.get(cc, f"cc={cc}")
        ax.plot(
            group["Round"],
            group["Rate"],
            marker="o",
            markersize=4,
            linewidth=2,
            color=COLORS.get(label),
            label=label,
        )
    ax.set_title("Recall em malicious_label por round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Recall")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_cc_malicious_label_recall_by_round.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_cc_type(args.system_dir, args.tail_rounds, SELECTED_CCS)
    fpr_frr = load_fpr_frr(args.system_dir, args.tail_rounds, SELECTED_CCS)
    accuracy = load_accuracy(args.results_dir, args.dataset, SELECTED_CCS)
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

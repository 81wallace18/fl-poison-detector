#!/usr/bin/env python3
"""Summarize MONZA threshold sweep runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _fpr_frr_io import normalize_columns

try:
    import h5py
except ImportError:  # pragma: no cover - optional at import time
    h5py = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("analysis_outputs/threshold_sweep"),
        help="Directory containing one subdirectory per threshold candidate.",
    )
    parser.add_argument(
        "--tail-rounds",
        type=int,
        default=30,
        help="Number of final rounds used for FPR/FRR and attack summaries.",
    )
    return parser.parse_args()


def latest_run(df: pd.DataFrame, min_rounds: int) -> pd.DataFrame:
    if "RunID" not in df.columns:
        return df.copy()
    run_ids = list(df["RunID"].drop_duplicates())
    for run_id in reversed(run_ids):
        run = df[df["RunID"] == run_id].copy()
        if run["Round"].astype(int).nunique() >= min_rounds:
            return run
    return df[df["RunID"] == run_ids[-1]].copy()


def read_accuracy(path: Path) -> tuple[float | None, float | None]:
    if h5py is None or not path.exists():
        return None, None
    with h5py.File(path, "r") as h5:
        if "rs_test_acc" not in h5:
            return None, None
        acc = h5["rs_test_acc"][:]
    if len(acc) == 0:
        return None, None
    return float(acc.max()), float(acc[-1])


def summarize_candidate(candidate_dir: Path, tail_rounds: int) -> dict | None:
    meta_path = candidate_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    cc = int(meta["cc"])

    fpr_path = candidate_dir / f"fpr_frr_results_{cc}.csv"
    type_path = candidate_dir / f"cc_type_results_{cc}.csv"
    if not fpr_path.exists() or not type_path.exists():
        return None

    fpr_df = latest_run(normalize_columns(pd.read_csv(fpr_path)), min_rounds=tail_rounds)
    type_df = latest_run(pd.read_csv(type_path), min_rounds=tail_rounds)

    fpr_tail_rounds = sorted(fpr_df["Round"].astype(int).unique())[-tail_rounds:]
    fpr_tail = fpr_df[fpr_df["Round"].isin(fpr_tail_rounds)]

    type_tail_rounds = sorted(type_df["Round"].astype(int).unique())[-tail_rounds:]
    type_tail = type_df[type_df["Round"].isin(type_tail_rounds)]

    row = {
        "Detector": meta["detector"],
        "CC": cc,
        "Threshold": float(meta["threshold"]),
        "Rounds": int(fpr_df["Round"].astype(int).nunique()),
        "DetectionFPR_mean_tail": float(fpr_tail["DetectionFPR"].mean()),
        "DetectionFRR_mean_tail": float(fpr_tail["DetectionFRR"].mean()),
        "QuarantineFPR_mean_tail": float(fpr_tail["QuarantineFPR"].mean()),
        "QuarantineFRR_mean_tail": float(fpr_tail["QuarantineFRR"].mean()),
    }

    for attack_type, group in type_tail.groupby("AttackType", sort=True):
        total = int(group["Total"].sum())
        removed = int(group["Removed"].sum())
        rate = float(removed / total) if total else 0.0
        key = "BenignUploadFPR" if attack_type == "benign" else f"Recall_{attack_type}"
        row[key] = rate
        row[f"Total_{attack_type}"] = total
        row[f"Removed_{attack_type}"] = removed

    best_acc, final_acc = read_accuracy(candidate_dir / "result.h5")
    row["BestAccuracy"] = best_acc
    row["FinalAccuracy"] = final_acc
    return row


def plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for detector, group in summary.groupby("Detector", sort=True):
        group = group.sort_values("Threshold")
        ax.plot(group["Threshold"], group[metric], marker="o", label=detector)
    ax.set_title(ylabel + " por threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for detector, group in summary.groupby("Detector", sort=True):
        ax.scatter(
            group["DetectionFPR_mean_tail"],
            group["Recall_malicious_label"],
            s=70,
            label=detector,
        )
        for _, row in group.iterrows():
            ax.annotate(f"{row['Threshold']:.2f}", (row["DetectionFPR_mean_tail"], row["Recall_malicious_label"]))
    ax.axvline(0.05, color="gray", linestyle=":", linewidth=1.2, label="DetectionFPR 5%")
    ax.set_title("Trade-off malicious_label recall vs DetectionFPR")
    ax.set_xlabel("DetectionFPR medio nas ultimas rodadas")
    ax.set_ylabel("Recall malicious_label")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = [
        row
        for candidate_dir in sorted(args.sweep_dir.iterdir())
        if candidate_dir.is_dir()
        for row in [summarize_candidate(candidate_dir, args.tail_rounds)]
        if row is not None
    ]
    if not rows:
        raise FileNotFoundError(f"Nenhum candidato valido encontrado em {args.sweep_dir}")

    summary = pd.DataFrame(rows).sort_values(["Detector", "Threshold"])
    args.sweep_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.sweep_dir / "threshold_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    required = {"Recall_malicious_label", "DetectionFPR_mean_tail", "FinalAccuracy"}
    missing = required - set(summary.columns)
    if not missing:
        plot_metric(summary, "Recall_malicious_label", "Recall malicious_label", args.sweep_dir / "plot_threshold_label_recall.png")
        plot_metric(summary, "DetectionFPR_mean_tail", "DetectionFPR medio", args.sweep_dir / "plot_threshold_upload_fpr.png")
        plot_metric(summary, "FinalAccuracy", "Acuracia final", args.sweep_dir / "plot_threshold_accuracy.png")
        plot_tradeoff(summary, args.sweep_dir / "plot_threshold_tradeoff.png")

    print(summary.to_string(index=False))
    print(f"\nResumo salvo em {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Shared I/O for MONZA fpr_frr_results_*.csv with old/new header normalization.

Two distinct metrics live in these CSVs:

- ``DetectionFPR`` / ``DetectionFRR`` -- per-round detection rate (paper Eq 14/15):
  computed over the clients that uploaded each round (``removed`` vs
  ``is_malicious_round``). This is the headline metric comparable to the paper's
  Table 4 (e.g. CIFAR-10 FPR 3.6% / FRR 7.68%).
- ``QuarantineFPR`` / ``QuarantineFRR`` -- cumulative quarantine-occupancy snapshot:
  fraction of all clients currently held in quarantine. Because quarantine grows
  exponentially (2**n rounds), recurrent benign clients stay quarantined forever and
  this saturates (CIFAR-10 ~0.72 over 300 rounds). Diagnostic only -- NOT the paper FPR.

Legacy CSVs use the column names ``UploadFPR``/``UploadFRR`` (per-round) and
``FPR``/``FRR`` (snapshot). ``load_fpr_frr`` maps both layouts to the new names so old
runs do not need to be re-executed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# legacy column name -> canonical name
_COLUMN_ALIASES = {
    "UploadFPR": "DetectionFPR",
    "UploadFRR": "DetectionFRR",
    "FPR": "QuarantineFPR",
    "FRR": "QuarantineFRR",
    "False Positive Rate": "QuarantineFPR",
    "False Rejection Rate": "QuarantineFRR",
}

DETECTION_COLS = ("DetectionFPR", "DetectionFRR")
QUARANTINE_COLS = ("QuarantineFPR", "QuarantineFRR")
_NUMERIC_COLS = (*DETECTION_COLS, *QUARANTINE_COLS)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with canonical Detection*/Quarantine* column names."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # only rename a legacy name when the canonical one is not already present
    rename = {
        old: new
        for old, new in _COLUMN_ALIASES.items()
        if old in df.columns and new not in df.columns
    }
    if rename:
        df = df.rename(columns=rename)
    return df


def load_fpr_frr(path: str | Path, min_rounds: int = 0) -> pd.DataFrame:
    """Load one ``fpr_frr_results_*.csv`` with normalized columns.

    Keeps only the latest run (by ``RunID``) that has at least ``min_rounds`` distinct
    rounds, mirroring the selection logic used by the plotting scripts. Numeric metric
    columns are coerced to float; missing metric columns are tolerated (older CSVs may
    lack the per-round Detection columns).
    """
    df = pd.read_csv(path)
    df = normalize_columns(df)
    if "Round" not in df.columns:
        raise ValueError(f"{Path(path).name} sem coluna 'Round'")
    df = df[df["Round"].astype(str) != "Round"].reset_index(drop=True)
    df["Round"] = df["Round"].astype(int)
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if min_rounds:
        df = _latest_run(df, min_rounds)
    return df


def _latest_run(df: pd.DataFrame, min_rounds: int) -> pd.DataFrame:
    if "RunID" not in df.columns or df.empty:
        return df.copy()
    run_ids = list(df["RunID"].drop_duplicates())
    for run_id in reversed(run_ids):
        run = df[df["RunID"] == run_id].copy()
        if run["Round"].nunique() >= min_rounds:
            return run
    return df[df["RunID"] == run_ids[-1]].copy()


def summarize_fpr_frr(df: pd.DataFrame, min_round: int = 0) -> dict:
    """Mean/std of Detection*/Quarantine* metrics over rounds ``>= min_round``.

    ``min_round`` should be the steady-state cutoff (e.g. ``round_init_atk``) so the
    warm-up rounds before attacks start do not bias the averages.
    """
    tail = df[df["Round"] >= min_round] if min_round else df
    out: dict = {"rounds_used": int(tail["Round"].nunique())}
    for col in _NUMERIC_COLS:
        if col in tail.columns and tail[col].notna().any():
            out[f"{col}_mean"] = float(tail[col].mean())
            out[f"{col}_std"] = float(tail[col].std(ddof=0))
        else:
            out[f"{col}_mean"] = float("nan")
            out[f"{col}_std"] = float("nan")
    return out


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import sys

    min_round = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    frame = load_fpr_frr(sys.argv[1])
    summary = summarize_fpr_frr(frame, min_round=min_round)
    for key, value in summary.items():
        print(f"{key}: {value}")

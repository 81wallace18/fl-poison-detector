#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=../lib/monza_common.sh
source "$ROOT/scripts/lib/monza_common.sh"
VENV_PY="${VENV_PY:-$ROOT/.venv/bin/python}"
SYSTEM_DIR="$ROOT/PFLlibMonza/system"
DATASET_NAME="${DATASET_NAME:-MNIST}"
MODEL="${MODEL:-CNN}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-50}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
NUM_MALICIOUS="${NUM_MALICIOUS:-30}"
JOIN_RATIO="${JOIN_RATIO:-1}"
DEVICE_ID="${DEVICE_ID:-0}"
LOCAL_STEPS="${LOCAL_STEPS:-1}"
TIMES="${TIMES:-1}"
RATE_FAKE="${RATE_FAKE:-1}"
ROUND_INIT_ATK="${ROUND_INIT_ATK:-5}"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT/artifacts}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DATASET_SLUG="$(monza_dataset_slug "$DATASET_NAME")"
RUN_OUTPUT="${RUN_OUTPUT:-$ARTIFACTS_ROOT/runs/$DATASET_SLUG/${RUN_ID}_threshold_sweep}"
BERT_DIR="${BERT_DIR:-$ARTIFACTS_ROOT/models/$DATASET_SLUG/bert}"
MLP_DIR="${MLP_DIR:-$ARTIFACTS_ROOT/models/$DATASET_SLUG/mlp}"
PUBLIC_VAL_DIR="${PUBLIC_VAL_DIR:-$ROOT/PFLlibMonza/dataset/$DATASET_NAME/public_val}"
SWEEP_DIR="${SWEEP_DIR:-$RUN_OUTPUT/analysis}"
TAIL_ROUNDS="${TAIL_ROUNDS:-30}"
RUN_LOG="${RUN_LOG:-$RUN_OUTPUT/run.log}"

BERT_THRESHOLDS="${BERT_THRESHOLDS:--0.45 -0.50 -0.55 -0.60 -0.65}"
MLP_THRESHOLDS="${MLP_THRESHOLDS:--1.80 -2.00 -2.20 -2.37 -2.55}"

export PUBLIC_VAL_DIR

safe_name() {
  printf '%s' "$1" | sed 's/-/m/g; s/\./p/g'
}

run_monza_threshold() {
  local detector="$1"
  local cc="$2"
  local threshold="$3"
  local detector_dir="$4"
  local candidate_dir="$SWEEP_DIR/${detector}_$(safe_name "$threshold")"

  monza_log "Run ${detector} cc=${cc} threshold=${threshold}"
  rm -rf "$candidate_dir"
  mkdir -p "$candidate_dir"
  rm -f \
    "$SYSTEM_DIR/fpr_frr_results_${cc}.csv" \
    "$SYSTEM_DIR/cc_detail_results_${cc}.csv" \
    "$SYSTEM_DIR/cc_type_results_${cc}.csv"

  cd "$SYSTEM_DIR"
  if [[ "$detector" == "bert" ]]; then
    "$VENV_PY" -u main.py \
      -m "$MODEL" -data "$DATASET_NAME" -nmc "$NUM_MALICIOUS" -nc "$NUM_CLIENTS" \
      -jr "$JOIN_RATIO" -atk all -ria "$ROUND_INIT_ATK" -cc "$cc" -gr "$GLOBAL_ROUNDS" \
      -t "$TIMES" -ls "$LOCAL_STEPS" -did "$DEVICE_ID" -rfake "$RATE_FAKE" \
      --detector_dir "$detector_dir" --bert_threshold_value "$threshold"
  else
    "$VENV_PY" -u main.py \
      -m "$MODEL" -data "$DATASET_NAME" -nmc "$NUM_MALICIOUS" -nc "$NUM_CLIENTS" \
      -jr "$JOIN_RATIO" -atk all -ria "$ROUND_INIT_ATK" -cc "$cc" -gr "$GLOBAL_ROUNDS" \
      -t "$TIMES" -ls "$LOCAL_STEPS" -did "$DEVICE_ID" -rfake "$RATE_FAKE" \
      --detector_dir "$detector_dir" --mlp_threshold_value "$threshold"
  fi
  cd "$ROOT"

  cp "$SYSTEM_DIR/fpr_frr_results_${cc}.csv" "$candidate_dir/"
  cp "$SYSTEM_DIR/cc_detail_results_${cc}.csv" "$candidate_dir/"
  cp "$SYSTEM_DIR/cc_type_results_${cc}.csv" "$candidate_dir/"
  local h5_file
  h5_file="$(find "$ROOT/PFLlibMonza/results" -maxdepth 1 -name "${DATASET_NAME}_FedAvg_${cc}_*_test_0.h5" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -n "$h5_file" ]]; then
    cp "$h5_file" "$candidate_dir/result.h5"
  fi
  "$VENV_PY" - <<PY
import json
from pathlib import Path
Path("$candidate_dir/meta.json").write_text(json.dumps({
    "detector": "$detector",
    "cc": $cc,
    "threshold": float("$threshold"),
    "global_rounds": int("$GLOBAL_ROUNDS"),
    "tail_rounds": int("$TAIL_ROUNDS"),
}, indent=2) + "\\n")
PY
}

main() {
  cd "$ROOT"
  monza_check_sync
  monza_log "START threshold sweep"
  monza_log "ROOT=$ROOT"
  monza_log "LOG=$RUN_LOG"
  mkdir -p "$SWEEP_DIR"

  if [[ ! -d "$BERT_DIR" ]]; then
    echo "Detector BERT ausente: $BERT_DIR" >&2
    exit 1
  fi
  if [[ ! -d "$MLP_DIR" ]]; then
    echo "Detector MLP ausente: $MLP_DIR" >&2
    exit 1
  fi
  if [[ ! -d "$PUBLIC_VAL_DIR" ]]; then
    echo "PUBLIC_VAL_DIR ausente: $PUBLIC_VAL_DIR" >&2
    exit 1
  fi

  for threshold in $BERT_THRESHOLDS; do
    run_monza_threshold bert 6 "$threshold" "$BERT_DIR"
  done
  for threshold in $MLP_THRESHOLDS; do
    run_monza_threshold mlp 7 "$threshold" "$MLP_DIR"
  done

  monza_log "Summarize sweep"
  "$VENV_PY" scripts/tools/summarize_threshold_sweep.py \
    --sweep-dir "$SWEEP_DIR" \
    --tail-rounds "$TAIL_ROUNDS"
  monza_log "DONE"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  monza_check_sync
  printf 'dataset=%s\nrun_output=%s\nbert_model=%s\nmlp_model=%s\nsweep=%s\nlog=%s\n' \
    "$DATASET_NAME" "$RUN_OUTPUT" "$BERT_DIR" "$MLP_DIR" "$SWEEP_DIR" "$RUN_LOG"
elif [[ "${1:-}" == "--background" ]]; then
  mkdir -p "$(dirname "$RUN_LOG")"
  nohup env \
    ROOT="$ROOT" \
    VENV_PY="$VENV_PY" \
    DATASET_NAME="$DATASET_NAME" \
    MODEL="$MODEL" \
    GLOBAL_ROUNDS="$GLOBAL_ROUNDS" \
    NUM_CLIENTS="$NUM_CLIENTS" \
    NUM_MALICIOUS="$NUM_MALICIOUS" \
    JOIN_RATIO="$JOIN_RATIO" \
    DEVICE_ID="$DEVICE_ID" \
    LOCAL_STEPS="$LOCAL_STEPS" \
    TIMES="$TIMES" \
    RATE_FAKE="$RATE_FAKE" \
    ROUND_INIT_ATK="$ROUND_INIT_ATK" \
    BERT_DIR="$BERT_DIR" \
    MLP_DIR="$MLP_DIR" \
    PUBLIC_VAL_DIR="$PUBLIC_VAL_DIR" \
    SWEEP_DIR="$SWEEP_DIR" \
    TAIL_ROUNDS="$TAIL_ROUNDS" \
    BERT_THRESHOLDS="$BERT_THRESHOLDS" \
    MLP_THRESHOLDS="$MLP_THRESHOLDS" \
    RUN_ID="$RUN_ID" RUN_OUTPUT="$RUN_OUTPUT" RUN_LOG="$RUN_LOG" \
    "$0" >"$RUN_LOG" 2>&1 &
  printf 'Started PID %s\nLog: %s\n' "$!" "$RUN_LOG"
else
  mkdir -p "$(dirname "$RUN_LOG")"
  main 2>&1 | tee "$RUN_LOG"
fi

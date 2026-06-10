#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PY="${VENV_PY:-$ROOT/.venv/bin/python}"
SYSTEM_DIR="$ROOT/PFLlibMonza/system"
DATASET_DIR="$ROOT/PFLlibMonza/dataset"
DATASET_NAME="${DATASET_NAME:-Cifar10}"
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
DUMP_START_ROUND="${DUMP_START_ROUND:-$((ROUND_INIT_ATK + 1))}"

STATE_DICTS_DIR="${STATE_DICTS_DIR:-$ROOT/state_dicts_monza_cifar10_cnn}"
MLP_DIR="${MLP_DIR:-$ROOT/detector_mlp_monza_cifar10_cnn}"
ANALYSIS_OUT="${ANALYSIS_OUT:-$ROOT/analysis_outputs_cifar10}"
PUBLIC_VAL_DIR="${PUBLIC_VAL_DIR:-$DATASET_DIR/$DATASET_NAME/public_val}"
RUN_LOG="${RUN_LOG:-$ROOT/rerun_cifar10_$(date +%Y%m%d_%H%M%S).log}"
MLP_THRESHOLD_KEY="${MLP_THRESHOLD_KEY:-combined_label_fpr05}"
MLP_THRESHOLD_VALUE="${MLP_THRESHOLD_VALUE:-}"

export PUBLIC_VAL_DIR DATASET_NAME

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_monza() {
  local cc="$1"
  shift
  log "Run cc=${cc}"
  cd "$SYSTEM_DIR"
  "$VENV_PY" -u main.py \
    -m "$MODEL" \
    -data "$DATASET_NAME" \
    -nmc "$NUM_MALICIOUS" \
    -nc "$NUM_CLIENTS" \
    -jr "$JOIN_RATIO" \
    -atk all \
    -ria "$ROUND_INIT_ATK" \
    -cc "$cc" \
    -gr "$GLOBAL_ROUNDS" \
    -t "$TIMES" \
    -ls "$LOCAL_STEPS" \
    -did "$DEVICE_ID" \
    -rfake "$RATE_FAKE" \
    "$@"
  cd "$ROOT"
}

main() {
  cd "$ROOT"
  if [[ "$DATASET_NAME" != "Cifar10" ]]; then
    echo "run_full_cifar10.sh suporta apenas DATASET_NAME=Cifar10; recebido: $DATASET_NAME" >&2
    exit 1
  fi

  log "START CIFAR10"
  log "ROOT=$ROOT"
  log "GIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  log "LOG=$RUN_LOG"

  log "Clean generated CIFAR10 artifacts"
  rm -rf \
    "$STATE_DICTS_DIR" \
    "$MLP_DIR"
  mkdir -p "$ANALYSIS_OUT"
  backup_dir="$ANALYSIS_OUT/pre_run_system_csv_backup_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$backup_dir"
  system_csvs=()
  for csv_path in \
    "$SYSTEM_DIR"/f.csv \
    "$SYSTEM_DIR"/fpr_frr_results_*.csv \
    "$SYSTEM_DIR"/cc_detail_results_*.csv \
    "$SYSTEM_DIR"/cc_type_results_*.csv
  do
    [[ -e "$csv_path" ]] && system_csvs+=("$csv_path")
  done
  if ((${#system_csvs[@]} > 0)); then
    cp "${system_csvs[@]}" "$backup_dir"/
  else
    rmdir "$backup_dir"
  fi
  rm -f \
    "$SYSTEM_DIR"/f.csv \
    "$SYSTEM_DIR"/fpr_frr_results_*.csv \
    "$SYSTEM_DIR"/cc_detail_results_*.csv \
    "$SYSTEM_DIR"/cc_type_results_*.csv

  log "Validate environment"
  "$VENV_PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY

  log "Generate ${DATASET_NAME} partition"
  rm -rf "$DATASET_DIR/$DATASET_NAME"
  cd "$DATASET_DIR"
  "$VENV_PY" generate_Cifar10.py noniid - dir --num-clients "$NUM_CLIENTS"
  cd "$ROOT"
  "$VENV_PY" scripts/create_label_flip_train_mal.py \
    --dataset-dir "$DATASET_DIR/$DATASET_NAME" \
    --num-classes 10
  printf 'train=%s train_mal=%s\n' \
    "$(find "$DATASET_DIR/$DATASET_NAME/train" -name '*.npz' | wc -l)" \
    "$(find "$DATASET_DIR/$DATASET_NAME/train_mal" -name '*.npz' | wc -l)"
  printf 'public_val=%s test=%s\n' \
    "$(find "$DATASET_DIR/$DATASET_NAME/public_val" -name '*.npz' | wc -l)" \
    "$(find "$DATASET_DIR/$DATASET_NAME/test" -name '*.npz' | wc -l)"

  log "Dump MONZA state_dicts"
  run_monza 5 --dump_state_dicts "$STATE_DICTS_DIR" --dump_start_round "$DUMP_START_ROUND"
  find "$STATE_DICTS_DIR" -name '*.json' | wc -l
  du -sh "$STATE_DICTS_DIR"

  log "Train MLP detector"
  STATE_DICTS_DIR="$STATE_DICTS_DIR" \
  PUBLIC_VAL_DIR="$PUBLIC_VAL_DIR" \
  DATASET_NAME="$DATASET_NAME" \
  ARTIFACTS_DIR="$MLP_DIR" \
    "$VENV_PY" -u src/detector_mlp.py

  log "Run baseline CCs"
  run_monza 2
  run_monza 3

  log "Run MLP detector CC"
  mlp_args=(--detector_dir "$MLP_DIR" --mlp_threshold_key "$MLP_THRESHOLD_KEY")
  if [[ -n "$MLP_THRESHOLD_VALUE" ]]; then
    mlp_args+=(--mlp_threshold_value "$MLP_THRESHOLD_VALUE")
  fi
  run_monza 7 "${mlp_args[@]}"

  log "Write CLI summaries"
  "$VENV_PY" scripts/plot_cc_attack_types.py \
    --system-dir PFLlibMonza/system \
    --out-dir "$ANALYSIS_OUT" \
    --tail-rounds 30
  cp "$SYSTEM_DIR"/fpr_frr_results_*.csv "$ANALYSIS_OUT"/
  cp "$SYSTEM_DIR"/cc_detail_results_*.csv "$ANALYSIS_OUT"/
  cp "$SYSTEM_DIR"/cc_type_results_*.csv "$ANALYSIS_OUT"/
  log "DONE CIFAR10"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

if [[ "${1:-}" == "--background" ]]; then
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
    DUMP_START_ROUND="$DUMP_START_ROUND" \
    STATE_DICTS_DIR="$STATE_DICTS_DIR" \
    MLP_DIR="$MLP_DIR" \
    ANALYSIS_OUT="$ANALYSIS_OUT" \
    PUBLIC_VAL_DIR="$PUBLIC_VAL_DIR" \
    MLP_THRESHOLD_KEY="$MLP_THRESHOLD_KEY" \
    MLP_THRESHOLD_VALUE="$MLP_THRESHOLD_VALUE" \
    RUN_LOG="$RUN_LOG" \
    "$0" >"$RUN_LOG" 2>&1 &
  printf 'Started PID %s\nLog: %s\n' "$!" "$RUN_LOG"
else
  main 2>&1 | tee "$RUN_LOG"
fi

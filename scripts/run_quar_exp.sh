#!/usr/bin/env bash
# Single execution script for 4-way Quarantine & Defense Benchmark
# Usage: bash scripts/run_quar_exp.sh <cifar10|mnist> [--dry-run|--background]

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=_monza_common.sh
source "$ROOT/scripts/_monza_common.sh"

usage() {
  echo "Uso: bash scripts/run_quar_exp.sh <mnist|cifar10> [--dry-run|--background]" >&2
}

PROFILE="${1:-cifar10}"
MODE="${2:-}"

case "$PROFILE" in
  mnist)
    DEFAULT_DATASET=MNIST
    GENERATOR=generate_MNIST.py
    ;;
  cifar10)
    DEFAULT_DATASET=Cifar10
    GENERATOR=generate_Cifar10.py
    ;;
  *)
    echo "Perfil invalido: $PROFILE" >&2
    usage
    exit 2
    ;;
esac

case "$MODE" in
  ""|--dry-run|--background) ;;
  *)
    echo "Opcao invalida: $MODE" >&2
    usage
    exit 2
    ;;
esac

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  VENV_PY="${VENV_PY:-$ROOT/.venv/bin/python}"
else
  VENV_PY="${VENV_PY:-$(which python3)}"
fi
JUPYTER="${JUPYTER:-$ROOT/.venv/bin/jupyter}"
SYSTEM_DIR="$ROOT/PFLlibMonza/system"
DATASET_DIR="$ROOT/PFLlibMonza/dataset"
RESULTS_DIR="$ROOT/PFLlibMonza/results"
DATASET_NAME="${DATASET_NAME:-$DEFAULT_DATASET}"
MODEL="${MODEL:-CNN}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-100}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
NUM_MALICIOUS="${NUM_MALICIOUS:-30}"
JOIN_RATIO="${JOIN_RATIO:-1}"
DEVICE_ID="${DEVICE_ID:-0}"
LOCAL_STEPS="${LOCAL_STEPS:-5}"
TIMES="${TIMES:-1}"
RATE_FAKE="${RATE_FAKE:-1}"
ROUND_INIT_ATK="${ROUND_INIT_ATK:-20}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.2}"
DUMP_GLOBAL_ROUNDS="${DUMP_GLOBAL_ROUNDS:-60}"
DUMP_TIMES="${DUMP_TIMES:-1}"
DUMP_START_ROUND="${DUMP_START_ROUND:-$((ROUND_INIT_ATK + 1))}"
KEEP_DUMP="${KEEP_DUMP:-0}"

ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT/artifacts}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DATASET_SLUG="$(monza_dataset_slug "$DATASET_NAME")"
RUN_OUTPUT="${RUN_OUTPUT:-$ARTIFACTS_ROOT/runs/$DATASET_SLUG/$RUN_ID}"
STATE_DICTS_DIR="${STATE_DICTS_DIR:-$ARTIFACTS_ROOT/dumps/$DATASET_SLUG/current}"
MLP_DIR="${MLP_DIR:-$ARTIFACTS_ROOT/models/$DATASET_SLUG/mlp}"
ANALYSIS_OUT="${ANALYSIS_OUT:-$RUN_OUTPUT/analysis}"
PUBLIC_VAL_DIR="${PUBLIC_VAL_DIR:-$DATASET_DIR/$DATASET_NAME/public_val}"
RUN_LOG="${RUN_LOG:-$RUN_OUTPUT/run.log}"

MLP_THRESHOLD_KEY="${MLP_THRESHOLD_KEY:-combined_label_fpr01}"
MLP_THRESHOLD_VALUE="${MLP_THRESHOLD_VALUE:-}"
OVERSAMPLE_LABEL_FACTOR="${OVERSAMPLE_LABEL_FACTOR:-6}"
LABEL_LOSS_WEIGHT="${LABEL_LOSS_WEIGHT:-1.0}"

export PUBLIC_VAL_DIR DATASET_NAME OVERSAMPLE_LABEL_FACTOR LABEL_LOSS_WEIGHT ROUND_INIT_ATK

print_config() {
  cat <<EOF
profile=$PROFILE
dataset=$DATASET_NAME
generator=$GENERATOR
model=$MODEL
rounds=$GLOBAL_ROUNDS
times=$TIMES
round_init_atk=$ROUND_INIT_ATK
run_output=$RUN_OUTPUT
state_dicts=$STATE_DICTS_DIR
mlp_model=$MLP_DIR
analysis=$ANALYSIS_OUT
log=$RUN_LOG
stages=sync-check,clean,dataset,dump,train-mlp,clean-baseline,poison-baseline,cc2-quar,cc2-noquar,cc3-quar,cc3-noquar,cc7-quar,cc7-noquar,cc8-quar,cc8-noquar,analysis
EOF
}

main() {
  cd "$ROOT"
  monza_check_sync

  monza_log "START Quarantine Experiment ($DATASET_NAME)"
  print_config

  mkdir -p "$RUN_OUTPUT" "$ANALYSIS_OUT"
  rm -rf "$STATE_DICTS_DIR" "$MLP_DIR" "$ANALYSIS_OUT"/*
  rm -f "$SYSTEM_DIR"/fpr_frr_results_*.csv "$SYSTEM_DIR"/cc_detail_results_*.csv "$SYSTEM_DIR"/cc_type_results_*.csv "$RESULTS_DIR"/*.h5 2>/dev/null || true

  monza_log "Validate GPU / Torch environment"
  "$VENV_PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY

  monza_log "Generate ${DATASET_NAME} partition (alpha=${DIRICHLET_ALPHA})"
  rm -rf "$DATASET_DIR/$DATASET_NAME"
  (
    cd "$DATASET_DIR"
    "$VENV_PY" "$GENERATOR" noniid - dir \
      --num-clients "$NUM_CLIENTS" --dirichlet-alpha "$DIRICHLET_ALPHA"
  )
  "$VENV_PY" "$ROOT/scripts/create_label_flip_train_mal.py" \
    --dataset-dir "$DATASET_DIR/$DATASET_NAME" --num-classes 10

  monza_log "Dump MONZA state_dicts"
  monza_run 5 "$NUM_MALICIOUS" "$DUMP_GLOBAL_ROUNDS" "$DUMP_TIMES" \
    --dump_state_dicts "$STATE_DICTS_DIR" --dump_start_round "$DUMP_START_ROUND"

  monza_log "Train MLP detector"
  STATE_DICTS_DIR="$STATE_DICTS_DIR" PUBLIC_VAL_DIR="$PUBLIC_VAL_DIR" \
  DATASET_NAME="$DATASET_NAME" ARTIFACTS_DIR="$MLP_DIR" \
  OVERSAMPLE_LABEL_FACTOR="$OVERSAMPLE_LABEL_FACTOR" LABEL_LOSS_WEIGHT="$LABEL_LOSS_WEIGHT" \
    "$VENV_PY" -u src/detector_mlp.py

  [[ "$KEEP_DUMP" == "1" ]] || rm -rf "$STATE_DICTS_DIR"

  monza_log "Stage 1: Clean Baseline (cc=5 nmal=0)"
  monza_run 5 0 "$GLOBAL_ROUNDS" "$TIMES"

  monza_log "Stage 2: Poisoned Baseline (cc=5 nmal=30)"
  monza_run 5 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES"

  monza_log "Stage 3: CC=2 (zPROBE Defense - With Quarantine)"
  monza_run 2 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES"

  monza_log "Stage 4: CC=2 (zPROBE Defense - Without Quarantine)"
  monza_run 2 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES" --disable_quarantine

  monza_log "Stage 5: CC=3 (MONZA Defense - With Quarantine)"
  monza_run 3 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES"

  monza_log "Stage 6: CC=3 (MONZA Defense - Without Quarantine)"
  monza_run 3 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES" --disable_quarantine

  local mlp_args=(--detector_dir "$MLP_DIR" --mlp_threshold_key "$MLP_THRESHOLD_KEY")
  [[ -z "$MLP_THRESHOLD_VALUE" ]] || mlp_args+=(--mlp_threshold_value "$MLP_THRESHOLD_VALUE")

  monza_log "Stage 7: CC=7 (MLP Defense - With Quarantine)"
  monza_run 7 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES" "${mlp_args[@]}"

  monza_log "Stage 8: CC=7 (MLP Defense - Without Quarantine)"
  monza_run 7 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES" "${mlp_args[@]}" --disable_quarantine

  monza_log "Stage 9: CC=8 (FedSIGN Defense - With Quarantine)"
  monza_run 8 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES"

  monza_log "Stage 10: CC=8 (FedSIGN Defense - Without Quarantine)"
  monza_run 8 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES" --disable_quarantine

  monza_log "Archive system CSVs & H5s to ANALYSIS_OUT"
  mkdir -p "$ANALYSIS_OUT"
  cp "$SYSTEM_DIR"/fpr_frr_results_*.csv "$ANALYSIS_OUT"/ 2>/dev/null || true
  cp "$SYSTEM_DIR"/cc_detail_results_*.csv "$ANALYSIS_OUT"/ 2>/dev/null || true
  cp "$SYSTEM_DIR"/cc_type_results_*.csv "$ANALYSIS_OUT"/ 2>/dev/null || true
  cp "$RESULTS_DIR"/*.h5 "$ANALYSIS_OUT"/ 2>/dev/null || true

  monza_log "Execute CLI summaries & PNG comparison plots"
  "$VENV_PY" "$ROOT/scripts/plot_cc_attack_types.py" \
    --system-dir "$ANALYSIS_OUT" --out-dir "$ANALYSIS_OUT" \
    --results-dir "$ANALYSIS_OUT" \
    --dataset "$DATASET_NAME" --tail-rounds 30 \
    --num-malicious "$NUM_MALICIOUS" || true

  monza_log "Copy PNG graphs to run output folder and root plots/ folder"
  cp "$ANALYSIS_OUT"/*.png "$RUN_OUTPUT/" 2>/dev/null || true
  mkdir -p "$ROOT/plots"
  cp "$ANALYSIS_OUT"/*.png "$ROOT/plots/" 2>/dev/null || true
  monza_log "DONE Experiment $DATASET_NAME"
}

if [[ "$MODE" == "--dry-run" ]]; then
  monza_check_sync
  print_config
elif [[ "$MODE" == "--background" ]]; then
  mkdir -p "$(dirname "$RUN_LOG")"
  nohup env ROOT="$ROOT" RUN_ID="$RUN_ID" RUN_OUTPUT="$RUN_OUTPUT" RUN_LOG="$RUN_LOG" \
    "$0" "$PROFILE" >"$RUN_LOG" 2>&1 &
  printf 'Started PID %s\nLog: %s\n' "$!" "$RUN_LOG"
else
  mkdir -p "$(dirname "$RUN_LOG")"
  main 2>&1 | tee "$RUN_LOG"
fi

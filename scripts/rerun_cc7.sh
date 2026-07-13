#!/usr/bin/env bash
# Re-run ONLY cc7 (MLP+features) on the EXISTING dataset partition, retraining the MLP detector
# with the new QuantileTransformer (fixes the CIFAR scaler-drift). Reuses the dataset on disk so
# results stay consistent with the kept cc3/cc5 baselines. Does not touch cc3/cc5 results.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=_monza_common.sh
source "$ROOT/scripts/_monza_common.sh"

usage() {
  echo "Uso: bash scripts/rerun_cc7.sh <mnist|cifar10> [--dry-run|--background]" >&2
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

PROFILE="${1:-}"
MODE="${2:-}"
if [[ -z "$PROFILE" || $# -gt 2 ]]; then
  usage
  exit 2
fi
case "$PROFILE" in
  mnist) DEFAULT_DATASET=MNIST ;;
  cifar10) DEFAULT_DATASET=Cifar10 ;;
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

VENV_PY="${VENV_PY:-$ROOT/.venv/bin/python}"
SYSTEM_DIR="$ROOT/PFLlibMonza/system"
DATASET_DIR="$ROOT/PFLlibMonza/dataset"
RESULTS_DIR="$ROOT/PFLlibMonza/results"
DATASET_NAME="${DATASET_NAME:-$DEFAULT_DATASET}"
MODEL="${MODEL:-CNN}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-300}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
NUM_MALICIOUS="${NUM_MALICIOUS:-30}"
JOIN_RATIO="${JOIN_RATIO:-1}"
DEVICE_ID="${DEVICE_ID:-0}"
LOCAL_STEPS="${LOCAL_STEPS:-1}"
TIMES="${TIMES:-10}"
RATE_FAKE="${RATE_FAKE:-1}"
ROUND_INIT_ATK="${ROUND_INIT_ATK:-5}"
DUMP_GLOBAL_ROUNDS="${DUMP_GLOBAL_ROUNDS:-60}"
DUMP_TIMES="${DUMP_TIMES:-1}"
DUMP_START_ROUND="${DUMP_START_ROUND:-$((ROUND_INIT_ATK + 1))}"
KEEP_DUMP="${KEEP_DUMP:-0}"

ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT/artifacts}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DATASET_SLUG="$(monza_dataset_slug "$DATASET_NAME")"
RUN_OUTPUT="${RUN_OUTPUT:-$ARTIFACTS_ROOT/runs/$DATASET_SLUG/${RUN_ID}_cc7}"
STATE_DICTS_DIR="${STATE_DICTS_DIR:-$ARTIFACTS_ROOT/dumps/$DATASET_SLUG/current}"
MLP_DIR="${MLP_DIR:-$ARTIFACTS_ROOT/models/$DATASET_SLUG/mlp}"
ANALYSIS_OUT="${ANALYSIS_OUT:-$RUN_OUTPUT/analysis}"
PUBLIC_VAL_DIR="${PUBLIC_VAL_DIR:-$DATASET_DIR/$DATASET_NAME/public_val}"
MLP_THRESHOLD_KEY="${MLP_THRESHOLD_KEY:-combined_label_fpr05}"
MLP_THRESHOLD_VALUE="${MLP_THRESHOLD_VALUE:-}"
# Activate the (otherwise inert) label-flip aux head: oversample the minority class and weight its loss.
OVERSAMPLE_LABEL_FACTOR="${OVERSAMPLE_LABEL_FACTOR:-4}"
LABEL_LOSS_WEIGHT="${LABEL_LOSS_WEIGHT:-4}"
RUN_LOG="${RUN_LOG:-$RUN_OUTPUT/run.log}"

export PUBLIC_VAL_DIR DATASET_NAME

validate_profile() {
  [[ "$DATASET_NAME" == "$DEFAULT_DATASET" ]] || {
    echo "Perfil $PROFILE requer DATASET_NAME=$DEFAULT_DATASET; recebido: $DATASET_NAME" >&2
    exit 2
  }
  [[ "$MODEL" == "CNN" ]] || {
    echo "O rerun suporta apenas MODEL=CNN; recebido: $MODEL" >&2
    exit 2
  }
}

main() {
  cd "$ROOT"
  validate_profile
  monza_check_sync
  # Reuse existing dataset (do NOT regenerate -> keeps consistency with kept cc3/cc5).
  if ! ls "$DATASET_DIR/$DATASET_NAME/train"/*.npz >/dev/null 2>&1; then
    echo "Dataset $DATASET_NAME nao existe no disco; rerun_cc7 reusa o existente. Abortando." >&2
    exit 1
  fi
  monza_log "START rerun cc7 ${DATASET_NAME} (gr=${GLOBAL_ROUNDS} times=${TIMES}) reusando dataset"

  # Protect kept baseline h5 from the cc5 dump (same _5_..._30_ filename).
  bak="$RESULTS_DIR/_rerun_cc7_bak_${DATASET_NAME}"
  mkdir -p "$bak"
  cp "$RESULTS_DIR"/"${DATASET_NAME}_FedAvg_5_100.0_${NUM_MALICIOUS}_test_"*.h5 "$bak"/ 2>/dev/null || true

  # Clean only cc7-related system CSVs; cc3 remains available for comparison.
  rm -f "$SYSTEM_DIR"/f.csv \
        "$SYSTEM_DIR"/fpr_frr_results_7.csv \
        "$SYSTEM_DIR"/cc_detail_results_7.csv \
        "$SYSTEM_DIR"/cc_type_results_7.csv

  rm -rf "$STATE_DICTS_DIR" "$MLP_DIR"
  monza_log "Dump cc5 (small) for MLP retraining"
  monza_run 5 "$NUM_MALICIOUS" "$DUMP_GLOBAL_ROUNDS" "$DUMP_TIMES" \
    --dump_state_dicts "$STATE_DICTS_DIR" --dump_start_round "$DUMP_START_ROUND"
  du -sh "$STATE_DICTS_DIR" || true

  monza_log "Train MLP detector (QuantileTransformer + label-flip head)"
  STATE_DICTS_DIR="$STATE_DICTS_DIR" PUBLIC_VAL_DIR="$PUBLIC_VAL_DIR" \
  DATASET_NAME="$DATASET_NAME" ARTIFACTS_DIR="$MLP_DIR" \
  OVERSAMPLE_LABEL_FACTOR="$OVERSAMPLE_LABEL_FACTOR" LABEL_LOSS_WEIGHT="$LABEL_LOSS_WEIGHT" \
    "$VENV_PY" -u src/detector_mlp.py

  [[ "$KEEP_DUMP" == "1" ]] || { monza_log "Free dump"; rm -rf "$STATE_DICTS_DIR"; }

  # Restore the kept baseline h5 that the dump may have overwritten.
  cp "$bak"/*.h5 "$RESULTS_DIR"/ 2>/dev/null || true
  rm -rf "$bak"

  monza_log "Run cc7 (paper scenario)"
  mlp_args=(--detector_dir "$MLP_DIR" --mlp_threshold_key "$MLP_THRESHOLD_KEY")
  [[ -n "$MLP_THRESHOLD_VALUE" ]] && mlp_args+=(--mlp_threshold_value "$MLP_THRESHOLD_VALUE")
  monza_run 7 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES" "${mlp_args[@]}"

  monza_log "Archive cc7 CSVs to ANALYSIS_OUT"
  mkdir -p "$ANALYSIS_OUT"
  cp "$SYSTEM_DIR"/fpr_frr_results_7.csv "$ANALYSIS_OUT"/ 2>/dev/null || true
  cp "$SYSTEM_DIR"/cc_detail_results_7.csv "$ANALYSIS_OUT"/ 2>/dev/null || true
  cp "$SYSTEM_DIR"/cc_type_results_7.csv "$ANALYSIS_OUT"/ 2>/dev/null || true
  monza_log "DONE rerun cc7 ${DATASET_NAME}"
}

validate_profile
if [[ "$MODE" == "--dry-run" ]]; then
  monza_check_sync
  printf 'profile=%s\ndataset=%s\nrun_output=%s\nstate_dicts=%s\nmlp_model=%s\nanalysis=%s\nlog=%s\n' \
    "$PROFILE" "$DATASET_NAME" "$RUN_OUTPUT" "$STATE_DICTS_DIR" "$MLP_DIR" "$ANALYSIS_OUT" "$RUN_LOG"
elif [[ "$MODE" == "--background" ]]; then
  mkdir -p "$(dirname "$RUN_LOG")"
  nohup env ROOT="$ROOT" VENV_PY="$VENV_PY" DATASET_NAME="$DATASET_NAME" MODEL="$MODEL" \
    GLOBAL_ROUNDS="$GLOBAL_ROUNDS" NUM_CLIENTS="$NUM_CLIENTS" NUM_MALICIOUS="$NUM_MALICIOUS" \
    JOIN_RATIO="$JOIN_RATIO" DEVICE_ID="$DEVICE_ID" LOCAL_STEPS="$LOCAL_STEPS" TIMES="$TIMES" \
    RATE_FAKE="$RATE_FAKE" ROUND_INIT_ATK="$ROUND_INIT_ATK" DUMP_GLOBAL_ROUNDS="$DUMP_GLOBAL_ROUNDS" \
    DUMP_TIMES="$DUMP_TIMES" DUMP_START_ROUND="$DUMP_START_ROUND" KEEP_DUMP="$KEEP_DUMP" \
    STATE_DICTS_DIR="$STATE_DICTS_DIR" MLP_DIR="$MLP_DIR" ANALYSIS_OUT="$ANALYSIS_OUT" \
    PUBLIC_VAL_DIR="$PUBLIC_VAL_DIR" MLP_THRESHOLD_KEY="$MLP_THRESHOLD_KEY" \
    MLP_THRESHOLD_VALUE="$MLP_THRESHOLD_VALUE" \
    OVERSAMPLE_LABEL_FACTOR="$OVERSAMPLE_LABEL_FACTOR" LABEL_LOSS_WEIGHT="$LABEL_LOSS_WEIGHT" \
    RUN_ID="$RUN_ID" RUN_OUTPUT="$RUN_OUTPUT" RUN_LOG="$RUN_LOG" \
    "$0" "$PROFILE" >"$RUN_LOG" 2>&1 &
  printf 'Started PID %s\nLog: %s\n' "$!" "$RUN_LOG"
else
  mkdir -p "$(dirname "$RUN_LOG")"
  main 2>&1 | tee "$RUN_LOG"
fi

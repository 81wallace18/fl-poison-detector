#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=_monza_common.sh
source "$ROOT/scripts/_monza_common.sh"

usage() {
  echo "Uso: bash scripts/run_full.sh <mnist|cifar10> [--dry-run|--background]" >&2
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

VENV_PY="${VENV_PY:-$ROOT/.venv/bin/python}"
JUPYTER="${JUPYTER:-$ROOT/.venv/bin/jupyter}"
SYSTEM_DIR="$ROOT/PFLlibMonza/system"
DATASET_DIR="$ROOT/PFLlibMonza/dataset"
RESULTS_DIR="$ROOT/PFLlibMonza/results"
DATASET_NAME="${DATASET_NAME:-$DEFAULT_DATASET}"
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

MLP_THRESHOLD_KEY="${MLP_THRESHOLD_KEY:-combined_label_fpr05}"
MLP_THRESHOLD_VALUE="${MLP_THRESHOLD_VALUE:-}"

export PUBLIC_VAL_DIR DATASET_NAME

print_config() {
  cat <<EOF
profile=$PROFILE
dataset=$DATASET_NAME
generator=$GENERATOR
model=$MODEL
rounds=$GLOBAL_ROUNDS
times=$TIMES
run_output=$RUN_OUTPUT
state_dicts=$STATE_DICTS_DIR
mlp_model=$MLP_DIR
analysis=$ANALYSIS_OUT
log=$RUN_LOG
stages=sync-check,clean,dataset,dump,train-mlp,baselines,cc3,cc7,analysis
EOF
}

validate_profile() {
  [[ "$DATASET_NAME" == "$DEFAULT_DATASET" ]] || {
    echo "Perfil $PROFILE requer DATASET_NAME=$DEFAULT_DATASET; recebido: $DATASET_NAME" >&2
    exit 2
  }
  [[ "$MODEL" == "CNN" ]] || {
    echo "O workflow completo suporta apenas MODEL=CNN; recebido: $MODEL" >&2
    exit 2
  }
}

archive_system_csvs() {
  local backup_dir="$RUN_OUTPUT/pre-run-system-csv"
  local csvs=()
  local path
  for path in \
    "$SYSTEM_DIR"/f.csv \
    "$SYSTEM_DIR"/fpr_frr_results_*.csv \
    "$SYSTEM_DIR"/cc_detail_results_*.csv \
    "$SYSTEM_DIR"/cc_type_results_*.csv
  do
    [[ -e "$path" ]] && csvs+=("$path")
  done
  if ((${#csvs[@]} > 0)); then
    mkdir -p "$backup_dir"
    cp "${csvs[@]}" "$backup_dir"/
  fi
}

main() {
  cd "$ROOT"
  validate_profile
  monza_check_sync

  monza_log "START $DATASET_NAME"
  print_config

  mkdir -p "$RUN_OUTPUT"
  archive_system_csvs
  rm -rf "$STATE_DICTS_DIR" "$MLP_DIR" "$ANALYSIS_OUT"
  rm -f \
    "$SYSTEM_DIR"/f.csv \
    "$SYSTEM_DIR"/fpr_frr_results_*.csv \
    "$SYSTEM_DIR"/cc_detail_results_*.csv \
    "$SYSTEM_DIR"/cc_type_results_*.csv \
    "$RESULTS_DIR"/"${DATASET_NAME}"_FedAvg_*.h5

  monza_log "Validate environment"
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
  find "$STATE_DICTS_DIR" -name '*.json' | wc -l
  du -sh "$STATE_DICTS_DIR"

  monza_log "Train MLP detector"
  STATE_DICTS_DIR="$STATE_DICTS_DIR" PUBLIC_VAL_DIR="$PUBLIC_VAL_DIR" \
  DATASET_NAME="$DATASET_NAME" ARTIFACTS_DIR="$MLP_DIR" \
    "$VENV_PY" -u src/detector_mlp.py

  [[ "$KEEP_DUMP" == "1" ]] || rm -rf "$STATE_DICTS_DIR"
  rm -f "$RESULTS_DIR"/"${DATASET_NAME}"_FedAvg_5_100.0_"${NUM_MALICIOUS}"_test_*.h5

  monza_log "Run baselines"
  monza_run 5 0 "$GLOBAL_ROUNDS" "$TIMES"
  monza_run 5 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES"
  monza_run 3 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES"

  local mlp_args=(--detector_dir "$MLP_DIR" --mlp_threshold_key "$MLP_THRESHOLD_KEY")
  [[ -z "$MLP_THRESHOLD_VALUE" ]] || mlp_args+=(--mlp_threshold_value "$MLP_THRESHOLD_VALUE")
  monza_run 7 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES" "${mlp_args[@]}"

  monza_log "Execute notebook plots"
  REPO_ROOT="$ROOT" ANALYSIS_OUT="$ANALYSIS_OUT" DATASET_NAME="$DATASET_NAME" \
    "$JUPYTER" nbconvert --to notebook --execute \
    notebooks/notebook_monza_analysis.ipynb \
    --output notebook-monza-analysis.executed.ipynb --output-dir "$RUN_OUTPUT" \
    || monza_log "WARN: nbconvert falhou; seguindo para os summaries CLI"

  monza_log "Write CLI summaries"
  "$VENV_PY" "$ROOT/scripts/plot_cc_attack_types.py" \
    --system-dir "$SYSTEM_DIR" --out-dir "$ANALYSIS_OUT" \
    --dataset "$DATASET_NAME" --tail-rounds 30 \
    --num-malicious "$NUM_MALICIOUS"
  cp "$SYSTEM_DIR"/fpr_frr_results_*.csv "$ANALYSIS_OUT"/ 2>/dev/null || true
  cp "$SYSTEM_DIR"/cc_detail_results_*.csv "$ANALYSIS_OUT"/ 2>/dev/null || true
  cp "$SYSTEM_DIR"/cc_type_results_*.csv "$ANALYSIS_OUT"/ 2>/dev/null || true
  monza_log "DONE $DATASET_NAME"
}

validate_profile
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

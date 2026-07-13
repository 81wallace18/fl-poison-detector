#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=../lib/monza_common.sh
source "$ROOT/scripts/lib/monza_common.sh"

PROFILE="${DATASET_PROFILE:-mnist}"
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
    echo "DATASET_PROFILE invalido: $PROFILE (use mnist ou cifar10)" >&2
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
SKIP_BERT="${SKIP_BERT:-0}"

ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT/artifacts}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DATASET_SLUG="$(monza_dataset_slug "$DATASET_NAME")"
RUN_OUTPUT="${RUN_OUTPUT:-$ARTIFACTS_ROOT/runs/$DATASET_SLUG/$RUN_ID}"
STATE_DICTS_DIR="${STATE_DICTS_DIR:-$ARTIFACTS_ROOT/dumps/$DATASET_SLUG/current}"
BERT_DIR="${BERT_DIR:-$ARTIFACTS_ROOT/models/$DATASET_SLUG/bert}"
MLP_DIR="${MLP_DIR:-$ARTIFACTS_ROOT/models/$DATASET_SLUG/mlp}"
RUN_DIR="${RUN_DIR:-$ARTIFACTS_ROOT/models/$DATASET_SLUG/bert-runs}"
ANALYSIS_OUT="${ANALYSIS_OUT:-$RUN_OUTPUT/analysis}"
PUBLIC_VAL_DIR="${PUBLIC_VAL_DIR:-$DATASET_DIR/$DATASET_NAME/public_val}"
RUN_LOG="${RUN_LOG:-$RUN_OUTPUT/run.log}"

BERT_THRESHOLD_KEY="${BERT_THRESHOLD_KEY:-combined_label_fpr05}"
BERT_THRESHOLD_VALUE="${BERT_THRESHOLD_VALUE:-}"
BERT_EPOCHS="${BERT_EPOCHS:-8}"
BERT_EARLY_STOPPING_PATIENCE="${BERT_EARLY_STOPPING_PATIENCE:-2}"
BERT_OVERSAMPLE_LABEL_FACTOR="${BERT_OVERSAMPLE_LABEL_FACTOR:-4}"
BERT_LABEL_LOSS_WEIGHT="${BERT_LABEL_LOSS_WEIGHT:-3.0}"
BERT_MAX_BENIGN_FPR="${BERT_MAX_BENIGN_FPR:-0.05}"
BERT_TRAIN_BATCH_SIZE="${BERT_TRAIN_BATCH_SIZE:-16}"
BERT_EVAL_BATCH_SIZE="${BERT_EVAL_BATCH_SIZE:-16}"
BERT_LEARNING_RATE="${BERT_LEARNING_RATE:-2e-4}"
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
bert_model=$BERT_DIR
mlp_model=$MLP_DIR
analysis=$ANALYSIS_OUT
log=$RUN_LOG
skip_bert=$SKIP_BERT
stages=sync-check,clean,dataset,dump,train,baselines,cc3,detectors,analysis
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
  rm -rf "$STATE_DICTS_DIR" "$BERT_DIR" "$MLP_DIR" "$RUN_DIR" "$ANALYSIS_OUT"
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
  "$VENV_PY" "$ROOT/scripts/tools/create_label_flip_train_mal.py" \
    --dataset-dir "$DATASET_DIR/$DATASET_NAME" --num-classes 10

  monza_log "Dump MONZA state_dicts"
  monza_run 5 "$NUM_MALICIOUS" "$DUMP_GLOBAL_ROUNDS" "$DUMP_TIMES" \
    --dump_state_dicts "$STATE_DICTS_DIR" --dump_start_round "$DUMP_START_ROUND"
  find "$STATE_DICTS_DIR" -name '*.json' | wc -l
  du -sh "$STATE_DICTS_DIR"

  if [[ "$SKIP_BERT" != "1" ]]; then
    monza_log "Train DistilBERT detector"
    STATE_DICTS_DIR="$STATE_DICTS_DIR" PUBLIC_VAL_DIR="$PUBLIC_VAL_DIR" \
    OVERSAMPLE_LABEL_FACTOR="$BERT_OVERSAMPLE_LABEL_FACTOR" \
    LABEL_LOSS_WEIGHT="$BERT_LABEL_LOSS_WEIGHT" BERT_EPOCHS="$BERT_EPOCHS" \
    BERT_EARLY_STOPPING_PATIENCE="$BERT_EARLY_STOPPING_PATIENCE" \
    BERT_MAX_BENIGN_FPR="$BERT_MAX_BENIGN_FPR" \
    BERT_TRAIN_BATCH_SIZE="$BERT_TRAIN_BATCH_SIZE" \
    BERT_EVAL_BATCH_SIZE="$BERT_EVAL_BATCH_SIZE" \
    BERT_LEARNING_RATE="$BERT_LEARNING_RATE" FINAL_MODEL_DIR="$BERT_DIR" \
    RUN_DIR="$RUN_DIR" "$VENV_PY" -u src/detector.py
  else
    monza_log "SKIP_BERT=1: pulando treino do DistilBERT (cc6)"
  fi

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

  local bert_args=(--detector_dir "$BERT_DIR" --bert_threshold_key "$BERT_THRESHOLD_KEY")
  local mlp_args=(--detector_dir "$MLP_DIR" --mlp_threshold_key "$MLP_THRESHOLD_KEY")
  [[ -z "$BERT_THRESHOLD_VALUE" ]] || bert_args+=(--bert_threshold_value "$BERT_THRESHOLD_VALUE")
  [[ -z "$MLP_THRESHOLD_VALUE" ]] || mlp_args+=(--mlp_threshold_value "$MLP_THRESHOLD_VALUE")
  if [[ "$SKIP_BERT" != "1" ]]; then
    monza_run 6 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES" "${bert_args[@]}"
  fi
  monza_run 7 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES" "${mlp_args[@]}"

  if [[ "$PROFILE" == "mnist" ]]; then
    monza_log "Execute notebook plots"
    REPO_ROOT="$ROOT" ANALYSIS_OUT="$ANALYSIS_OUT" \
      "$JUPYTER" nbconvert --to notebook --execute \
      notebooks/notebook_monza_analysis.ipynb \
      --output notebook-monza-analysis.executed.ipynb --output-dir "$RUN_OUTPUT" \
      || monza_log "WARN: nbconvert falhou; seguindo para os summaries CLI"
  fi

  monza_log "Write CLI summaries"
  "$VENV_PY" "$ROOT/scripts/tools/plot_cc_attack_types.py" \
    --system-dir "$SYSTEM_DIR" --out-dir "$ANALYSIS_OUT" \
    --dataset "$DATASET_NAME" --tail-rounds 30
  cp "$SYSTEM_DIR"/fpr_frr_results_*.csv "$ANALYSIS_OUT"/ 2>/dev/null || true
  cp "$SYSTEM_DIR"/cc_detail_results_*.csv "$ANALYSIS_OUT"/ 2>/dev/null || true
  cp "$SYSTEM_DIR"/cc_type_results_*.csv "$ANALYSIS_OUT"/ 2>/dev/null || true
  monza_log "DONE $DATASET_NAME"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

validate_profile
if [[ "${1:-}" == "--dry-run" ]]; then
  monza_check_sync
  print_config
elif [[ "${1:-}" == "--background" ]]; then
  mkdir -p "$(dirname "$RUN_LOG")"
  nohup env ROOT="$ROOT" DATASET_PROFILE="$PROFILE" RUN_ID="$RUN_ID" \
    RUN_OUTPUT="$RUN_OUTPUT" RUN_LOG="$RUN_LOG" "$0" >"$RUN_LOG" 2>&1 &
  printf 'Started PID %s\nLog: %s\n' "$!" "$RUN_LOG"
else
  mkdir -p "$(dirname "$RUN_LOG")"
  main 2>&1 | tee "$RUN_LOG"
fi

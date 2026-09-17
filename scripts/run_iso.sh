#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=_monza_common.sh
source "$ROOT/scripts/_monza_common.sh"

usage() {
  echo "Uso: bash scripts/run_iso.sh <mnist|cifar10> [--dry-run|--background]" >&2
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

# --- ISOLATED RUN CONFIGURATION ---
NUM_CLIENTS="${NUM_CLIENTS:-50}"                # Reduced to 50 clients for faster prototyping
TARGET_ATTACK="${TARGET_ATTACK:-label}"         # Run only 1 attack (label, random, shuffle, or zeros)
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-50}"            # Shorter 50-round evaluation
ROUND_INIT_ATK="${ROUND_INIT_ATK:-30}"          # Give the model 30 rounds to learn BEFORE the attack hits
DUMP_GLOBAL_ROUNDS="${DUMP_GLOBAL_ROUNDS:-40}"  # Stop dump at round 40
NUM_MALICIOUS="${NUM_MALICIOUS:-15}"            # 15 attackers = 30% of 50 clients
DUMP_NUM_MALICIOUS="${DUMP_NUM_MALICIOUS:-20}"  # 20 attackers = 40% of 50 clients
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.3}"       # Slightly increased to prevent 0-sample crashes
# ----------------------------------

JOIN_RATIO="${JOIN_RATIO:-1}"
DEVICE_ID="${DEVICE_ID:-0}"
LOCAL_STEPS="${LOCAL_STEPS:-1}"
TIMES="${TIMES:-1}"
RATE_FAKE="${RATE_FAKE:-1}"
DUMP_TIMES="${DUMP_TIMES:-1}"
DUMP_START_ROUND="${DUMP_START_ROUND:-$((ROUND_INIT_ATK + 1))}"
KEEP_DUMP="${KEEP_DUMP:-0}"

ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT/artifacts}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$TARGET_ATTACK}"
DATASET_SLUG="$(monza_dataset_slug "$DATASET_NAME")"
RUN_OUTPUT="${RUN_OUTPUT:-$ARTIFACTS_ROOT/runs/$DATASET_SLUG/$RUN_ID}"
STATE_DICTS_DIR="${STATE_DICTS_DIR:-$ARTIFACTS_ROOT/dumps/$DATASET_SLUG/current}"
MLP_DIR="${MLP_DIR:-$ARTIFACTS_ROOT/models/$DATASET_SLUG/mlp}"
ANALYSIS_OUT="${ANALYSIS_OUT:-$RUN_OUTPUT/analysis}"
PUBLIC_VAL_DIR="${PUBLIC_VAL_DIR:-$DATASET_DIR/$DATASET_NAME/public_val}"
RUN_LOG="${RUN_LOG:-$RUN_OUTPUT/run.log}"

MLP_THRESHOLD_KEY="${MLP_THRESHOLD_KEY:-combined_label_fpr05}"
MLP_THRESHOLD_VALUE="${MLP_THRESHOLD_VALUE:-}"
OVERSAMPLE_LABEL_FACTOR="${OVERSAMPLE_LABEL_FACTOR:-4}"
LABEL_LOSS_WEIGHT="${LABEL_LOSS_WEIGHT:-4}"

export PUBLIC_VAL_DIR DATASET_NAME

print_config() {
  cat <<EOF
profile=$PROFILE
dataset=$DATASET_NAME
generator=$GENERATOR
model=$MODEL
target_attack=$TARGET_ATTACK
rounds=$GLOBAL_ROUNDS
attack_starts_at=$ROUND_INIT_ATK
dump_rounds=$DUMP_GLOBAL_ROUNDS
num_clients=$NUM_CLIENTS
eval_malicious=$NUM_MALICIOUS
dump_malicious=$DUMP_NUM_MALICIOUS
times=$TIMES
run_output=$RUN_OUTPUT
state_dicts=$STATE_DICTS_DIR
mlp_model=$MLP_DIR
analysis=$ANALYSIS_OUT
log=$RUN_LOG
oversample_label_factor=$OVERSAMPLE_LABEL_FACTOR
label_loss_weight=$LABEL_LOSS_WEIGHT
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

  monza_log "START $DATASET_NAME - ISOLATED ATTACK: $TARGET_ATTACK"
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

  # Check if the dataset already exists before deleting and generating
  if [ ! -d "$DATASET_DIR/$DATASET_NAME" ]; then
      monza_log "Generate ${DATASET_NAME} partition (alpha=${DIRICHLET_ALPHA})"
      rm -rf "$DATASET_DIR/$DATASET_NAME"
      (
        cd "$DATASET_DIR"
        "$VENV_PY" "$GENERATOR" noniid - dir \
          --num-clients "$NUM_CLIENTS" --dirichlet-alpha "$DIRICHLET_ALPHA"
      )
      "$VENV_PY" "$ROOT/scripts/create_label_flip_train_mal.py" \
        --dataset-dir "$DATASET_DIR/$DATASET_NAME" --num-classes 10
  else
      monza_log "Dataset ${DATASET_NAME} already exists! Skipping generation to allow parallel runs."
  fi

  monza_log "Dump MONZA state_dicts (Isolated attack: $TARGET_ATTACK, ${DUMP_NUM_MALICIOUS} Malicious Clients)"
  mkdir -p "$STATE_DICTS_DIR"
  ATTACKS=("$TARGET_ATTACK")
  
  for atk in "${ATTACKS[@]}"; do
    monza_log "-> Dumping state dicts for: $atk"
    
    TMP_DUMP="$STATE_DICTS_DIR/$atk"
    mkdir -p "$TMP_DUMP"
    
    ATTACK_TYPE="$atk" monza_run 5 "$DUMP_NUM_MALICIOUS" "$DUMP_GLOBAL_ROUNDS" "$DUMP_TIMES" \
      --dump_state_dicts "$TMP_DUMP" --dump_start_round "$DUMP_START_ROUND" -at "$atk"
      
    # Direct move: No renaming, no sed edits. Preserves all labels and safetensors perfectly!
    mv "$TMP_DUMP"/* "$STATE_DICTS_DIR/" 2>/dev/null || true
    rm -rf "$TMP_DUMP"
  done

  find "$STATE_DICTS_DIR" -name '*.json' | wc -l
  du -sh "$STATE_DICTS_DIR"

  monza_log "Train MLP detector"
  STATE_DICTS_DIR="$STATE_DICTS_DIR" PUBLIC_VAL_DIR="$PUBLIC_VAL_DIR" \
  DATASET_NAME="$DATASET_NAME" ARTIFACTS_DIR="$MLP_DIR" \
  OVERSAMPLE_LABEL_FACTOR="$OVERSAMPLE_LABEL_FACTOR" LABEL_LOSS_WEIGHT="$LABEL_LOSS_WEIGHT" \
    "$VENV_PY" -u src/detector_mlp.py

  [[ "$KEEP_DUMP" == "1" ]] || rm -rf "$STATE_DICTS_DIR"
  rm -f "$RESULTS_DIR"/"${DATASET_NAME}"_FedAvg_5_100.0_"${DUMP_NUM_MALICIOUS}"_test_*.h5

  monza_log "Run baselines (${NUM_MALICIOUS} Malicious Clients)"
  monza_run 5 0 "$GLOBAL_ROUNDS" "$TIMES" -at "none"
  ATTACK_TYPE="$TARGET_ATTACK" monza_run 5 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES" -at "$TARGET_ATTACK" || true
  ATTACK_TYPE="$TARGET_ATTACK" monza_run 3 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES" -at "$TARGET_ATTACK"

  local mlp_args=(--detector_dir "$MLP_DIR" --mlp_threshold_key "$MLP_THRESHOLD_KEY")
  [[ -z "$MLP_THRESHOLD_VALUE" ]] || mlp_args+=(--mlp_threshold_value "$MLP_THRESHOLD_VALUE")

  monza_log "========================================"
  monza_log "   STARTING CC=7 (MLP Defense vs $TARGET_ATTACK)"
  monza_log "========================================"

  for atk in "${ATTACKS[@]}"; do
      monza_log "   -> Running CC=7 against: $atk"
      ATTACK_TYPE="$atk" monza_run 7 "$NUM_MALICIOUS" "$GLOBAL_ROUNDS" "$TIMES" -at "$atk" "${mlp_args[@]}"
  done

  monza_log "Execute notebook plots"
  REPO_ROOT="$ROOT" ANALYSIS_OUT="$ANALYSIS_OUT" DATASET_NAME="$DATASET_NAME" \
    "$JUPYTER" nbconvert --to notebook --execute \
    notebooks/notebook_monza_analysis.ipynb \
    --output notebook-monza-analysis.executed.ipynb --output-dir "$RUN_OUTPUT" \
    || monza_log "WARN: nbconvert falhou; seguindo para os summaries CLI"

  monza_log "Write CLI summaries"
  "$VENV_PY" "$ROOT/scripts/plot_cc_attack_types.py" \
    --system-dir "$SYSTEM_DIR" --results-dir "$RESULTS_DIR" --out-dir "$ANALYSIS_OUT" \
    --dataset "$DATASET_NAME" --tail-rounds 15 \
    --num-malicious "$NUM_MALICIOUS" || true
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
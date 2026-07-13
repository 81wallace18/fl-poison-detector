#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV_PY="${VENV_PY:-$ROOT/.venv/bin/python}"
SYSTEM_DIR="$ROOT/PFLlibMonza/system"
DATASET_DIR="$ROOT/PFLlibMonza/dataset"
RESULTS_DIR="$ROOT/PFLlibMonza/results"
LOCK_FILE="${LOCK_FILE:-$ROOT/.run_paper_cc3.lock}"

DATASET_NAME="${DATASET_NAME:-MNIST}"
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
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.2}"
ANALYSIS_OUT="${ANALYSIS_OUT:-$ROOT/artifacts/runs/legacy/paper_cc3_${DATASET_NAME}}"
RUN_LOG="${RUN_LOG:-$ANALYSIS_OUT/run.log}"
RUN_STATUS_FILE="$ANALYSIS_OUT/RUN_STATUS.txt"
CSV_BACKUP_DIR="$ANALYSIS_OUT/pre_run_system_csv_backup"
GENERATED_CSV_DIR="$ANALYSIS_OUT/system_csvs"
STAGE_LOG_DIR="$ANALYSIS_OUT/stage_logs"
RESTORE_SYSTEM_CSVS=0

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

backup_system_csvs() {
  mkdir -p "$CSV_BACKUP_DIR" "$GENERATED_CSV_DIR" "$STAGE_LOG_DIR"
  shopt -s nullglob
  local csvs=(
    "$SYSTEM_DIR"/f.csv
    "$SYSTEM_DIR"/fpr_frr_results_*.csv
    "$SYSTEM_DIR"/cc_detail_results_*.csv
    "$SYSTEM_DIR"/cc_type_results_*.csv
  )
  if ((${#csvs[@]} > 0)); then
    cp "${csvs[@]}" "$CSV_BACKUP_DIR"/
  fi
  rm -f "${csvs[@]}"
  RESTORE_SYSTEM_CSVS=1
  shopt -u nullglob
}

restore_system_csvs() {
  if [[ "$RESTORE_SYSTEM_CSVS" != "1" ]]; then
    return 0
  fi
  rm -f \
    "$SYSTEM_DIR"/f.csv \
    "$SYSTEM_DIR"/fpr_frr_results_*.csv \
    "$SYSTEM_DIR"/cc_detail_results_*.csv \
    "$SYSTEM_DIR"/cc_type_results_*.csv
  if [[ -d "$CSV_BACKUP_DIR" ]]; then
    shopt -s nullglob
    local backups=("$CSV_BACKUP_DIR"/*.csv)
    if ((${#backups[@]} > 0)); then
      cp "${backups[@]}" "$SYSTEM_DIR"/
    fi
    shopt -u nullglob
  fi
}

mark_interrupted() {
  local status=$?
  if [[ "$status" -ne 0 && -d "$ANALYSIS_OUT" ]]; then
    printf 'invalid\nexit_code=%s\ninterrupted_or_failed_at=%s\n' "$status" "$(date '+%Y-%m-%d %H:%M:%S')" >"$RUN_STATUS_FILE"
  fi
  restore_system_csvs
  exit "$status"
}

copy_generated_csvs() {
  shopt -s nullglob
  local csvs=(
    "$SYSTEM_DIR"/f.csv
    "$SYSTEM_DIR"/fpr_frr_results_*.csv
    "$SYSTEM_DIR"/cc_detail_results_*.csv
    "$SYSTEM_DIR"/cc_type_results_*.csv
  )
  if ((${#csvs[@]} > 0)); then
    cp "${csvs[@]}" "$GENERATED_CSV_DIR"/
  fi
  shopt -u nullglob
}

run_monza() {
  local stage="$1"
  local cc="$2"
  local nmal="$3"
  shift 3
  log "Run ${DATASET_NAME} stage=${stage} cc=${cc} nmal=${nmal}"
  local stage_log="$STAGE_LOG_DIR/${stage}.log"
  cd "$SYSTEM_DIR"
  "$VENV_PY" -u main.py \
    -m "$MODEL" \
    -data "$DATASET_NAME" \
    -nmc "$nmal" \
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
    "$@" 2>&1 | tee "$stage_log"
  cd "$ROOT"
}

generate_dataset() {
  log "Generate ${DATASET_NAME} partition alpha=${DIRICHLET_ALPHA}"
  rm -rf "$DATASET_DIR/$DATASET_NAME"
  cd "$DATASET_DIR"
  case "$DATASET_NAME" in
    MNIST)
      "$VENV_PY" generate_MNIST.py noniid - dir \
        --num-clients "$NUM_CLIENTS" \
        --dirichlet-alpha "$DIRICHLET_ALPHA"
      ;;
    Cifar10)
      "$VENV_PY" generate_Cifar10.py noniid - dir \
        --num-clients "$NUM_CLIENTS" \
        --dirichlet-alpha "$DIRICHLET_ALPHA"
      ;;
    *)
      echo "DATASET_NAME suportado: MNIST ou Cifar10; recebido: $DATASET_NAME" >&2
      exit 1
      ;;
  esac
  cd "$ROOT"
  "$VENV_PY" scripts/create_label_flip_train_mal.py \
    --dataset-dir "$DATASET_DIR/$DATASET_NAME" \
    --num-classes 10
  printf 'train=%s train_mal=%s public_val=%s test=%s\n' \
    "$(find "$DATASET_DIR/$DATASET_NAME/train" -name '*.npz' | wc -l)" \
    "$(find "$DATASET_DIR/$DATASET_NAME/train_mal" -name '*.npz' | wc -l)" \
    "$(find "$DATASET_DIR/$DATASET_NAME/public_val" -name '*.npz' | wc -l)" \
    "$(find "$DATASET_DIR/$DATASET_NAME/test" -name '*.npz' | wc -l)"
}

write_summary() {
  log "Write paper CC3 summary"
  mkdir -p "$ANALYSIS_OUT"
  RESULTS_DIR="$RESULTS_DIR" \
  ANALYSIS_OUT="$ANALYSIS_OUT" \
  DATASET_NAME="$DATASET_NAME" \
  NUM_MALICIOUS="$NUM_MALICIOUS" \
    "$VENV_PY" - <<'PY'
import csv
import glob
import os

import h5py

dataset = os.environ["DATASET_NAME"]
results_dir = os.environ["RESULTS_DIR"]
out_dir = os.environ["ANALYSIS_OUT"]
rows = []
for label, cc, nmal in [
    ("default_clean", 5, 0),
    ("without_defense", 5, int(os.environ["NUM_MALICIOUS"])),
    ("monza_cc3", 3, int(os.environ["NUM_MALICIOUS"])),
]:
    pattern = os.path.join(results_dir, f"{dataset}_FedAvg_{cc}_100.0_{nmal}_test_*.h5")
    for path in sorted(glob.glob(pattern)):
        with h5py.File(path, "r") as hf:
            acc = list(map(float, hf["rs_test_acc"][:]))
        rows.append({
            "dataset": dataset,
            "label": label,
            "cc": cc,
            "nmal": nmal,
            "run": os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[-1],
            "rounds": len(acc) - 1,
            "final_acc": acc[-1],
            "best_acc": max(acc),
        })

out_path = os.path.join(out_dir, "paper_cc3_accuracy_summary.csv")
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "dataset", "label", "cc", "nmal", "run", "rounds", "final_acc", "best_acc",
    ])
    writer.writeheader()
    writer.writerows(rows)
print(out_path)
PY

  # Paper-comparable FPR/FRR: mean per-round DetectionFPR/DetectionFRR over steady-state
  # rounds (>= ROUND_INIT_ATK), plus the QuarantineFPR/FRR diagnostic. Reads the cc=3
  # CSV copied into system_csvs; tolerant of legacy column names via scripts/_fpr_frr_io.
  ROOT="$ROOT" \
  ANALYSIS_OUT="$ANALYSIS_OUT" \
  GENERATED_CSV_DIR="$GENERATED_CSV_DIR" \
  DATASET_NAME="$DATASET_NAME" \
  ROUND_INIT_ATK="$ROUND_INIT_ATK" \
    "$VENV_PY" - <<'PY'
import csv
import os
import sys

root = os.environ["ROOT"]
sys.path.insert(0, os.path.join(root, "scripts"))
from _fpr_frr_io import load_fpr_frr, summarize_fpr_frr  # noqa: E402

out_dir = os.environ["ANALYSIS_OUT"]
csv_dir = os.environ["GENERATED_CSV_DIR"]
dataset = os.environ["DATASET_NAME"]
min_round = int(os.environ["ROUND_INIT_ATK"])

fields = [
    "dataset", "cc", "rounds_used",
    "DetectionFPR", "DetectionFRR", "QuarantineFPR", "QuarantineFRR",
]
rows = []
for cc in (3,):
    path = os.path.join(csv_dir, f"fpr_frr_results_{cc}.csv")
    if not os.path.exists(path):
        continue
    summary = summarize_fpr_frr(load_fpr_frr(path), min_round=min_round)
    rows.append({
        "dataset": dataset,
        "cc": cc,
        "rounds_used": summary["rounds_used"],
        "DetectionFPR": summary["DetectionFPR_mean"],
        "DetectionFRR": summary["DetectionFRR_mean"],
        "QuarantineFPR": summary["QuarantineFPR_mean"],
        "QuarantineFRR": summary["QuarantineFRR_mean"],
    })

out_path = os.path.join(out_dir, "paper_cc3_fpr_frr_summary.csv")
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(out_path)
PY
}

main() {
  cd "$ROOT"
  if [[ "$MODEL" != "CNN" ]]; then
    echo "run_paper_cc3.sh usa MODEL=CNN para comparar com o paper; recebido: $MODEL" >&2
    exit 1
  fi

  exec 9>"$LOCK_FILE"
  log "Waiting for lock $LOCK_FILE"
  flock 9

  log "START PAPER CC3"
  log "ROOT=$ROOT"
  log "DATASET_NAME=$DATASET_NAME MODEL=$MODEL GLOBAL_ROUNDS=$GLOBAL_ROUNDS TIMES=$TIMES"
  log "DIRICHLET_ALPHA=$DIRICHLET_ALPHA NUM_CLIENTS=$NUM_CLIENTS NUM_MALICIOUS=$NUM_MALICIOUS"
  log "LOG=$RUN_LOG"

  rm -rf "$ANALYSIS_OUT"
  mkdir -p "$ANALYSIS_OUT"
  printf 'running\nstarted_at=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" >"$RUN_STATUS_FILE"
  trap mark_interrupted EXIT

  backup_system_csvs

  rm -f \
    "$RESULTS_DIR"/"${DATASET_NAME}_FedAvg_5_100.0_0_test_"*.h5 \
    "$RESULTS_DIR"/"${DATASET_NAME}_FedAvg_5_100.0_${NUM_MALICIOUS}_test_"*.h5 \
    "$RESULTS_DIR"/"${DATASET_NAME}_FedAvg_3_100.0_${NUM_MALICIOUS}_test_"*.h5

  log "Validate environment"
  "$VENV_PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY

  generate_dataset
  run_monza default_clean 5 0
  run_monza without_defense 5 "$NUM_MALICIOUS"
  run_monza monza_cc3 3 "$NUM_MALICIOUS"

  copy_generated_csvs
  cp "$RESULTS_DIR"/"${DATASET_NAME}_FedAvg_5_100.0_0_test_"*.h5 "$ANALYSIS_OUT"/ 2>/dev/null || true
  cp "$RESULTS_DIR"/"${DATASET_NAME}_FedAvg_5_100.0_${NUM_MALICIOUS}_test_"*.h5 "$ANALYSIS_OUT"/ 2>/dev/null || true
  cp "$RESULTS_DIR"/"${DATASET_NAME}_FedAvg_3_100.0_${NUM_MALICIOUS}_test_"*.h5 "$ANALYSIS_OUT"/ 2>/dev/null || true

  write_summary
  printf 'complete\nfinished_at=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" >"$RUN_STATUS_FILE"
  restore_system_csvs
  RESTORE_SYSTEM_CSVS=0
  trap - EXIT
  log "DONE PAPER CC3"
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
    DIRICHLET_ALPHA="$DIRICHLET_ALPHA" \
    ANALYSIS_OUT="$ANALYSIS_OUT" \
    RUN_LOG="$RUN_LOG" \
    "$0" >"$RUN_LOG" 2>&1 &
  printf 'Started PID %s\nLog: %s\n' "$!" "$RUN_LOG"
else
  main 2>&1 | tee "$RUN_LOG"
fi

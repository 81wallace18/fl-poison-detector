#!/usr/bin/env bash

monza_log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

monza_dataset_slug() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

monza_check_sync() {
  python3 "$ROOT/scripts/_check_runtime_sync.py"
}

# monza_run <cc> <nmal> <rounds> <times> [main.py args...]
monza_run() {
  local cc="$1" nmal="$2" rounds="$3" times="$4"
  shift 4
  monza_log "Run cc=${cc} nmal=${nmal} gr=${rounds} times=${times}"
  (
    cd "$SYSTEM_DIR"
    "$VENV_PY" -u main.py \
      -m "$MODEL" -data "$DATASET_NAME" -nmc "$nmal" -nc "$NUM_CLIENTS" \
      -jr "$JOIN_RATIO" -atk all -ria "$ROUND_INIT_ATK" -cc "$cc" \
      -gr "$rounds" -t "$times" -ls "$LOCAL_STEPS" -did "$DEVICE_ID" \
      -rfake "$RATE_FAKE" "$@"
  )
}

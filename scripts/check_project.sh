#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON=python3
[[ ! -x .venv/bin/python ]] || PYTHON=.venv/bin/python

bash -n scripts/*.sh scripts/lib/*.sh scripts/workflows/*.sh scripts/legacy/*.sh
for script in \
  scripts/run_full_monza.sh scripts/run_full_cifar10.sh scripts/rerun_cc7.sh \
  scripts/sweep_monza_thresholds.sh scripts/lib/*.sh scripts/workflows/*.sh \
  scripts/legacy/*.sh
do
  # shellcheck disable=SC1090
  (source "$script")
done
"$PYTHON" -m py_compile src/*.py scripts/*.py scripts/tools/*.py scripts/legacy/*.py tests/*.py
"$PYTHON" scripts/check_runtime_sync.py
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" scripts/tools/check_markdown_links.py
"$PYTHON" - <<'PY'
import json
from pathlib import Path

for path in Path("notebooks").glob("*.ipynb"):
    json.loads(path.read_text(encoding="utf-8"))
    print(f"notebook json ok: {path}")
PY

for script in scripts/run_full_monza.sh scripts/run_full_cifar10.sh \
  scripts/rerun_cc7.sh scripts/sweep_monza_thresholds.sh
do
  bash "$script" --dry-run >/dev/null
done

if [[ "$PYTHON" == ".venv/bin/python" ]]; then
  "$PYTHON" scripts/create_label_flip_train_mal.py --help >/dev/null
  "$PYTHON" scripts/plot_cc_attack_types.py --help >/dev/null
  "$PYTHON" scripts/summarize_threshold_sweep.py --help >/dev/null
else
  echo "WARN: .venv ausente; smoke de dependencias cientificas nao executado" >&2
fi

echo "project checks ok"

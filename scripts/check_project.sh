#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON=python3
[[ ! -x .venv/bin/python ]] || PYTHON=.venv/bin/python
export PYTHONPYCACHEPREFIX="$ROOT/artifacts/cache/pycache"

bash -n scripts/*.sh
for script in scripts/_monza_common.sh scripts/run_full.sh scripts/rerun_cc7.sh; do
  # shellcheck disable=SC1090
  (source "$script")
done

"$PYTHON" -m py_compile \
  src/*.py scripts/*.py tests/*.py \
  PFLlibMonza/system/main.py \
  PFLlibMonza/system/flcore/detector/*.py \
  PFLlibMonza/system/flcore/servers/serveravg.py
"$PYTHON" scripts/_check_runtime_sync.py
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" scripts/_check_markdown_links.py

for profile in mnist cifar10; do
  bash scripts/run_full.sh "$profile" --dry-run >/dev/null
  bash scripts/rerun_cc7.sh "$profile" --dry-run >/dev/null
done

if [[ "$PYTHON" == ".venv/bin/python" ]]; then
  "$PYTHON" scripts/create_label_flip_train_mal.py --help >/dev/null
  "$PYTHON" scripts/plot_cc_attack_types.py --help >/dev/null
  "$PYTHON" PFLlibMonza/system/main.py --help >/dev/null
else
  echo "WARN: .venv ausente; smoke de dependencias cientificas nao executado" >&2
fi

echo "project checks ok"

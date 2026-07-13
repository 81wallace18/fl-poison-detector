#!/usr/bin/env bash
set -euo pipefail
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROOT DATASET_PROFILE=mnist
exec "$ROOT/scripts/workflows/run_full.sh" "$@"

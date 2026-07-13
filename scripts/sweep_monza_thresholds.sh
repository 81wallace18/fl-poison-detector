#!/usr/bin/env bash
set -euo pipefail
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROOT
exec "$ROOT/scripts/workflows/sweep_monza_thresholds.sh" "$@"

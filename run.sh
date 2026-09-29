#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "smoke" ]]; then
  shift
  exec "${PYTHON:-python3}" "$script_dir/test_smoke.py" "$@"
fi
exec "${PYTHON:-python3}" "$script_dir/run.py" "$@"

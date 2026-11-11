#!/usr/bin/env bash
# crispz-motion - install (Linux / macOS). The work is in cm_setup.py:
#   .venv the app (Pillow), film/.venv FILM + ESRGAN (torch, spandrel),
#   film/models/film_net_fp32.pt (138 MB, Apache 2.0), config.json when missing.
# Linux gets torch cu128 (NVIDIA); macOS has no CUDA: FILM runs on the CPU (slow).
set -euo pipefail
cd "$(dirname "$0")"
PY=""
for c in python3.11 python3.12 python3.13 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
        PY="$c"; break
    fi
done
[ -n "$PY" ] || { echo "[install] Python 3.10 or later not found"; exit 1; }
echo "[install] Python: $PY"
"$PY" cm_setup.py install

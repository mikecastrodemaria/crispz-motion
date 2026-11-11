#!/usr/bin/env bash
# crispz-motion - update (Linux / macOS): git pull --ff-only (git refuses to overwrite a
# file changed here), then reinstalls only the dependencies whose file changed; torch
# must stay a CUDA build on Linux. config.json is never touched: new keys are listed.
#   --force-deps   reinstall the dependencies anyway
main() {
    cd "$(dirname "$0")"
    local py=.venv/bin/python
    [ -x .venv/Scripts/python.exe ] && py=.venv/Scripts/python.exe   # git-bash on Windows
    if [ ! -x "$py" ]; then
        ./install.sh
        exit
    fi
    "$py" cm_setup.py update "$@"
}
main "$@"
exit

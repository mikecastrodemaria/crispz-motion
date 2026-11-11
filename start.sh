#!/usr/bin/env bash
# crispz-motion - start (Linux / macOS).
#   1. first run: install
#   2. GitHub has new commits: asks "Update first?" (10 s, no answer = no)
#   3. no argument: the web page (cm_server.py, Ctrl+C to stop); arguments: cm_cli.py
# The body is a function on purpose: bash reads a script while it runs, and a git pull
# that changes this file would otherwise make it read the wrong line.
main() {
    cd "$(dirname "$0")"
    local py=.venv/bin/python new ans
    # git-bash / msys2 on Windows: the venv keeps its python in Scripts/
    [ -x .venv/Scripts/python.exe ] && py=.venv/Scripts/python.exe
    if [ ! -x "$py" ]; then
        ./install.sh || exit 1
        [ -x .venv/Scripts/python.exe ] && py=.venv/Scripts/python.exe
    fi
    new="$("$py" cm_setup.py check-update 2>/dev/null || echo 0)"
    case "$new" in ''|*[!0-9]*) new=0 ;; esac
    if [ "$new" -gt 0 ]; then
        echo "[start] $new new commit(s) on GitHub."
        if ! read -r -t 10 -p "[start] Update first? [y/N] (10 s, default N) " ans; then
            ans=n; echo
        fi
        case "$ans" in [yYoO]*) ./update.sh ;; esac
    fi
    [ $# -eq 0 ] && exec "$py" cm_server.py
    exec "$py" cm_cli.py "$@"
}
main "$@"
exit

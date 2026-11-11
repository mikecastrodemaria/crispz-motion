"""crispz-motion - install, update and update check, shared by the .bat / .ps1 / .sh.

The shell scripts only find a Python and ask the questions; the work is here, once, so
the three platforms cannot drift apart.

    python cm_setup.py install        two venvs, the FILM weights, config.json
    python cm_setup.py update         git pull --ff-only, then only what changed
    python cm_setup.py check-update   prints how many commits GitHub has that we lack

The two venvs:
    .venv        the app: stdlib + Pillow (requirements.txt)
    film/.venv   FILM and ESRGAN: torch cu128 + spandrel (film/requirements.txt)

The torch trap (seen on lot 0): a torchvision taken from PyPI replaces the CUDA torch with
a CPU one. torch and torchvision always come from the PyTorch index first, and the
install says so when torch is not a CUDA build. On macOS there is no CUDA: FILM runs on
the CPU (slow) and the install says it.
"""

import os
import sys
import json
import shutil
import hashlib
import platform
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
APP_VENV = os.path.join(HERE, ".venv")
FILM_VENV = os.path.join(HERE, "film", ".venv")
FILM_URL = ("https://github.com/dajes/frame-interpolation-pytorch/releases/download/"
            "v1.0.2/film_net_fp32.pt")
FILM_FILE = os.path.join(HERE, "film", "models", "film_net_fp32.pt")
FILM_SIZE = 137922129
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
IS_MAC = platform.system() == "Darwin"


def say(msg):
    print(f"[setup] {msg}", flush=True)


def vpy(venv):
    for p in (os.path.join(venv, "Scripts", "python.exe"), os.path.join(venv, "bin", "python")):
        if os.path.isfile(p):
            return p
    return None


def run(cmd, check=True, **kw):
    p = subprocess.run(cmd, cwd=HERE, **kw)
    if check and p.returncode != 0:
        raise SystemExit(f"[setup] FAILED ({p.returncode}): {' '.join(map(str, cmd))}")
    return p


def out(cmd):
    try:
        return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def file_md5(path):
    if not os.path.isfile(path):
        return None
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def make_venv(venv):
    if not vpy(venv):
        say(f"creating {os.path.relpath(venv, HERE)} with Python {platform.python_version()}")
        run([sys.executable, "-m", "venv", venv])
    py = vpy(venv)
    run([py, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    return py


def torch_version(py):
    return out([py, "-c", "import torch; print(torch.__version__)"])


def install_app():
    py = make_venv(APP_VENV)
    say("app venv: pip install -r requirements.txt")
    run([py, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"])


def install_film(force=False):
    py = make_venv(FILM_VENV)
    tv = torch_version(py)
    if force or not tv or (not IS_MAC and "+cu" not in tv):
        say("FILM venv: torch + torchvision" + ("" if IS_MAC else " (cu128)"))
        cmd = [py, "-m", "pip", "install", "torch", "torchvision", "--quiet"]
        if not IS_MAC:
            cmd += ["--index-url", TORCH_INDEX]
            if tv:
                cmd += ["--force-reinstall"]
        run(cmd)
    say("FILM venv: pip install -r film/requirements.txt")
    run([py, "-m", "pip", "install", "-r", os.path.join("film", "requirements.txt"), "--quiet"])
    tv = torch_version(py)
    if IS_MAC:
        say(f"torch {tv}: no CUDA on macOS, FILM and ESRGAN run on the CPU (slow)")
    elif "+cu" not in tv:
        raise SystemExit(f"[setup] torch {tv} is a CPU build: run the install again")
    else:
        gpu = out([py, "-c", "import torch; print(torch.cuda.get_device_name(0) "
                             "if torch.cuda.is_available() else 'none')"])
        say(f"torch {tv}, GPU: {gpu}")


def download_weights():
    if os.path.isfile(FILM_FILE) and os.path.getsize(FILM_FILE) == FILM_SIZE:
        say("FILM weights: present")
        return
    os.makedirs(os.path.dirname(FILM_FILE), exist_ok=True)
    say("FILM weights: downloading film_net_fp32.pt v1.0.2 (138 MB, Apache 2.0, "
        "dajes/frame-interpolation-pytorch)")
    tmp = FILM_FILE + ".part"
    with urllib.request.urlopen(FILM_URL, timeout=60) as r, open(tmp, "wb") as fh:
        total, done, last = int(r.headers.get("Content-Length") or 0), 0, -1
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            pct = done * 100 // total if total else 0
            if pct // 10 != last:
                last = pct // 10
                print(f"  {pct} %", flush=True)
    if os.path.getsize(tmp) != FILM_SIZE:
        os.remove(tmp)
        raise SystemExit(f"[setup] the FILM weights have the wrong size (a proxy page?): "
                         f"run the install again")
    os.replace(tmp, FILM_FILE)


def config_report(create=False):
    cfg, sample = os.path.join(HERE, "config.json"), os.path.join(HERE, "config-sample.json")
    if not os.path.isfile(cfg):
        if create:
            shutil.copyfile(sample, cfg)
            say("config.json created from config-sample.json (edit engines.moss.dir if "
                "crispz-moss is not next to this folder)")
        return
    try:
        a = json.load(open(cfg, encoding="utf-8"))
        b = json.load(open(sample, encoding="utf-8"))
    except (OSError, ValueError) as e:
        say(f"config.json unreadable: {e}")
        return
    new = [k for k in b if k not in a and not k.startswith("_")]
    if new:
        say(f"new config keys in config-sample.json: {', '.join(new)} (config.json is "
            f"never overwritten; the defaults apply until you add them)")


def check_ffmpeg():
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    hint = {"Windows": "winget install Gyan.FFmpeg", "Darwin": "brew install ffmpeg"}.get(
        platform.system(), "sudo apt install ffmpeg")
    say(f"WARNING: ffmpeg / ffprobe not on the PATH: the exports need them ({hint})")


def install():
    install_app()
    install_film()
    download_weights()
    config_report(create=True)
    check_ffmpeg()
    say("ready. Start: start.bat / start.ps1 / ./start.sh")


def update(force_deps=False):
    reqs = {"app": "requirements.txt", "film": os.path.join("film", "requirements.txt")}
    before = {k: file_md5(os.path.join(HERE, v)) for k, v in reqs.items()}
    if not shutil.which("git") or not os.path.isdir(os.path.join(HERE, ".git")):
        say("no git (or not a git clone): pull skipped, update the files by hand")
    else:
        say("git pull --ff-only (git refuses to overwrite a file you changed here)")
        p = run(["git", "pull", "--ff-only"], check=False)
        if p.returncode != 0:
            raise SystemExit("[setup] git pull failed: local changes in the way, or the "
                             "branch diverged. Commit or stash them, then update again")
    after = {k: file_md5(os.path.join(HERE, v)) for k, v in reqs.items()}
    if force_deps or before["app"] != after["app"] or not vpy(APP_VENV):
        install_app()
    else:
        say("requirements.txt unchanged")
    tv = torch_version(vpy(FILM_VENV)) if vpy(FILM_VENV) else ""
    if force_deps or before["film"] != after["film"] or not tv:
        install_film()
    else:
        say(f"film/requirements.txt unchanged (torch {tv})")
        if not IS_MAC and "+cu" not in tv:
            say("WARNING: torch in film/.venv is a CPU build: run update with --force-deps")
    download_weights()
    config_report()
    check_ffmpeg()
    if os.path.isfile(os.path.join(HERE, "CHANGELOG.md")):
        say("what's new: CHANGELOG.md")
    say("update done")


def check_update():
    """Prints the number of commits upstream has that HEAD lacks (0 when unknown)."""
    if not shutil.which("git") or not os.path.isdir(os.path.join(HERE, ".git")):
        print(0)
        return
    try:
        subprocess.run(["git", "fetch", "--quiet"], cwd=HERE, timeout=15,
                       capture_output=True)
    except (OSError, subprocess.TimeoutExpired):
        print(0)
        return
    n = out(["git", "rev-list", "--count", "HEAD..@{u}"])
    print(int(n) if n.isdigit() else 0)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if sys.version_info < (3, 10):
        raise SystemExit(f"[setup] Python 3.10 or later is needed (this is "
                         f"{platform.python_version()})")
    if cmd == "install":
        install()
    elif cmd == "update":
        update(force_deps="--force-deps" in sys.argv)
    elif cmd == "check-update":
        check_update()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

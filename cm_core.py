"""crispz-motion - shared basics: paths, config, log, atomic JSON writes.

stdlib only. The config is config.json next to the app (gitignored); missing keys come
from DEFAULTS, so an old config.json keeps working after an update. config-sample.json
is the versioned template.
"""

import os
import sys
import json
import time
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
LOG_DIR = os.path.join(HERE, "logs")

DEFAULTS = {
    "engine": "moss",
    "engines": {
        "moss": {"dir": "../crispz-moss", "url": "http://127.0.0.1:7860", "launch": True},
    },
    # Turbo LoRA, 4 steps, CFG 1 (the licence choice: MOSS-Image + its Turbo LoRA, Apache 2.0)
    "turbo": {"steps": 4, "guidance": 1.0, "loras": ["moss-turbo-4-steps:1"]},
    # img2img keys (chain, mid): more steps than gen, since the denoise is a fraction of
    # them (4 steps only allow 0.25 / 0.5 / 0.75 / 1)
    "chain": {"steps": 8},
    # edit keys: the Moss-Image-Edit author's settings
    "edit": {"steps": 50, "guidance": 4.0},
    # distance guard between consecutive keys (cm_guard): mean difference of 64 px
    # thumbnails, 0-255
    "guard": {"warn": 12, "risk": 25},
    "gen_size": [1088, 1920],
    "projects_dir": "projects",
    "work_mpx": 2.1,            # keys and FILM run at this size at most (order b, Q7)
    "film": {"python": "", "model": "film/models/film_net_fp32.pt", "fp16": True},
    "esrgan_dir": "",           # empty = the engine's own ESRGAN folder
    "esrgan_model": "4x-ClearRealityV1.safetensors",
    "ffmpeg": "ffmpeg",
    "stop_engine_before_film": True,
}

_LOCK = threading.Lock()
_LOG_FILE = None


def deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None):
    path = path or CONFIG_PATH
    cfg = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    cfg = deep_merge(DEFAULTS, {k: v for k, v in cfg.items() if not k.startswith("_")})
    # relative paths are relative to the app folder, never to the caller's cwd
    for name, e in cfg["engines"].items():
        if isinstance(e, dict) and e.get("dir"):
            e["dir"] = abspath(e["dir"])
    cfg["projects_dir"] = abspath(cfg["projects_dir"])
    cfg["film"]["model"] = abspath(cfg["film"]["model"])
    if cfg.get("esrgan_dir"):
        cfg["esrgan_dir"] = abspath(cfg["esrgan_dir"])
    return cfg


def abspath(p):
    p = os.path.expanduser(str(p))
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(HERE, p))


def set_log_file(path):
    """Every log line also goes to this file (the project's render.log)."""
    global _LOG_FILE
    _LOG_FILE = path


def log(msg, tag="motion"):
    line = f"[{tag}] {msg}"
    with _LOCK:
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode(), flush=True)
        if _LOG_FILE:
            try:
                with open(_LOG_FILE, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%H:%M:%S')} {line}\n")
            except OSError:
                pass


def write_json(path, data):
    """Atomic: a temp file then os.replace, so a crash never leaves half a file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def venv_python(venv_dir):
    """The interpreter of a venv on Windows (Scripts/) or Unix (bin/), or None."""
    for p in (os.path.join(venv_dir, "Scripts", "python.exe"),
              os.path.join(venv_dir, "bin", "python")):
        if os.path.isfile(p):
            return p
    return None


def is_windows():
    return sys.platform.startswith("win")

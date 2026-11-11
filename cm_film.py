"""crispz-motion - drives the two GPU workers in the FILM venv (film/.venv).

  - cm_film_worker.py: FILM, route A (PyTorch port, fp16; decision Q6 after lot 0).
  - cm_upscale_worker.py: ESRGAN on every frame for outputs above the work size
    (order "FILM at 2 Mpx, then ESRGAN", decision Q7).

The app itself never imports torch. A worker is a subprocess; its JSON lines come back
through on_event, its stderr goes to the project's render.log on failure.

Route C (the official TensorFlow FILM on CPU) is not wired in yet: when route A is not
installed the render stops with the reason, it never falls back in silence.
"""

import os
import json
import hashlib
import subprocess

from cm_core import HERE, venv_python, log, check_cancel, register_proc, unregister_proc

FILM_DIR = os.path.join(HERE, "film")


def film_python(cfg):
    p = (cfg.get("film") or {}).get("python") or venv_python(os.path.join(FILM_DIR, ".venv"))
    return p if p and os.path.isfile(p) else None


def check(cfg):
    """None when route A can run, else the reason."""
    py = film_python(cfg)
    if not py:
        return "the FILM venv (film/.venv) is missing: run install.bat / install.sh"
    if not os.path.isfile(cfg["film"]["model"]):
        return f"the FILM weights are missing ({cfg['film']['model']}): run install again"
    return None


def file_hash(path, _memo={}):
    st = os.stat(path)
    k = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if k not in _memo:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _memo[k] = h.hexdigest()
    return _memo[k]


def segment_id(a_path, b_path, times, cfg):
    h = hashlib.sha1()
    for part in (file_hash(a_path), file_hash(b_path), str(times),
                 os.path.basename(cfg["film"]["model"]), str(bool(cfg["film"]["fp16"]))):
        h.update(part.encode())
    return h.hexdigest()[:12]


def segment_complete(folder, times):
    return os.path.isfile(os.path.join(folder, f"f{2 ** times:05d}.png"))


def _run(cmd, on_event, log_path, cwd=None):
    """Runs a worker, streams its JSON lines. Returns (ok, last error text)."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    check_cancel()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         encoding="utf-8", errors="replace", cwd=cwd or HERE, env=env,
                         creationflags=flags)
    register_proc(p)
    try:
        return _stream(p, cmd, on_event, log_path)
    finally:
        unregister_proc(p)


def _stream(p, cmd, on_event, log_path):
    error = None
    for line in p.stdout:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("event") == "error":
            error = ev.get("error")
        if ev.get("event") == "warning":
            log(f"WARNING: {ev.get('warning')}")
        if on_event:
            on_event(ev)
    stderr = p.stderr.read()
    p.wait()
    check_cancel()              # killed by the Cancel button: not a worker failure
    if p.returncode != 0:
        if log_path and stderr:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"--- stderr of {os.path.basename(cmd[1])} ---\n{stderr[-6000:]}\n")
        return False, error or f"worker exited with code {p.returncode}: {stderr[-500:]}"
    return True, None


def run_film(cfg, jobs, workdir, on_event=None, log_path=None):
    """jobs: [{"a", "b", "out", "times"}]. Only the missing segments are passed."""
    if not jobs:
        return True, None
    jobs_path = os.path.join(workdir, "film_jobs.json")
    with open(jobs_path, "w", encoding="utf-8") as fh:
        json.dump(jobs, fh, indent=1)
    cmd = [film_python(cfg), os.path.join(HERE, "cm_film_worker.py"), "--model",
           cfg["film"]["model"], "--jobs", jobs_path]
    if cfg["film"].get("fp16", True):
        cmd.append("--fp16")
    ok, err = _run(cmd, on_event, log_path)
    try:
        os.remove(jobs_path)
    except OSError:
        pass
    return ok, err


def run_esrgan(cfg, model_path, size, jobs, workdir, on_event=None, log_path=None):
    """jobs: [{"src", "dst"}]; size: final (w, h)."""
    if not jobs:
        return True, None
    jobs_path = os.path.join(workdir, "esrgan_jobs.json")
    with open(jobs_path, "w", encoding="utf-8") as fh:
        json.dump(jobs, fh, indent=1)
    cmd = [film_python(cfg), os.path.join(HERE, "cm_upscale_worker.py"), "--model",
           model_path, "--size", str(size[0]), str(size[1]), "--jobs", jobs_path]
    ok, err = _run(cmd, on_event, log_path)
    try:
        os.remove(jobs_path)
    except OSError:
        pass
    return ok, err


def esrgan_dir(cfg):
    """config esrgan_dir > the engine's own folder (crispz-moss preferences.json
    esrgan_dir, else <engine>/upscale_models)."""
    if cfg.get("esrgan_dir"):
        return cfg["esrgan_dir"]
    eng = (cfg.get("engines") or {}).get(cfg.get("engine")) or {}
    d = eng.get("dir") or ""
    prefs = os.path.join(d, "preferences.json")
    if os.path.isfile(prefs):
        try:
            with open(prefs, encoding="utf-8") as fh:
                p = json.load(fh).get("esrgan_dir")
            if p and os.path.isdir(p):
                return p
        except (OSError, ValueError):
            pass
    return os.path.join(d, "upscale_models") if d else ""


def esrgan_model(cfg):
    """(path, None) or (None, reason)."""
    d = esrgan_dir(cfg)
    name = cfg.get("esrgan_model") or "4x-ClearRealityV1.safetensors"
    p = name if os.path.isabs(name) else os.path.join(d, name)
    if os.path.isfile(p):
        return p, None
    return None, f"ESRGAN model not found: {p} (set esrgan_dir / esrgan_model in config.json)"

"""crispz-motion - client AND launcher of the crispz family engines.

Taken from crispz-dataset-studio's ds_engines.py (same contract, same _gradio_call):
crispz-motion renders no key itself, every key comes from a family tool (crispz-moss by
default) through the CLI protocol v1, the hidden Gradio endpoints cli_caps / cli_gen /
cli_upscale / cli_edit / cli_inpaint. img2img is op "upscale" with factor 1 and a denoise.

The engine's install folder is in the config, so when nothing answers at its URL we start
its app.py ourselves (port through GRADIO_SERVER_PORT, log to logs/<engine>.log, pid file
to adopt it after a restart of crispz-motion). An engine the user started by hand is never
stopped. One family app at a time on the GPU: a second launch is refused while a child we
started is alive.

Everything is stdlib. Failures come back as {ok: false, error}.
"""

import os
import re
import json
import time
import shutil
import subprocess
import threading
import urllib.request
import urllib.parse

from cm_core import LOG_DIR, log as _core_log

PROTOCOL = 1


def _log(msg):
    _core_log(msg, "engines")


def _gradio_call(url, name, data, post_timeout=10, get_timeout=3600):
    """POST /gradio_api/call/<name> -> event_id, then GET the stream and return the
    first output, JSON-decoded when it is a JSON string."""
    req = urllib.request.Request(
        f"{url.rstrip('/')}/gradio_api/call/{name}",
        data=json.dumps({"data": data}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=post_timeout) as r:
        j = json.loads(r.read().decode("utf-8"))
    eid = j.get("event_id") or j.get("hash")
    if not eid:
        raise ValueError("no event id (endpoint missing? app too old?)")
    with urllib.request.urlopen(f"{url.rstrip('/')}/gradio_api/call/{name}/{eid}",
                                timeout=get_timeout) as r:
        text = r.read().decode("utf-8")
    m = re.search(r"data:\s*(\[[\s\S]*?\])\s*(?:\n|$)", text)
    if not m:
        raise ValueError(f"empty API response: {text[-300:]}")
    out = json.loads(m.group(1))[0]
    if isinstance(out, str):
        try:
            return json.loads(out)
        except ValueError:
            return out
    return out


def probe(url, timeout=4):
    """caps of the instance at `url`, None when nothing answers."""
    try:
        caps = _gradio_call(url, "cli_caps", [], post_timeout=timeout,
                            get_timeout=max(timeout, 8))
        return caps if isinstance(caps, dict) and caps.get("ok") else None
    except Exception:
        return None


def _port_of(url):
    try:
        return int(urllib.parse.urlparse(url).port or 7860)
    except Exception:
        return 7860


def _pid_alive(pid):
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"],
                                 capture_output=True, text=True, timeout=10).stdout
            return str(int(pid)) in out
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _kill_pid(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                       capture_output=True, timeout=20)
    else:
        import signal
        os.kill(int(pid), signal.SIGTERM)


def resolve_python(engine_dir, explicit=None):
    """(argv prefix, label) of the interpreter that runs <engine_dir>/app.py, or
    (None, reason). Order: explicit > .venv > Pinokio env > py -3 / python3."""
    if explicit:
        if os.path.isfile(explicit):
            return [explicit], f"python key ({explicit})"
        return None, f"python key points to a missing file: {explicit}"
    win = os.name == "nt"
    for label, sub in ((".venv", ".venv"), ("Pinokio env", "env")):
        p = os.path.join(engine_dir, sub, "Scripts" if win else "bin",
                         "python.exe" if win else "python")
        if os.path.isfile(p):
            return [p], f"{label} ({p})"
    if win and shutil.which("py"):
        return ["py", "-3"], "py -3 launcher (no .venv in the engine folder)"
    for name in ("python3", "python"):
        p = shutil.which(name)
        if p:
            return [p], f"{name} on PATH (no .venv in the engine folder)"
    return None, "no Python found (no .venv, no env, no py/python3 on PATH)"


class Engine:
    """One configured family tool: how to reach it, and how to start it."""

    def __init__(self, name, cfg):
        self.name = name
        self.dir = str(cfg.get("dir") or "").strip()
        self.url = str(cfg.get("url") or "http://127.0.0.1:7860").strip()
        self.python = str(cfg.get("python") or "").strip() or None
        self.launch_allowed = bool(cfg.get("launch", True))
        self.boot_timeout = float(cfg.get("boot_timeout", 240))
        self.proc = None
        self.log_path = os.path.join(LOG_DIR, f"{name}.log")
        self.pid_path = os.path.join(LOG_DIR, f"{name}.pid")
        self._lock = threading.Lock()
        # a child started by a previous crispz-motion process: adopted through its pid
        # file, so it can still be stopped instead of holding the VRAM as an orphan
        self.adopted_pid = None
        try:
            with open(self.pid_path, encoding="utf-8") as fh:
                pid = int(fh.read().strip() or 0)
            if pid > 0 and _pid_alive(pid):
                self.adopted_pid = pid
            else:
                os.remove(self.pid_path)
        except (OSError, ValueError):
            pass

    def caps(self, timeout=4):
        return probe(self.url, timeout=timeout)

    def started_by_us(self):
        if self.proc is not None:
            return self.proc.poll() is None
        if self.adopted_pid and _pid_alive(self.adopted_pid):
            return True
        self.adopted_pid = None
        return False

    def pid(self):
        return self.proc.pid if self.proc is not None else self.adopted_pid

    def status(self):
        caps = self.caps()
        sup = (caps or {}).get("supports") or {}
        return {"name": self.name, "url": self.url, "dir": self.dir,
                "running": caps is not None, "tool": (caps or {}).get("tool"),
                "version": (caps or {}).get("version"),
                "guidance": bool(sup.get("guidance")), "edit": bool(sup.get("edit")),
                "started_by_us": self.started_by_us(), "pid": self.pid()}

    def launch(self):
        """Starts <dir>/app.py when nothing answers at url, then waits until it does."""
        with self._lock:
            caps = probe(self.url)
            if caps:
                return {"ok": True, "started": False, "caps": caps}
            if not self.launch_allowed:
                return {"ok": False, "error": f"engine '{self.name}' is not running at "
                        f"{self.url} and 'launch' is off in its config: start it by hand"}
            if not self.dir or not os.path.isfile(os.path.join(self.dir, "app.py")):
                return {"ok": False, "error": f"engine '{self.name}': no app.py in "
                        f"'{self.dir or 'empty'}'; set engines.{self.name}.dir in config.json"}
            py, label = resolve_python(self.dir, self.python)
            if not py:
                return {"ok": False, "error": f"engine '{self.name}': {label}"}
            if self.started_by_us():
                return {"ok": False, "error": f"engine '{self.name}' was started by us (pid "
                        f"{self.pid()}) but does not answer at {self.url}; see {self.log_path}"}
            os.makedirs(LOG_DIR, exist_ok=True)
            env = dict(os.environ, GRADIO_SERVER_PORT=str(_port_of(self.url)),
                       GRADIO_SERVER_NAME="127.0.0.1")
            env.setdefault("PYTHONUTF8", "1")
            logf = open(self.log_path, "ab")
            logf.write(f"\n=== launched by crispz-motion {time.strftime('%Y-%m-%d %H:%M:%S')} :: "
                       f"{' '.join(py)} app.py (port {env['GRADIO_SERVER_PORT']})\n".encode())
            flags = 0
            if os.name == "nt":
                flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | \
                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                self.proc = subprocess.Popen(py + ["app.py"], cwd=self.dir, env=env,
                                             stdout=logf, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, creationflags=flags)
            except OSError as e:
                logf.close()
                return {"ok": False, "error": f"engine '{self.name}': cannot start: {e}"}
            try:
                with open(self.pid_path, "w", encoding="utf-8") as fh:
                    fh.write(str(self.proc.pid))
            except OSError:
                pass
            _log(f"{self.name}: started pid {self.proc.pid} with {label}, port "
                 f"{env['GRADIO_SERVER_PORT']}, log {self.log_path}")
        deadline = time.time() + self.boot_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return {"ok": False, "error": f"engine '{self.name}' exited with code "
                        f"{self.proc.returncode} during boot; see {self.log_path}"}
            caps = probe(self.url, timeout=3)
            if caps:
                _log(f"{self.name}: {caps.get('tool')} {caps.get('version')} ready at "
                     f"{self.url} (the model loads at the first key, which takes longer)")
                return {"ok": True, "started": True, "caps": caps}
            time.sleep(1.0)
        return {"ok": False, "error": f"engine '{self.name}' did not answer at {self.url} "
                f"within {int(self.boot_timeout)} s; see {self.log_path}"}

    def stop(self, timeout=30):
        """Terminates the child WE started; an engine started by hand is left alone."""
        if not self.started_by_us():
            self.proc = None
            if probe(self.url, timeout=2):
                return {"ok": False, "error": f"engine '{self.name}' at {self.url} was not "
                        f"started by crispz-motion: close it yourself"}
            return {"ok": True, "stopped": False}
        pid = self.pid()
        try:
            if self.proc is not None:
                if os.name == "nt":
                    _kill_pid(pid)      # /T: the app's own children too
                else:
                    self.proc.terminate()
                try:
                    self.proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            else:
                _kill_pid(pid)
                deadline = time.time() + timeout
                while _pid_alive(pid) and time.time() < deadline:
                    time.sleep(0.5)
        except Exception as e:
            return {"ok": False, "error": f"engine '{self.name}': stop failed: {e}"}
        _log(f"{self.name}: stopped pid {pid}")
        self.proc, self.adopted_pid = None, None
        try:
            os.remove(self.pid_path)
        except OSError:
            pass
        time.sleep(2)           # let the driver give the VRAM back before FILM
        return {"ok": True, "stopped": True}

    def call(self, op, spec, timeout=3600):
        """One protocol op (gen, upscale, edit, inpaint): protocol and op are filled in."""
        spec = dict(spec, protocol=PROTOCOL, op=op)
        try:
            out = _gradio_call(self.url, f"cli_{op}", [json.dumps(spec)], get_timeout=timeout)
        except Exception as e:
            return {"ok": False, "error": f"{self.name} (cli_{op}): {type(e).__name__}: {e}"}
        if not isinstance(out, dict):
            return {"ok": False, "error": f"{self.name} (cli_{op}): unexpected answer {out!r}"}
        return out


class Engines:
    def __init__(self, config):
        self.by_name = {n: Engine(n, c) for n, c in (config.get("engines") or {}).items()
                        if isinstance(c, dict)}
        self.default = str(config.get("engine") or "") or next(iter(self.by_name), "")

    def get(self, name=None):
        name = name or self.default
        if name not in self.by_name:
            raise KeyError(f"unknown engine '{name}' (configured: "
                           f"{', '.join(self.by_name) or 'none'})")
        return self.by_name[name]

    def ensure(self, name=None):
        """The engine, running: probes, launches when allowed, refuses a second child."""
        e = self.get(name)
        if e.caps():
            return e, {"ok": True, "started": False}
        for other in self.by_name.values():
            if other is not e and other.started_by_us():
                return e, {"ok": False, "error": f"engine '{other.name}' (started by "
                           f"crispz-motion) is still running: two family apps do not fit "
                           f"on one GPU. Stop it first"}
        return e, e.launch()

    def stop_ours(self):
        """Stops every engine crispz-motion started (before FILM: the GPU is shared)."""
        for e in self.by_name.values():
            if e.started_by_us():
                e.stop()

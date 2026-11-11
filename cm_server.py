"""crispz-motion - the web page: a stdlib HTTP server, a JSON API, one job at a time.

    python cm_server.py [--port 7870] [--no-browser]

Serves assets/index.html and the project files (keys, thumbnails, videos with Range
requests so the player can seek). Everything that needs the engine or the GPU (keys,
middle keys, regeneration, the Animator plan, the render) is a JOB: one at a time, in a
background thread, with its log lines, its progress per step, a preview image and a
Cancel button. Cancel stops at the next safe point: between two keys (an engine call in
flight finishes first), or at once for FILM / ESRGAN (their worker is killed).

While a job runs on a project, the page cannot change that project (409): the job's own
saves would overwrite the change.

The engine started by crispz-motion stays up between jobs (no model reload between two
keys); a render stops it before FILM, and so does closing the server. An engine started
by hand is never touched.

Bound to 127.0.0.1 by default (config "server"): the API can start apps and delete
projects, it is not meant for a network.
"""

import argparse
import base64
import io
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import cm_core
from cm_core import HERE, load_config, log
from cm_project import Project, cover, DEFAULT_OUTPUT
from cm_engines import Engines
import cm_animator
import cm_film
import cm_guard
import cm_keys
import cm_render
import cm_timeline

ASSETS = os.path.join(HERE, "assets")
STEPS = ("keys", "plan", "film", "timing", "size", "export")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ApiError(Exception):
    def __init__(self, msg, code=400):
        super().__init__(msg)
        self.code = code


# -- jobs ------------------------------------------------------------------------------
class Job:
    def __init__(self, jid, kind, project, label):
        self.id, self.kind, self.project, self.label = jid, kind, project, label
        self.state = "running"
        self.lines = []
        self.steps = {}
        self.preview = None
        self.result = None
        self.error = None
        self.started = time.time()
        self.ended = None

    def view(self, since=0):
        return {"id": self.id, "kind": self.kind, "project": self.project,
                "label": self.label, "state": self.state, "steps": self.steps,
                "preview": self.preview, "result": self.result, "error": self.error,
                "elapsed": round((self.ended or time.time()) - self.started, 1),
                "lines": self.lines[since:], "n": len(self.lines)}


class Jobs:
    def __init__(self):
        self.current = None
        self.last = None
        self.n = 0
        self.lock = threading.Lock()

    def busy_on(self, pid):
        j = self.current
        return j is not None and j.state == "running" and j.project == pid

    def start(self, kind, pid, label, fn):
        with self.lock:
            if self.current is not None and self.current.state == "running":
                raise ApiError(f"a job is running ({self.current.label}): wait or cancel it",
                               409)
            self.n += 1
            job = Job(self.n, kind, pid, label)
            self.current = job
        cm_core.CANCEL.clear()

        def on_line(line):
            job.lines.append(f"{time.strftime('%H:%M:%S')} {line}")
            if len(job.lines) > 5000:
                del job.lines[:1000]

        def on_progress(step, done, total, preview):
            job.steps[step] = {"done": done, "total": total}
            if preview:
                job.preview = preview

        def run():
            cm_core.add_listener(on_line)
            cm_core.add_progress(on_progress)
            try:
                job.result = fn()
                job.state = "done"
            except cm_core.Cancelled:
                job.state = "cancelled"
                log("cancelled")
            except (cm_render.RenderError, cm_timeline.PlanError, cm_animator.AnimatorError,
                    RuntimeError, ValueError, KeyError, FileNotFoundError) as e:
                job.state, job.error = "error", str(e)
                log(f"ERROR: {e}")
            except Exception as e:                      # a bug: keep the trace in the log
                job.state, job.error = "error", f"{type(e).__name__}: {e}"
                log("ERROR: " + traceback.format_exc())
            finally:
                job.ended = time.time()
                cm_core.remove_listener(on_line)
                cm_core.remove_progress(on_progress)
                self.last = job
        threading.Thread(target=run, daemon=True, name=f"job-{job.id}").start()
        return job

    def cancel(self):
        j = self.current
        if j is None or j.state != "running":
            return False
        cm_core.CANCEL.set()
        cm_core.kill_procs()
        log("cancel asked: stopping at the next safe point")
        return True


# -- the app ---------------------------------------------------------------------------
class App:
    def __init__(self, cfg):
        self.cfg = cfg
        self.engines = Engines(cfg)
        self.jobs = Jobs()
        self._status = (0, None)
        os.makedirs(cfg["projects_dir"], exist_ok=True)

    # projects
    def pdir(self, pid):
        if not pid or not _ID.match(pid) or ".." in pid:
            raise ApiError(f"bad project id '{pid}'")
        p = os.path.join(self.cfg["projects_dir"], pid)
        if not os.path.isfile(os.path.join(p, "project.json")):
            raise ApiError(f"no project '{pid}'", 404)
        return p

    def project(self, pid):
        return Project(self.pdir(pid))

    def editable(self, pid):
        if self.jobs.busy_on(pid):
            raise ApiError("a job is running on this project: wait or cancel it", 409)
        return self.project(pid)

    def list_projects(self):
        out = []
        root = self.cfg["projects_dir"]
        for d in sorted(os.listdir(root), reverse=True):
            f = os.path.join(root, d, "project.json")
            if d.startswith(".") or not os.path.isfile(f):
                continue
            data = cm_core.read_json(f, {}) or {}
            keys = data.get("keys") or []
            lr = data.get("last_render") or {}
            out.append({"id": d, "name": data.get("name") or d,
                        "created": data.get("created"), "keys": len(keys),
                        "thumb": keys[0]["id"] if keys else None,
                        "planned": len((data.get("plan") or {}).get("keys") or []),
                        "rendered": lr.get("date"),
                        "mtime": os.path.getmtime(f)})
        out.sort(key=lambda x: x["mtime"], reverse=True)
        return out

    def project_view(self, pid):
        p = self.project(pid)
        keys = []
        for k in p.keys:
            path = p.key_path(k)
            r = k.get("recipe") or {}
            spec = r.get("spec") or {}
            e = None
            if spec.get("denoise") is not None and r.get("op") in ("chain", "mid"):
                run, n = cm_keys.effective_denoise(spec["denoise"], spec.get("steps", 4))
                e = round(run / n, 3)
            keys.append(dict(k, v=int(os.path.getmtime(path)) if os.path.isfile(path) else 0,
                             missing=not os.path.isfile(path), prompt=cm_keys.prompt_of(k),
                             seed=cm_keys.seed_of(k), effective_denoise=e))
        guard = []
        if len(p.keys) >= 2 and all(not k["missing"] for k in keys):
            try:
                guard = cm_guard.check(p, self.cfg)
            except Exception as ex:
                guard = [{"error": str(ex)}]
        o = p.output
        info = {}
        if p.work_size:
            try:
                info["output_size"] = list(cm_timeline.output_size(p.work_size, o["res"]))
            except Exception as ex:
                info["output_error"] = str(ex)
        if len(p.keys) >= 2:
            try:
                pl = cm_timeline.plan(p.keys, int(o["fps"]), o["loop"], o.get("close"))
                info.update(frames=len(pl["frames"]), duration_s=pl["duration_s"])
            except Exception as ex:
                info["plan_error"] = str(ex)
        files = {}
        for f in ("video.mp4", "video.webm", "video.gif", "render.log"):
            fp = os.path.join(p.path, f)
            if os.path.isfile(fp):
                files[f] = int(os.path.getmtime(fp))
        plan = p.data.get("plan")
        return {"id": pid, "name": p.data.get("name"), "path": p.path,
                "work_size": p.work_size, "keys": keys, "output": o, "guard": guard,
                "info": info, "files": files, "plan": plan,
                "pending": len(cm_animator.pending(p)) if plan else 0,
                "last_render": p.data.get("last_render"),
                "busy": self.jobs.busy_on(pid)}

    def status(self, fresh=False):
        t, s = self._status
        if s is not None and not fresh and time.time() - t < 4:
            return s
        film = cm_film.check(self.cfg)
        model, why = cm_film.esrgan_model(self.cfg)
        engines = []
        for e in self.engines.by_name.values():
            try:
                engines.append(e.status())
            except Exception as ex:
                engines.append({"name": e.name, "running": False, "error": str(ex)})
        s = {"film": film or "ready", "film_ok": not film,
             "esrgan": os.path.basename(model) if model else why, "esrgan_ok": bool(model),
             "engines": engines, "engine": self.engines.default,
             "projects_dir": self.cfg["projects_dir"],
             "chain_steps": int((self.cfg.get("chain") or {}).get("steps", 8)),
             "animator": self.cfg.get("animator"), "eases": list(cm_timeline.EASES),
             "loops": list(cm_timeline.LOOPS), "edit_licence": cm_keys.EDIT_LICENCE,
             "guard": self.cfg.get("guard")}
        self._status = (time.time(), s)
        return s

    # -- jobs ------------------------------------------------------------------------
    def job(self, pid, body):
        kind = body.get("kind")
        cfg, eng = self.cfg, self.engines
        engine = body.get("engine") or None
        if kind == "engine_start":
            return self.jobs.start(kind, None, "start the engine",
                                   lambda: _ok(eng.ensure(engine)[1]))
        p = self.editable(pid)

        def n(v, f=float):
            return None if v in (None, "") else f(v)

        if kind == "gen":
            prompt = (body.get("prompt") or "").strip()
            if not prompt:
                raise ApiError("a prompt is needed")
            size = None
            if body.get("size") and not p.work_size:
                w, h = str(body["size"]).lower().split("x")
                size = (int(w), int(h))
            label = f"gen: {prompt[:40]}"
            fn = lambda: _kid(cm_keys.gen_key(p, eng, cfg, prompt, body.get("negative", ""),
                                              n(body.get("seed"), int), size, engine,
                                              t=n(body.get("t"))))
        elif kind == "chain":
            prompts = [x.strip() for x in (body.get("prompts") or "").splitlines() if x.strip()]
            count = max(1, int(body.get("count") or 1))
            frm = body.get("from") or None
            d = float(body.get("denoise") or 0.625)
            label = f"chain x{count} at {d}"

            def fn():
                made, src = [], frm
                for i in range(count):
                    cm_core.check_cancel()
                    cm_core.progress("keys", i, count)
                    pr = prompts[min(i, len(prompts) - 1)] if prompts else None
                    k = cm_keys.chain_key(p, eng, cfg, src, pr, denoise=d,
                                          t=_t_after(p, src), engine=engine)
                    made.append(k["id"])
                    src = k["id"]
                    cm_core.progress("keys", i + 1, count, p.key_path(k))
                return {"keys": made}
        elif kind == "edit":
            ins = (body.get("instruction") or "").strip()
            if not ins:
                raise ApiError("an instruction is needed")
            frm = body.get("from") or None
            label = f"edit: {ins[:40]}"
            fn = lambda: _kid(cm_keys.edit_key(p, eng, cfg, ins, frm, n(body.get("seed"), int),
                                               _t_after(p, frm), engine))
        elif kind == "inpaint":
            prompt = (body.get("prompt") or "").strip()
            if not prompt or not body.get("mask"):
                raise ApiError("a mask and a prompt are needed")
            frm = body.get("from") or None
            mask = _save_mask(p, body["mask"], float(body.get("feather") or 0))
            label = f"inpaint: {prompt[:40]}"
            fn = lambda: _kid(cm_keys.inpaint_key(p, eng, cfg, mask, prompt, frm,
                                                  float(body.get("denoise") or 1.0),
                                                  n(body.get("seed"), int), _t_after(p, frm),
                                                  engine))
        elif kind == "mid":
            a, b = body.get("a"), body.get("b")
            label = f"middle key {a} | {b}"
            fn = lambda: _kid(cm_keys.mid_key(p, eng, cfg, a, b,
                                              float(body.get("denoise", 0.2)), None, None,
                                              engine))
        elif kind == "regen":
            kid = body.get("key")
            seed = n(body.get("seed"), int)
            if body.get("new_seed"):
                seed = int.from_bytes(os.urandom(4), "little")
            label = f"regenerate {kid}" + (" and the keys built on it" if body.get("cascade")
                                           else "")
            fn = lambda: {"keys": [k["id"] for k in cm_keys.regen(
                p, eng, cfg, kid, seed, bool(body.get("cascade")), engine)]}
        elif kind == "build":
            label = "make the planned keys"
            fn = lambda: {"keys": [k["id"] for k in cm_animator.build(p, eng, cfg, engine)]}
        elif kind == "render":
            ov = {k: body.get(k) for k in ("res", "fps", "loop", "close", "formats", "codec",
                                          "crf", "upscaler", "order")}
            label = f"render {p.output.get('res')}"
            fn = lambda: cm_render.render(p, cfg, eng, ov)
        else:
            raise ApiError(f"unknown job '{kind}'")
        return self.jobs.start(kind, pid, label, fn)


def _ok(r):
    if not r.get("ok"):
        raise RuntimeError(r.get("error"))
    return {k: v for k, v in r.items() if k != "caps"}


def _kid(k):
    return {"keys": [k["id"]]}


def _t_after(p, kid):
    """Time of a key made from `kid`: after the last key when kid is the last one (or
    None), else halfway to the key after it."""
    if not kid or kid == p.keys[-1]["id"]:
        return None
    i = [k["id"] for k in p.keys].index(kid)
    t = (float(p.keys[i]["t"]) + float(p.keys[i + 1]["t"])) / 2
    return round(t, 4)


def _save_mask(p, data_url, feather):
    """The mask drawn on the page (a PNG data URL, white = redraw), at the work size,
    feathered by `feather` px (a soft edge: lot 2's inpaint stopped hard at the mask)."""
    from PIL import Image, ImageFilter
    raw = base64.b64decode(data_url.split(",", 1)[-1])
    with Image.open(io.BytesIO(raw)) as im:
        m = im.convert("L")
    if p.work_size:
        m = cover(m, *p.work_size)
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    os.makedirs(os.path.join(p.path, "masks"), exist_ok=True)
    path = os.path.join(p.path, "masks", f"mask_{time.strftime('%Y%m%d_%H%M%S')}.png")
    m.save(path)
    return path


# -- HTTP --------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    app = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):         # quiet: the page polls every second
        pass

    # helpers
    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        return self._raw

    def _jbody(self):
        b = self._body()
        try:
            return json.loads(b.decode("utf-8")) if b else {}
        except ValueError:
            raise ApiError("bad JSON")

    def _file(self, path, ctype=None, cache=False):
        if not os.path.isfile(path):
            raise ApiError("not found", 404)
        size = os.path.getsize(path)
        ctype = ctype or mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "max-age=31536000" if cache else "no-cache")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
                left -= len(chunk)

    def _inside(self, path):
        root = os.path.normcase(os.path.abspath(self.app.cfg["projects_dir"]))
        ap = os.path.normcase(os.path.abspath(path))
        if not ap.startswith(root + os.sep):
            raise ApiError("outside the projects folder", 403)
        return path

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def _route(self, method):
        # the body is always read, used or not: HTTP/1.1 keeps the connection, and an
        # unread body would be taken as the start of the next request
        n = int(self.headers.get("Content-Length") or 0)
        self._raw = self.rfile.read(n) if n else b""
        u = urllib.parse.urlsplit(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        parts = [urllib.parse.unquote(x) for x in u.path.strip("/").split("/") if x]
        try:
            self._dispatch(method, parts, q)
        except ApiError as e:
            self._json({"ok": False, "error": str(e)}, e.code)
        except (cm_timeline.PlanError, cm_animator.AnimatorError, ValueError, KeyError,
                FileNotFoundError) as e:
            self._json({"ok": False, "error": str(e)}, 400)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            traceback.print_exc()
            self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

    def _dispatch(self, method, parts, q):
        app = self.app
        if method == "GET" and (not parts or parts == ["index.html"]):
            return self._file(os.path.join(ASSETS, "index.html"), "text/html; charset=utf-8")
        if method == "GET" and parts[0] == "assets" and len(parts) == 2 and _ID.match(parts[1]):
            return self._file(os.path.join(ASSETS, parts[1]))
        if parts[0] == "p" and method == "GET" and len(parts) >= 3:
            return self._project_file(parts[1], parts[2:], q)
        if parts[0] == "preview" and method == "GET":
            return self._file(self._inside(q.get("path", "")))
        if parts[0] != "api":
            raise ApiError("not found", 404)
        r = parts[1:]

        if method == "GET":
            if r == ["status"]:
                return self._json(app.status(q.get("fresh") == "1"))
            if r == ["projects"]:
                return self._json(app.list_projects())
            if len(r) == 2 and r[0] == "p":
                return self._json(app.project_view(r[1]))
            if r == ["job"]:
                j = app.jobs.current
                return self._json(j.view(int(q.get("since") or 0)) if j else None)
            raise ApiError("not found", 404)

        # POST
        if r == ["projects"]:
            b = self._jbody()
            p = Project.create(app.cfg["projects_dir"], (b.get("name") or "motion").strip())
            return self._json({"id": os.path.basename(p.path)})
        if len(r) == 3 and r[0] == "projects" and r[2] == "duplicate":
            src = app.pdir(r[1])
            p = Project.create(app.cfg["projects_dir"],
                               (cm_core.read_json(os.path.join(src, "project.json"), {})
                                .get("name") or r[1]) + " copy")
            for sub in ("keys", "masks"):
                if os.path.isdir(os.path.join(src, sub)):
                    shutil.copytree(os.path.join(src, sub), os.path.join(p.path, sub),
                                    dirs_exist_ok=True)
            data = cm_core.read_json(os.path.join(src, "project.json"), {})
            data["name"] = p.data["name"]
            data["created"] = p.data["created"]
            data.pop("last_render", None)
            cm_core.write_json(p.file, data)
            return self._json({"id": os.path.basename(p.path)})
        if len(r) == 3 and r[0] == "projects" and r[2] == "trash":
            if app.jobs.busy_on(r[1]):
                raise ApiError("a job is running on this project", 409)
            src = app.pdir(r[1])
            trash = os.path.join(app.cfg["projects_dir"], ".trash")
            os.makedirs(trash, exist_ok=True)
            dst = os.path.join(trash, r[1])
            if os.path.exists(dst):
                dst += time.strftime("-%Y%m%d%H%M%S")
            shutil.move(src, dst)
            log(f"project {r[1]} moved to {dst}")
            return self._json({"ok": True, "trash": dst})
        if r == ["animator"]:
            b = self._jbody()
            if not b.get("settings"):
                raise ApiError("settings.txt is needed")
            # parse first: a bad file makes no empty project
            cm_animator.make_plan(cm_animator.read_settings(b["settings"]),
                                  cm_animator.read_keyframes(b["keyframes"])
                                  if b.get("keyframes") else None, app.cfg, b.get("seg"))
            p = Project.create(app.cfg["projects_dir"], (b.get("name") or "animator").strip())
            cm_animator.import_files(p, b["settings"], b.get("keyframes"), app.cfg,
                                     b.get("seg"), {"settings": b.get("settings_name"),
                                                    "keyframes": b.get("keyframes_name")})
            return self._json({"id": os.path.basename(p.path)})
        if r == ["job", "cancel"]:
            return self._json({"ok": app.jobs.cancel()})
        if r == ["engine", "stop"]:
            if app.jobs.current is not None and app.jobs.current.state == "running":
                raise ApiError("a job is running", 409)
            out = {}
            for e in app.engines.by_name.values():
                out[e.name] = e.stop() if e.started_by_us() else {"ok": True, "stopped": False}
            app._status = (0, None)
            return self._json(out)
        if r == ["job"]:
            b = self._jbody()
            return self._json(app.job(b.get("project"), b).view())
        if len(r) >= 3 and r[0] == "p":
            return self._json(self._project_post(r[1], r[2:], q))
        raise ApiError("not found", 404)

    def _project_post(self, pid, r, q):
        app = self.app
        p = app.editable(pid)
        if r == ["output"]:
            b = self._jbody()
            o = p.output
            for k in DEFAULT_OUTPUT:
                if k in b and k != "seg":
                    o[k] = b[k]
            if o.get("close") in ("", 0):
                o["close"] = None
            for k, f in (("fps", int), ("crf", int), ("webm_crf", int), ("gif_width", int)):
                if o.get(k) not in (None, ""):
                    o[k] = f(o[k])
            if isinstance(o.get("formats"), str):
                o["formats"] = [x for x in o["formats"].split(",") if x]
            if not o.get("formats"):
                raise ApiError("choose at least one format")
            cm_timeline.output_size(p.work_size or (1088, 1920), o["res"])   # validates
            if b.get("seg"):
                p.respace(float(b["seg"]))
            elif b.get("total") and len(p.keys) > 1:
                p.respace(float(b["total"]) / (len(p.keys) - 1))
            p.save()
            return {"ok": True}
        if r == ["rename"]:
            b = self._jbody()
            p.data["name"] = (b.get("name") or "").strip() or p.data.get("name")
            p.save()
            return {"ok": True}
        if len(r) == 2 and r[0] == "key":
            b = self._jbody()
            k = p.key(r[1])
            if "t" in b:
                t = round(float(b["t"]), 4)
                if t < 0:
                    raise ApiError("a time is 0 or more")
                if any(abs(float(x["t"]) - t) < 1e-6 for x in p.keys if x is not k):
                    raise ApiError(f"a key is already at {t} s")
                k["t"] = t
            if "hold" in b:
                k["hold"] = max(0, int(b["hold"]))
            if "ease" in b:
                if b["ease"] not in cm_timeline.EASES:
                    raise ApiError(f"unknown ease {b['ease']}")
                k["ease"] = b["ease"]
            p.keys.sort(key=lambda x: x["t"])
            p.save()
            return {"ok": True}
        if r == ["order"]:
            ids = self._jbody().get("ids") or []
            if sorted(ids) != sorted(k["id"] for k in p.keys):
                raise ApiError("the order must name every key once")
            times = sorted(float(k["t"]) for k in p.keys)
            for t, kid in zip(times, ids):
                p.key(kid)["t"] = t
            p.keys.sort(key=lambda x: x["t"])
            p.save()
            log(f"keys reordered: {' '.join(ids)}")
            return {"ok": True}
        if len(r) == 2 and r[0] == "remove":
            p.remove_key(r[1])
            log(f"key {r[1]} removed")
            return {"ok": True}
        if r == ["upload"]:
            data = self._body()
            name = os.path.basename(q.get("name") or "upload.png")
            if not _ID.match(name):
                name = re.sub(r"[^A-Za-z0-9._-]", "_", name) or "upload.png"
            up = os.path.join(p.path, "uploads")
            os.makedirs(up, exist_ok=True)
            path = os.path.join(up, name)
            with open(path, "wb") as fh:
                fh.write(data)
            t = q.get("t")
            try:
                k = cm_keys.import_key(p, path, t=float(t) if t not in (None, "") else None,
                                       max_mpx=app.cfg["work_mpx"])
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
            return {"ok": True, "key": k["id"]}
        if r == ["plan"]:
            b = self._jbody()
            plan = p.data.get("plan")
            if not plan:
                raise ApiError("no plan")
            by_i = {k["i"]: k for k in plan["keys"]}
            for e in b.get("keys") or []:
                pk = by_i.get(int(e["i"]))
                if pk is None:
                    continue
                for f in ("prompt", "negative"):
                    if f in e:
                        pk[f] = str(e[f])
                if e.get("seed") not in (None, ""):
                    pk["seed"] = int(e["seed"])
                if e.get("denoise") not in (None, "") and pk["op"] == "chain":
                    pk["denoise"] = float(e["denoise"])
            if b.get("all_prompt") is not None and str(b["all_prompt"]).strip():
                for pk in plan["keys"]:
                    pk["prompt"] = str(b["all_prompt"]).strip()
            if b.get("all_denoise") not in (None, ""):
                for pk in plan["keys"]:
                    if pk["op"] == "chain":
                        pk["denoise"] = float(b["all_denoise"])
            p.save()
            return {"ok": True}
        if r == ["plan", "clear"]:
            p.data.pop("plan", None)
            p.save()
            return {"ok": True}
        raise ApiError("not found", 404)

    def _project_file(self, pid, rest, q):
        p = self.app.project(pid)
        if rest[0] == "thumb" and len(rest) == 2:
            k = p.key(rest[1])
            return self._file(_thumb(p, k, int(q.get("w") or 320)), "image/jpeg", cache=True)
        path = os.path.join(p.path, *rest)
        self._inside(path)
        if os.path.normcase(os.path.abspath(path)).startswith(
                os.path.normcase(os.path.abspath(p.path)) + os.sep):
            return self._file(path)
        raise ApiError("not found", 404)


def _thumb(p, k, w):
    from PIL import Image
    src = p.key_path(k)
    st = os.stat(src)
    d = os.path.join(p.path, "thumbs")
    os.makedirs(d, exist_ok=True)
    w = max(64, min(1024, w))
    path = os.path.join(d, f"{k['id']}-{st.st_mtime_ns}-{w}.jpg")
    if not os.path.isfile(path):
        for f in os.listdir(d):
            if f.startswith(k["id"] + "-"):
                try:
                    os.remove(os.path.join(d, f))
                except OSError:
                    pass
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((w, w * 4))
            im.save(path + ".tmp.jpg", quality=86)
        os.replace(path + ".tmp.jpg", path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cm_server.py", description="crispz-motion web page")
    ap.add_argument("--config")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    s = cfg.get("server") or {}
    host, port = a.host or s.get("host", "127.0.0.1"), a.port or int(s.get("port", 7870))
    app = App(cfg)
    Handler.app = app
    try:
        srv = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        print(f"[server] cannot listen on {host}:{port}: {e}. Is crispz-motion already "
              f"open? Else set server.port in config.json", flush=True)
        return 1
    srv.daemon_threads = True
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/"
    log(f"web page: {url}  (Ctrl+C to stop)", "server")
    if s.get("open_browser", True) and not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        log("stopping", "server")
        app.jobs.cancel()
        srv.server_close()
        app.engines.stop_ours()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""crispz-motion - one render: keys -> FILM -> retiming -> (ESRGAN) -> ffmpeg.

    plan     cm_timeline: which FILM frame goes in each output frame
    film     only the segments missing from the cache (film/<hash>/)
    timing   an output frame between two FILM frames is their blend (blend/<hash>/),
             so every frame sits at its exact time (see cm_timeline)
    size     output above the work size: ESRGAN on every frame used (Q7), cached in
             final/<WxH>-<model>/<hash>/; below it: Lanczos; equal: the FILM frame as is
    frames   frames/f00000.png ... hard links to those files (a copy when links fail)
    export   ffmpeg, make*.bat / make*.sh, project.json "last_render"

Every degradation is logged (cropped key, Lanczos instead of ESRGAN, engine stopped).
"""

import os
import time
import shutil

from PIL import Image

import cm_film
import cm_keys
import cm_guard
import cm_video
import cm_timeline
from cm_core import log, set_log_file, progress, check_cancel
from cm_project import cover


class RenderError(RuntimeError):
    pass


def _name(pos):
    """File name of a frame at a FILM position (1/POS_GRID steps): p00000256.png."""
    return f"p{int(round(pos * cm_timeline.POS_GRID)):08d}.png"


def _link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _progress(label, every=8, step=None, total=0, preview=None):
    """The worker's JSON events: to the log, and to the page's progress (step, done,
    total; preview(ev) names an image to show)."""
    state = {"n": 0}

    def on_event(ev):
        e = ev.get("event")
        if e == "ready":
            log(f"{label}: {ev.get('device')} {ev.get('gpu') or ''} "
                f"{ev.get('dtype') or ('fp16' if ev.get('fp16') else '')} "
                f"loaded in {ev.get('load_s')} s")
        elif e == "frame":
            state["n"] += 1
            if state["n"] % every == 0:
                log(f"{label}: {state['n']} frames")
            if step:
                progress(step, state["n"], total)
        elif e == "segment":
            if step and preview:
                progress(step, state["n"], total, preview(ev))
            log(f"{label}: segment {ev['seg'] + 1} done, {ev['frames']} frames, "
                f"{ev['per_frame_s']} s/frame, peak {ev.get('peak_gb')} GB")
        elif e == "done":
            log(f"{label}: done, {ev.get('frames')} frames in {ev.get('wall_s')} s")
    return on_event


def _keys_upscaled(project, cfg, engines, size):
    """Order a: every key upscaled by the engine (ESRGAN + a Turbo refine at denoise 0.2,
    op upscale), cropped to the output size, cached in keys_up/ by content."""
    if engines is None:
        raise RenderError("order keys_first needs the engine (crispz-moss)")
    w, h = size
    d = os.path.join(project.path, "keys_up")
    os.makedirs(d, exist_ok=True)
    ww, wh = project.work_size
    out = []
    for i, k in enumerate(project.keys):
        check_cancel()
        progress("keys", i, len(project.keys))
        src = project.key_path(k)
        p = os.path.join(d, f"{k['id']}_{cm_film.file_hash(src)[:12]}_{w}x{h}.png")
        if not os.path.isfile(p):
            spec = dict(cfg["turbo"], input=src, factor=round(max(w / ww, h / wh), 3),
                        denoise=0.2, model=os.path.basename(cfg.get("esrgan_model") or ""),
                        prompt=cm_keys.prompt_of(k), seed=cm_keys.seed_of(k) or 0)
            spec.update(cfg.get("chain") or {})
            log(f"keys first: {k['id']} upscaled x{spec['factor']} by the engine")
            local, _res, _caps = cm_keys._engine_call(project, engines, cfg, "upscale", spec)
            with Image.open(local) as im:
                cover(im.convert("RGB"), w, h).save(p)
        out.append(p)
        progress("keys", i + 1, len(project.keys), p)
    for f in os.listdir(d):         # older versions of the keys
        if os.path.join(d, f) not in out:
            os.remove(os.path.join(d, f))
    return out


def black_frames(paths, threshold=3.0):
    """Frames whose mean brightness is below threshold (0-255): a black frame."""
    bad = []
    for p in paths:
        with Image.open(p) as im:
            g = im.convert("L").resize((64, 64))
            if sum(g.getdata()) / 4096.0 < threshold:
                bad.append(p)
    return bad


def render(project, cfg, engines=None, overrides=None):
    o = project.output
    o.update({k: v for k, v in (overrides or {}).items() if v is not None})
    project.save()
    log_path = os.path.join(project.path, "render.log")
    set_log_file(log_path)
    t_all = time.time()
    timings = {}
    try:
        log(f"=== render {os.path.basename(project.path)}: {len(project.keys)} keys, "
            f"{o['res']}, {o['fps']} fps, loop {o['loop']}, {','.join(o['formats'])}")
        if len(project.keys) < 2:
            raise RenderError("at least 2 keys are needed")
        why = cm_film.check(cfg)
        if why:
            raise RenderError(why)
        for k in project.keys:
            if not os.path.isfile(project.key_path(k)):
                raise RenderError(f"key {k['id']}: missing file {k['file']}")

        # 1. plan
        try:
            pl = cm_timeline.plan(project.keys, int(o["fps"]), o["loop"], o.get("close"))
        except cm_timeline.PlanError as e:
            raise RenderError(str(e))
        work = project.work_size
        out_w, out_h = cm_timeline.output_size(work, o["res"])
        log(f"plan: {len(pl['segments'])} segments, {len(pl['frames'])} frames "
            f"({pl['duration_s']} s), work {work[0]}x{work[1]} -> output {out_w}x{out_h}")
        progress("plan", 1, 1)

        # guard: keys too far apart for FILM (said, never blocking)
        guard = cm_guard.check(project, cfg, o["loop"])
        for e in guard:
            msg = cm_guard.advice(e, os.path.basename(project.path))
            if msg:
                log(f"guard: {msg}")

        # order a (option): the engine upscales the keys, FILM runs at the output size
        key_src = [project.key_path(k) for k in project.keys]
        film_size = tuple(work)
        bigger = max(out_w, out_h) > max(work) * 1.1
        if o.get("order") == "keys_first" and bigger:
            t0 = time.time()
            key_src = _keys_upscaled(project, cfg, engines, (out_w, out_h))
            film_size = (out_w, out_h)
            timings["keys_upscale_s"] = round(time.time() - t0, 2)

        # 2. the GPU is shared: an engine we started goes before FILM
        if engines is not None and cfg.get("stop_engine_before_film", True):
            for e in engines.by_name.values():
                if e.started_by_us():
                    log(f"stopping {e.name} (started by crispz-motion) to free the GPU for FILM")
                    e.stop()

        # 3. FILM, missing segments only
        t0 = time.time()
        seg_dirs, jobs = [], []
        for s in pl["segments"]:
            a, b = key_src[s["a"]], key_src[s["b"]]
            d = os.path.join(project.path, "film", cm_film.segment_id(a, b, s["times"], cfg))
            seg_dirs.append(d)
            if not cm_film.segment_complete(d, s["times"]):
                jobs.append({"a": a, "b": b, "out": d, "times": s["times"]})
        log(f"FILM: {len(jobs)} of {len(seg_dirs)} segments to compute "
            f"({len(seg_dirs) - len(jobs)} cached)")
        os.makedirs(os.path.join(project.path, "film"), exist_ok=True)
        check_cancel()
        n_film = sum(2 ** j["times"] - 1 for j in jobs)
        progress("film", 0, n_film)

        def film_preview(ev):
            j = jobs[ev["seg"]]
            return os.path.join(j["out"], f"f{2 ** (j['times'] - 1):05d}.png")
        ok, err = cm_film.run_film(cfg, jobs, project.path,
                                   _progress("FILM", step="film", total=n_film,
                                             preview=film_preview), log_path)
        if not ok:
            raise RenderError(f"FILM failed: {err}")
        progress("film", n_film, n_film)
        timings["film_s"] = round(time.time() - t0, 2)

        # 4. exact times: a position between two FILM frames is their blend
        t0 = time.time()
        refs = sorted(set(pl["frames"]))
        src, keep, n_blend = {}, list(seg_dirs), 0
        for i, r in enumerate(refs):
            if i % 16 == 0:
                check_cancel()
                progress("timing", i, len(refs))
            s, pos = r
            if pos == int(pos):
                src[r] = os.path.join(seg_dirs[s], f"f{int(pos):05d}.png")
                continue
            bdir = os.path.join(project.path, "blend", os.path.basename(seg_dirs[s]))
            src[r] = os.path.join(bdir, _name(pos))
            keep.append(bdir)
            if not os.path.isfile(src[r]):
                lo = int(pos)
                os.makedirs(bdir, exist_ok=True)
                with Image.open(os.path.join(seg_dirs[s], f"f{lo:05d}.png")) as ia, \
                        Image.open(os.path.join(seg_dirs[s], f"f{lo + 1:05d}.png")) as ib:
                    Image.blend(ia.convert("RGB"), ib.convert("RGB"), pos - lo).save(
                        src[r] + ".tmp.png", compress_level=1)
                os.replace(src[r] + ".tmp.png", src[r])
                n_blend += 1
        progress("timing", len(refs), len(refs))
        timings["blend_s"] = round(time.time() - t0, 2)
        log(f"timing: {sum(1 for r in refs if r[1] != int(r[1]))} of {len(refs)} frames "
            f"between two FILM frames ({n_blend} blended now)")

        # 5. output size
        t0 = time.time()
        if (out_w, out_h) == film_size:
            final = src
            method = "FILM frames as is" if film_size == tuple(work) else \
                "keys upscaled by the engine, FILM at the output size (order a)"
        else:
            up = max(out_w, out_h) > max(work) * 1.1
            model, why = (cm_film.esrgan_model(cfg) if up and o.get("upscaler") == "esrgan"
                          else (None, None))
            if up and model is None:
                log(f"WARNING: {why or 'upscaler off'}: Lanczos instead of ESRGAN, the 4K "
                    f"frames will be soft")
            tag = f"{out_w}x{out_h}-" + (os.path.splitext(os.path.basename(model))[0]
                                          if model else "lanczos")
            root = os.path.join(project.path, "final", tag)
            final = {r: os.path.join(root, os.path.basename(seg_dirs[r[0]]), _name(r[1]))
                     for r in refs}
            keep += sorted({os.path.dirname(p) for p in final.values()})
            todo = [r for r in refs if not os.path.isfile(final[r])]
            method = f"ESRGAN {os.path.basename(model)}" if model else "Lanczos"
            log(f"size: {method} to {out_w}x{out_h}, {len(todo)} of {len(refs)} frames to do")
            check_cancel()
            progress("size", 0, len(todo))
            if model:
                ok, err = cm_film.run_esrgan(cfg, model, (out_w, out_h),
                                             [{"src": src[r], "dst": final[r]} for r in todo],
                                             project.path,
                                             _progress("ESRGAN", 32, step="size",
                                                       total=len(todo)), log_path)
                if not ok:
                    raise RenderError(f"ESRGAN failed: {err}")
            else:
                for i, r in enumerate(todo):
                    if i % 8 == 0:
                        check_cancel()
                        progress("size", i, len(todo))
                    os.makedirs(os.path.dirname(final[r]), exist_ok=True)
                    with Image.open(src[r]) as im:
                        cover(im.convert("RGB"), out_w, out_h).save(final[r] + ".tmp.png",
                                                                    compress_level=1)
                    os.replace(final[r] + ".tmp.png", final[r])
        n_size = 0 if (out_w, out_h) == film_size else len(todo)
        progress("size", n_size, n_size)
        timings["size_s"] = round(time.time() - t0, 2)

        # 6. checks, then frames/
        bad = black_frames([final[r] for r in refs])
        if bad:
            raise RenderError(f"{len(bad)} black frame(s), first: {bad[0]}")
        fdir = os.path.join(project.path, "frames")
        shutil.rmtree(fdir, ignore_errors=True)
        os.makedirs(fdir)
        for i, r in enumerate(pl["frames"]):
            _link(final[r], os.path.join(fdir, f"f{i:05d}.png"))

        # 7. export
        check_cancel()
        progress("export", 0, len(o["formats"]))
        t0 = time.time()
        files = cm_video.export(project.path, int(o["fps"]), o["formats"], o,
                                cfg.get("ffmpeg") or "ffmpeg")
        progress("export", len(o["formats"]), len(o["formats"]))
        timings["export_s"] = round(time.time() - t0, 2)
        timings["total_s"] = round(time.time() - t_all, 2)
        project.prune_cache(keep)
        summary = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "size": [out_w, out_h],
                   "guard": guard,
                   "frames": len(pl["frames"]), "duration_s": pl["duration_s"],
                   "segments": len(pl["segments"]), "film_segments_computed": len(jobs),
                   "method": method, "files": {k: os.path.basename(v) for k, v in files.items()},
                   "timings": timings}
        project.data["last_render"] = summary
        project.save()
        log(f"=== done in {timings['total_s']} s: {', '.join(files.values())}")
        return summary
    finally:
        set_log_file(None)

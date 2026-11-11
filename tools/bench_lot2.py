"""crispz-motion lot 2 - the three levers for subject motion, then the acceptance clip.

    .venv\\Scripts\\python tools\\bench_lot2.py levers       (about 10 minutes)
    .venv\\Scripts\\python tools\\bench_lot2.py accept [--denoise D] [--prompts FILE]

levers: one k0 (prompt of the historic example 1, Turbo), then from that same k0:
  A. img2img chains of 2 keys at denoise 0.5 / 0.625 / 0.75 / 0.875 (8 steps)
  B. a chain with an evolving prompt (head turned left, then looking up) at 0.625
  C. two edits by instruction (Moss-Image-Edit LoRA, 50 steps, CFG 4)
Each branch is its own project; the guard score of each pair and the time of each key
go to bench/lot2/levers.json, a contact sheet to bench/lot2/sheet_levers.jpg, and each
branch is rendered (work size, once, 1 s per segment) for the eye.

accept: the acceptance of lot 2, 11 keys in an img2img chain from the example 1 prompt,
10 s, rendered in 4K; total time in bench/lot2/accept.json.

crispz-moss is started by crispz-motion if it is not running, and stopped before FILM.
"""

import os
import sys
import json
import time
import shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from PIL import Image, ImageDraw

import cm_keys
import cm_guard
import cm_render
from cm_core import load_config, log
from cm_project import Project
from cm_engines import Engines

OUT = os.path.join(HERE, "bench", "lot2")
PROMPT = ("a pencil drawing of a man with a mustache and a beard, long hair, open jacket, "
          "tall grass in the wind, smoke coming out of his head and a lot of smoke coming "
          "out of his mouth, white paper background")
SEED = 2763806460


def timed(fn, *a, **kw):
    t0 = time.time()
    r = fn(*a, **kw)
    return r, round(time.time() - t0, 2)


def fresh(cfg, name):
    for d in os.listdir(cfg["projects_dir"]) if os.path.isdir(cfg["projects_dir"]) else []:
        if d.endswith("-" + name):
            shutil.rmtree(os.path.join(cfg["projects_dir"], d), ignore_errors=True)
    return Project.create(cfg["projects_dir"], name)


def branch_from(cfg, name, k0_path, k0_key):
    """A project whose first key is k0, recipe included (so chain finds its prompt)."""
    p = fresh(cfg, name)
    p.add_key(k0_path, "gen", dict(k0_key["recipe"]))
    return p


def sheet(rows, path, thumb=180):
    """rows: [(label, [image paths])] -> one jpg."""
    cols = max(len(r[1]) for r in rows)
    w, h = thumb, round(thumb * 1920 / 1088)
    img = Image.new("RGB", (cols * w + 170, len(rows) * h), "white")
    d = ImageDraw.Draw(img)
    for i, (label, paths) in enumerate(rows):
        d.text((6, i * h + 6), label, fill="black")
        for j, p in enumerate(paths):
            with Image.open(p) as im:
                img.paste(im.convert("RGB").resize((w, h)), (170 + j * w, i * h))
    img.save(path, quality=88)


def levers(cfg, engines):
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "branches": {}}
    base = fresh(cfg, "lot2-k0")
    k0, s = timed(cm_keys.gen_key, base, engines, cfg, PROMPT, "text, logo, watermark", SEED)
    res["k0_s"] = s
    k0_path = base.key_path(k0)
    branches = {f"A-d{int(d * 1000)}": ("chain", d) for d in (0.5, 0.625, 0.75, 0.875)}
    branches["B-prompt"] = ("prompts", 0.625)
    branches["C-edit"] = ("edit", None)
    rows = []
    for name, (kind, d) in branches.items():
        p = branch_from(cfg, f"lot2-{name}", k0_path, k0)
        times = []
        if kind == "chain":
            for _ in range(2):
                times.append(timed(cm_keys.chain_key, p, engines, cfg, denoise=d)[1])
        elif kind == "prompts":
            for extra in (", his head turned to the left, seen in three-quarter view",
                          ", looking up at the sky, mouth wide open"):
                times.append(timed(cm_keys.chain_key, p, engines, cfg,
                                   prompt=PROMPT + extra, denoise=d)[1])
        else:
            for ins in ("turn his head to the left", "make him look up at the sky with "
                                                     "his mouth wide open"):
                times.append(timed(cm_keys.edit_key, p, engines, cfg, ins)[1])
        g = cm_guard.check(p, cfg, loop="once")
        res["branches"][name] = {"project": os.path.basename(p.path), "key_s": times,
                                 "guard": g}
        log(f"{name}: keys {times} s, guard {[e['score'] for e in g]}")
        rows.append((f"{name}\n{' '.join(str(e['score']) for e in g)}",
                     [p.key_path(k) for k in p.keys]))
    sheet(rows, os.path.join(OUT, "sheet_levers.jpg"))
    engines.stop_ours()
    for name, b in res["branches"].items():
        p = Project.find(cfg["projects_dir"], b["project"])
        s = cm_render.render(p, cfg, engines, {"res": "1080p", "loop": "pingpong",
                                               "formats": ["mp4"]})
        b["render_s"] = s["timings"]["total_s"]
        shutil.copyfile(os.path.join(p.path, "video.mp4"), os.path.join(OUT, f"{name}.mp4"))
    with open(os.path.join(OUT, "levers.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    return res


def accept(cfg, engines, denoise, prompts=None):
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "denoise": denoise,
           "prompts": prompts}
    t_all = time.time()
    p = fresh(cfg, "lot2-accept")
    _, res["k0_s"] = timed(cm_keys.gen_key, p, engines, cfg, PROMPT,
                           "text, logo, watermark", SEED)
    _, res["chain_s"] = timed(cm_keys.chain, p, engines, cfg, 10, prompts, denoise)
    res["keys_s"] = round(time.time() - t_all, 2)
    res["guard"] = cm_guard.check(p, cfg, loop="pingpong")
    s = cm_render.render(p, cfg, engines, {"res": "4k", "fps": 24, "loop": "once",
                                           "formats": ["mp4"]})
    res["render"] = s
    res["total_s"] = round(time.time() - t_all, 2)
    shutil.copyfile(os.path.join(p.path, "video.mp4"), os.path.join(OUT, "accept_4k.mp4"))
    sheet([("accept", [p.key_path(k) for k in p.keys[:6]]),
           ("", [p.key_path(k) for k in p.keys[6:]])], os.path.join(OUT, "sheet_accept.jpg"))
    with open(os.path.join(OUT, "accept.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    log(f"accept: keys {res['keys_s']} s, render {s['timings']['total_s']} s, "
        f"total {res['total_s']} s")
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    engines = Engines(cfg)
    what = sys.argv[1] if len(sys.argv) > 1 else "levers"
    try:
        if what == "levers":
            levers(cfg, engines)
        else:
            d = float(sys.argv[sys.argv.index("--denoise") + 1]) if "--denoise" in sys.argv \
                else 0.625
            prompts = None
            if "--prompts" in sys.argv:
                with open(sys.argv[sys.argv.index("--prompts") + 1], encoding="utf-8") as fh:
                    prompts = [l.strip() for l in fh if l.strip()]
            accept(cfg, engines, d, prompts)
    finally:
        engines.stop_ours()


if __name__ == "__main__":
    main()

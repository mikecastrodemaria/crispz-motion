"""crispz-motion - command line (the web page is cm_server.py: start with no argument).

    cm_cli.py status
    cm_cli.py new NAME [--keys a.png b.png ...] [--seg 1.0]
    cm_cli.py add PROJECT a.png [b.png ...] [--at SECONDS] [--hold N] [--ease E]
    cm_cli.py gen PROJECT --prompt "..." [--negative ...] [--seed N] [--size WxH]
    cm_cli.py info PROJECT
    cm_cli.py render PROJECT [--res 1080p|4k|2mpx|1mpx|work|WxH] [--fps 24]
                             [--loop once|pingpong|closed] [--formats mp4,webm,gif]
                             [--seg S] [--codec x264|x265] [--crf 17]
    cm_cli.py engine start|stop|status [--engine moss]

PROJECT is a folder, a folder name under projects/, or a project name (newest match).
`quick` makes a project from images and renders it in one go:

    cm_cli.py quick a.png b.png --res 4k --loop pingpong --seg 2
"""

import os
import sys
import json
import argparse

from cm_core import load_config, log
from cm_project import Project
from cm_engines import Engines
import cm_keys
import cm_film
import cm_guard
import cm_render
import cm_timeline


def _size(s):
    w, h = s.lower().split("x", 1)
    return int(w), int(h)


def _render_opts(p):
    p.add_argument("--res")
    p.add_argument("--fps", type=int)
    p.add_argument("--loop", choices=cm_timeline.LOOPS)
    p.add_argument("--formats", help="comma list: mp4,webm,gif")
    p.add_argument("--seg", type=float, help="re-space the keys every SEG seconds")
    p.add_argument("--close", type=float, help="closed loop: last -> first, in seconds")
    p.add_argument("--codec", choices=("x264", "x265"))
    p.add_argument("--crf", type=int)
    p.add_argument("--upscaler", choices=("esrgan", "lanczos"))
    p.add_argument("--order", choices=("film_first", "keys_first"),
                   help="film_first: FILM then ESRGAN on every frame (default); "
                        "keys_first: the engine upscales the keys, FILM at output size")


def _overrides(a):
    return {"res": a.res, "fps": a.fps, "loop": a.loop, "codec": a.codec, "crf": a.crf,
            "close": a.close, "upscaler": a.upscaler, "order": a.order,
            "formats": [f.strip() for f in a.formats.split(",") if f.strip()]
            if a.formats else None}


def cmd_status(cfg, a):
    print(f"projects: {cfg['projects_dir']}")
    why = cm_film.check(cfg)
    print(f"FILM (route A): {'ready' if not why else why}")
    model, why = cm_film.esrgan_model(cfg)
    print(f"ESRGAN: {model or why}")
    for e in Engines(cfg).by_name.values():
        st = e.status()
        print(f"engine {st['name']}: {'running ' + str(st['tool']) + ' ' + str(st['version']) if st['running'] else 'not running'}"
              f" at {st['url']} (dir {st['dir'] or '-'}"
              f"{', started by crispz-motion' if st['started_by_us'] else ''})")
    return 0


def cmd_info(cfg, a):
    p = Project.find(cfg["projects_dir"], a.project)
    print(f"{p.path}\nwork size: {p.work_size}")
    for k in p.keys:
        r = k.get("recipe") or {}
        what = cm_keys.prompt_of(k) or r.get("file", "")
        if r.get("from"):
            what = f"<- {r['from']} " + (f"d={r['spec'].get('denoise')} "
                                        if (r.get("spec") or {}).get("denoise") is not None
                                        else "") + what
        if r.get("client") is False:
            what += "  [not for client work]"
        print(f"  {k['id']}  t={k['t']:<6} hold={k['hold']} ease={k['ease']:<11} "
              f"{k['source']:<7} {what[:90]}")
    print("output: " + json.dumps(p.output))
    if p.data.get("last_render"):
        print("last render: " + json.dumps(p.data["last_render"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cm_cli.py", description="crispz-motion")
    ap.add_argument("--config", help="another config.json")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("status")
    p = sub.add_parser("new")
    p.add_argument("name")
    p.add_argument("--keys", nargs="*", default=[])
    p.add_argument("--seg", type=float, default=1.0)
    p = sub.add_parser("add")
    p.add_argument("project")
    p.add_argument("images", nargs="+")
    p.add_argument("--at", type=float)
    p.add_argument("--hold", type=int, default=0)
    p.add_argument("--ease", default="linear", choices=list(cm_timeline.EASES))
    p = sub.add_parser("gen")
    p.add_argument("project")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="")
    p.add_argument("--seed", type=int)
    p.add_argument("--size", type=_size)
    p.add_argument("--engine")
    p.add_argument("--at", type=float)
    p = sub.add_parser("chain", help="img2img keys in a row, each from the one before")
    p.add_argument("project")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--denoise", type=float, default=0.45)
    p.add_argument("--prompt", help="the same prompt for every new key")
    p.add_argument("--prompts", help="a text file, one prompt per new key (the last repeats)")
    p.add_argument("--from", dest="from_kid", help="start from this key (default: the last)")
    p.add_argument("--engine")
    p = sub.add_parser("edit", help="the next key by instruction (Moss-Image-Edit LoRA)")
    p.add_argument("project")
    p.add_argument("instruction")
    p.add_argument("--from", dest="from_kid")
    p.add_argument("--seed", type=int)
    p.add_argument("--at", type=float)
    p.add_argument("--engine")
    p = sub.add_parser("inpaint", help="the next key with a masked area redrawn")
    p.add_argument("project")
    p.add_argument("--mask", required=True, help="PNG, white = redraw")
    p.add_argument("--prompt", required=True)
    p.add_argument("--denoise", type=float, default=1.0)
    p.add_argument("--from", dest="from_kid")
    p.add_argument("--seed", type=int)
    p.add_argument("--engine")
    p = sub.add_parser("mid", help="a middle key between two keys (FILM + img2img)")
    p.add_argument("project")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--denoise", type=float, default=0.2)
    p.add_argument("--prompt")
    p.add_argument("--engine")
    p = sub.add_parser("regen", help="make a key again from its recipe")
    p.add_argument("project")
    p.add_argument("key")
    p.add_argument("--seed", type=int, help="a new seed (default: the same)")
    p.add_argument("--cascade", action="store_true",
                   help="then redo the keys built on this one, in order")
    p.add_argument("--engine")
    p = sub.add_parser("remove", help="delete a key")
    p.add_argument("project")
    p.add_argument("key")
    p = sub.add_parser("guard", help="distance between consecutive keys")
    p.add_argument("project")
    p = sub.add_parser("info")
    p.add_argument("project")
    p = sub.add_parser("render")
    p.add_argument("project")
    _render_opts(p)
    p = sub.add_parser("quick")
    p.add_argument("images", nargs="+")
    p.add_argument("--name", default="quick")
    _render_opts(p)
    p = sub.add_parser("engine")
    p.add_argument("action", choices=("start", "stop", "status"))
    p.add_argument("--engine")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    try:
        if a.cmd is None:
            cmd_status(cfg, a)
            print("\ncommands: new, add, gen, chain, edit, inpaint, mid, regen, remove, "
                  "guard, info, render, quick, engine, status (cm_cli.py -h, or "
                  "<command> -h)")
            return 0
        if a.cmd == "status":
            return cmd_status(cfg, a)
        if a.cmd == "info":
            return cmd_info(cfg, a)
        if a.cmd == "new":
            p = Project.create(cfg["projects_dir"], a.name)
            p.output["seg"] = a.seg
            p.save()
            for img in a.keys:
                cm_keys.import_key(p, img, max_mpx=cfg["work_mpx"])
            print(p.path)
            return 0
        if a.cmd == "add":
            p = Project.find(cfg["projects_dir"], a.project)
            for i, img in enumerate(a.images):
                t = None if a.at is None else a.at + i * float(p.output["seg"])
                cm_keys.import_key(p, img, t=t, hold=a.hold, ease=a.ease,
                                   max_mpx=cfg["work_mpx"])
            return 0
        if a.cmd == "gen":
            p = Project.find(cfg["projects_dir"], a.project)
            k = cm_keys.gen_key(p, Engines(cfg), cfg, a.prompt, a.negative, a.seed, a.size,
                                a.engine, t=a.at)
            print(p.key_path(k))
            return 0
        if a.cmd == "chain":
            p = Project.find(cfg["projects_dir"], a.project)
            prompts = None
            if a.prompts:
                with open(a.prompts, encoding="utf-8") as fh:
                    prompts = [line.strip() for line in fh if line.strip()]
            elif a.prompt:
                prompts = [a.prompt]
            if a.from_kid:
                cm_keys.chain_key(p, Engines(cfg), cfg, a.from_kid,
                                  prompts[0] if prompts else None, denoise=a.denoise,
                                  engine=a.engine)
                return 0
            cm_keys.chain(p, Engines(cfg), cfg, a.count, prompts, a.denoise, a.engine)
            return 0
        if a.cmd == "edit":
            p = Project.find(cfg["projects_dir"], a.project)
            cm_keys.edit_key(p, Engines(cfg), cfg, a.instruction, a.from_kid, a.seed, a.at,
                             a.engine)
            return 0
        if a.cmd == "inpaint":
            p = Project.find(cfg["projects_dir"], a.project)
            cm_keys.inpaint_key(p, Engines(cfg), cfg, a.mask, a.prompt, a.from_kid,
                                a.denoise, a.seed, engine=a.engine)
            return 0
        if a.cmd == "mid":
            p = Project.find(cfg["projects_dir"], a.project)
            cm_keys.mid_key(p, Engines(cfg), cfg, a.a, a.b, a.denoise, a.prompt,
                            engine=a.engine)
            return 0
        if a.cmd == "regen":
            p = Project.find(cfg["projects_dir"], a.project)
            done = cm_keys.regen(p, Engines(cfg), cfg, a.key, a.seed, a.cascade, a.engine)
            print("regenerated: " + ", ".join(k["id"] for k in done))
            return 0
        if a.cmd == "remove":
            p = Project.find(cfg["projects_dir"], a.project)
            p.remove_key(a.key)
            return 0
        if a.cmd == "guard":
            p = Project.find(cfg["projects_dir"], a.project)
            for e in cm_guard.check(p, cfg):
                msg = cm_guard.advice(e, os.path.basename(p.path))
                print(f"  {e['a']} -> {e['b']}  {e['score']:6.2f}  {e['level']}"
                      + (f"   {msg}" if e["level"] == "risk" else ""))
            return 0
        if a.cmd in ("render", "quick"):
            engines = Engines(cfg)
            if a.cmd == "quick":
                p = Project.create(cfg["projects_dir"], a.name)
                for img in a.images:
                    cm_keys.import_key(p, img, max_mpx=cfg["work_mpx"])
            else:
                p = Project.find(cfg["projects_dir"], a.project)
            if a.seg:
                p.respace(a.seg)
            s = cm_render.render(p, cfg, engines, _overrides(a))
            print(json.dumps(s, indent=2))
            return 0
        if a.cmd == "engine":
            engines = Engines(cfg)
            e = engines.get(a.engine)
            if a.action == "status":
                print(json.dumps(e.status(), indent=2))
                return 0
            r = engines.ensure(a.engine)[1] if a.action == "start" else e.stop()
            print(json.dumps({k: v for k, v in r.items() if k != "caps"}, indent=2))
            return 0 if r.get("ok") else 1
    except (cm_render.RenderError, FileNotFoundError, KeyError, RuntimeError, ValueError,
            cm_timeline.PlanError) as e:
        log(f"ERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""crispz-motion - where keys come from.

    import   a file                                             (no engine)
    gen      a prompt                                           op gen
    chain    the previous key (or any key) passed again          op upscale, factor 1,
             with a denoise: the "sparse loopback"                   denoise, seed + 1
    edit     a key + an instruction ("turn his head left")       op edit (LoRA Moss-Image-Edit)
    inpaint  a key + a mask + a local prompt                     op inpaint
    mid      FILM's midpoint of two keys, cleaned by img2img     FILM worker + op upscale
             at a low denoise: the assisted middle key

A key keeps its RECIPE: {"op", "spec" (what was sent, without the paths), "from" (the id
of the key it starts from), "engine", "tool"}. regen() plays it again from the CURRENT
image of its "from" key, so a chain can be redone from any link (cascade=True also redoes
the keys built on it, in order).

Licence: the Moss-Image-Edit LoRA has no licence for its weights (crispz-moss lot 4b).
Every edit says so in the log and the recipe carries "client": false.
"""

import os
import shutil

import cm_film
from cm_core import log

EDIT_LICENCE = ("Moss-Image-Edit LoRA: no licence is stated for its weights; a key made "
                "by edit is not for client work")


def _engine_call(project, engines, cfg, op, spec, engine=None):
    """One protocol call; the result copied into the project's raw/. Returns
    (local path, the answer, caps). Raises with the engine's own reason."""
    eng, st = engines.ensure(engine)
    if not st.get("ok"):
        raise RuntimeError(st["error"])
    caps = st.get("caps") or eng.caps() or {}
    sup = caps.get("supports") or {}
    need = {"edit": "edit", "inpaint": "inpaint"}.get(op)
    if op == "upscale" and float(spec.get("factor", 2)) <= 1.0:
        need = "img2img"
    if need and need in sup and not sup.get(need):
        raise RuntimeError(f"{eng.name} ({caps.get('tool')}) cannot do '{op}' here "
                           f"(supports.{need} is false)"
                           + ("; for edit, the Moss-Image-Edit LoRA must be on disk"
                              if op == "edit" else ""))
    if not sup.get("guidance"):
        log(f"{eng.name}: supports.guidance is absent: the CFG of the spec may be ignored")
    raw = os.path.join(project.path, "raw")
    os.makedirs(raw, exist_ok=True)
    res = eng.call(op, dict(spec, out_dir=raw))
    if not res.get("ok"):
        raise RuntimeError(f"{eng.name} {op} failed: {res.get('error')}")
    for w in res.get("warnings") or []:
        log(f"{eng.name}: warning: {w}")
    src = res["images"][0]
    local = os.path.join(raw, os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(local):
        shutil.copyfile(src, local)
    return local, res, dict(caps, engine=eng.name)


def _recipe(op, spec, res, caps, src_key=None, **extra):
    s = {k: v for k, v in spec.items() if k not in ("input", "mask", "out_dir")}
    if res.get("seed_used") is not None:
        s["seed"] = res["seed_used"]
    r = {"op": op, "spec": s, "engine": caps.get("engine"), "tool": caps.get("tool")}
    if src_key is not None:
        r["from"] = src_key["id"]
    r.update(extra)
    return r


def prompt_of(key):
    r = key.get("recipe") or {}
    return (r.get("spec") or {}).get("prompt") or r.get("prompt") or ""


def seed_of(key):
    r = key.get("recipe") or {}
    s = (r.get("spec") or {}).get("seed", r.get("seed"))
    return int(s) if s is not None else None


# -- sources ---------------------------------------------------------------------------
def import_key(project, path, t=None, hold=0, ease="linear", max_mpx=2.1):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return project.add_key(path, "import", {"op": "import", "file": os.path.basename(path)},
                           t=t, hold=hold, ease=ease, max_mpx=max_mpx)


def gen_key(project, engines, cfg, prompt, negative="", seed=None, size=None, engine=None,
            t=None, extra=None):
    """A key from a prompt (op gen), at the project's work size once it has one."""
    w, h = size or project.work_size or tuple(cfg["gen_size"])
    spec = dict(cfg["turbo"], prompt=prompt, negative=negative or "", width=int(w),
                height=int(h))
    if seed is not None:
        spec["seed"] = int(seed)
    spec.update(extra or {})
    log(f"gen {w}x{h}, {spec.get('steps')} steps, CFG {spec.get('guidance')}, "
        f"loras {spec.get('loras')}")
    local, res, caps = _engine_call(project, engines, cfg, "gen", spec, engine)
    return project.add_key(local, "gen", _recipe("gen", spec, res, caps), t=t)


def effective_denoise(denoise, steps):
    """What the engine really plays. crispz-moss runs `n - int(n - n * denoise)` of its
    n steps (at least one): with the Turbo LoRA at 4 steps, denoise 0.35 and 0.50 both
    run 2 steps and give the SAME image (found on lot 2). Returns (steps run, n)."""
    n = max(1, int(steps))
    start = min(int(max(n - n * float(denoise), 0)), n - 1)
    return n - start, n


def _chain_spec(cfg, project, src, prompt, negative, denoise, seed):
    prompt = prompt if prompt is not None else prompt_of(src)
    if not prompt:
        raise ValueError(f"key {src['id']} has no prompt (an imported key): give one")
    if seed is None:
        seed = (seed_of(src) or 0) + 1
    neg = negative if negative is not None else \
        ((src.get("recipe") or {}).get("spec") or {}).get("negative", "")
    spec = dict(cfg["turbo"], input=project.key_path(src), factor=1.0,
                denoise=float(denoise), prompt=prompt, negative=neg or "", seed=int(seed))
    # img2img runs a fraction of the steps: 4 steps leave only 0.25 / 0.5 / 0.75 / 1,
    # so the chain uses more (cfg["chain"]["steps"], 8 by default: also the Turbo
    # author's count above 1 Mpx)
    spec.update(cfg.get("chain") or {})
    run, n = effective_denoise(denoise, spec.get("steps", 4))
    if abs(run / n - float(denoise)) > 0.01:
        log(f"denoise {denoise} runs {run} of {n} steps: effective {run / n:.3f}")
    return spec


def chain_key(project, engines, cfg, from_kid=None, prompt=None, negative=None,
              denoise=0.45, seed=None, t=None, engine=None):
    """The next key of the sparse loopback: a key passed again through img2img
    (op upscale, factor 1) at `denoise`, seed + 1, its prompt unless another is given."""
    src = project.key(from_kid) if from_kid else project.keys[-1]
    spec = _chain_spec(cfg, project, src, prompt, negative, denoise, seed)
    log(f"chain from {src['id']}: denoise {spec['denoise']}, seed {spec['seed']}, "
        f"prompt '{spec['prompt'][:60]}'")
    local, res, caps = _engine_call(project, engines, cfg, "upscale", spec, engine)
    return project.add_key(local, "chain", _recipe("chain", spec, res, caps, src), t=t)


def chain(project, engines, cfg, count, prompts=None, denoise=0.45, engine=None,
          on_key=None):
    """`count` keys in a row, each from the one before. `prompts`: one per new key
    (the last one repeats), or None to keep each key's own prompt."""
    made = []
    for i in range(int(count)):
        p = None
        if prompts:
            p = prompts[min(i, len(prompts) - 1)]
        k = chain_key(project, engines, cfg, prompt=p, denoise=denoise, engine=engine)
        made.append(k)
        if on_key:
            on_key(k)
    return made


def edit_key(project, engines, cfg, instruction, from_kid=None, seed=None, t=None,
             engine=None):
    """The next key by instruction (op edit, LoRA Moss-Image-Edit, the author's 50 steps
    at CFG 4 by default: cfg["edit"]). Keeps the character and the set best."""
    src = project.key(from_kid) if from_kid else project.keys[-1]
    e = cfg.get("edit") or {}
    spec = {"prompt": instruction, "input": project.key_path(src),
            "steps": int(e.get("steps", 50)), "guidance": float(e.get("guidance", 4.0)),
            "seed": int(seed if seed is not None else (seed_of(src) or 0) + 1)}
    if e.get("loras"):
        spec["loras"] = list(e["loras"])
    log(f"edit from {src['id']}: '{instruction}', {spec['steps']} steps, CFG "
        f"{spec['guidance']}. {EDIT_LICENCE}")
    if "style" not in instruction.lower():
        # lot 2: "turn his head to the left" turned a pencil drawing into a colour image;
        # "..., keep the black and white pencil drawing style" kept it
        log("edit: the instruction does not name the style; the edit may change it (a "
            "pencil drawing came back in colour). Add e.g. 'keep the <style> style'")
    local, res, caps = _engine_call(project, engines, cfg, "edit", spec, engine)
    return project.add_key(local, "edit", _recipe("edit", spec, res, caps, src,
                                                   client=False), t=t)


def inpaint_key(project, engines, cfg, mask_path, prompt, from_kid=None, denoise=1.0,
                seed=None, t=None, engine=None):
    """The next key with one area redrawn (white = redraw): smoke, mouth, sky."""
    from PIL import Image
    from cm_project import cover
    src = project.key(from_kid) if from_kid else project.keys[-1]
    os.makedirs(os.path.join(project.path, "masks"), exist_ok=True)
    rel = f"masks/{os.path.splitext(os.path.basename(mask_path))[0]}.png"
    with Image.open(mask_path) as m:
        cover(m.convert("L"), *project.work_size).save(os.path.join(project.path, rel))
    spec = dict(cfg["turbo"], input=project.key_path(src),
                mask=os.path.join(project.path, rel), prompt=prompt,
                denoise=float(denoise),
                seed=int(seed if seed is not None else (seed_of(src) or 0) + 1))
    local, res, caps = _engine_call(project, engines, cfg, "inpaint", spec, engine)
    return project.add_key(local, "inpaint", _recipe("inpaint", spec, res, caps, src,
                                                      mask=rel), t=t)


def mid_key(project, engines, cfg, a_kid, b_kid, denoise=0.2, prompt=None, seed=None,
            engine=None):
    """The assisted middle key: FILM's midpoint of a and b, cleaned by img2img at a low
    denoise (0.15 to 0.25: it repairs the ghosts, keeps the in-between), placed halfway
    in time. denoise 0 keeps FILM's midpoint as is."""
    ka, kb = project.key(a_kid), project.key(b_kid)
    why = cm_film.check(cfg)
    if why:
        raise RuntimeError(why)
    tmp = os.path.join(project.path, "mid_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    ok, err = cm_film.run_film(cfg, [{"a": project.key_path(ka), "b": project.key_path(kb),
                                      "out": os.path.join(tmp, "seg"), "times": 1}], tmp)
    if not ok:
        raise RuntimeError(f"FILM midpoint failed: {err}")
    mid = os.path.join(tmp, "seg", "f00001.png")
    t = (float(ka["t"]) + float(kb["t"])) / 2
    recipe = {"op": "mid", "a": a_kid, "b": b_kid, "denoise": float(denoise)}
    if denoise > 0:
        spec = _chain_spec(cfg, project, ka, prompt, None, denoise, seed)
        spec["input"] = mid
        log(f"mid {a_kid}|{b_kid}: FILM midpoint cleaned at denoise {denoise}")
        local, res, caps = _engine_call(project, engines, cfg, "upscale", spec, engine)
        recipe.update(_recipe("mid", spec, res, caps))
    else:
        local = os.path.join(project.path, "raw", f"mid_{a_kid}_{b_kid}.png")
        os.makedirs(os.path.dirname(local), exist_ok=True)
        shutil.copyfile(mid, local)
        log(f"mid {a_kid}|{b_kid}: FILM midpoint as is")
    key = project.add_key(local, "mid", recipe, t=t)
    shutil.rmtree(tmp, ignore_errors=True)
    return key


# -- regeneration ------------------------------------------------------------------------
def regen(project, engines, cfg, kid, seed=None, cascade=False, engine=None):
    """Plays a key's recipe again, from the current image of its source key. A new seed
    gives a new image; the same seed redoes the same one (useful when the source changed).
    cascade=True then redoes, in time order, every key built on this one."""
    k = project.key(kid)
    r = k.get("recipe") or {}
    op = r.get("op")
    if op in (None, "import"):
        raise ValueError(f"key {kid} is imported: nothing to regenerate")
    if op == "mid":
        b, a = r["b"], r["a"]
        project.remove_key(kid)
        nk = mid_key(project, engines, cfg, a, b, r.get("denoise", 0.2),
                     (r.get("spec") or {}).get("prompt"), seed, engine)
        log(f"key {kid} regenerated as {nk['id']}")
        return [nk]
    spec = dict(r.get("spec") or {})
    if seed is not None:
        spec["seed"] = int(seed)
    if r.get("from"):
        spec["input"] = project.key_path(project.key(r["from"]))
    if op == "inpaint":
        spec["mask"] = os.path.join(project.path, r["mask"])
    engine_op = {"chain": "upscale", "gen": "gen", "edit": "edit", "inpaint": "inpaint"}[op]
    if op == "edit":
        log(EDIT_LICENCE)
    log(f"regen {kid} ({op}), seed {spec.get('seed')}")
    local, res, caps = _engine_call(project, engines, cfg, engine_op, spec, engine)
    extra = {x: r[x] for x in ("mask", "client") if x in r}
    project.replace_key_image(kid, local, _recipe(op, spec, res, caps,
                                                  project.key(r["from"]) if r.get("from")
                                                  else None, **extra))
    done = [project.key(kid)]
    if cascade:
        for nxt in sorted((x for x in project.keys
                           if (x.get("recipe") or {}).get("from") == kid),
                          key=lambda x: x["t"]):
            done += regen(project, engines, cfg, nxt["id"], cascade=True, engine=engine)
    return done

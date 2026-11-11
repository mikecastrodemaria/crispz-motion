"""crispz-motion - where keys come from.

Lot 1: imported files and generated keys (op gen, Turbo LoRA, CFG 1 by default). The
img2img chain, edit, inpaint and the assisted middle key come with lot 2; they will
write the same kind of recipe.

A recipe is the spec sent to the engine, without out_dir: enough to make the key again.
"""

import os
import shutil

from cm_core import log


def import_key(project, path, t=None, hold=0, ease="linear", max_mpx=2.1):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return project.add_key(path, "import", {"file": os.path.basename(path)}, t=t,
                           hold=hold, ease=ease, max_mpx=max_mpx)


def gen_key(project, engines, cfg, prompt, negative="", seed=None, size=None, engine=None,
            t=None, extra=None):
    """Asks the engine for one image (op gen) and adds it as the next key."""
    eng, st = engines.ensure(engine)
    if not st.get("ok"):
        raise RuntimeError(st["error"])
    caps = st.get("caps") or eng.caps() or {}
    if not (caps.get("supports") or {}).get("guidance"):
        log(f"{eng.name}: this engine does not announce supports.guidance: the CFG of the "
            f"spec may be ignored (crispz-moss needs commit e250d99 or later)")
    if size is None:
        size = project.work_size or tuple(cfg["gen_size"])
    w, h = size
    spec = dict(cfg["turbo"])
    spec.update({"prompt": prompt, "negative": negative or "", "width": int(w),
                 "height": int(h)})
    if seed is not None:
        spec["seed"] = int(seed)
    spec.update(extra or {})
    raw = os.path.join(project.path, "raw")
    os.makedirs(raw, exist_ok=True)
    log(f"{eng.name}: gen {w}x{h}, {spec.get('steps')} steps, CFG {spec.get('guidance')}, "
        f"loras {spec.get('loras')}")
    res = eng.call("gen", dict(spec, out_dir=raw))
    if not res.get("ok"):
        raise RuntimeError(f"{eng.name} gen failed: {res.get('error')}")
    for w_ in res.get("warnings") or []:
        log(f"{eng.name}: warning: {w_}")
    src = res["images"][0]
    recipe = dict(spec, op="gen", engine=eng.name, tool=caps.get("tool"),
                  seed_used=res.get("seed_used"))
    if res.get("seed_used") is not None:
        recipe["seed"] = res["seed_used"]
    # keep a copy of what the engine returned inside the project (its file may move)
    local = os.path.join(raw, os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(local):
        shutil.copyfile(src, local)
    return project.add_key(local, "gen", recipe, t=t)

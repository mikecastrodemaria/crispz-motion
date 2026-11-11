"""crispz-motion - import of the A1111 Animator settings (settings.txt, keyframes.csv).

The Animator extension diffused EVERY frame: frame i = frame i-1 through img2img at a low
denoise (0.3), seed + 1, the prompt of the moment (`A :w AND B :1-w`). crispz-motion
keeps the idea on SPACED keys (one every `seg` seconds, 1 s by default) and FILM fills
the gaps, so the import turns those files into a PLAN of keys:

    key 0       op gen (or the Animator's initial image when it is on disk)
    key i > 0   op chain from key i-1: its prompt, its seed, a chain denoise

What is read, and how:

  settings.txt   width/height -> the ratio of the keys (made at about work_mpx);
                 total_time, fps -> keys at 0, seg, 2 seg ... total_time; output fps;
                 seed (+1 per frame), denoising_strength, tmpl_pos / tmpl_neg;
                 vid_mp4 / vid_webm / vid_gif -> output formats
  keyframes.csv  one row per frame: pos_prompt / neg_prompt (`A :w AND B :w'`; the
                 part with the larger weight wins), denoise, seed_start, cfg_scale,
                 x_shift / y_shift / zoom / rotation

What changes, said in the plan's notes (never silently):

  - the denoise: 0.3 per frame at 25 fps is not 0.3 per key one second apart. The
    settings' own denoise maps to animator.denoise (0.625: lot 2 measured 0.625 to
    0.75 keeping the character, 0.875 losing it); a row's denoise moves it in
    proportion, within animator.denoise_min / denoise_max;
  - SD 1.5 LoRA tags (<lora:...>) and negative embeddings mean nothing to MOSS: they
    are removed from the prompts;
  - the CFG of the rows: the keys use the Turbo LoRA at CFG 1, where the negative
    prompt has no effect either;
  - camera moves (x/y shift, zoom, rotation) are v2: ignored, with a note when a row
    has one; noise, restore_faces, smoothing and final_fps are not used (FILM makes the
    in-betweens at the output fps).

Pure functions plus build(), which makes the planned keys through cm_keys.
"""

import csv
import io
import json
import os
import re

from cm_core import log, check_cancel, progress

# common SD 1.5 negative embeddings: plain words to MOSS
NEG_EMBEDDINGS = {
    "bad-artist-anime", "bad-artist", "fastnegativeembedding", "badream", "baddream",
    "verybadimagenegative_v1.2-6400", "verybadimagenegative_v1.3", "bad_pictures",
    "easynegative", "badhandv4", "ng_deepnegative_v1_75t", "unrealisticdream",
    "bad-hands-5", "bad_prompt_version2", "bad_prompt", "negative_hand-neg",
    "fastnegativev2", "bad-image-v2-39000",
}
_TAG = re.compile(r"<\s*(lora|lyco|hypernet)\s*:[^>]*>", re.I)
_PART = re.compile(r"^(.*?)[\s,]*:\s*(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)\s*$", re.S)


class AnimatorError(ValueError):
    pass


# -- reading ---------------------------------------------------------------------------
def read_settings(text):
    """settings.txt is JSON (both historic examples). Returns a dict."""
    try:
        s = json.loads(text)
    except ValueError as e:
        raise AnimatorError(f"settings.txt is not JSON: {e}")
    if not isinstance(s, dict):
        raise AnimatorError("settings.txt: a JSON object was expected")
    return s


def read_keyframes(text):
    """keyframes.csv: one dict per frame, numbers as floats where they parse."""
    rows = []
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    for r in reader:
        row = {}
        for k, v in r.items():
            if k is None:
                continue
            k = k.strip()
            v = (v or "").strip()
            try:
                row[k] = float(v) if v != "" and k not in ("pos_prompt", "neg_prompt",
                                                          "pos1", "neg1", "pos2", "neg2",
                                                          "seed_str") else v
            except ValueError:
                row[k] = v
        rows.append(row)
    if not rows:
        raise AnimatorError("keyframes.csv has no rows")
    return rows


def weighted_parts(s):
    """`A :0.7 AND B :0.3` -> [("A", 0.7), ("B", 0.3)]. A part without a weight
    counts 1."""
    out = []
    for part in re.split(r"\s+AND\s+", str(s or "")):
        m = _PART.match(part.strip())
        if m:
            out.append((m.group(1).strip(" ,"), float(m.group(2))))
        elif part.strip():
            out.append((part.strip(" ,"), 1.0))
    return out


def dominant(s):
    """The text of the part with the largest weight ('' when none has text)."""
    parts = [p for p in weighted_parts(s) if p[0]]
    if not parts:
        return ""
    return max(parts, key=lambda p: p[1])[0]


def clean_prompt(text, negative=False):
    """(clean text, [removed tags/words]). LoRA tags go; in a negative, the SD 1.5
    embeddings go too."""
    removed = [m.group(0) for m in _TAG.finditer(text or "")]
    t = _TAG.sub(" ", text or "")
    if negative:
        words = []
        for chunk in re.split(r"(\s+|,)", t):
            if chunk.strip().lower() in NEG_EMBEDDINGS:
                removed.append(chunk.strip())
                continue
            words.append(chunk)
        t = "".join(words)
    t = re.sub(r"\s*,\s*(,\s*)+", ", ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,")
    t = re.sub(r"\s+,", ",", t)
    return t, removed


def key_size(width, height, mpx=2.1, max_side=2048, mult=16):
    """The keys' size at the Animator's ratio, about `mpx` megapixels, multiples of 16,
    no side above max_side (crispz-moss renders at most 2048 per side)."""
    r = float(width) / float(height)
    h = (mpx * 1e6 / r) ** 0.5
    w = h * r
    s = min(1.0, max_side / max(w, h))
    w, h = w * s, h * s
    return max(mult, int(w) // mult * mult), max(mult, int(h) // mult * mult)


# -- the plan --------------------------------------------------------------------------
def make_plan(settings, rows=None, cfg=None, seg=None, names=None):
    """The plan of spaced keys, the output settings and the notes. `rows` may be None
    (settings only: one prompt, the settings' denoise, seed + 1 per frame)."""
    a = dict({"denoise": 0.625, "denoise_min": 0.25, "denoise_max": 0.75, "seg": 1.0},
             **((cfg or {}).get("animator") or {}))
    seg = float(seg or a["seg"])
    if seg <= 0:
        raise AnimatorError("seg must be above 0")
    notes = []
    fps = float(settings.get("fps") or 24)
    total = float(settings.get("total_time") or 0)
    n_frames = len(rows) if rows else int(round(total * fps)) + 1
    if total <= 0:
        total = (n_frames - 1) / fps
    base_d = float(settings.get("denoising_strength") or 0.3)
    seed0 = int(float(settings.get("seed") or 0))
    tmpl_pos = settings.get("tmpl_pos") or ""
    tmpl_neg = settings.get("tmpl_neg") or ""
    w, h = int(settings.get("width") or 512), int(settings.get("height") or 512)
    size = key_size(w, h, float((cfg or {}).get("work_mpx", 2.1)))
    notes.append(f"Animator {w}x{h}, {total:g} s at {fps:g} fps ({n_frames} frames, one "
                 f"diffusion each): here {size[0]}x{size[1]} keys every {seg:g} s, FILM "
                 f"between them")

    n_keys = int(round(total / seg)) + 1
    times = [round(min(i * seg, total), 4) for i in range(n_keys)]
    if len(times) > 1 and times[-1] - times[-2] < seg * 0.25:
        times.pop(-2)               # a last gap far shorter than seg: merged
    removed_all, camera, cfgs = set(), set(), set()
    keys = []
    for i, t in enumerate(times):
        f = min(int(round(t * fps)), n_frames - 1)
        row = rows[f] if rows else {}
        pos = dominant(row.get("pos_prompt")) if row.get("pos_prompt") else tmpl_pos
        neg = dominant(row.get("neg_prompt")) if row.get("neg_prompt") else tmpl_neg
        pos, rp = clean_prompt(pos)
        neg, rn = clean_prompt(neg, negative=True)
        removed_all.update(rp + rn)
        seed = row.get("seed_start")
        seed = int(seed) if isinstance(seed, float) else seed0 + f
        hd = float(row["denoise"]) if isinstance(row.get("denoise"), float) else base_d
        d = a["denoise"] * hd / base_d if base_d > 0 else a["denoise"]
        d = round(min(a["denoise_max"], max(a["denoise_min"], d)), 3)
        for c in ("x_shift", "y_shift", "rotation"):
            if isinstance(row.get(c), float) and abs(row[c]) > 1e-6:
                camera.add(c)
        if isinstance(row.get("zoom"), float) and abs(row["zoom"] - 1.0) > 1e-6:
            camera.add("zoom")
        if isinstance(row.get("cfg_scale"), float):
            cfgs.add(row["cfg_scale"])
        keys.append({"i": i, "t": t, "frame": f, "op": "gen" if i == 0 else "chain",
                     "prompt": pos, "negative": neg, "seed": seed,
                     "denoise": None if i == 0 else d, "hist_denoise": hd, "kid": None})

    init = settings.get("initial_img")
    if isinstance(init, str) and init and os.path.isfile(init):
        keys[0]["op"] = "import"
        keys[0]["file"] = init
        notes.append(f"key 0 is the Animator's initial image ({os.path.basename(init)})")
    elif init:
        notes.append("the Animator's initial image is not on this disk: key 0 is generated")

    notes.append(f"denoise: the Animator's {base_d:g} per frame becomes "
                 f"{a['denoise']:g} per key (between {a['denoise_min']:g} and "
                 f"{a['denoise_max']:g}; lot 2: 0.625 to 0.75 keeps the character)")
    if removed_all:
        notes.append("removed from the prompts (SD 1.5 only): "
                     + ", ".join(sorted(removed_all)))
    if cfgs:
        notes.append(f"CFG {', '.join(f'{c:g}' for c in sorted(cfgs))} of the Animator "
                     f"not used: Turbo LoRA at CFG 1, where the negative prompt has no "
                     f"effect")
    if camera:
        notes.append(f"camera moves ({', '.join(sorted(camera))}) are not in v1: ignored")
    if settings.get("add_noise"):
        notes.append("add_noise is not used")
    if settings.get("restore_faces"):
        notes.append("restore_faces is not used")
    if settings.get("film_interpolation") or settings.get("smoothing"):
        notes.append(f"the Animator's own FILM smoothing (final {settings.get('final_fps')} "
                     f"fps) is replaced by crispz-motion's FILM at the output fps")
    if len({k["prompt"] for k in keys}) > 1:
        notes.append(f"{len({k['prompt'] for k in keys})} different prompts along the keys")
    if not any(k["prompt"] for k in keys):
        notes.append("no prompt in these files: give one to each planned key before "
                     "making them")

    formats = [f for f, on in (("mp4", settings.get("vid_mp4", True)),
                               ("webm", settings.get("vid_webm")),
                               ("gif", settings.get("vid_gif"))) if on] or ["mp4"]
    out_fps = int(round(fps))
    output = {"fps": out_fps, "loop": "once", "seg": seg, "formats": formats}
    return {"source": "animator", "files": names or {}, "size": list(size),
            "keys": keys, "notes": notes}, output


def import_files(project, settings_text, keyframes_text=None, cfg=None, seg=None,
                 names=None):
    """Writes the plan and the output settings into the project. Returns the plan."""
    settings = read_settings(settings_text)
    rows = read_keyframes(keyframes_text) if keyframes_text else None
    plan, output = make_plan(settings, rows, cfg, seg, names)
    project.data["plan"] = plan
    project.output.update(output)
    project.save()
    for n in plan["notes"]:
        log(f"animator: {n}")
    log(f"animator: {len(plan['keys'])} keys planned")
    return plan


# -- making the planned keys ------------------------------------------------------------
def pending(project):
    plan = project.data.get("plan") or {}
    have = {k["id"] for k in project.keys}
    return [k for k in plan.get("keys", []) if not k.get("kid") or k["kid"] not in have]


def build(project, engines, cfg, engine=None):
    """Makes the planned keys still missing, in order (a key whose image was deleted is
    made again). Resumable: each made key is saved in the plan at once."""
    import cm_keys
    plan = project.data.get("plan")
    if not plan:
        raise AnimatorError("this project has no plan")
    keys = plan["keys"]
    have = {k["id"] for k in project.keys}
    todo = pending(project)
    log(f"plan: {len(todo)} of {len(keys)} keys to make")
    done = len(keys) - len(todo)
    for pk in keys:
        if pk.get("kid") in have:
            continue
        check_cancel()
        progress("keys", done, len(keys))
        if not pk.get("prompt") and pk["op"] != "import":
            raise AnimatorError(f"planned key {pk['i']} has no prompt")
        if pk["op"] == "import":
            k = cm_keys.import_key(project, pk["file"], t=pk["t"],
                                   max_mpx=cfg.get("work_mpx", 2.1))
        elif pk["op"] == "gen":
            k = cm_keys.gen_key(project, engines, cfg, pk["prompt"], pk.get("negative", ""),
                                pk.get("seed"), tuple(plan["size"]), engine, t=pk["t"])
        else:
            prev = keys[pk["i"] - 1].get("kid")
            if prev not in {x["id"] for x in project.keys}:
                raise AnimatorError(f"planned key {pk['i']}: the key before it is missing")
            k = cm_keys.chain_key(project, engines, cfg, prev, pk["prompt"],
                                  pk.get("negative", ""), pk["denoise"], pk.get("seed"),
                                  t=pk["t"], engine=engine)
        pk["kid"] = k["id"]
        project.save()
        done += 1
        progress("keys", done, len(keys), project.key_path(k))
    return [project.key(pk["kid"]) for pk in keys]

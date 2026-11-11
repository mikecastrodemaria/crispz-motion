"""crispz-motion - distance guard: warns BEFORE a render when two consecutive keys are
too far apart for FILM, which then fills the gap with ghosts (two half-transparent
copies) instead of motion.

The score: mean absolute difference of the two keys as 64 px wide grey thumbnails,
0-255. Pillow only (the app venv). Calibrated on lot 2 (bench/lot2/REPORT.md):

    photo pair, head turn + hair, ghosts seen          47
    crispz-moss img2img chain, smoke and grass only     3 to 6

    ok    below cfg guard.warn (12)
    warn  up to cfg guard.risk (25): watch the segment
    risk  above: insert a middle key (cm_cli.py mid PROJECT ka kb), which FILM and an
          img2img pass at a low denoise build between the two

The guard never blocks: it says, and the render goes on.
"""

from PIL import Image, ImageChops, ImageStat

SIZE = 64


def _thumb(path):
    with Image.open(path) as im:
        g = im.convert("L")
        w, h = g.size
        return g.resize((SIZE, max(1, round(SIZE * h / w))), Image.BOX)


def distance(a_path, b_path):
    a, b = _thumb(a_path), _thumb(b_path)
    if a.size != b.size:
        b = b.resize(a.size, Image.BOX)
    return round(ImageStat.Stat(ImageChops.difference(a, b)).mean[0], 2)


def level(score, cfg):
    g = cfg.get("guard") or {}
    if score >= float(g.get("risk", 25)):
        return "risk"
    if score >= float(g.get("warn", 12)):
        return "warn"
    return "ok"


def check(project, cfg, loop=None):
    """One entry per segment of the project (the closing one too for a closed loop)."""
    keys = project.keys
    pairs = [(keys[i], keys[i + 1]) for i in range(len(keys) - 1)]
    if (loop or project.output.get("loop")) == "closed" and len(keys) > 2:
        pairs.append((keys[-1], keys[0]))
    out = []
    for ka, kb in pairs:
        s = distance(project.key_path(ka), project.key_path(kb))
        out.append({"a": ka["id"], "b": kb["id"], "score": s, "level": level(s, cfg)})
    return out


def advice(entry, project_ref):
    if entry["level"] == "risk":
        return (f"{entry['a']} -> {entry['b']}: distance {entry['score']}, FILM will "
                f"likely ghost; insert a middle key: cm_cli.py mid {project_ref} "
                f"{entry['a']} {entry['b']}")
    if entry["level"] == "warn":
        return (f"{entry['a']} -> {entry['b']}: distance {entry['score']}, large motion: "
                f"check this segment")
    return None

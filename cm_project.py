"""crispz-motion - a project on disk.

    projects/<YYYY-MM-DD>-<name>/
      project.json     keys, recipes, output settings, last render
      keys/            k000.png, k001.png ...   (the work size, all keys alike)
      raw/             what the engine returned, before any crop
      film/<hash>/     FILM frames of one segment (f00000.png ... both keys included)
      final/<tag>/     frames at the output size (ESRGAN or Lanczos), same layout
      frames/          the output frames in order (hard links into film/ or final/)
      video.mp4 ...    exports, and makemp4.bat / makewebm.bat / makegif.bat
      render.log

A key keeps its RECIPE (source, prompt, seed, spec...) next to its image, so it can be
made again (lot 2). FILM caches by content: a segment folder is named after the hash of
its two key images and its settings, so changing a key redoes only its two segments.
"""

import os
import re
import time
import shutil

from PIL import Image

from cm_core import write_json, read_json, log

VERSION = 1

DEFAULT_OUTPUT = {
    "res": "1080p", "fps": 24, "loop": "pingpong", "seg": 1.0, "close": None,
    "formats": ["mp4"], "codec": "x264", "crf": 17, "preset": "medium",
    "webm_crf": 32, "gif_width": 720, "upscaler": "esrgan",
}


def slug(name):
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", str(name).strip()).strip("-").lower()
    return s or "motion"


def cover(img, w, h):
    """Lanczos resize so that img covers w x h, then centre crop."""
    if img.size == (w, h):
        return img
    s = max(w / img.width, h / img.height)
    r = img.resize((max(w, round(img.width * s)), max(h, round(img.height * s))),
                   Image.LANCZOS)
    x, y = (r.width - w) // 2, (r.height - h) // 2
    return r.crop((x, y, x + w, y + h))


def work_size_for(w, h, max_mpx):
    """The first key's size, scaled down when above max_mpx, multiples of 8."""
    s = min(1.0, (max_mpx * 1e6 / (w * h)) ** 0.5)
    return max(64, int(w * s) // 8 * 8), max(64, int(h * s) // 8 * 8)


class Project:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.file = os.path.join(self.path, "project.json")
        data = read_json(self.file)
        if data is None:
            raise FileNotFoundError(f"no project.json in {self.path}")
        self.data = data
        self.data.setdefault("keys", [])
        self.data["output"] = dict(DEFAULT_OUTPUT, **(self.data.get("output") or {}))

    # -- creation / lookup ------------------------------------------------------
    @classmethod
    def create(cls, projects_dir, name):
        base = f"{time.strftime('%Y-%m-%d')}-{slug(name)}"
        path, i = os.path.join(projects_dir, base), 2
        while os.path.exists(path):
            path, i = os.path.join(projects_dir, f"{base}-{i}"), i + 1
        os.makedirs(os.path.join(path, "keys"))
        write_json(os.path.join(path, "project.json"),
                   {"version": VERSION, "name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "work_size": None, "keys": [], "output": dict(DEFAULT_OUTPUT)})
        log(f"project created: {path}")
        return cls(path)

    @staticmethod
    def find(projects_dir, ref):
        """A project from a path, a folder name, or a name (the newest match wins)."""
        if os.path.isfile(os.path.join(ref, "project.json")):
            return Project(ref)
        p = os.path.join(projects_dir, ref)
        if os.path.isfile(os.path.join(p, "project.json")):
            return Project(p)
        if os.path.isdir(projects_dir):
            hits = sorted(d for d in os.listdir(projects_dir)
                          if (d == ref or d.endswith("-" + slug(ref)) or
                              re.sub(r"-\d+$", "", d).endswith("-" + slug(ref)))
                          and os.path.isfile(os.path.join(projects_dir, d, "project.json")))
            if hits:
                return Project(os.path.join(projects_dir, hits[-1]))
        raise FileNotFoundError(f"no project '{ref}' (in {projects_dir})")

    def save(self):
        write_json(self.file, self.data)

    # -- keys ---------------------------------------------------------------------
    @property
    def keys(self):
        return self.data["keys"]

    @property
    def output(self):
        return self.data["output"]

    @property
    def work_size(self):
        ws = self.data.get("work_size")
        return tuple(ws) if ws else None

    def key_path(self, key):
        return os.path.join(self.path, key["file"])

    def _next_id(self):
        n = max([int(k["id"][1:]) for k in self.keys] + [-1]) + 1
        return f"k{n:03d}"

    def add_key(self, image_path, source, recipe=None, t=None, hold=0, ease="linear",
                max_mpx=2.1):
        """Adds a key at the end. The first key sets the work size (its own, capped at
        max_mpx); a later key of another size is cropped to the centre and resized, and
        the log says so (never a silent change)."""
        with Image.open(image_path) as im:
            img = im.convert("RGB")
        notes = []
        if self.work_size is None:
            ws = work_size_for(img.width, img.height, max_mpx)
            self.data["work_size"] = list(ws)
            if ws != img.size:
                notes.append(f"first key {img.width}x{img.height} -> work size "
                             f"{ws[0]}x{ws[1]}")
        ws = self.work_size
        if img.size != ws:
            r_in, r_ws = img.width / img.height, ws[0] / ws[1]
            if abs(r_in / r_ws - 1) > 0.01:
                notes.append(f"{os.path.basename(image_path)} {img.width}x{img.height} is "
                             f"cropped at the centre to the ratio of the first key")
            img = cover(img, *ws)
        kid = self._next_id()
        rel = f"keys/{kid}.png"
        os.makedirs(os.path.join(self.path, "keys"), exist_ok=True)
        img.save(os.path.join(self.path, rel))
        if t is None:
            t = 0.0 if not self.keys else float(self.keys[-1]["t"]) + float(self.output["seg"])
        key = {"id": kid, "file": rel, "source": source, "t": round(float(t), 4),
               "hold": int(hold), "ease": ease, "recipe": recipe or {}}
        self.keys.append(key)
        self.keys.sort(key=lambda k: k["t"])
        self.save()
        for n in notes:
            log(n)
        log(f"key {kid} ({source}) at {key['t']} s")
        return key

    def remove_key(self, kid):
        k = next((k for k in self.keys if k["id"] == kid), None)
        if k is None:
            raise KeyError(f"no key {kid}")
        self.keys.remove(k)
        try:
            os.remove(self.key_path(k))
        except OSError:
            pass
        if not self.keys:
            self.data["work_size"] = None
        self.save()

    def respace(self, seg):
        """Puts the keys every `seg` seconds, in their current order."""
        for i, k in enumerate(self.keys):
            k["t"] = round(i * float(seg), 4)
        self.output["seg"] = float(seg)
        self.save()

    def prune_cache(self, keep_dirs):
        """Deletes film/ and final/ folders no plan uses any more."""
        keep = {os.path.normcase(os.path.abspath(d)) for d in keep_dirs}
        for top in ("film", "blend", "final"):
            root = os.path.join(self.path, top)
            if not os.path.isdir(root):
                continue
            for dirpath, dirnames, _files in os.walk(root, topdown=False):
                if dirpath == root:
                    continue
                if os.path.normcase(dirpath) in keep:
                    continue
                if any(os.path.normcase(k).startswith(os.path.normcase(dirpath) + os.sep)
                       for k in keep):
                    continue
                shutil.rmtree(dirpath, ignore_errors=True)

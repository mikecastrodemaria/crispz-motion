"""cm_project + cm_keys on CPU: work size, crop of a key of another ratio, recipes, and a
generated key through a fake engine (no GPU, no family app)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from PIL import Image

import cm_keys
from cm_project import Project, work_size_for


def img(path, w, h, color=(200, 100, 50)):
    Image.new("RGB", (w, h), color).save(path)
    return str(path)


def test_work_size_is_capped_and_a_multiple_of_8():
    assert work_size_for(1088, 1920, 2.1) == (1088, 1920)
    w, h = work_size_for(2160, 3840, 2.1)
    assert w % 8 == 0 and h % 8 == 0 and w * h <= 2.1e6
    assert abs(w / h - 2160 / 3840) < 0.01


def test_keys_share_the_first_size_and_get_times(tmp_path):
    p = Project.create(str(tmp_path), "Mon Projet!")
    assert os.path.basename(p.path).endswith("-mon-projet")
    k0 = cm_keys.import_key(p, img(tmp_path / "a.png", 2160, 3840))
    k1 = cm_keys.import_key(p, img(tmp_path / "b.png", 1000, 1000))     # other ratio
    assert p.work_size == tuple(Image.open(p.key_path(k0)).size)
    assert Image.open(p.key_path(k1)).size == p.work_size               # cropped, resized
    assert [k["t"] for k in p.keys] == [0.0, 1.0]
    assert k0["recipe"] == {"file": "a.png"}
    p.respace(2.5)
    assert [k["t"] for k in Project(p.path).keys] == [0.0, 2.5]         # saved


def test_find_by_name(tmp_path):
    p = Project.create(str(tmp_path), "loop")
    assert Project.find(str(tmp_path), "loop").path == p.path
    with pytest.raises(FileNotFoundError):
        Project.find(str(tmp_path), "nothing")


class FakeEngine:
    name = "moss"

    def __init__(self, out):
        self.out, self.specs = out, []

    def caps(self):
        return {"ok": True, "tool": "crispz-moss", "supports": {"guidance": True}}

    def call(self, op, spec):
        self.specs.append((op, spec))
        path = img(os.path.join(spec["out_dir"], "render_001.png"), spec["width"],
                   spec["height"])
        return {"ok": True, "images": [path], "seed_used": 1234, "warnings": []}


class FakeEngines:
    def __init__(self, eng):
        self.eng = eng

    def ensure(self, name=None):
        return self.eng, {"ok": True, "started": False, "caps": self.eng.caps()}


def test_gen_key_sends_the_turbo_spec_and_keeps_the_recipe(tmp_path):
    p = Project.create(str(tmp_path), "gen")
    eng = FakeEngine(tmp_path)
    cfg = {"turbo": {"steps": 4, "guidance": 1.0, "loras": ["moss-turbo-4-steps:1"]},
           "gen_size": [1088, 1920]}
    k = cm_keys.gen_key(p, FakeEngines(eng), cfg, "a pencil drawing", seed=7)
    op, spec = eng.specs[0]
    assert op == "gen"
    assert spec["steps"] == 4 and spec["guidance"] == 1.0
    assert spec["loras"] == ["moss-turbo-4-steps:1"]
    assert (spec["width"], spec["height"]) == (1088, 1920) and spec["seed"] == 7
    assert k["source"] == "gen" and k["recipe"]["prompt"] == "a pencil drawing"
    assert k["recipe"]["seed"] == 1234 and "out_dir" not in k["recipe"]
    # the second key is asked at the project's work size
    cm_keys.gen_key(p, FakeEngines(eng), cfg, "next")
    assert (eng.specs[1][1]["width"], eng.specs[1][1]["height"]) == p.work_size

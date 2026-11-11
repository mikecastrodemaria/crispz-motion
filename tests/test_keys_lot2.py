"""Lot 2 on CPU: chain, edit, regen (with cascade), effective denoise, distance guard,
through a fake engine that writes a recognisable image per call."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from PIL import Image, ImageChops, ImageStat

import cm_keys
import cm_guard
from cm_project import Project

CFG = {"turbo": {"steps": 4, "guidance": 1.0, "loras": ["moss-turbo-4-steps:1"]},
       "chain": {"steps": 8}, "edit": {"steps": 50, "guidance": 4.0},
       "gen_size": [64, 96], "guard": {"warn": 12, "risk": 25}}


class FakeEngine:
    """gen: a grey image of the seed's level; upscale / edit: the input brightened by
    10 (a visible change per link of the chain)."""
    name = "moss"

    def __init__(self, edit=True):
        self.calls, self.edit = [], edit

    def caps(self):
        return {"ok": True, "tool": "crispz-moss",
                "supports": {"guidance": True, "img2img": True, "inpaint": True,
                             "edit": self.edit}}

    def call(self, op, spec):
        self.calls.append((op, dict(spec)))
        n = len(self.calls)
        path = os.path.join(spec["out_dir"], f"r{n:03d}.png")
        if op == "gen":
            Image.new("RGB", (spec["width"], spec["height"]),
                      (spec.get("seed", 0) % 200,) * 3).save(path)
        else:
            with Image.open(spec["input"]) as im:
                im.point(lambda v: min(255, v + 10)).save(path)
        return {"ok": True, "images": [path], "seed_used": spec.get("seed", 7)}


class FakeEngines:
    def __init__(self, eng):
        self.eng = eng

    def ensure(self, name=None):
        return self.eng, {"ok": True, "caps": self.eng.caps()}


def mean(path):
    with Image.open(path) as im:
        return ImageStat.Stat(im.convert("L")).mean[0]


def make(tmp_path, edit=True):
    p = Project.create(str(tmp_path), "lot2")
    eng = FakeEngine(edit)
    cm_keys.gen_key(p, FakeEngines(eng), CFG, "a pencil drawing", seed=50)
    return p, eng


def test_effective_denoise_is_quantised_by_the_steps():
    assert cm_keys.effective_denoise(0.35, 4) == (2, 4)
    assert cm_keys.effective_denoise(0.50, 4) == (2, 4)      # the same image (lot 2)
    assert cm_keys.effective_denoise(0.35, 8) == (3, 8)
    assert cm_keys.effective_denoise(0.0, 8) == (1, 8)       # at least one step
    assert cm_keys.effective_denoise(1.0, 8) == (8, 8)


def test_chain_keeps_prompt_adds_one_to_the_seed_and_records_its_source(tmp_path):
    p, eng = make(tmp_path)
    made = cm_keys.chain(p, FakeEngines(eng), CFG, 3, denoise=0.45)
    assert [k["id"] for k in p.keys] == ["k000", "k001", "k002", "k003"]
    ops = [c[0] for c in eng.calls]
    assert ops == ["gen", "upscale", "upscale", "upscale"]
    spec = eng.calls[1][1]
    assert spec["factor"] == 1.0 and spec["denoise"] == 0.45 and spec["steps"] == 8
    assert spec["seed"] == 51 and eng.calls[3][1]["seed"] == 53
    assert spec["input"] == p.key_path(p.keys[0])
    assert made[2]["recipe"]["from"] == "k002"
    assert "input" not in made[2]["recipe"]["spec"]
    assert [k["t"] for k in p.keys] == [0.0, 1.0, 2.0, 3.0]


def test_evolving_prompts_one_per_key_the_last_repeats(tmp_path):
    p, eng = make(tmp_path)
    cm_keys.chain(p, FakeEngines(eng), CFG, 3, prompts=["looks left", "looks up"])
    assert [c[1]["prompt"] for c in eng.calls[1:]] == ["looks left", "looks up", "looks up"]


def test_an_imported_key_needs_a_prompt_to_chain(tmp_path):
    p = Project.create(str(tmp_path), "imp")
    Image.new("RGB", (64, 96), (90, 90, 90)).save(tmp_path / "a.png")
    cm_keys.import_key(p, str(tmp_path / "a.png"))
    with pytest.raises(ValueError):
        cm_keys.chain_key(p, FakeEngines(FakeEngine()), CFG)


def test_edit_uses_the_authors_settings_and_is_marked_not_for_clients(tmp_path):
    p, eng = make(tmp_path)
    k = cm_keys.edit_key(p, FakeEngines(eng), CFG, "turn his head to the left")
    op, spec = eng.calls[-1]
    assert op == "edit" and spec["steps"] == 50 and spec["guidance"] == 4.0
    assert spec["prompt"] == "turn his head to the left" and "loras" not in spec
    assert k["recipe"]["client"] is False


def test_edit_is_refused_by_name_when_the_engine_cannot(tmp_path):
    p, eng = make(tmp_path, edit=False)
    with pytest.raises(RuntimeError, match="Moss-Image-Edit"):
        cm_keys.edit_key(p, FakeEngines(eng), CFG, "smile")


def test_regen_cascade_redoes_the_chain_from_the_new_image(tmp_path):
    p, eng = make(tmp_path)
    cm_keys.chain(p, FakeEngines(eng), CFG, 2)
    before = [mean(p.key_path(k)) for k in p.keys]
    # a new seed for the first key: a brighter gen, then the chain follows
    done = cm_keys.regen(p, FakeEngines(eng), CFG, "k000", seed=120, cascade=True)
    assert [k["id"] for k in done] == ["k000", "k001", "k002"]
    after = [mean(p.key_path(k)) for k in p.keys]
    assert after[0] == pytest.approx(120, abs=1) and before[0] == pytest.approx(50, abs=1)
    assert after[1] == pytest.approx(after[0] + 10, abs=1)
    assert after[2] == pytest.approx(after[0] + 20, abs=1)
    assert [k["t"] for k in p.keys] == [0.0, 1.0, 2.0]          # places kept


def test_regen_without_cascade_touches_one_key(tmp_path):
    p, eng = make(tmp_path)
    cm_keys.chain(p, FakeEngines(eng), CFG, 2)
    k2 = mean(p.key_path(p.keys[2]))
    cm_keys.regen(p, FakeEngines(eng), CFG, "k001")
    assert mean(p.key_path(p.keys[2])) == k2
    with pytest.raises(ValueError):
        Image.new("RGB", (64, 96)).save(tmp_path / "i.png")
        cm_keys.import_key(p, str(tmp_path / "i.png"), t=9)
        cm_keys.regen(p, FakeEngines(eng), CFG, p.keys[-1]["id"])


def test_a_key_time_cannot_be_taken_twice(tmp_path):
    p, _ = make(tmp_path)
    Image.new("RGB", (64, 96)).save(tmp_path / "i.png")
    with pytest.raises(ValueError):
        cm_keys.import_key(p, str(tmp_path / "i.png"), t=0.0)


def test_guard_scores_and_levels(tmp_path):
    a, b, c = (str(tmp_path / f"{n}.png") for n in "abc")
    Image.new("RGB", (128, 128), (100, 100, 100)).save(a)
    Image.new("RGB", (128, 128), (105, 105, 105)).save(b)
    Image.new("RGB", (128, 128), (160, 160, 160)).save(c)
    assert cm_guard.distance(a, b) == 5.0
    assert cm_guard.level(cm_guard.distance(a, b), CFG) == "ok"
    assert cm_guard.level(15, CFG) == "warn"
    assert cm_guard.level(cm_guard.distance(a, c), CFG) == "risk"
    p = Project.create(str(tmp_path), "g")
    for f in (a, b, c):
        cm_keys.import_key(p, f)
    res = cm_guard.check(p, CFG, loop="closed")
    assert [(e["a"], e["b"], e["level"]) for e in res] == \
        [("k000", "k001", "ok"), ("k001", "k002", "risk"), ("k002", "k000", "risk")]
    assert "mid" in cm_guard.advice(res[1], "g")

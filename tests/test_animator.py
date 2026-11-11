"""Lot 3 on CPU: the Animator import (settings.txt + keyframes.csv -> a plan of spaced
keys) and building that plan through a fake engine."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import cm_animator as A
from cm_project import Project
from test_keys_lot2 import CFG, FakeEngine, FakeEngines

TMPL = ("a drawing of a 1man with a beard and smoke  <lora:add_detail:1> "
        "<lora:detailed_eye:1>")
NEG = "bad-artist-anime FastNegativeEmbedding BadDream bad_pictures, text, logo, watermark"
HEAD = ",pos1,neg1,pos2,neg2,prompt,denoise,noise,x_shift,y_shift,zoom,rotation,cfg_scale," \
       "px0,py0,px1,py1,px2,py2,px3,py3,seed_start,seed_end,seed_str,pos_prompt,neg_prompt"


def settings(**kw):
    s = {"width": 512, "height": 512, "total_time": 2.0, "fps": 5.0, "seed": 100,
         "denoising_strength": 0.3, "tmpl_pos": TMPL, "tmpl_neg": NEG, "cfg_scale": 5,
         "film_interpolation": True, "smoothing": 2, "final_fps": 97.0,
         "restore_faces": True, "vid_mp4": True, "vid_webm": True, "vid_gif": False,
         "initial_img": None}
    s.update(kw)
    return json.dumps(s)


def keyframes(n=11, denoise=0.3, zoom=1.0, second=None):
    rows = [HEAD]
    for i in range(n):
        w = 1 - i / (n - 1)
        a = TMPL
        b = second or TMPL
        pos = f'"{a}, :{w} AND {b},:{1 - w}"'
        neg = f'"{NEG},:{w} AND {NEG},:{1 - w}"'
        rows.append(f"{i},,,,,{w},{denoise},0.11,0.0,0.0,{zoom},0.0,5.0,0,0,0,0,0,0,0,0,"
                    f"{100 + i}.0,,0,{pos},{neg}")
    return "\n".join(rows) + "\n"


def test_weighted_parts_and_the_dominant_one():
    s = "a cat :0.7 AND a dog, :0.30000000000000004"
    assert A.weighted_parts(s) == [("a cat", 0.7), ("a dog", 0.30000000000000004)]
    assert A.dominant(s) == "a cat"
    assert A.dominant(":1.0 AND :0.0") == ""            # example 2: no prompt at all


def test_clean_prompt_drops_sd15_loras_and_negative_embeddings():
    pos, rp = A.clean_prompt(TMPL)
    assert pos == "a drawing of a 1man with a beard and smoke"
    assert len(rp) == 2
    neg, rn = A.clean_prompt(NEG, negative=True)
    assert neg == "text, logo, watermark"
    assert "FastNegativeEmbedding" in rn


def test_key_size_keeps_the_ratio_in_multiples_of_16():
    assert A.key_size(512, 512, 2.1) == (1440, 1440)
    w, h = A.key_size(512, 768, 2.1)
    assert w % 16 == 0 and h % 16 == 0 and abs(w / h - 2 / 3) < 0.02
    assert max(A.key_size(4000, 500, 2.1)) <= 2048


def test_plan_of_example_1_shape():
    plan, out = A.make_plan(json.loads(settings()), A.read_keyframes(keyframes()), CFG)
    keys = plan["keys"]
    assert [k["t"] for k in keys] == [0.0, 1.0, 2.0]                # one key per second
    assert [k["frame"] for k in keys] == [0, 5, 10]
    assert [k["seed"] for k in keys] == [100, 105, 110]            # the rows' seeds
    assert keys[0]["op"] == "gen" and keys[1]["op"] == "chain"
    assert keys[0]["denoise"] is None and keys[1]["denoise"] == 0.625
    assert keys[1]["prompt"] == "a drawing of a 1man with a beard and smoke"
    assert keys[1]["negative"] == "text, logo, watermark"
    assert out == {"fps": 5, "loop": "once", "seg": 1.0, "formats": ["mp4", "webm"]}
    notes = " ".join(plan["notes"])
    for word in ("<lora:add_detail:1>", "CFG 5", "restore_faces", "final 97.0 fps"):
        assert word in notes


def test_the_row_denoise_moves_the_chain_denoise_within_bounds():
    plan, _ = A.make_plan(json.loads(settings()), A.read_keyframes(keyframes(denoise=0.15)),
                          CFG)
    assert plan["keys"][1]["denoise"] == 0.312                     # 0.625 * 0.15 / 0.3
    plan, _ = A.make_plan(json.loads(settings()), A.read_keyframes(keyframes(denoise=0.9)),
                          CFG)
    assert plan["keys"][1]["denoise"] == 0.75                      # capped


def test_prompt_travel_camera_and_seg():
    kf = keyframes(zoom=1.02, second="a drawing of a woman")
    plan, _ = A.make_plan(json.loads(settings()), A.read_keyframes(kf), CFG, seg=0.5)
    assert [k["t"] for k in plan["keys"]] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert plan["keys"][0]["prompt"].startswith("a drawing of a 1man")
    assert plan["keys"][-1]["prompt"] == "a drawing of a woman"    # the weight crossed
    assert any("camera moves (zoom)" in n for n in plan["notes"])


def test_settings_only_and_no_prompt():
    plan, _ = A.make_plan(json.loads(settings(tmpl_pos="", tmpl_neg="")), None, CFG)
    assert len(plan["keys"]) == 3 and plan["keys"][2]["seed"] == 110   # seed + frame
    assert any("no prompt" in n for n in plan["notes"])


def test_a_bad_settings_file_is_refused():
    with pytest.raises(A.AnimatorError):
        A.read_settings("not json")


def test_import_then_build_through_the_engine(tmp_path):
    p = Project.create(str(tmp_path), "anim")
    plan = A.import_files(p, settings(), keyframes(), CFG)
    assert p.output["fps"] == 5 and p.output["loop"] == "once"
    plan["size"] = [64, 64]                                        # keep the test small
    p.save()
    eng = FakeEngine()
    keys = A.build(p, FakeEngines(eng), CFG)
    assert [k["id"] for k in keys] == ["k000", "k001", "k002"]
    assert [op for op, _ in eng.calls] == ["gen", "upscale", "upscale"]
    assert eng.calls[1][1]["denoise"] == 0.625 and eng.calls[1][1]["seed"] == 105
    assert eng.calls[0][1]["width"] == 64
    assert [k["t"] for k in p.keys] == [0.0, 1.0, 2.0]
    assert A.pending(p) == []
    # a key removed is made again, from the one before it
    p.remove_key("k002")
    assert len(A.pending(Project(p.path))) == 1
    A.build(Project(p.path), FakeEngines(eng), CFG)
    assert [op for op, _ in eng.calls][-1] == "upscale"
    assert len(Project(p.path).keys) == 3

"""cm_timeline: the retiming is pure, tested without FILM or GPU."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import cm_timeline as tl


def keys(*ts, **kw):
    return [dict({"t": t}, **kw) for t in ts]


def test_recursions_oversample_the_frames():
    assert tl.recursions_for(24) == 6          # 2 x 24 = 48 -> 64
    assert tl.recursions_for(8) == 4
    assert tl.recursions_for(1) == 1
    assert tl.recursions_for(24, "ease-in") == 7   # slope 2: 96 -> 128
    assert tl.recursions_for(10_000) == tl.MAX_TIMES


def test_linear_steps_are_even_no_stutter():
    """48 output frames over 128 FILM frames: every step is 2.666, not 2, 3, 3."""
    p = tl.plan(keys(0, 2), 24, "once")
    pos = [x for s, x in p["frames"]]
    steps = [b - a for a, b in zip(pos, pos[1:])]
    assert max(steps) - min(steps) < 2 / tl.POS_GRID


def test_once_shows_each_key_once_and_ends_on_the_last():
    p = tl.plan(keys(0, 1, 2), 24, "once")
    assert len(p["frames"]) == 49
    assert p["frames"][0] == (0, 0)
    assert p["frames"][24] == (1, 0)                       # key 1, once
    assert (0, 2 ** p["segments"][0]["times"]) not in p["frames"]
    assert p["frames"][-1] == (1, 2 ** p["segments"][1]["times"])


def test_frames_move_forward_within_a_segment():
    p = tl.plan(keys(0, 1), 24, "once")
    idx = [j for s, j in p["frames"]]
    assert idx == sorted(idx) and len(set(idx)) == len(idx)   # linear: no repeat


def test_pingpong_never_repeats_an_end():
    p = tl.plan(keys(0, 1), 24, "pingpong")
    f = p["frames"]
    assert len(f) == 2 * 25 - 2
    loop = f + f[:1]
    assert all(a != b for a, b in zip(loop, loop[1:]))       # incl. the wrap


def test_closed_loop_adds_last_to_first_and_drops_the_repeat():
    p = tl.plan(keys(0, 1, 2), 24, "closed")
    assert len(p["segments"]) == 3
    assert p["segments"][-1]["a"] == 2 and p["segments"][-1]["b"] == 0
    assert len(p["frames"]) == 72
    assert p["frames"][-1] != (2, 2 ** p["segments"][2]["times"])


def test_holds_and_rounding():
    k = keys(0, 0.5, 1.25)
    k[0]["hold"] = 3
    k[2]["hold"] = 2
    p = tl.plan(k, 25, "once")
    assert p["frames"][:4] == [(0, 0)] * 4
    assert [s["n"] for s in p["segments"]] == [12, 19]      # round(12.5)=12, 31-12
    assert len(p["frames"]) == 3 + 12 + 19 + 1 + 2


def test_ease_in_starts_slow():
    p = tl.plan(keys(0, 1, ease="ease-in"), 24, "once")
    idx = [j for s, j in p["frames"] if s == 0]
    assert idx[1] - idx[0] < idx[-1] - idx[-2]


def test_bad_plans():
    with pytest.raises(tl.PlanError):
        tl.plan(keys(0), 24)
    with pytest.raises(tl.PlanError):
        tl.plan(keys(0, 1, 1), 24)
    with pytest.raises(tl.PlanError):
        tl.plan(keys(0, 1, ease="bounce"), 24)
    with pytest.raises(tl.PlanError):
        tl.plan(keys(0, 1), 24, "spiral")


def test_output_sizes():
    assert tl.output_size((1088, 1920), "1080p") == (1080, 1920)
    assert tl.output_size((1088, 1920), "4k") == (2160, 3840)
    assert tl.output_size((1920, 1088), "4k") == (3840, 2160)
    assert tl.output_size((1024, 1024), "4k") == (2160, 2160)    # inside 3840 x 2160
    assert tl.output_size((1440, 1440), "1080p") == (1080, 1080)
    assert tl.output_size((1440, 1440), "1mpx") == (720, 720)
    assert tl.output_size((1088, 1920), "work") == (1088, 1920)
    assert tl.output_size((1088, 1920), "720x1280") == (720, 1280)
    w, h = tl.output_size((832, 1248), "1080p")       # 2:3 keeps its ratio, inside 1080 x 1920
    assert (w, h) == (1080, 1620)
    with pytest.raises(tl.PlanError):
        tl.output_size((1088, 1920), "8k")

"""crispz-motion - retiming: which FILM frame goes in each output frame.

Pure functions, stdlib only (tested on CPU without FILM).

The model:
  - keys carry a time in seconds (t, strictly increasing, the first at 0), a hold
    (extra frozen frames shown on the key) and an ease (the speed curve of the segment
    that STARTS at that key);
  - a segment a -> b gets n output frames, n = round(t_b * fps) - round(t_a * fps),
    so the rounding never drifts over a long video;
  - FILM computes 2^N + 1 frames per segment (both keys included), by recursive
    midpoints only: the network ignores any other time (measured on lot 1, a frame
    asked at t = 0.02 is 15 dB away from the key), so a frame can only be asked at
    k / 2^N;
  - output frame j of a segment (0 <= j < n) sits EXACTLY at pos = ease(j / n) * 2^N,
    a position between two FILM frames. A whole pos is that FILM frame; otherwise the
    two neighbours are blended in proportion. Taking the nearest FILM frame instead
    made a 1, 1, 2 step pattern (48 output frames over 64 FILM frames): the motion
    stuttered 8 times a second. N is the smallest count with 2^N >= OVERSAMPLE * n *
    (max slope of the ease), so the blended neighbours are at most half an output
    step apart (a slight motion blur, no stutter);
  - the end key of a segment is the start of the next one, so it is shown once: no
    doubled frame at a key.

Loops:
  - once: k0 -> ... -> kN, the last key included.
  - pingpong: the forward run, then backwards without repeating either end, so the
    video loops with no frame shown twice in a row.
  - closed: one more segment kN -> k0 ("close" seconds, default the mean segment); the
    last frame (k0 again) is dropped, so the loop joins seamlessly.

A plan is a list of output frames, each a (segment index, position) reference, the
position a float in FILM frame units (rounded to 1/POS_GRID so that it can name a cache
file), plus the segment list. The renderer maps those references to files.
"""

import math

EASES = {
    # name: (curve, max slope over [0, 1])
    "linear": (lambda u: u, 1.0),
    "ease-in": (lambda u: u * u, 2.0),
    "ease-out": (lambda u: 1.0 - (1.0 - u) ** 2, 2.0),
    "ease-in-out": (lambda u: u * u * (3.0 - 2.0 * u), 1.5),
}
LOOPS = ("once", "pingpong", "closed")
MAX_TIMES = 8           # 257 FILM frames per segment: 5 s at 24 fps, oversampled x2
OVERSAMPLE = 2          # FILM frames per output frame, at least
POS_GRID = 256          # positions are rounded to 1/256 of a FILM frame


class PlanError(ValueError):
    pass


def recursions_for(n_frames, ease="linear"):
    slope = EASES.get(ease, EASES["linear"])[1]
    need = max(2, math.ceil(OVERSAMPLE * n_frames * slope))
    return max(1, min(MAX_TIMES, math.ceil(math.log2(need))))


def _pos(x):
    r = round(x * POS_GRID) / POS_GRID
    return int(r) if r == int(r) else r


def plan(keys, fps, loop="pingpong", close=None):
    """keys: [{"t": seconds, "hold": frames, "ease": name}, ...] (at least 2).
    Returns {"segments": [{"a", "b", "n", "times", "ease"}], "frames": [(seg, idx)],
    "duration_s", "forward_frames"}."""
    if len(keys) < 2:
        raise PlanError("at least 2 keys are needed")
    if fps <= 0:
        raise PlanError("fps must be positive")
    if loop not in LOOPS:
        raise PlanError(f"loop must be one of {', '.join(LOOPS)}")
    times = [float(k.get("t", 0.0)) for k in keys]
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            raise PlanError(f"key {i} is at {times[i]} s, not after key {i - 1} "
                            f"({times[i - 1]} s): times must increase")
    pairs = [(i, i + 1, times[i], times[i + 1]) for i in range(len(keys) - 1)]
    if loop == "closed":
        span = times[-1] - times[0]
        c = float(close) if close else span / (len(keys) - 1)
        if c <= 0:
            raise PlanError("the closing segment needs a positive duration")
        pairs.append((len(keys) - 1, 0, times[-1], times[-1] + c))

    segments, frames = [], []
    for s, (a, b, ta, tb) in enumerate(pairs):
        n = max(1, round(tb * fps) - round(ta * fps))
        ease = keys[a].get("ease") or "linear"
        if ease not in EASES:
            raise PlanError(f"key {a}: unknown ease '{ease}' ({', '.join(EASES)})")
        nt = recursions_for(n, ease)
        segments.append({"a": a, "b": b, "n": n, "times": nt, "ease": ease})
        last = 2 ** nt
        curve = EASES[ease][0]
        frames += [(s, 0)] * max(0, int(keys[a].get("hold") or 0))   # hold on key a
        frames += [(s, _pos(curve(j / n) * last)) for j in range(n)]
    if loop != "closed":
        s = len(segments) - 1
        last = 2 ** segments[s]["times"]
        frames += [(s, last)] * (1 + max(0, int(keys[-1].get("hold") or 0)))
    forward = len(frames)
    if loop == "pingpong" and len(frames) > 2:
        frames = frames + frames[-2:0:-1]
    return {"segments": segments, "frames": frames, "forward_frames": forward,
            "duration_s": round(len(frames) / fps, 3)}


def output_size(work_size, res):
    """Final (w, h) for a resolution choice. The key ratio is kept and the frame fits
    inside the preset's 16:9 frame turned like the keys (1mpx 1280 x 720, 1080p / 2mpx
    1920 x 1080, 4k 3840 x 2160): a square key gives 720, 1080 or 2160 square. A ratio
    within 2 % of 16:9 or 9:16 snaps to the standard frame (1088 x 1920 keys -> 1080 x
    1920, 2160 x 3840).
    "WxH" is taken as is (the frames are cropped to it). "work" keeps the key size.
    Always even (yuv420p)."""
    ww, wh = work_size
    res = str(res or "1080p").lower().strip()
    if res == "work":
        return ww - ww % 2, wh - wh % 2
    longs = {"1mpx": 1280, "720p": 1280, "1080p": 1920, "2mpx": 1920, "4k": 3840}
    if res not in longs and "x" in res:     # "1mpx" has an x too: the presets come first
        try:
            w, h = (int(v) for v in res.split("x", 1))
        except ValueError:
            raise PlanError(f"bad size {res}")
        if w < 16 or h < 16:
            raise PlanError(f"bad size {res}")
        return w - w % 2, h - h % 2
    if res not in longs:
        raise PlanError(f"unknown resolution '{res}' (1mpx, 1080p, 2mpx, 4k, work, WxH)")
    # the frame fits INSIDE the standard 16:9 frame turned like the keys: a square 4K is
    # 2160 x 2160, not 3840 x 3840 (decided by Mike on lot 3), a square 1080p 1080 x 1080
    box_long = longs[res]
    box_short = box_long * 9 // 16
    ratio = min(ww, wh) / max(ww, wh)
    if abs(ratio / (9 / 16) - 1) < 0.02:
        long_side, short = box_long, box_short
    else:
        s = min(box_long / max(ww, wh), box_short / min(ww, wh))
        long_side = 2 * round(max(ww, wh) * s / 2)
        short = 2 * round(min(ww, wh) * s / 2)
    return (long_side, short) if ww >= wh else (short, long_side)

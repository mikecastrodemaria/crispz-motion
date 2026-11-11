"""crispz-motion - FILM worker (route A: the PyTorch / TorchScript port).

Runs with the FILM venv (film/.venv, torch cu128), never with the app's own Python:
the app stays stdlib + Pillow, torch lives here only.

What it does: loads the FILM "Style" network once (TorchScript export of the official
Google weights, dajes/frame-interpolation-pytorch, Apache 2.0), then for each pair of
consecutive keys runs FILM's recursive midpoint interpolation (the network was trained
to predict t = 0.5; the official repo interpolates the same way) and writes every frame
of the segment, BOTH keys included, so that a segment is a self-contained cache entry:

    out_dir/seg000/f00000.png ... f0000{2^N}.png      keys[0] -> keys[1]
    out_dir/seg001/...                                  keys[1] -> keys[2]

One JSON line per event on stdout (the caller parses them, a human can read them):
    {"event": "ready", "device": ..., "load_s": ...}
    {"event": "frame", "seg": 0, "done": 3, "total": 7, "infer_s": 0.41}
    {"event": "segment", "seg": 0, "frames": 9, "infer_s": ..., "save_s": ..., "peak_gb": ...}
    {"event": "done", ...}  or  {"event": "error", "error": ...}

Usage:
    film/.venv/Scripts/python cm_film_worker.py --model film/models/film_net_fp32.pt \
        --times 3 --out out_dir key0.png key1.png [key2.png ...] [--fp16] [--cpu]

    film/.venv/Scripts/python cm_film_worker.py --model ... --fp16 --jobs jobs.json

--jobs (what the app uses, lot 1): a JSON list of segments to compute, each
{"a": key_a, "b": key_b, "out": folder, "times": N}. The app gives only the segments
whose cache is missing, each with its own folder and its own N. A segment is written
to "<out>.tmp" then renamed, so a folder that exists is always complete. In that mode
the "seg" of the events is the index in the jobs list.
"""

import os
import sys
import json
import time
import shutil
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ALIGN = 64          # FILM's pyramid needs sizes that divide by 64 (the official 'align')


def emit(**kw):
    print(json.dumps(kw), flush=True)


def load_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def to_tensor(img, device, dtype):
    a = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=dtype)


def pad_to_align(t):
    """Replicate-pads (B, C, H, W) to multiples of ALIGN; returns (padded, (h, w)).
    The result is channels_last (NHWC): measured on lot 2 (RTX 5090, fp16, 4K), FILM
    runs 0.45 s per frame on NHWC inputs and 3.3 to 3.6 s on the NCHW tensor F.pad
    returns. A key of 2176 px (no padding, NHWC straight from numpy) was fast and a key
    of 2160 px (padded) was 8 times slower, for the same tensor size."""
    h, w = t.shape[-2:]
    ph, pw = (-h) % ALIGN, (-w) % ALIGN
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="replicate")
    return t.contiguous(memory_format=torch.channels_last), (h, w)


def to_image(t, hw):
    h, w = hw
    a = t[0, :, :h, :w].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))


class Film:
    def __init__(self, model_path, device, fp16):
        self.device = device
        self.dtype = torch.float16 if (fp16 and device.type == "cuda") else torch.float32
        t0 = time.time()
        m = torch.jit.load(model_path, map_location="cpu")
        m.eval()
        # the weights stay as they are: converting the model to channels_last made every
        # frame slower (5 s in 4K, 0.28 s at 2 Mpx); only the INPUTS are (pad_to_align)
        self.model = m.to(device=device, dtype=self.dtype)
        self.load_s = time.time() - t0

    @torch.no_grad()
    def mid(self, x0, x1):
        dt = x0.new_full((1, 1), 0.5)
        return self.model(x0, x1, dt).clamp(0, 1)

    def recursive(self, x0, x1, n, on_frame=None):
        """[x0, ..., x1]: 2^n + 1 frames, the official recursive midpoint order."""
        if n == 0:
            return [x0, x1]
        frames = [x0, x1]
        for _level in range(n):
            nxt = [frames[0]]
            for a, b in zip(frames[:-1], frames[1:]):
                t0 = time.time()
                m = self.mid(a, b)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                if on_frame:
                    on_frame(time.time() - t0)
                nxt += [m, b]
            frames = nxt
        return frames


def main(argv=None):
    ap = argparse.ArgumentParser(description="crispz-motion FILM worker (PyTorch)")
    ap.add_argument("keys", nargs="*", help="key images, in order (2 or more, same size)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--times", type=int, default=3, help="recursions: 2^N - 1 frames per pair")
    ap.add_argument("--out", help="output folder (with positional keys)")
    ap.add_argument("--jobs", help="JSON list of {a, b, out, times} (the app's mode)")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="time only (benchmarks)")
    a = ap.parse_args(argv)

    if a.jobs:
        try:
            with open(a.jobs, encoding="utf-8") as fh:
                jobs = json.load(fh)
            jobs = [{"a": j["a"], "b": j["b"], "out": j["out"],
                     "times": int(j.get("times", a.times))} for j in jobs]
        except (OSError, ValueError, KeyError, TypeError) as e:
            emit(event="error", error=f"bad --jobs file {a.jobs}: {e}")
            return 2
    else:
        if len(a.keys) < 2 or not a.out:
            emit(event="error", error="need at least 2 keys and --out (or --jobs)")
            return 2
        jobs = [{"a": a.keys[i], "b": a.keys[i + 1], "times": a.times,
                 "out": os.path.join(a.out, f"seg{i:03d}")} for i in range(len(a.keys) - 1)]
    if not jobs:
        emit(event="done", segments=0, frames=0, infer_s=0, save_s=0, wall_s=0)
        return 0
    for j in jobs:
        if not 0 <= j["times"] <= 8:
            emit(event="error", error="times must be 0 to 8")
            return 2
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cpu" and not a.cpu:
        # never a silent fallback: FILM on the CPU is 10 to 50 times slower
        emit(event="warning", warning=f"no CUDA in this torch ({torch.__version__}): FILM "
                                      f"runs on the CPU. A CPU build of torch? run "
                                      f"update with --force-deps")
    try:
        film = Film(a.model, device, a.fp16)
    except Exception as e:
        emit(event="error", error=f"cannot load {a.model}: {type(e).__name__}: {e}")
        return 1
    emit(event="ready", device=str(device),
         gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
         dtype=str(film.dtype).replace("torch.", ""), load_s=round(film.load_s, 2),
         torch=torch.__version__)

    cache = {}

    def image(path):
        if path not in cache:
            if len(cache) > 4:      # consecutive jobs share a key; keep a few, not all
                cache.clear()
            cache[path] = load_rgb(path)
        return cache[path]

    size = None
    totals = {"infer_s": 0.0, "save_s": 0.0, "frames": 0}
    t_all = time.time()
    try:
        for seg, job in enumerate(jobs):
            ia, ib = image(job["a"]), image(job["b"])
            size = ia.size          # the two keys of a pair share one size
            if ib.size != size:
                emit(event="error", error=f"{job['b']} is {ib.size[0]}x{ib.size[1]}, "
                                          f"{job['a']} is {size[0]}x{size[1]}: the two "
                                          f"keys of a pair must share one size")
                return 2
            per_pair = 2 ** job["times"] - 1
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            x0, hw = pad_to_align(to_tensor(ia, device, film.dtype))
            x1, _ = pad_to_align(to_tensor(ib, device, film.dtype))
            state = {"done": 0, "infer": 0.0}

            def on_frame(dt_s, seg=seg, state=state, per_pair=per_pair):
                state["done"] += 1
                state["infer"] += dt_s
                emit(event="frame", seg=seg, done=state["done"], total=per_pair,
                     infer_s=round(dt_s, 3))

            frames = film.recursive(x0, x1, job["times"], on_frame)
            t_s = time.time()
            if not a.no_save:
                d = job["out"]
                tmp = d + ".tmp"
                shutil.rmtree(tmp, ignore_errors=True)
                os.makedirs(tmp, exist_ok=True)
                for i, f in enumerate(frames):
                    to_image(f, hw).save(os.path.join(tmp, f"f{i:05d}.png"), compress_level=1)
                shutil.rmtree(d, ignore_errors=True)
                os.replace(tmp, d)
            save_s = time.time() - t_s
            peak = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None
            totals["infer_s"] += state["infer"]
            totals["save_s"] += save_s
            totals["frames"] += len(frames)
            emit(event="segment", seg=seg, frames=len(frames), size=list(size),
                 infer_s=round(state["infer"], 2),
                 per_frame_s=round(state["infer"] / max(1, per_pair), 3),
                 save_s=round(save_s, 2),
                 peak_gb=round(peak, 2) if peak is not None else None)
            del frames, x0, x1
    except torch.cuda.OutOfMemoryError as e:
        emit(event="error", error=f"out of GPU memory at {size[0]}x{size[1]}: {e}")
        return 1
    except Exception as e:
        emit(event="error", error=f"{type(e).__name__}: {e}")
        return 1
    emit(event="done", segments=len(jobs), frames=totals["frames"],
         infer_s=round(totals["infer_s"], 2), save_s=round(totals["save_s"], 2),
         wall_s=round(time.time() - t_all, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

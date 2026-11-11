"""crispz-motion - ESRGAN worker: the 4K step of the render (order "FILM at 2 Mpx, then
ESRGAN on every frame", decided after lot 0, question Q7).

Runs with the FILM venv (film/.venv: torch cu128 + spandrel), like cm_film_worker.py.
Loads one ESRGAN model (any spandrel architecture: 4x-ClearRealityV1 by default) once,
then for each job upscales the source frame, resizes it on the GPU (bicubic,
antialiased) so that it covers the final size, and crops the centre. No diffusion touches the frames, so nothing
flickers between them.

    film/.venv/Scripts/python cm_upscale_worker.py --model 4x-ClearRealityV1.safetensors \
        --size 2160 3840 --jobs jobs.json [--cpu]

jobs.json: a list of {"src": frame.png, "dst": out.png}. A dst is written to a temporary
name then renamed, so a file that exists is complete. One JSON line per event on stdout:
ready, frame, done, error, warning (the same shape as cm_film_worker.py).
"""

import os
import sys
import json
import math
import time
import argparse

import numpy as np
import torch
from PIL import Image


def emit(**kw):
    print(json.dumps(kw), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="crispz-motion ESRGAN worker")
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, nargs=2, required=True, metavar=("W", "H"))
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    w, h = a.size
    try:
        with open(a.jobs, encoding="utf-8") as fh:
            jobs = [(j["src"], j["dst"]) for j in json.load(fh)]
    except (OSError, ValueError, KeyError, TypeError) as e:
        emit(event="error", error=f"bad --jobs file {a.jobs}: {e}")
        return 2

    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cpu" and not a.cpu:
        emit(event="warning", warning=f"no CUDA in this torch ({torch.__version__}): ESRGAN "
                                      f"runs on the CPU (very slow in 4K)")
    t0 = time.time()
    try:
        from spandrel import ModelLoader
        model = ModelLoader().load_from_file(a.model).to(device).eval()
    except Exception as e:
        emit(event="error", error=f"cannot load {a.model}: {type(e).__name__}: {e}")
        return 1
    half = device.type == "cuda" and bool(getattr(model, "supports_half", False))
    if half:
        model = model.half()
    emit(event="ready", device=str(device), model=os.path.basename(a.model),
         scale=getattr(model, "scale", None), fp16=half, load_s=round(time.time() - t0, 2))

    def save(arr, dst):
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        tmp = dst + ".tmp.png"
        Image.fromarray(arr).save(tmp, compress_level=1)
        os.replace(tmp, dst)

    # The x4 frame is 33 Mpx for a 2 Mpx key: resizing it on the CPU (and moving it
    # there in float) cost 20 times the network itself. The cover resize runs on the
    # GPU (bicubic, antialiased), only the final uint8 frame comes back, and the PNG
    # writes overlap the next frame.
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=3)
    pending = []
    t_all = time.time()
    infer = 0.0
    try:
        for i, (src, dst) in enumerate(jobs):
            with Image.open(src) as im:
                img = im.convert("RGB")
            t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
            t = t.permute(2, 0, 1).unsqueeze(0).to(device)
            if half:
                t = t.half()
            t1 = time.time()
            with torch.no_grad():
                up = model(t).float()
                uh, uw = up.shape[-2:]
                s = max(w / uw, h / uh)
                rw, rh = max(w, math.ceil(uw * s)), max(h, math.ceil(uh * s))
                if (rw, rh) != (uw, uh):
                    up = torch.nn.functional.interpolate(up, size=(rh, rw), mode="bicubic",
                                                         antialias=True, align_corners=False)
                x, y = (rw - w) // 2, (rh - h) // 2
                up = up[0, :, y:y + h, x:x + w]
                arr = (up.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).permute(1, 2, 0)
                arr = arr.contiguous().cpu().numpy()
            infer += time.time() - t1
            pending.append(pool.submit(save, arr, dst))
            while len(pending) > 6:
                pending.pop(0).result()
            emit(event="frame", done=i + 1, total=len(jobs))
        for f in pending:
            f.result()
        pool.shutdown()
    except torch.cuda.OutOfMemoryError as e:
        emit(event="error", error=f"out of GPU memory: {e}")
        return 1
    except Exception as e:
        emit(event="error", error=f"{type(e).__name__}: {e}")
        return 1
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
    emit(event="done", frames=len(jobs), infer_s=round(infer, 2),
         wall_s=round(time.time() - t_all, 2),
         peak_gb=round(peak, 2) if peak is not None else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

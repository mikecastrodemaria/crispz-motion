"""crispz-motion lot 0 - FILM measurements on this machine.

Run with the FILM venv:  film\\.venv\\Scripts\\python tools\\bench_lot0.py

1. Route A (PyTorch port, GPU): one pair at 1 Mpx, 2 Mpx and 4K (portrait 9:16),
   3 recursions (7 frames between the keys), fp32 and fp16: time per frame, peak VRAM.
2. Route C (official TensorFlow, CPU on native Windows): the midpoint at 1 Mpx, the
   quality reference; PSNR of route A's midpoint against it.
3. The 4K order: FILM at 2 Mpx then ESRGAN x4 (downscaled to 4K) on every frame, timed
   per frame, to set against "keys upscaled first, then FILM in 4K" (step 1).
4. A ping-pong mp4 of the 2 Mpx segment and of the 4K one, for the eye.

Everything goes to bench/lot0/ (gitignored); bench/lot0/results.json gathers the numbers.
"""
import os
import sys
import json
import math
import time
import shutil
import subprocess

import numpy as np
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 64 * 1024 * 1024   # FILM's sample PNGs carry big text chunks

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "bench", "lot0")
MODEL = os.path.join(HERE, "film", "models", "film_net_fp32.pt")
WORKER = os.path.join(HERE, "cm_film_worker.py")
FILM_TF_DIR = os.environ.get("FILM_TF_DIR") or os.path.join(os.path.dirname(HERE), "frame-interpolation-main")
FILM_TF_PY = os.path.join(FILM_TF_DIR, ".venv", "Scripts", "python.exe")
ESRGAN_DIR = os.environ.get("ESRGAN_DIR", "")   # folder of ESRGAN models; empty = step skipped
ESRGAN_MODELS = ["4x-ClearRealityV1.safetensors", "4x-UltraSharp.pth"]
PAIR = [os.path.join(FILM_TF_DIR, "photos", "a_woman_with_long_wav_1.png"),
        os.path.join(FILM_TF_DIR, "photos", "a_woman_with_long_wav_2.png")]
SIZES = {"1mpx": (768, 1344), "2mpx": (1080, 1920), "4k": (2160, 3840)}
TIMES = 3


def log(msg):
    print(f"[lot0] {msg}", flush=True)


def cover(img, w, h):
    """Resize + centre crop to exactly w x h (Lanczos)."""
    s = max(w / img.width, h / img.height)
    r = img.resize((math.ceil(img.width * s), math.ceil(img.height * s)), Image.LANCZOS)
    x, y = (r.width - w) // 2, (r.height - h) // 2
    return r.crop((x, y, x + w, y + h))


def prep_inputs():
    d = os.path.join(OUT, "inputs")
    os.makedirs(d, exist_ok=True)
    src = [Image.open(p).convert("RGB") for p in PAIR]
    out = {}
    for tag, (w, h) in SIZES.items():
        paths = []
        for i, im in enumerate(src):
            p = os.path.join(d, f"{tag}_{'ab'[i]}.png")
            cover(im, w, h).save(p)
            paths.append(p)
        out[tag] = paths
    return out


def run_worker(keys, out_dir, fp16, save=True):
    cmd = [sys.executable, WORKER, *keys, "--model", MODEL, "--times", str(TIMES),
           "--out", out_dir]
    if fp16:
        cmd.append("--fp16")
    if not save:
        cmd.append("--no-save")
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    events = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            events.append(json.loads(line))
    res = {"wall_s": round(time.time() - t0, 2), "returncode": p.returncode}
    for e in events:
        if e["event"] in ("ready", "segment", "done", "error", "warning"):
            res[e["event"]] = e
    if p.returncode != 0 and "error" not in res:
        res["stderr"] = p.stderr[-1500:]
    return res


def psnr(a_path, b_path):
    a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.float64)
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else round(10 * math.log10(255.0 ** 2 / mse), 2)


def esrgan_bench(frames_dir, out_dir):
    """ESRGAN x4 on every 2 Mpx frame, then Lanczos down to 4K: time per frame."""
    import torch
    from spandrel import ModelLoader
    dev = torch.device("cuda")
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    res = {}
    for name in ESRGAN_MODELS:
        path = os.path.join(ESRGAN_DIR, name)
        if not os.path.isfile(path):
            res[name] = {"error": f"missing: {path}"}
            continue
        t0 = time.time()
        m = ModelLoader().load_from_file(path).to(dev).eval()
        half = bool(getattr(m, "supports_half", False))
        if half:
            m = m.half()
        load_s = time.time() - t0
        torch.cuda.reset_peak_memory_stats()
        times = []
        d = os.path.join(out_dir, os.path.splitext(name)[0])
        os.makedirs(d, exist_ok=True)
        for i, f in enumerate(files):
            img = Image.open(os.path.join(frames_dir, f)).convert("RGB")
            t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
            t = t.permute(2, 0, 1).unsqueeze(0).to(dev)
            if half:
                t = t.half()
            t1 = time.time()
            with torch.no_grad():
                up = m(t)
            torch.cuda.synchronize()
            times.append(time.time() - t1)
            if i in (0, len(files) // 2):
                a = up[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                big = Image.fromarray((a * 255 + 0.5).astype(np.uint8))
                big.resize(SIZES["4k"], Image.LANCZOS).save(os.path.join(d, f))
        res[name] = {"load_s": round(load_s, 2), "fp16": half,
                     "per_frame_s": round(sum(times[1:]) / max(1, len(times) - 1), 3),
                     "first_frame_s": round(times[0], 3),
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        del m
        torch.cuda.empty_cache()
    return res


def pingpong_mp4(frames_dir, out_path, fps=24):
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    seq = files + files[-2:0:-1]
    tmp = out_path + "_list"
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(seq * 3):
        shutil.copyfile(os.path.join(frames_dir, f), os.path.join(tmp, f"{i:05d}.png"))
    cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i",
           os.path.join(tmp, "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "17", "-preset", "veryfast", out_path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return p.returncode == 0


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {"date": time.strftime("%Y-%m-%d %H:%M"), "times": TIMES,
               "pair": [os.path.basename(p) for p in PAIR], "route_a": {}}
    import torch
    if not torch.cuda.is_available():
        log(f"STOP: this torch ({torch.__version__}) has no CUDA; run install_film.bat")
        return 1
    inputs = prep_inputs()
    log(f"inputs ready: {', '.join(f'{k} {w}x{h}' for k, (w, h) in SIZES.items())}")

    for tag in SIZES:
        for fp16 in (False, True):
            key = f"{tag}_{'fp16' if fp16 else 'fp32'}"
            out_dir = os.path.join(OUT, f"A_{key}")
            log(f"route A {key} ...")
            r = run_worker(inputs[tag], out_dir, fp16)
            results["route_a"][key] = r
            seg = r.get("segment") or {}
            log(f"  -> {seg.get('per_frame_s')} s/frame, peak {seg.get('peak_gb')} GB, "
                f"load {(r.get('ready') or {}).get('load_s')} s"
                + (f", ERROR {r['error']['error']}" if 'error' in r else ""))

    # Route C: the official TensorFlow midpoint, CPU
    mid_c = os.path.join(OUT, "C_1mpx_mid.png")
    if os.path.isfile(FILM_TF_PY):
        log("route C (official TF, CPU) midpoint at 1 Mpx ...")
        p = subprocess.run([FILM_TF_PY, os.path.join(HERE, "tools", "film_tf_mid.py"),
                            *inputs["1mpx"], mid_c, "--film-dir", FILM_TF_DIR],
                           capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("{")]
        results["route_c"] = json.loads(line[-1]) if line else {"error": p.stderr[-1500:]}
        log(f"  -> {results['route_c']}")
        if os.path.isfile(mid_c):
            mid_index = 2 ** (TIMES - 1)
            results["psnr_vs_tf"] = {}
            for prec in ("fp32", "fp16"):
                a = os.path.join(OUT, f"A_1mpx_{prec}", "seg000", f"f{mid_index:05d}.png")
                if os.path.isfile(a):
                    results["psnr_vs_tf"][prec] = psnr(a, mid_c)
            log(f"  PSNR of route A's midpoint against TF: {results['psnr_vs_tf']}")
    else:
        results["route_c"] = {"error": f"no FILM TF python at {FILM_TF_PY}"}

    # The 4K order "FILM at 2 Mpx, then ESRGAN on every frame"
    frames_2m = os.path.join(OUT, "A_2mpx_fp16", "seg000")
    if os.path.isdir(frames_2m):
        log("ESRGAN x4 -> 4K on every 2 Mpx frame ...")
        results["esrgan_after_film"] = esrgan_bench(frames_2m, os.path.join(OUT, "B_esrgan"))
        log(f"  -> {results['esrgan_after_film']}")

    for tag in ("2mpx", "4k"):
        d = os.path.join(OUT, f"A_{tag}_fp16", "seg000")
        if os.path.isdir(d):
            ok = pingpong_mp4(d, os.path.join(OUT, f"pingpong_{tag}.mp4"))
            log(f"pingpong_{tag}.mp4: {'ok' if ok else 'ffmpeg failed'}")

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"results: {os.path.join(OUT, 'results.json')}")


if __name__ == "__main__":
    main()

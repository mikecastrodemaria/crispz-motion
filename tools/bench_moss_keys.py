"""crispz-motion lot 0 - keys from crispz-moss (Turbo LoRA, CFG 1) through the protocol,
then FILM between them.

    film\\.venv\\Scripts\\python tools\\bench_moss_keys.py

1. Starts crispz-moss's app (its own .venv, port 7860) unless one already answers there;
   stops it at the end only if this script started it.
2. k0 = gen, Turbo, 1088x1920, the prompt of the historic example 1 (its SD 1.5 LoRA tags
   and embeddings removed: they do not exist for MOSS).
3. The "sparse loopback": k1 = img2img of k0, k2 = img2img of k1 (upscale op, factor 1,
   Turbo, seed + 1 per key), at two denoise values.
4. 4K keys: upscale x2 of k0 and k1 (ESRGAN 4x-ClearRealityV1, Turbo refine, low denoise).
5. FILM (cm_film_worker, fp16, 3 recursions) over k0 -> k1 -> k2 at 2 Mpx and k0 -> k1 in
   4K, and ping-pong mp4 of both.

Writes bench/lot0/moss/ and bench/lot0/moss/results.json. The spec of every call is kept
in the results (a key is a recipe).
"""
import os
import re
import sys
import json
import time
import shutil
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "bench", "lot0", "moss")
MOSS_DIR = os.environ.get("MOSS_DIR") or os.path.join(os.path.dirname(HERE), "crispz-moss")
URL = "http://127.0.0.1:7860"
WORKER = os.path.join(HERE, "cm_film_worker.py")
MODEL = os.path.join(HERE, "film", "models", "film_net_fp32.pt")

PROMPT = ("a pencil drawing of a man with a mustache and a beard, long hair, open jacket, "
          "tall grass in the wind, smoke coming out of his head and a lot of smoke coming "
          "out of his mouth, white paper background")
NEGATIVE = "text, logo, watermark"
SEED = 2763806460
W, H = 1088, 1920
TURBO = {"steps": 4, "guidance": 1.0, "loras": ["moss-turbo-4-steps:1"]}


def log(msg):
    print(f"[moss-keys] {msg}", flush=True)


def gradio_call(name, data, get_timeout=3600):
    req = urllib.request.Request(f"{URL}/gradio_api/call/{name}",
                                 data=json.dumps({"data": data}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        eid = json.loads(r.read().decode())["event_id"]
    with urllib.request.urlopen(f"{URL}/gradio_api/call/{name}/{eid}",
                                timeout=get_timeout) as r:
        text = r.read().decode("utf-8")
    m = re.search(r"data:\s*(\[[\s\S]*?\])\s*(?:\n|$)", text)
    if not m:
        raise RuntimeError(f"empty answer from {name}: {text[-400:]}")
    out = json.loads(m.group(1))[0]
    return json.loads(out) if isinstance(out, str) else out


def probe():
    try:
        caps = gradio_call("cli_caps", [], get_timeout=10)
        return caps if isinstance(caps, dict) and caps.get("ok") else None
    except Exception:
        return None


def start_moss():
    py = os.path.join(MOSS_DIR, ".venv", "Scripts", "python.exe")
    env = dict(os.environ, GRADIO_SERVER_PORT="7860", GRADIO_SERVER_NAME="127.0.0.1",
               PYTHONUTF8="1")
    logf = open(os.path.join(OUT, "moss.log"), "ab")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    p = subprocess.Popen([py, "app.py"], cwd=MOSS_DIR, env=env, stdout=logf,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         creationflags=flags)
    t0 = time.time()
    while time.time() - t0 < 240:
        if p.poll() is not None:
            raise RuntimeError(f"crispz-moss exited with {p.returncode}, see moss.log")
        caps = probe()
        if caps:
            log(f"{caps.get('tool')} {caps.get('version')} ready in {time.time() - t0:.0f} s "
                f"(supports.guidance={caps.get('supports', {}).get('guidance')})")
            return p
        time.sleep(2)
    p.terminate()
    raise RuntimeError("crispz-moss did not answer within 240 s")


def call(op, spec, name):
    spec = dict(spec, protocol=1, op=op, out_dir=os.path.join(OUT, "raw"))
    t0 = time.time()
    res = gradio_call({"gen": "cli_gen", "upscale": "cli_upscale"}[op], [json.dumps(spec)])
    wall = time.time() - t0
    if not res.get("ok"):
        raise RuntimeError(f"{name}: {res.get('error')}")
    dst = os.path.join(OUT, f"{name}.png")
    shutil.copyfile(res["images"][0], dst)
    log(f"{name}: {wall:.1f} s  {res.get('size') or ''}  guidance={res.get('guidance')} "
        f"warnings={res.get('warnings')}")
    return {"spec": spec, "wall_s": round(wall, 2), "timings": res.get("timings"),
            "guidance": res.get("guidance"), "warnings": res.get("warnings"),
            "seed_used": res.get("seed_used"), "path": dst}


def film(keys, out_dir):
    cmd = [sys.executable, WORKER, *keys, "--model", MODEL, "--times", "3", "--fp16",
           "--out", out_dir]
    p = subprocess.run(cmd, capture_output=True, text=True)
    ev = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    return {"returncode": p.returncode,
            "segments": [e for e in ev if e["event"] == "segment"],
            "done": next((e for e in ev if e["event"] == "done"), None),
            "error": next((e for e in ev if e["event"] == "error"), None)}


def mp4_from_segments(film_dir, out_path, fps=24, pingpong=True):
    segs = sorted(d for d in os.listdir(film_dir) if d.startswith("seg"))
    files = []
    for i, s in enumerate(segs):
        fs = sorted(os.listdir(os.path.join(film_dir, s)))
        fs = fs if i == len(segs) - 1 else fs[:-1]      # shared key once
        files += [os.path.join(film_dir, s, f) for f in fs]
    seq = files + (files[-2:0:-1] if pingpong else [])
    tmp = out_path + "_list"
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(seq * 2):
        shutil.copyfile(f, os.path.join(tmp, f"{i:05d}.png"))
    p = subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i",
                        os.path.join(tmp, "%05d.png"), "-c:v", "libx264", "-pix_fmt",
                        "yuv420p", "-crf", "17", "-preset", "veryfast", out_path],
                       capture_output=True, text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return p.returncode == 0


def main():
    os.makedirs(OUT, exist_ok=True)
    started = None
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "keys": {}}
    if not probe():
        log("starting crispz-moss ...")
        started = start_moss()
    try:
        k = res["keys"]
        k["k0"] = call("gen", dict(TURBO, prompt=PROMPT, negative=NEGATIVE, width=W,
                                   height=H, seed=SEED), "k0")
        k["k0_again"] = call("gen", dict(TURBO, prompt=PROMPT, negative=NEGATIVE, width=W,
                                         height=H, seed=SEED), "k0_again")
        for den in (0.35, 0.5):
            tag = f"d{int(den * 100)}"
            prev = k["k0"]["path"]
            for i in (1, 2):
                name = f"k{i}_{tag}"
                k[name] = call("upscale", dict(TURBO, input=prev, factor=1.0, denoise=den,
                                               prompt=PROMPT, seed=SEED + i), name)
                prev = k[name]["path"]
        for src in ("k0", "k1_d35"):
            k[f"{src}_4k"] = call("upscale", dict(TURBO, input=k[src]["path"], factor=2.0,
                                                  denoise=0.2, model="4x-ClearRealityV1.safetensors",
                                                  prompt=PROMPT, seed=SEED), f"{src}_4k")
    finally:
        if started is not None:
            log("stopping the crispz-moss app this script started")
            started.terminate()
            try:
                started.wait(timeout=30)
            except subprocess.TimeoutExpired:
                started.kill()
            time.sleep(3)

    res["film"] = {}
    for tag in ("d35", "d50"):
        keys = [res["keys"]["k0"]["path"], res["keys"][f"k1_{tag}"]["path"],
                res["keys"][f"k2_{tag}"]["path"]]
        d = os.path.join(OUT, f"film_2mpx_{tag}")
        res["film"][f"2mpx_{tag}"] = film(keys, d)
        mp4_from_segments(d, os.path.join(OUT, f"loopback_2mpx_{tag}.mp4"))
        log(f"FILM 2 Mpx {tag}: {res['film'][f'2mpx_{tag}']['done']}")
    keys4k = [res["keys"]["k0_4k"]["path"], res["keys"]["k1_d35_4k"]["path"]]
    d = os.path.join(OUT, "film_4k_d35")
    res["film"]["4k_d35"] = film(keys4k, d)
    mp4_from_segments(d, os.path.join(OUT, "loopback_4k_d35.mp4"))
    log(f"FILM 4K: {res['film']['4k_d35']['done'] or res['film']['4k_d35']['error']}")
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    log(f"results: {os.path.join(OUT, 'results.json')}")


if __name__ == "__main__":
    main()

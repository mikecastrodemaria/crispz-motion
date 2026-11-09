"""crispz-motion lot 0 - route C reference: the OFFICIAL FILM (TensorFlow) midpoint.

Runs with the Python of an official FILM install (frame-interpolation-main/.venv),
which on native Windows has no GPU (TensorFlow >= 2.11): it is the quality reference
for route A (the PyTorch port) and a timing of the CPU fallback.

    <film>/.venv/Scripts/python tools/film_tf_mid.py --film-dir <film> \
        --model Style a.png b.png out_mid.png

Prints one JSON line: {"ok": true, "load_s", "infer_s", "size"}.
"""
import os
import sys
import json
import time
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    ap.add_argument("--film-dir", required=True)
    ap.add_argument("--model", default="Style")
    a = ap.parse_args()
    sys.path.insert(0, a.film_dir)
    import numpy as np
    from eval import interpolator, util
    t0 = time.time()
    path = os.path.join(a.film_dir, "pretrained_models", "film_net", a.model, "saved_model")
    inter = interpolator.Interpolator(path, align=64)
    load_s = time.time() - t0
    x0 = util.read_image(a.a)
    x1 = util.read_image(a.b)
    dt = np.full((1,), 0.5, dtype=np.float32)
    t1 = time.time()
    mid = inter(x0[np.newaxis, ...], x1[np.newaxis, ...], dt)[0]
    infer_s = time.time() - t1
    util.write_image(a.out, mid)
    print(json.dumps({"ok": True, "load_s": round(load_s, 2), "infer_s": round(infer_s, 2),
                      "size": [int(x0.shape[1]), int(x0.shape[0])]}), flush=True)


if __name__ == "__main__":
    main()

"""crispz-motion - checks a rendered loop: size, frame count, black frames, jumps.

    film\\.venv\\Scripts\\python tools\\check_video.py video.mp4 [--loop]

Decodes the video with ffmpeg to small grey frames and measures the mean absolute
difference between consecutive frames (with --loop, also last -> first, the seam).
  - black: a frame whose mean is below 3 / 255;
  - repeat: a difference under 0.05, a frame shown twice (a stutter);
  - jump: a difference above 4 times the median, a cut the eye sees.
Prints one JSON line; exit code 1 when anything is found.
"""
import sys
import json
import argparse
import subprocess

import numpy as np

SMALL = 96


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=width,height,r_frame_rate", "-of", "json", a.video],
                        capture_output=True, text=True)
    st = json.loads(pr.stdout)["streams"][0]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", a.video, "-vf",
                          f"scale={SMALL}:{SMALL}:flags=area,format=gray", "-f", "rawvideo",
                          "-"], capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, SMALL, SMALL).astype(np.float32)
    seq = np.concatenate([fr, fr[:1]]) if a.loop else fr
    d = np.abs(np.diff(seq, axis=0)).mean(axis=(1, 2))
    med = float(np.median(d))
    res = {"video": a.video, "size": [st["width"], st["height"]], "fps": st["r_frame_rate"],
           "frames": int(len(fr)),
           "black": [int(i) for i in np.where(fr.mean(axis=(1, 2)) < 3)[0]],
           "repeats": [int(i) for i in np.where(d < 0.05)[0]],
           "jumps": [int(i) for i in np.where(d > 4 * max(med, 0.1))[0]],
           "diff_median": round(med, 3), "diff_max": round(float(d.max()), 3),
           # regularity of the steps: std / mean. A 1, 1, 2 stutter gives about 0.35.
           "step_cv": round(float(d.std() / max(d.mean(), 1e-6)), 3),
           "seam_diff": round(float(d[-1]), 3) if a.loop else None}
    print(json.dumps(res))
    return 1 if res["black"] or res["repeats"] or res["jumps"] else 0


if __name__ == "__main__":
    sys.exit(main())

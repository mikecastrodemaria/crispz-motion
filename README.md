# crispz-motion

Short videos, up to 4K, from a handful of key images made by the crispz family, with
FILM (Frame Interpolation for Large Motion, Google Research, ECCV 2022) computing every
frame in between. No video model: the images come from the crispz image engines through
the family CLI protocol v1, FILM makes the motion, ffmpeg writes the file.

> **Status: lot 0 (measurements).** The FILM worker and the benchmarks are in; the app
> (keys, timeline, UI) comes with lots 1 to 3.

## What is here

| File | Role |
|---|---|
| `cm_film_worker.py` | runs FILM (the PyTorch / TorchScript port of the official "Style" weights) over consecutive keys, recursive midpoint, one folder of frames per segment, JSON progress lines |
| `install_film.bat` | creates `film\.venv` (torch cu128) and downloads the FILM weights (138 MB) |
| `tools/bench_lot0.py` | FILM timings at 1 Mpx, 2 Mpx and 4K, fp32 / fp16, against the official TensorFlow FILM |
| `tools/bench_moss_keys.py` | keys from crispz-moss (Turbo LoRA, CFG 1) through the protocol, then FILM between them |
| `tools/film_tf_mid.py` | the official FILM midpoint (TensorFlow), the quality reference |

## Install (lot 0)

```bat
install_film.bat
film\.venv\Scripts\python cm_film_worker.py --model film\models\film_net_fp32.pt --times 3 --fp16 --out out a.png b.png
```

## Licences

- FILM weights and model: Apache 2.0 (google-research/frame-interpolation); PyTorch port
  and TorchScript export: Apache 2.0 (dajes/frame-interpolation-pytorch).
- Default image engine: crispz-moss (MOSS-Image, Apache 2.0) with the Turbo 4-step LoRA
  (akhaliq/MOSS-Image-SFT-4-steps, Apache 2.0).

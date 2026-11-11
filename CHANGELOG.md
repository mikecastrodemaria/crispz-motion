# Changelog

## 2026-11-09 - lot 1

- Projects on disk with keys, recipes and output settings (`cm_project.py`).
- Keys imported from files or generated through the engine: op gen, Turbo LoRA, 4 steps,
  CFG 1 (`cm_keys.py`, `cm_engines.py`).
- Retiming with exact frame times: blends two neighbouring FILM frames instead of taking
  the nearest one, which stuttered. Eases, holds, and once / ping-pong / closed loops
  that join without a repeated frame (`cm_timeline.py`).
- FILM worker: a jobs file, so only the segments missing from the cache are computed;
  each segment is written to a temporary folder then renamed.
- 4K: ESRGAN on every frame (4x-ClearRealityV1) with the resize on the GPU, about
  0.14 s per 4K frame (`cm_upscale_worker.py`).
- Exports mp4 (x264 / x265), webm, gif, and the make scripts in the project
  (`cm_video.py`).
- Command line (`cm_cli.py`): new, add, gen, info, render, quick, engine, status.
- install, start and update in .bat, .ps1 and .sh (`cm_setup.py`); start offers the
  update when GitHub has new commits. `install_film.bat` is replaced by install.
- `tools/check_video.py`; CPU tests in `tests/`.

## 2026-11-09 - lot 0

- FILM worker (PyTorch port, fp16) and the measurement scripts.

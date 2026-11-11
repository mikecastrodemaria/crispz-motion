# crispz-motion

Short videos, up to 4K, from a handful of key images made by the crispz family, with
FILM (Frame Interpolation for Large Motion, Google Research, ECCV 2022) computing every
frame in between. No video model: the keys come from the crispz image engines through
the family CLI protocol v1 (crispz-moss by default), FILM makes the motion, ESRGAN takes
the frames to 4K, ffmpeg writes the file.

> **Status: lot 1.** Projects, imported and generated keys, FILM, retiming, 4K, exports
> and the command line work. Next: the img2img key chain (lot 2), then the web page
> (lot 3).

## Install, start, update

| Windows (cmd) | Windows (PowerShell) | Linux / macOS |
|---|---|---|
| `install.bat` | `install.ps1` | `./install.sh` |
| `start.bat` | `start.ps1` | `./start.sh` |
| `update.bat` | `update.ps1` | `./update.sh` |

- **install** creates two venvs: `.venv` for the app (Pillow only) and `film/.venv` for
  FILM and ESRGAN (torch cu128, spandrel). It downloads the FILM weights (138 MB) and
  writes `config.json` from `config-sample.json`. Python 3.10 to 3.13.
- **start** installs on the first run. When GitHub has new commits it asks
  `Update first? [y/N]`; after 10 s without an answer it starts without updating. Its
  arguments go to `cm_cli.py`; with none, it prints the status.
- **update** runs `git pull --ff-only`, so git refuses to overwrite a file you changed.
  It reinstalls only the dependencies whose file changed, and checks that torch is
  still a CUDA build. `config.json` is never touched; new keys of the sample are listed.
  `--force-deps` (`-ForceDeps` in PowerShell) reinstalls them anyway.
- PowerShell may block scripts: `powershell -ExecutionPolicy Bypass -File start.ps1`.
- ffmpeg and ffprobe must be on the PATH (`winget install Gyan.FFmpeg`,
  `sudo apt install ffmpeg`, `brew install ffmpeg`).
- macOS has no CUDA: FILM and ESRGAN run on the CPU there, slowly.

## Command line

```bat
start.bat quick a.png b.png --seg 2 --res 4k --loop pingpong
start.bat new smoke --seg 1
start.bat add smoke k0.png k1.png k2.png
start.bat gen smoke --prompt "a pencil drawing of a man with a beard" --seed 2763806460
start.bat render smoke --res 1080p --fps 24 --loop closed --formats mp4,webm,gif
start.bat info smoke
start.bat engine start|stop|status
```

`PROJECT` is a folder, a folder name under `projects/`, or a project name (the newest
match wins). Options of `render` are saved in the project.

| Option | Values |
|---|---|
| `--res` | `1mpx`, `1080p` (= `2mpx`), `4k`, `work` (the key size), `WxH` |
| `--fps` | 24, 25, 30, 60 or any integer |
| `--loop` | `once`, `pingpong`, `closed` (adds a last-to-first segment, `--close` seconds) |
| `--seg` | re-spaces the keys every SEG seconds |
| `--formats` | `mp4`, `webm`, `gif` (comma list) |
| `--codec`, `--crf` | `x264` or `x265`, CRF 17 by default |
| `--upscaler` | `esrgan` (default above the key size) or `lanczos` |

A key also has a hold (frozen frames) and an ease for the segment it starts (`linear`,
`ease-in`, `ease-out`, `ease-in-out`): `add PROJECT img.png --hold 6 --ease ease-in-out`.

## How a render works

1. **Keys**: 2 or more images, all at the work size (the first key's size, at most
   2.1 Mpx). A key of another ratio is cropped at the centre, and the log says so. A
   generated key keeps its recipe (prompt, seed, the spec sent to the engine).
2. **FILM** runs recursive midpoints between each pair, in fp16. The network only
   answers t = 0.5 well, so it computes 2^N + 1 frames per segment, at least two per
   output frame.
3. **Retiming**: each output frame sits at its exact time. When that time falls between
   two FILM frames, the two are blended in proportion. Taking the nearest FILM frame
   instead made the motion stutter (steps of 1, 1, 2). A key is shown once: no doubled
   frame at a key, and the ping-pong and closed loops join without a repeat.
4. **Size**: above the work size, ESRGAN (4x-ClearRealityV1) runs on every frame, then
   a GPU resize to the exact size. No frame goes through diffusion, so nothing flickers.
   Below it, Lanczos.
5. **Export**: ffmpeg mp4 (x264 or x265, CRF 17, yuv420p), webm (VP9), gif (720 px).
   The project gets `makemp4.bat` / `.sh` and the others, to rebuild a file from
   `frames/` with other settings.

Everything is cached by content. A FILM segment is named after the hash of its two key
images and its settings, so changing a key redoes only its two segments, and changing
only the output size redoes only ESRGAN.

The GPU is shared: an engine that crispz-motion started is stopped before FILM. An
engine you started yourself is never touched.

## Projects

```
projects/<date>-<name>/
  project.json         keys, recipes, output settings, last render
  keys/  raw/          the keys at the work size, what the engine returned
  film/  blend/  final/   caches: FILM frames, blended frames, frames at output size
  frames/              the output frames in order (hard links)
  video.mp4  video.webm  video.gif  make*.bat  make*.sh  render.log
```

## Files

| File | Role |
|---|---|
| `cm_cli.py` | the command line |
| `cm_render.py` | one render: plan, FILM, retiming, size, checks, export |
| `cm_timeline.py` | retiming: frame times, eases, holds, loops, output sizes |
| `cm_project.py` | a project on disk, its keys and recipes |
| `cm_keys.py` | keys: import, generation through the engine |
| `cm_engines.py` | CLI protocol v1 client and launcher of the family apps |
| `cm_film.py` | drives the two GPU workers, FILM cache, ESRGAN model lookup |
| `cm_film_worker.py` | FILM (PyTorch port), runs in `film/.venv` |
| `cm_upscale_worker.py` | ESRGAN on every frame, runs in `film/.venv` |
| `cm_video.py` | ffmpeg exports and the make scripts |
| `cm_setup.py` | install, update, update check (used by the shell scripts) |
| `cm_core.py` | config, log, atomic JSON |
| `tools/check_video.py` | checks a video: black frames, repeats, jumps, step regularity |
| `tools/bench_*.py` | lot 0 measurements |

Tests (CPU, no GPU): `film\.venv\Scripts\python -m pytest tests`.

## Licences

- FILM weights and model: Apache 2.0 (google-research/frame-interpolation); PyTorch port
  and TorchScript export: Apache 2.0 (dajes/frame-interpolation-pytorch).
- Default image engine: crispz-moss (MOSS-Image, Apache 2.0) with the Turbo 4-step LoRA
  (akhaliq/MOSS-Image-SFT-4-steps, Apache 2.0).
- 4x-ClearRealityV1: see its own licence; any spandrel ESRGAN model can be set in
  `config.json`.

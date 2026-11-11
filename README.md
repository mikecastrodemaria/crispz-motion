# crispz-motion

Short videos, up to 4K, from a handful of key images made by the crispz family, with
FILM (Frame Interpolation for Large Motion, Google Research, ECCV 2022) computing every
frame in between. No video model: the keys come from the crispz image engines through
the family CLI protocol v1 (crispz-moss by default), FILM makes the motion, ESRGAN takes
the frames to 4K, ffmpeg writes the file.

> **Status: lot 2.** Projects, keys imported or made by the family (generation, img2img
> chain, edit, inpaint, assisted middle key, regeneration), distance guard, FILM,
> retiming, 4K, exports and the command line work. Next: the web page (lot 3).

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

### Keys made by the family

```bat
start.bat gen smoke --prompt "a pencil drawing of a man ..." --seed 2763806460
start.bat chain smoke --count 10 --denoise 0.625
start.bat chain smoke --count 3 --prompts prompts.txt
start.bat edit smoke "turn his head to the left"
start.bat inpaint smoke --mask mouth.png --prompt "open mouth, smoke coming out"
start.bat guard smoke
start.bat mid smoke k003 k004
start.bat regen smoke k002 --seed 12 --cascade
start.bat remove smoke k005
```

| Command | Engine op | What it makes |
|---|---|---|
| `gen` | gen | a key from a prompt (Turbo LoRA, 4 steps, CFG 1) |
| `chain` | upscale, factor 1 | the "sparse loopback": each new key is the previous one passed again through img2img at `--denoise`, seed + 1. `--prompts` gives one prompt per new key (the last repeats) |
| `edit` | edit | the next key by instruction (Moss-Image-Edit LoRA, 50 steps, CFG 4) |
| `inpaint` | inpaint | the next key with a masked area redrawn (white = redraw) |
| `mid` | FILM + upscale | the assisted middle key: FILM's midpoint of two keys, cleaned by img2img at a low denoise (0.2), placed halfway |
| `regen` | the key's own | the key made again from its recipe and the current image of its source key; `--cascade` then redoes the keys built on it |
| `guard` | none | the distance between consecutive keys (see below) |

The denoise of an img2img key is a fraction of the steps the engine plays, so it moves
in steps: with 4 steps, 0.35 and 0.50 both play 2 and give the same image. Chain and mid
keys therefore run 8 steps (`chain.steps` in `config.json`); the log says the effective
value when it differs from the one asked.

**What moves the subject** (measured on lot 2, from the prompt of the historic example
1): the chain's denoise makes the image live (0.625 to 0.75 keeps the character, 0.875
loses it); a new prompt alone barely moves the pose at those values; an edit is what
turns a head or opens a mouth, at about 20 times the cost of a chain key. Name the style
in the instruction ("turn his head to the left, keep the black and white pencil drawing
style"), or the edit may change it.

**Licence of edit keys.** The Moss-Image-Edit LoRA states no licence for its weights. An
edit key says so in the log and in its recipe (`"client": false`, shown by `info`): not
for client work.

**Distance guard.** Before FILM, the render scores each pair of consecutive keys (mean
difference of 64 px grey thumbnails, 0 to 255). Above `guard.warn` (12) it warns, above
`guard.risk` (25) it advises a middle key: FILM fills a large gap with ghosts instead of
motion. The guard never blocks a render.

**Order of the 4K step.** `--order film_first` (default) runs FILM at the work size and
ESRGAN on every frame. `--order keys_first` has the engine upscale the keys (ESRGAN plus a
light Turbo refine) and runs FILM at the output size: sharper keys, softer in-betweens.

An engine started by crispz-motion keeps running between commands (so a chain of
commands does not reload the model), and a render stops it before FILM. `start.bat engine
stop` stops it by hand; an engine you started yourself is never touched.

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
| `cm_keys.py` | keys: import, gen, chain, edit, inpaint, mid, regeneration |
| `cm_guard.py` | distance guard between consecutive keys |
| `cm_engines.py` | CLI protocol v1 client and launcher of the family apps |
| `cm_film.py` | drives the two GPU workers, FILM cache, ESRGAN model lookup |
| `cm_film_worker.py` | FILM (PyTorch port), runs in `film/.venv` |
| `cm_upscale_worker.py` | ESRGAN on every frame, runs in `film/.venv` |
| `cm_video.py` | ffmpeg exports and the make scripts |
| `cm_setup.py` | install, update, update check (used by the shell scripts) |
| `cm_core.py` | config, log, atomic JSON |
| `tools/check_video.py` | checks a video: black frames, repeats, jumps, step regularity |
| `tools/bench_*.py` | measurements of lots 0 and 2 |

Tests (CPU, no GPU): `film\.venv\Scripts\python -m pytest tests`.

## Licences

- FILM weights and model: Apache 2.0 (google-research/frame-interpolation); PyTorch port
  and TorchScript export: Apache 2.0 (dajes/frame-interpolation-pytorch).
- Default image engine: crispz-moss (MOSS-Image, Apache 2.0) with the Turbo 4-step LoRA
  (akhaliq/MOSS-Image-SFT-4-steps, Apache 2.0).
- 4x-ClearRealityV1: see its own licence; any spandrel ESRGAN model can be set in
  `config.json`.

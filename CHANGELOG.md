# Changelog

## 2026-11-11 - lot 3

- The web page (`cm_server.py`, `assets/index.html`): projects (open, duplicate, delete
  to `projects/.trash`), the keys strip with the distance guard and drag-and-drop
  reorder, a key's time / hold / ease / recipe, regeneration, the next key by chain, edit
  or inpaint (mask painted on the image, soft edge), middle keys, image import and
  generation, the output settings, the render with progress per step, preview, player,
  log and Cancel. One job at a time, in the background.
- `start` with no argument opens the web page; with arguments it runs the command line.
- Import of the A1111 Animator settings (`cm_animator.py`): `settings.txt` and
  `keyframes.csv` become a plan of keys one every second (gen, then img2img chain), with
  every change written in the plan's notes; the plan is edited on the page, then made.
- Cancel: FILM and ESRGAN workers are killed at once, keys stop between two engine calls.
- Output sizes fit inside the 16:9 frame turned like the keys: square keys give 2160 x
  2160 in 4K and 1080 x 1080 in 1080p (were 3840 and 1920 on the long side).
- `--res 1mpx` works (it was read as a WxH size and refused).
- Distance guard: thumbnails cached by file and date.
- CPU tests `tests/test_animator.py`, `tests/test_server.py`.

## 2026-11-11 - lot 2

- Keys made by the family: `chain` (img2img from the previous key, seed + 1, one prompt
  per key optional), `edit` (Moss-Image-Edit LoRA, marked not for client work), `inpaint`,
  `mid` (FILM's midpoint cleaned by img2img), `regen` (from the recipe and the current
  image of the source key, `--cascade` redoes the keys built on it), `remove`.
- Recipes keep the spec sent and the id of the source key.
- Distance guard (`cm_guard.py`, `guard` command): scores consecutive keys before FILM,
  advises a middle key above the risk level, never blocks.
- Effective denoise: img2img keys run 8 steps (`chain.steps`); the log says the value the
  engine really plays.
- Render option `--order keys_first`: the engine upscales the keys, FILM runs at the
  output size (`film_first`, the default, unchanged).
- FILM worker: inputs in channels_last, 7 times faster on a padded 4K key (0.5 s per frame
  instead of 3.5), 0.11 s per frame at 1088 x 1920; pairs of different sizes accepted in
  one jobs file (each pair keeps one size).
- `tools/bench_lot2.py`, `tools/bench_lot2_tools.ps1`; CPU tests `tests/test_keys_lot2.py`.
- Needs crispz-moss with whole-image variations (the protocol's upscale at factor 1 no
  longer follows the 4K tile rules).

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

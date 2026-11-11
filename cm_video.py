"""crispz-motion - exports with ffmpeg, and the make*.bat / make*.sh beside them.

The frames are frames/f00000.png ... at the output size. Like the historic A1111
outputs, the project gets one script per format that rebuilds the file from frames/
(edit the CRF, run it again).

  mp4  : x264 (or x265), CRF 17, yuv420p, +faststart
  webm : VP9, CRF 32 by default, yuv420p
  gif  : 720 px wide by default, one palette for the whole clip
"""

import os
import subprocess

from cm_core import log


def _args(fmt, fps, o):
    src = ["-framerate", str(fps), "-i", os.path.join("frames", "f%05d.png")]
    if fmt == "mp4":
        if o.get("codec") == "x265":
            v = ["-c:v", "libx265", "-crf", str(o["crf"]), "-preset", o["preset"],
                 "-tag:v", "hvc1"]
        else:
            v = ["-c:v", "libx264", "-crf", str(o["crf"]), "-preset", o["preset"]]
        return src + v + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", "video.mp4"]
    if fmt == "webm":
        return src + ["-c:v", "libvpx-vp9", "-crf", str(o["webm_crf"]), "-b:v", "0",
                      "-row-mt", "1", "-pix_fmt", "yuv420p", "video.webm"]
    if fmt == "gif":
        vf = (f"scale={int(o['gif_width'])}:-2:flags=lanczos,split[a][b];"
              f"[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a")
        return src + ["-vf", vf, "-loop", "0", "video.gif"]
    raise ValueError(f"unknown format '{fmt}' (mp4, webm, gif)")


def _quote(a):
    return f'"{a}"' if any(c in a for c in ' ;,[]=') else a


def write_scripts(project_dir, fps, o, ffmpeg="ffmpeg"):
    for fmt in ("mp4", "webm", "gif"):
        args = " ".join(_quote(a) for a in _args(fmt, fps, o))
        bat = (f"@echo off\r\nREM crispz-motion: rebuilds video.{fmt} from frames\\ "
               f"(edit the options, run again)\r\ncd /d \"%~dp0\"\r\n"
               f"{ffmpeg} -y {args.replace('%05d', '%%05d')}\r\n")
        sh = (f"#!/usr/bin/env sh\n# crispz-motion: rebuilds video.{fmt} from frames/\n"
              f"cd \"$(dirname \"$0\")\"\n{ffmpeg} -y {args.replace(os.sep, '/')}\n")
        with open(os.path.join(project_dir, f"make{fmt}.bat"), "w", encoding="utf-8",
                  newline="") as fh:
            fh.write(bat)
        with open(os.path.join(project_dir, f"make{fmt}.sh"), "w", encoding="utf-8",
                  newline="\n") as fh:
            fh.write(sh)


def export(project_dir, fps, formats, o, ffmpeg="ffmpeg"):
    """Returns {fmt: path} of the files written; raises on an ffmpeg failure."""
    out = {}
    for fmt in formats:
        cmd = [ffmpeg, "-y", "-v", "error"] + _args(fmt, fps, o)
        p = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg {fmt} failed: {p.stderr.strip()[-800:]}")
        path = os.path.join(project_dir, f"video.{fmt}")
        out[fmt] = path
        log(f"export {fmt}: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    write_scripts(project_dir, fps, o, ffmpeg)
    return out


def probe(path, ffprobe="ffprobe"):
    """(width, height, frame count, duration s) of a video, through ffprobe."""
    p = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=width,height,nb_read_frames:format=duration",
                        "-of", "default=nw=1", path], capture_output=True, text=True)
    vals = dict(line.split("=", 1) for line in p.stdout.split() if "=" in line)
    return (int(vals.get("width", 0)), int(vals.get("height", 0)),
            int(vals.get("nb_read_frames", 0) or 0), float(vals.get("duration", 0) or 0))

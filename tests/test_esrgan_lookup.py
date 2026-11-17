"""Where the ESRGAN model is looked up: config esrgan_dir when set, else the app's
models/esrgan (the install puts it there), else the engine's ESRGAN folder."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cm_film

NAME = "4x-test.safetensors"


def cfg(tmp_path, esrgan_dir=""):
    eng = tmp_path / "engine"
    (eng / "upscale_models").mkdir(parents=True, exist_ok=True)
    return {"esrgan_dir": esrgan_dir, "esrgan_model": NAME, "engine": "moss",
            "engines": {"moss": {"dir": str(eng)}}}


def test_lookup_order(tmp_path, monkeypatch):
    own = tmp_path / "own"
    own.mkdir()
    monkeypatch.setattr(cm_film, "ESRGAN_OWN", str(own))
    c = cfg(tmp_path)
    p, why = cm_film.esrgan_model(c)
    assert p is None and "not found" in why
    engine_file = tmp_path / "engine" / "upscale_models" / NAME
    engine_file.write_bytes(b"x")
    assert cm_film.esrgan_model(c)[0] == str(engine_file)
    (own / NAME).write_bytes(b"x")
    assert cm_film.esrgan_model(c)[0] == str(own / NAME)          # the app's own first
    custom = tmp_path / "custom"
    custom.mkdir()
    c = cfg(tmp_path, str(custom))
    assert cm_film.esrgan_model(c)[0] is None                       # esrgan_dir only
    (custom / NAME).write_bytes(b"x")
    assert cm_film.esrgan_model(c)[0] == str(custom / NAME)

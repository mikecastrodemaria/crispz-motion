"""Lot 3 on CPU: the web server's API, on a temporary projects folder (no engine, no
GPU). One kept-alive connection on purpose: a POST body left unread used to be taken as
the start of the next request."""
import http.client
import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from PIL import Image

import cm_core
import cm_server
from test_animator import settings, keyframes


@pytest.fixture()
def server(tmp_path):
    cfgfile = tmp_path / "config.json"
    cfgfile.write_text(json.dumps({
        "projects_dir": str(tmp_path / "projects"),
        "engines": {"moss": {"dir": "", "url": "http://127.0.0.1:9", "launch": False}}}))
    app = cm_server.App(cm_core.load_config(str(cfgfile)))
    cm_server.Handler.app = app
    srv = ThreadingHTTPServer(("127.0.0.1", 0), cm_server.Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=20)
    yield conn, tmp_path
    conn.close()
    srv.shutdown()
    srv.server_close()


def call(conn, method, path, body=None, raw=None, headers=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    conn.request(method, path, body=data, headers=headers or {})
    r = conn.getresponse()
    out = r.read()
    if r.getheader("Content-Type", "").startswith("application/json"):
        return r.status, json.loads(out)
    return r.status, out


def png(path, level, size=(96, 64)):
    Image.new("RGB", size, (level,) * 3).save(path)
    return path.read_bytes()


def test_project_life_on_one_connection(server):
    conn, tmp = server
    st, s = call(conn, "GET", "/api/status")
    assert st == 200 and s["engines"][0]["running"] is False
    st, r = call(conn, "POST", "/api/projects", {"name": "web test"})
    pid = r["id"]
    for i, lv in enumerate((40, 90, 140)):
        st, r = call(conn, "POST", f"/api/p/{pid}/upload?name=k{i}.png",
                     raw=png(tmp / f"k{i}.png", lv))
        assert st == 200, r
    st, v = call(conn, "GET", f"/api/p/{pid}")
    assert [k["id"] for k in v["keys"]] == ["k000", "k001", "k002"]
    assert v["work_size"] == [96, 64] and len(v["guard"]) == 2
    assert v["info"]["frames"] > 0

    # a POST whose body the route does not use, then a GET on the same connection
    st, r = call(conn, "POST", "/api/job/cancel", {"unused": True})
    assert st == 200 and r["ok"] is False
    st, r = call(conn, "GET", "/api/projects")
    assert st == 200 and r[0]["id"] == pid

    st, r = call(conn, "POST", f"/api/p/{pid}/order", {"ids": ["k002", "k000", "k001"]})
    st, v = call(conn, "GET", f"/api/p/{pid}")
    assert [(k["id"], k["t"]) for k in v["keys"]] == [("k002", 0.0), ("k000", 1.0),
                                                      ("k001", 2.0)]
    st, r = call(conn, "POST", f"/api/p/{pid}/key/k001", {"t": 1.0})
    assert st == 400 and "already" in r["error"]
    st, r = call(conn, "POST", f"/api/p/{pid}/key/k001", {"hold": 4, "ease": "ease-in"})
    st, r = call(conn, "POST", f"/api/p/{pid}/output",
                 {"res": "work", "loop": "closed", "formats": ["mp4", "gif"], "total": 5})
    assert st == 200, r
    st, v = call(conn, "GET", f"/api/p/{pid}")
    assert v["keys"][-1]["t"] == 5.0 and v["output"]["loop"] == "closed"
    assert v["keys"][-1]["hold"] == 4 and len(v["guard"]) == 3     # closing pair too
    st, r = call(conn, "POST", f"/api/p/{pid}/output", {"formats": []})
    assert st == 400

    # files: a thumbnail, a key with a Range request
    st, b = call(conn, "GET", f"/p/{pid}/thumb/k000?w=64")
    assert st == 200 and b[:2] == b"\xff\xd8"
    conn.request("GET", f"/p/{pid}/keys/k000.png", headers={"Range": "bytes=0-3"})
    r = conn.getresponse()
    assert r.status == 206 and r.read() == b"\x89PNG"
    st, r = call(conn, "GET", f"/p/{pid}/../../config.json")
    assert st in (403, 404)

    st, r = call(conn, "POST", f"/api/p/{pid}/remove/k000", {})
    st, v = call(conn, "GET", f"/api/p/{pid}")
    assert len(v["keys"]) == 2

    st, r = call(conn, "POST", f"/api/projects/{pid}/duplicate", {})
    dup = r["id"]
    st, v = call(conn, "GET", f"/api/p/{dup}")
    assert len(v["keys"]) == 2 and v["last_render"] is None
    st, r = call(conn, "POST", f"/api/projects/{dup}/trash", {})
    assert st == 200 and ".trash" in r["trash"]
    st, r = call(conn, "GET", "/api/projects")
    assert [p["id"] for p in r] == [pid]


def test_animator_import_makes_a_planned_project(server):
    conn, tmp = server
    st, r = call(conn, "POST", "/api/animator", {"settings": "{bad", "name": "x"})
    assert st == 400
    st, r = call(conn, "GET", "/api/projects")
    assert r == []                                   # a bad file makes no empty project
    st, r = call(conn, "POST", "/api/animator", {"settings": settings(),
                                                 "keyframes": keyframes(), "name": "ex1"})
    assert st == 200
    pid = r["id"]
    st, v = call(conn, "GET", f"/api/p/{pid}")
    assert v["pending"] == 3 and len(v["plan"]["keys"]) == 3
    assert v["output"]["fps"] == 5
    st, r = call(conn, "POST", f"/api/p/{pid}/plan",
                 {"keys": [{"i": 1, "prompt": "another", "denoise": "0.75"}],
                  "all_prompt": ""})
    st, v = call(conn, "GET", f"/api/p/{pid}")
    assert v["plan"]["keys"][1]["prompt"] == "another"
    assert v["plan"]["keys"][1]["denoise"] == 0.75
    st, r = call(conn, "POST", f"/api/p/{pid}/plan", {"all_denoise": 0.5})
    st, v = call(conn, "GET", f"/api/p/{pid}")
    assert [k["denoise"] for k in v["plan"]["keys"]] == [None, 0.5, 0.5]


def test_a_job_needs_a_known_kind_and_a_prompt(server):
    conn, tmp = server
    st, r = call(conn, "POST", "/api/projects", {"name": "jobs"})
    pid = r["id"]
    st, r = call(conn, "POST", "/api/job", {"project": pid, "kind": "nope"})
    assert st == 400
    st, r = call(conn, "POST", "/api/job", {"project": pid, "kind": "gen", "prompt": " "})
    assert st == 400 and "prompt" in r["error"]
    st, r = call(conn, "POST", "/api/job", {"project": "../x", "kind": "render"})
    assert st == 400

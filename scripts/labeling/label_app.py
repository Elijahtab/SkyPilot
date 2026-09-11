"""
Local review app for the size-gated Kaggle crops. Serves a grid of pre-rendered
tiles and records one class decision per crop.

Complements scripts/labeling/manual_label.py: that tool draws boxes one image at
a time (OpenCV, desktop); this one classifies existing boxes 60 to a page in a
browser. Same 7-class schema, same 1-7 key order, different output path.

Deliberately stdlib-only (http.server + sqlite3): no pip install, no build step,
runs offline on the machine that already holds the data. The multi-user version
in docs/labeling-web-app-research.md is NOT justified for this corpus -- at the
48px gate the whole job is ~1,313 crops, about 25-40 minutes for one person, and
recruiting/calibrating volunteers costs far more than that. Build the web
version for the next, higher-resolution corpus.

What the app is, precisely: a fixed set of immutable crops and a table of which
key was pressed on each. Nobody draws, moves or resizes a box. So there is no
canvas, no geometry, no image editing.

Decisions are an APPEND-ONLY sqlite log -- every keypress is a new row, so the
session is resumable, every change is auditable, and undo is just "ignore the
last row". Current label = latest row per crop.

⚠ The manifest is read ONCE at startup. If build_crops.py is re-run while this
  is serving, the browser keeps writing the OLD schema's class ids into the new
  store. That silently corrupted 53 decisions on 2026-09-08 (id 3 flipped from
  Truck to SUV). Restart the app after any rebuild.

Usage:
    python scripts/labeling/label_app.py              # http://127.0.0.1:8000
    python scripts/labeling/label_app.py --port 8080
    python scripts/labeling/label_app.py --user luke
    python scripts/labeling/label_app.py --progress   # print status, don't serve
"""

import argparse
import json
import mimetypes
import sqlite3
import sys
import threading
import time
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _paths import KAGGLE, REPO

REVIEW   = REPO / "Labeling" / "review"
CROPS    = REVIEW / "crops"
MANIFEST = REVIEW / "manifest.json"
DB       = REVIEW / "decisions.sqlite"
INDEX    = Path(__file__).with_name("label_app.html")


def db_connect():
    con = sqlite3.connect(DB, check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            row_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_id  TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            user     TEXT NOT NULL,
            ts       REAL NOT NULL,
            active   INTEGER NOT NULL DEFAULT 1
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_crop ON decisions(crop_id)")
    con.commit()
    return con


def current_labels(con):
    """Latest active decision per crop."""
    rows = con.execute("""
        SELECT crop_id, class_id FROM decisions d
        WHERE active = 1 AND row_id = (
            SELECT MAX(row_id) FROM decisions
            WHERE crop_id = d.crop_id AND active = 1)
    """).fetchall()
    return dict(rows)


class Handler(BaseHTTPRequestHandler):
    manifest = None
    con = None
    lock = threading.Lock()
    user = "local"

    def log_message(self, *a):
        pass                                    # keep the console readable

    def _send(self, code, body, ctype="application/json", cache=False):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # tiles are immutable once built; the page is not
        self.send_header("Cache-Control",
                         "public, max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, cache=False):
        if not path.is_file():
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype, cache=cache)

    # -- GET ---------------------------------------------------------
    def do_GET(self):
        route = unquote(urlparse(self.path).path)

        if route in ("/", "/index.html"):
            return self._file(INDEX)

        if route == "/api/manifest":
            with self.lock:
                labels = current_labels(self.con)
            return self._send(200, {**self.manifest, "labels": labels,
                                    "user": self.user})

        if route.startswith("/crop/"):
            name = Path(route[len("/crop/"):]).name       # no traversal
            return self._file(CROPS / name, cache=True)

        if route.startswith("/frame/"):
            # full 416 frame for the context view: /frame/<split>/<image name>
            parts = route[len("/frame/"):].split("/", 1)
            if len(parts) != 2 or parts[0] not in ("valid", "test"):
                return self._send(404, {"error": "bad frame path"})
            return self._file(KAGGLE / parts[0] / "images" / Path(parts[1]).name,
                              cache=True)

        return self._send(404, {"error": "no route"})

    # -- POST --------------------------------------------------------
    def do_POST(self):
        route = unquote(urlparse(self.path).path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad json"})

        if route == "/api/decide":
            # [{id, class_id}, ...] -- one append-only row each
            items = payload.get("items") or []
            now = time.time()
            rows = [(i["id"], int(i["class_id"]), self.user, now)
                    for i in items if "id" in i and "class_id" in i]
            if not rows:
                return self._send(400, {"error": "no items"})
            with self.lock:
                self.con.executemany(
                    "INSERT INTO decisions (crop_id, class_id, user, ts) "
                    "VALUES (?,?,?,?)", rows)
                self.con.commit()
                done = len(current_labels(self.con))
            return self._send(200, {"ok": True, "written": len(rows), "done": done})

        if route == "/api/undo":
            # deactivate the most recent batch of rows; the log keeps them
            n = int(payload.get("count") or 0)
            with self.lock:
                if n > 0:
                    ids = [r[0] for r in self.con.execute(
                        "SELECT row_id FROM decisions WHERE active = 1 "
                        "ORDER BY row_id DESC LIMIT ?", (n,)).fetchall()]
                    if ids:
                        self.con.executemany(
                            "UPDATE decisions SET active = 0 WHERE row_id = ?",
                            [(i,) for i in ids])
                        self.con.commit()
                labels = current_labels(self.con)
            return self._send(200, {"ok": True, "labels": labels,
                                    "done": len(labels)})

        return self._send(404, {"error": "no route"})


def print_progress(manifest, con):
    labels = current_labels(con)
    crops = manifest["crops"]
    classes = manifest["classes"]
    print(f"\n  {len(labels)} / {len(crops)} crops decided "
          f"({len(labels) / max(len(crops), 1) * 100:.1f}%)")
    if labels:
        dist = Counter(classes[c] for c in labels.values())
        print(f"  decided distribution: {dict(dist)}")
    if manifest.get("has_prior"):
        prior = {c["id"]: c["prior"] for c in crops}
        both = [(prior[k], v) for k, v in labels.items() if k in prior]
        if both:
            agree = sum(p == v for p, v in both) / len(both)
            print(f"  agreement with {manifest['prior_model']} prior: "
                  f"{agree:.3f} over {len(both)} crops")
            print("    (this is the pre-fill bias number -- an agreement near "
                  "1.000\n     means the pass mostly rubber-stamped the model)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--user", default="local", help="tagged on every decision row")
    ap.add_argument("--progress", action="store_true", help="print status and exit")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not MANIFEST.exists():
        sys.exit("[ERR] no manifest. Run first:\n"
                 "  .\\myenv\\Scripts\\python.exe scripts\\labeling\\build_crops.py")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    con = db_connect()

    if args.progress:
        return print_progress(manifest, con)

    Handler.manifest = manifest
    Handler.con = con
    Handler.user = args.user

    url = f"http://{args.host}:{args.port}"
    n = len(manifest["crops"])
    pages = (n + manifest["page_size"] - 1) // manifest["page_size"]
    print(f"\n  Review corpus : {n} crops >= {manifest['min_size_px']}px, "
          f"{pages} pages")
    print(f"  Decisions     : {DB}")
    print(f"  Serving       : {url}")
    print("  Classes       : " + "  ".join(
        f"{i + 1}={n}" for i, n in enumerate(manifest["classes"])))
    print("\n  Keys: 1-7 assign | Space accept page | u undo | arrows page | "
          "click tile select")
    print("  Ctrl-C to stop. Progress is saved after every keypress.\n")

    if not args.no_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
        print_progress(manifest, con)
        print("\n  Export when done:\n"
              "    .\\myenv\\Scripts\\python.exe scripts\\labeling\\export_labels.py")


if __name__ == "__main__":
    main()

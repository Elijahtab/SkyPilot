"""
Run the two-stage pipeline over frames and store every detection in a
searchable SQLite index. Search it with search_vehicles.py.

    python scripts/pipeline/index_frames.py <dir or image> [...]
    python scripts/pipeline/index_frames.py --list frames.txt
    python scripts/pipeline/index_frames.py images_dir --db preds/my_index.sqlite --reindex

Frames already in the index are skipped unless --reindex, so an interrupted run
resumes. The index stores boxes, not crops: search re-crops from the frame on
disk, so moving the frames breaks the sheets but not the results.
"""

import argparse
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _paths import REPO
from pipeline.two_stage import TYPE_CONF, TwoStagePipeline

DEFAULT_DB = REPO / "preds" / "vehicle_index.sqlite"
EXTS = (".jpg", ".jpeg", ".png")

SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    frame_id   INTEGER PRIMARY KEY,
    path       TEXT UNIQUE NOT NULL,
    width      INTEGER, height INTEGER,
    indexed_at REAL, models TEXT
);
CREATE TABLE IF NOT EXISTS detections (
    det_id      INTEGER PRIMARY KEY,
    frame_id    INTEGER NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    det_conf    REAL,
    size_px     REAL,             -- box long side, native px
    type        TEXT NOT NULL,    -- a TYPE_CLASSES name, or 'Vehicle'
    type_status TEXT NOT NULL,    -- typed | unsure | too_small
    type_guess  TEXT,             -- classifier top-1, even when unsure
    type_conf   REAL,
    color       TEXT,             -- NULL below the colour gate
    color_conf  REAL
);
CREATE INDEX IF NOT EXISTS idx_det_type_color ON detections(type, color);
CREATE INDEX IF NOT EXISTS idx_det_frame ON detections(frame_id);
"""


def gather(sources, list_file):
    paths = []
    if list_file:
        paths += [Path(l.strip()) for l in Path(list_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    for s in sources:
        p = Path(s)
        paths += sorted(q for q in p.rglob("*") if q.suffix.lower() in EXTS) if p.is_dir() else [p]
    return [p.resolve() for p in paths if p.suffix.lower() in EXTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*", help="image files or directories (recursive)")
    ap.add_argument("--list", help="text file with one image path per line")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--type-conf", type=float, default=TYPE_CONF)
    ap.add_argument("--reindex", action="store_true", help="re-run frames already indexed")
    args = ap.parse_args()

    frames = gather(args.sources, args.list)
    if not frames:
        sys.exit("[ERR] no images given; pass directories/files or --list")

    db = Path(args.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    done = {r[0] for r in con.execute("SELECT path FROM frames")}
    todo = frames if args.reindex else [f for f in frames if str(f) not in done]
    print(f"{len(frames)} frames given, {len(todo)} to index -> {db}")
    if not todo:
        return

    pipe = TwoStagePipeline(type_conf=args.type_conf)
    status, types, colors, t0 = Counter(), Counter(), Counter(), time.time()
    for n, f in enumerate(todo, 1):
        dets, (w, h) = pipe(f)
        with con:
            con.execute("DELETE FROM frames WHERE path = ?", (str(f),))     # cascades on --reindex
            fid = con.execute("INSERT INTO frames (path, width, height, indexed_at, models) "
                              "VALUES (?, ?, ?, ?, ?)", (str(f), w, h, time.time(), pipe.versions)).lastrowid
            con.executemany(
                "INSERT INTO detections (frame_id, x1, y1, x2, y2, det_conf, size_px, type, type_status, "
                "type_guess, type_conf, color, color_conf) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(fid, *d["box"], d["det_conf"], d["size_px"], d["type"], d["type_status"],
                  d["type_guess"], d["type_conf"], d["color"], d["color_conf"]) for d in dets])
        for d in dets:
            status[d["type_status"]] += 1
            types[d["type"]] += 1
            colors[d["color"] or "(too small)"] += 1
        if n % 50 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} frames  {sum(status.values())} vehicles  "
                  f"{(time.time() - t0) / n * 1000:.0f} ms/frame")
    con.close()

    total = sum(status.values())
    print(f"\n  {total} vehicles in {len(todo)} frames")
    print(f"  typed {status['typed']} ({status['typed'] / max(total, 1):.0%}), "
          f"unsure {status['unsure']}, too small to type {status['too_small']}")
    print(f"  types : {dict(types.most_common())}")
    print(f"  colors: {dict(colors.most_common())}")
    print(f'\nNext:  .\\myenv\\Scripts\\python.exe scripts\\pipeline\\search_vehicles.py "red suv"')


if __name__ == "__main__":
    main()

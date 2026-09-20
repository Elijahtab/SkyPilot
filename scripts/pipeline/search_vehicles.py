"""
Search the vehicle index by colour and type: "red suv", "white van", "black car".

    python scripts/pipeline/search_vehicles.py "red suv"
    python scripts/pipeline/search_vehicles.py "white van" --limit 30
    python scripts/pipeline/search_vehicles.py "blue" --min-type-conf 0.8

Writes a contact sheet of the matches to preds/search/<query>.jpg.

Query words are matched against a small vocabulary (COLOR_WORDS, TYPE_WORDS);
unknown words are reported, never silently ignored. "car" means any passenger
car -- SUV or Standard Car -- because that is what people mean by it; "sedan"
means Standard Car only. A query with only a colour searches every type.

Detections the pipeline could not type (too small, or classifier unsure) are
stored as 'Vehicle' and are excluded from a typed query; the count of those that
match the colour is printed so they are not invisible.
"""

import argparse
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageOps

from _color import COLORS
from _crops import TYPE_CLASSES, square_crop
from _paths import REPO
from pipeline.index_frames import DEFAULT_DB

OUT = REPO / "preds" / "search"

COLOR_WORDS = {c: c for c in COLORS} | {
    "grey": "gray", "silver": "gray", "maroon": "red", "burgundy": "red", "navy": "blue",
    "tan": "brown", "beige": "brown", "gold": "brown", "champagne": "brown", "bronze": "brown",
}
TYPE_WORDS = {
    "suv": ["SUV"], "crossover": ["SUV"], "jeep": ["SUV"],
    "car": ["SUV", "Standard Car"], "sedan": ["Standard Car"], "coupe": ["Standard Car"],
    "hatchback": ["Standard Car"],
    "truck": ["Truck"], "pickup": ["Truck"], "lorry": ["Truck"], "semi": ["Truck"],
    "van": ["Van"], "minivan": ["Van"],
    "bus": ["Bus"], "coach": ["Bus"],
    "motorcycle": ["Motorcycle"], "motorbike": ["Motorcycle"], "bike": ["Motorcycle"],
    "scooter": ["Motorcycle"], "moped": ["Motorcycle"],
    "vehicle": list(TYPE_CLASSES),
}


def parse(query):
    colors, types, unknown = set(), set(), []
    for w in re.findall(r"[a-z]+", query.lower()):
        stem = w[:-1] if w.endswith("s") and w[:-1] in TYPE_WORDS else w   # "suvs", "vans"
        if w in COLOR_WORDS:
            colors.add(COLOR_WORDS[w])
        elif stem in TYPE_WORDS:
            types.update(TYPE_WORDS[stem])
        else:
            unknown.append(w)
    return colors, types, unknown


def fmt(v):
    return "--" if v is None else f"{v:.2f}"


def sheet(rows, out, tile=128, cols=8):
    im_cache = {}
    s = Image.new("RGB", (cols * tile, ((len(rows) + cols - 1) // cols) * (tile + 28)), "white")
    d = ImageDraw.Draw(s)
    for j, r in enumerate(rows):
        if r["path"] not in im_cache:
            im_cache = {r["path"]: ImageOps.exif_transpose(Image.open(r["path"])).convert("RGB")}
        crop = square_crop(im_cache[r["path"]], (r["x1"], r["y1"], r["x2"], r["y2"])).resize((tile, tile))
        x, y = (j % cols) * tile, (j // cols) * (tile + 28)
        s.paste(crop, (x, y))
        d.text((x + 2, y + tile + 1), f"{r['type'][:12]} {fmt(r['type_conf'])}", fill="black")
        d.text((x + 2, y + tile + 14), f"{r['color'] or '-'} {fmt(r['color_conf'])}  {r['size_px']:.0f}px",
               fill=(90, 90, 90))
    out.parent.mkdir(parents=True, exist_ok=True)
    s.save(out, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--min-type-conf", type=float, default=0.0,
                    help="extra filter on top of the threshold used at index time")
    ap.add_argument("--min-color-conf", type=float, default=0.0)
    args = ap.parse_args()

    colors, types, unknown = parse(args.query)
    if unknown:
        print(f"[WARN] not understood: {unknown}\n"
              f"       colours: {sorted(COLOR_WORDS)}\n       types: {sorted(TYPE_WORDS)}")
    if not colors and not types:
        sys.exit("[ERR] query has no colour or type words")
    if not Path(args.db).exists():
        sys.exit(f"[ERR] no index at {args.db}; run scripts/pipeline/index_frames.py first")

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    where, params = [], []
    if colors:
        where.append(f"d.color IN ({','.join('?' * len(colors))}) AND d.color_conf >= ?")
        params += [*colors, args.min_color_conf]
    color_sql, color_params = " AND ".join(where), list(params)
    if types:
        where.append(f"d.type IN ({','.join('?' * len(types))}) AND d.type_conf >= ?")
        params += [*types, args.min_type_conf]

    base = "FROM detections d JOIN frames f USING (frame_id) WHERE " + " AND ".join(where)
    total = con.execute(f"SELECT COUNT(*) {base}", params).fetchone()[0]
    rows = con.execute(
        f"SELECT f.path, d.* {base} ORDER BY COALESCE(d.type_conf, 1) * d.det_conf DESC, "
        f"d.color_conf DESC LIMIT ?", params + [args.limit]).fetchall()

    want = " ".join(filter(None, ["/".join(sorted(colors)), "/".join(sorted(types))]))
    print(f'"{args.query}" -> {want}: {total} matches'
          f"{f', showing {len(rows)}' if total > len(rows) else ''}")
    if types and colors:
        untyped = con.execute(
            f"SELECT COUNT(*) FROM detections d WHERE {color_sql} AND d.type = 'Vehicle'",
            color_params).fetchone()[0]
        if untyped:
            print(f"  (+{untyped} {'/'.join(sorted(colors))} vehicles too small or unclear to type)")
    for r in rows[:15]:
        print(f"  {Path(r['path']).name[:52]:52s} {r['type']:13s}{fmt(r['type_conf'])}  "
              f"{r['color'] or '-':6s}{fmt(r['color_conf'])}  {r['size_px']:4.0f}px")

    if rows:
        out = OUT / (re.sub(r"[^a-z0-9]+", "_", args.query.lower()).strip("_") + ".jpg")
        sheet(rows, out)
        print(f"\n  sheet -> {out}")


if __name__ == "__main__":
    main()

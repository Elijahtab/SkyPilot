"""
Export the review decisions to YOLO labels for the Phase 1 pilot pool.

Writes into a NEW pool directory and never touches the Kaggle originals -- the
same discipline that let the poisoned yolov8n batch be quarantined instead of
lost (Labeling/quarantine_branchB_yolov8n/README.md).

The size gate has a consequence that has to be handled explicitly: boxes BELOW
the gate were never shown to a human, so there is no reviewed class for them.
Two policies, and the default is the safe one:

  --below generic (default)  keep them, labeled Vehicle. Complete annotation,
      unspecific on the small boxes. 'Vehicle' is the umbrella covering every
      type, so this is a true-but-unspecific label, not a wrong one.
  --below drop               omit them entirely. DO NOT USE without thinking:
      unlabeled real vehicles train as background, which is exactly the
      under-annotation that made branch B poison v5/v6/v7.

⚠ Sub-gate boxes all becoming 'Vehicle' means the exported pool teaches "large
  car = SUV, small car = Vehicle" -- size, not appearance. That is a real defect
  of this export, not a rounding detail. See docs/v4-integration-plan.md §2.

Usage:
    python scripts/labeling/export_labels.py                  # -> images/labels kaggle_review
    python scripts/labeling/export_labels.py --check          # report only
    python scripts/labeling/export_labels.py --name my_pool

Then, before merging anything into training -- never skip this:
    python scripts/evaluation/diagnose_labels.py images/kaggle_review
"""

import argparse
import json
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _paths import KAGGLE, REPO

REVIEW   = REPO / "Labeling" / "review"
MANIFEST = REVIEW / "manifest.json"
DB       = REVIEW / "decisions.sqlite"


def current_labels(db):
    if not db.exists():
        sys.exit(f"[ERR] no decision log at {db}; nothing has been labeled yet")
    con = sqlite3.connect(db)
    rows = con.execute("""
        SELECT crop_id, class_id FROM decisions d
        WHERE active = 1 AND row_id = (
            SELECT MAX(row_id) FROM decisions
            WHERE crop_id = d.crop_id AND active = 1)
    """).fetchall()
    con.close()
    return dict(rows)


def read_lines(lbl):
    return [l for l in lbl.read_text().splitlines() if l.split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="kaggle_review",
                    help="pool dir name under images/ and labels/")
    ap.add_argument("--below", choices=("generic", "drop"), default="generic",
                    help="what to do with boxes under the size gate")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--partial", action="store_true",
                    help="export even if some crops are still undecided")
    args = ap.parse_args()

    if not MANIFEST.exists():
        sys.exit("[ERR] no manifest; run build_crops.py first")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    crops = manifest["crops"]
    classes = manifest["classes"]
    generic = manifest["generic_id"]
    labels = current_labels(DB)

    decided = [c for c in crops if c["id"] in labels]
    print(f"  gate           : >= {manifest['min_size_px']}px")
    print(f"  reviewed crops : {len(decided)} / {len(crops)}")

    # Pre-fill bias, made visible. If this sits at ~1.000 the pass rubber-stamped
    # the model and the labels carry no independent information -- which is the
    # failure mode that training on your own predictions produces.
    if manifest.get("has_prior") and decided:
        agree = sum(labels[c["id"]] == c["prior"] for c in decided) / len(decided)
        print(f"  agreement with {manifest['prior_model']} prior: {agree:.3f}")
        changed = Counter()
        for c in decided:
            if labels[c["id"]] != c["prior"]:
                changed[f"{classes[c['prior']]} -> {classes[labels[c['id']]]}"] += 1
        if changed:
            print("  corrections:")
            for k, v in changed.most_common():
                print(f"    {k:32s} {v}")

    if len(decided) < len(crops) and not args.partial:
        sys.exit(f"\n[STOP] {len(crops) - len(decided)} crops still undecided.\n"
                 f"       Finish the pass, or pass --partial to export anyway.")

    # -- assemble per-frame label files ------------------------------
    reviewed = defaultdict(dict)                 # (split, stem) -> {box_index: cls}
    for c in decided:
        reviewed[(c["split"], c["stem"])][c["box_index"]] = labels[c["id"]]

    img_out = REPO / "images" / args.name
    lbl_out = REPO / "labels" / args.name
    dist, n_frames, n_boxes, n_below = Counter(), 0, 0, 0

    if not args.check:
        for d in (img_out, lbl_out):
            d.mkdir(parents=True, exist_ok=True)

    for (split, stem), box_map in sorted(reviewed.items()):
        src_lbl = KAGGLE / split / "labels" / f"{stem}.txt"
        if not src_lbl.exists():
            print(f"  [warn] missing source label: {src_lbl.name}")
            continue
        src_img = next((KAGGLE / split / "images" / f"{stem}{e}"
                        for e in (".jpg", ".jpeg", ".png")
                        if (KAGGLE / split / "images" / f"{stem}{e}").exists()), None)
        if src_img is None:
            print(f"  [warn] missing source image for {stem}")
            continue

        out = []
        for i, line in enumerate(read_lines(src_lbl)):
            geom = line.split()[1:]              # keep the box exactly as drawn
            if i in box_map:
                cls = box_map[i]
            elif args.below == "generic":
                cls = generic
                n_below += 1
            else:
                continue
            out.append(" ".join([str(cls), *geom]))
            dist[classes[cls]] += 1
        n_frames += 1
        n_boxes += len(out)

        if not args.check:
            (lbl_out / f"{stem}.txt").write_text("\n".join(out) + "\n",
                                                 encoding="utf-8")
            shutil.copy2(src_img, img_out / src_img.name)

    print(f"\n  frames         : {n_frames}")
    print(f"  boxes written  : {n_boxes}  "
          f"({n_boxes - n_below} reviewed, {n_below} sub-gate as generic)")
    print(f"  distribution   : {dict(dist)}")

    if args.check:
        return print("\n  --check: nothing written.")

    print(f"\n  images -> {img_out}\n  labels -> {lbl_out}")
    print("\n  GATE THIS BATCH BEFORE TRAINING ON IT:")
    print(f"    .\\myenv\\Scripts\\python.exe scripts\\evaluation\\diagnose_labels.py "
          f"images\\{args.name}")


if __name__ == "__main__":
    main()

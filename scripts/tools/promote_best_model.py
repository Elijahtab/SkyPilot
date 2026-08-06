"""
Promote the best checkpoint from the runs tree into weights/released/ with a
timestamped name.

    python scripts/tools/promote_best_model.py                # best by val mAP50-95
    python scripts/tools/promote_best_model.py --newest       # most recently modified
    python scripts/tools/promote_best_model.py --run Vehicle_type_detection_v4

Default is now "best by recorded mAP50-95", not "most recently modified". The
old newest-file behaviour would have promoted v7 (0.363) over v4 (0.430), since
the regressions are the more recent runs.
"""

import argparse
import csv
import datetime
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _paths import RELEASED, VTD_RUNS, best_weights


def best_map(run_dir: Path):
    """Peak val mAP50-95 recorded in a run's results.csv, or None."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        vals = [float(r["metrics/mAP50-95(B)"]) for r in csv.DictReader(f)
                if r.get("metrics/mAP50-95(B)")]
    return max(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="promote this run by name")
    ap.add_argument("--newest", action="store_true",
                    help="pick the most recently modified checkpoint instead")
    args = ap.parse_args()

    if args.run:
        src = best_weights(args.run)
        if not src.exists():
            sys.exit(f"[ERR] not found: {src}")
        chosen = args.run

    elif args.newest:
        cands = sorted(VTD_RUNS.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
        if not cands:
            cands = sorted(VTD_RUNS.glob("*/weights/last.pt"), key=lambda p: p.stat().st_mtime)
        if not cands:
            sys.exit(f"[ERR] no best.pt or last.pt under {VTD_RUNS}")
        src = cands[-1]
        chosen = src.parent.parent.name

    else:
        scored = [(best_map(d), d) for d in sorted(VTD_RUNS.iterdir()) if d.is_dir()]
        scored = [(m, d) for m, d in scored if m is not None and (d / "weights" / "best.pt").exists()]
        if not scored:
            sys.exit(f"[ERR] no run under {VTD_RUNS} has both results.csv and best.pt")

        print(f"{'run':<45s}{'best mAP50-95':>15s}")
        for m, d in sorted(scored, reverse=True):
            print(f"{d.name:<45s}{m:>15.4f}")

        m, run_dir = max(scored)
        src = run_dir / "weights" / "best.pt"
        chosen = run_dir.name
        print(f"\nSelected: {chosen}  (mAP50-95 {m:.4f})")

    RELEASED.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = RELEASED / f"vehicle_{chosen}_{ts}.pt"
    shutil.copy2(src, dst)
    print(f"\nCopied {src}\n    -> {dst}")


if __name__ == "__main__":
    main()

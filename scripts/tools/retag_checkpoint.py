"""
Sync the class names baked into a trained .pt with configs/vehicle_7class.yaml.

Ultralytics stores `names` inside the checkpoint, and `val`/`predict` read them
from there — not from the data yaml. So renaming a class in configs/ does not
change what an existing model reports. This rewrites the names dict only; the
weights are untouched.

By default it writes a COPY next to the original (best.pt → best_vehicle.pt) so
proven checkpoints are never modified in place.

    python scripts/tools/retag_checkpoint.py --all           # every run, as copies
    python scripts/tools/retag_checkpoint.py --all --in-place
    python scripts/tools/retag_checkpoint.py path/to/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from _paths import CLASS_NAMES, VTD_RUNS


def retag(pt: Path, in_place: bool, suffix: str = "_vehicle") -> bool:
    ckpt = torch.load(pt, map_location="cpu", weights_only=False)

    model = ckpt.get("model")
    if model is None or not hasattr(model, "names"):
        print(f"  ⚠ {pt}: no model.names, skipped")
        return False

    old = [model.names[i] for i in sorted(model.names)]
    if len(old) != len(CLASS_NAMES):
        print(f"  ⚠ {pt}: {len(old)} classes vs {len(CLASS_NAMES)} in config, skipped")
        return False
    if old == CLASS_NAMES:
        print(f"  = {pt.name}: already in sync")
        return False

    new_names = dict(enumerate(CLASS_NAMES))
    model.names = new_names
    ckpt["model"] = model
    if isinstance(ckpt.get("ema"), type(model)) and hasattr(ckpt["ema"], "names"):
        ckpt["ema"].names = new_names

    out = pt if in_place else pt.with_name(f"{pt.stem}{suffix}{pt.suffix}")
    torch.save(ckpt, out)

    changed = [f"{o}→{n}" for o, n in zip(old, CLASS_NAMES) if o != n]
    print(f"  ✓ {pt.name} → {out.name}   ({', '.join(changed)})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="checkpoint paths")
    ap.add_argument("--all", action="store_true", help="every best.pt/last.pt under runs/")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the original instead of writing a copy")
    args = ap.parse_args()

    targets = [Path(p) for p in args.paths]
    if args.all:
        targets = sorted(VTD_RUNS.glob("*/weights/*.pt"))
        targets = [p for p in targets if not p.stem.endswith("_vehicle")]

    if not targets:
        ap.error("give checkpoint paths or --all")

    print(f"target names: {CLASS_NAMES}")
    print(f"mode: {'IN-PLACE' if args.in_place else 'copy alongside original'}\n")

    n = sum(retag(p, args.in_place) for p in targets if p.exists())
    print(f"\n{n} checkpoint(s) retagged.")


if __name__ == "__main__":
    main()

"""
Single source of truth for repo paths and the class schema.

Every script under scripts/ imports from here instead of hardcoding absolute
paths, so the repo works from any checkout location.

Usage from a script one level down (e.g. scripts/training/foo.py):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import REPO, CLASS_NAMES, build_data_yaml
"""

from pathlib import Path
import tempfile
import yaml

# ── repo layout ──────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]

CONFIGS    = REPO / "configs"
WEIGHTS    = REPO / "weights"
PRETRAINED = WEIGHTS / "pretrained"
RELEASED   = WEIGHTS / "released"

# vehicle-type dataset (current 7-class work)
VTD        = REPO / "Vehicle_type_detection"
VTD_IMAGES = VTD / "images"
VTD_LABELS = VTD / "labels"
VTD_RUNS   = VTD / "runs"

VTD_TRAIN = VTD_IMAGES / "images_train"
VTD_VAL   = VTD_IMAGES / "images_val"
VTD_TEST  = VTD_IMAGES / "images_test"

# auto-labeling
LABELING   = REPO / "Labeling"
KAGGLE     = LABELING / "kaggle_dataset"
KAGGLE_IMG = KAGGLE / "train" / "images"
KAGGLE_LBL = KAGGLE / "train" / "labels"        # original Kaggle boxes (class 0 = vehicle)
KAGGLE_GPT = KAGGLE / "train" / "labels_gpt"    # GPT-assigned classes on those boxes
KAGGLE_PREVIEW = KAGGLE / "preview"

# merged auto-label pool consumed by training
POOL_IMG = REPO / "images" / "kaggle_gpt"
POOL_LBL = REPO / "labels" / "kaggle_gpt"

# human-reviewed pool from the >=48px crop review (scripts/labeling/label_app.py)
REVIEW_IMG = REPO / "images" / "kaggle_review"
REVIEW_LBL = REPO / "labels" / "kaggle_review"

# legacy 11-class Stanford/streetcam runs
LEGACY_RUNS = REPO / "runs"

# ── class schema (7-class, current) ──────────────────────────────────
VEHICLE_SCHEMA = CONFIGS / "vehicle_7class.yaml"

with VEHICLE_SCHEMA.open("r", encoding="utf-8") as _f:
    _schema = yaml.safe_load(_f)

CLASS_NAMES = list(_schema["names"])   # ['Bus', 'Vehicle', 'Motorcycle', ...]
NC = len(CLASS_NAMES)

# 'Vehicle' is the UMBRELLA class covering every type in the list, so several
# scripts need its index explicitly. It is not a sibling bucket — never merge
# SUV / Standard Car into it. See configs/vehicle_7class.yaml.
VEHICLE_ID = CLASS_NAMES.index("Vehicle")

ID2NAME = dict(enumerate(CLASS_NAMES))


# ── helpers ──────────────────────────────────────────────────────────
def best_weights(run_name: str) -> Path:
    """Path to best.pt for a named run under Vehicle_type_detection/runs."""
    return VTD_RUNS / run_name / "weights" / "best.pt"


def build_data_yaml(name: str, train=None, val=None, test=None) -> Path:
    """
    Write a temp ultralytics data yaml with ABSOLUTE split paths.

    Ultralytics resolves a relative `path:` against its global datasets_dir, not
    against the yaml's own location, so relative paths in a checked-in config are
    a portability trap. Scripts call this instead and get correct paths anywhere.

    `train` accepts a list to keep multiple sources in their own folders (and to
    oversample by repeating an entry).
    """
    def norm(v):
        if v is None:
            return None
        return [str(Path(p)) for p in v] if isinstance(v, (list, tuple)) else str(Path(v))

    # ultralytics' check_det_dataset() hard-requires both 'train' and 'val' keys
    # even for a val-only run, so mirror whichever one is missing.
    if train is None:
        train = val
    if val is None:
        val = train
    if train is None:
        raise ValueError("build_data_yaml needs at least one of train= or val=")

    cfg = {"names": CLASS_NAMES, "nc": NC}
    for key, val_ in (("train", train), ("val", val), ("test", test)):
        if val_ is not None:
            cfg[key] = norm(val_)

    out = Path(tempfile.gettempdir()) / f"skypilot_{name}.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def labels_for(img_dir) -> Path:
    """
    Map an image directory to its YOLO label directory, mirroring the
    ultralytics `/images/ -> /labels/` convention used across this repo.
    """
    img_dir = Path(img_dir)
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    # e.g. .../images/kaggle_gpt handled above; fall back to sibling 'labels'
    return img_dir.parent / "labels"


def require(*paths) -> None:
    """Exit with a readable message if any required directory is missing."""
    import sys
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print("[ERR] Missing required paths:")
        for p in missing:
            print("  -", p)
        sys.exit(1)

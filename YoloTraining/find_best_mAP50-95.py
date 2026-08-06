from pathlib import Path
import shutil, datetime

ROOT = Path(__file__).resolve().parents[1]        # adjust if needed
runs_root = ROOT / "runs"           # where YOLO puts detect runs
candidates = list(runs_root.glob("train*/weights/best.pt"))

if not candidates:  # fallback to last.pt if no best.pt
    candidates = list(runs_root.glob("train*/weights/last.pt"))
    if not candidates:
        raise FileNotFoundError("No best.pt or last.pt found under runs/detect")

src = max(candidates, key=lambda p: p.stat().st_mtime)

dst_dir = ROOT / "models"
dst_dir.mkdir(exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
dst = dst_dir / f"vehicle_yolov8_best_{ts}.pt"

shutil.copy2(src, dst)
print(f"Copied {src} -> {dst}")
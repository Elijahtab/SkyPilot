from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    # 1. Find the newest best.pt (fallback to last.pt if needed)
    weights_dir = Path(r"S:/GitHub/SkyPilot/runs")
    bests = sorted(weights_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    if not bests:
        lasts = sorted(weights_dir.rglob("last.pt"), key=lambda p: p.stat().st_mtime)
        if not lasts:
            raise FileNotFoundError("No best.pt or last.pt found under runs/detect")
        weight_path = lasts[-1]
    else:
        weight_path = bests[-1]

    print(f"Using weights: {weight_path}")
    model = YOLO(str(weight_path))

    # 2. Evaluate
    metrics = model.val(
        data=r"S:\GitHub\SkyPilot\YoloTraining\data.yaml",
        split="val",
        imgsz=640,
        batch=16
    )
    print(metrics.results_dict)

    for i, ap in enumerate(metrics.box.maps):
        print(f"{i:3d} {model.names[i]:40s} {ap:.3f}")
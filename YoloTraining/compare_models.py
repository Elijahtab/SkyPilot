from ultralytics import YOLO
from pathlib import Path

# Paths to models
V4_PATH = r"S:\GitHub\SkyPilot\Vehicle_type_detection\runs\Vehicle_type_detection_v4\weights\best.pt"
V5_PATH = r"S:\GitHub\SkyPilot\Vehicle_type_detection\runs\Vehicle_type_detection_v5\weights\best.pt"
DATA_YAML = r"S:\GitHub\SkyPilot\YoloTraining\data_vehicle_type_detection_v4.yaml"

def evaluate_model(path, name):
    print(f"\n--- Evaluating {name} ---")
    model = YOLO(path)
    # Using the test split for a final comparison
    metrics = model.val(data=DATA_YAML, split='test', imgsz=640, batch=16, plots=False)
    return {
        "Model": name,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr
    }

def print_table(results):
    header = f"{'Model':<10} | {'mAP50':<10} | {'mAP50-95':<10} | {'Precision':<10} | {'Recall':<10}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"{r['Model']:<10} | {r['mAP50']:<10.4f} | {r['mAP50-95']:<10.4f} | {r['Precision']:<10.4f} | {r['Recall']:<10.4f}")

if __name__ == "__main__":
    results = []
    
    # Evaluate V4
    if Path(V4_PATH).exists():
        results.append(evaluate_model(V4_PATH, "v4"))
    else:
        print(f"V4 weights not found at {V4_PATH}")

    # Evaluate V5
    if Path(V5_PATH).exists():
        results.append(evaluate_model(V5_PATH, "v5"))
    else:
        print(f"V5 weights not found at {V5_PATH}")

    if results:
        print("\n=== Model Comparison (Test Set) ===")
        print_table(results)
        
        # Calculate Delta if both exist
        if len(results) == 2:
            print("\n=== Improvement (v5 - v4) ===")
            metrics_keys = ["mAP50", "mAP50-95", "Precision", "Recall"]
            for k in metrics_keys:
                diff = results[1][k] - results[0][k]
                color = "+" if diff >= 0 else ""
                print(f"{k:10s}: {color}{diff:.4f}")

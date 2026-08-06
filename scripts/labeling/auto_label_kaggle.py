"""
Assign vehicle subtypes to the Kaggle traffic-camera dataset.

The Kaggle labels are single-class (`car`, id 0) but the boxes are human-drawn
and trustworthy. This script keeps every box exactly as-is and only asks GPT-4o
to name the crop, writing 7-class labels to kaggle_dataset/train/labels_gpt/.

Requires OPENAI_API_KEY in the environment.

⚠ TWO KNOWN ISSUES — read before running a large batch:

1. COST. One API call per box, ~15.2 boxes/image, 5248 images ≈ 80k vision calls
   for a full pass. Only 207 images are done so far. Also, 62% of the Kaggle
   train split is Roboflow augmentation duplicates (2015 unique source frames →
   5248 images), so deduping by source frame first cuts the bill by ~62%.

2. AUGMENTED IMAGERY. Every image in that split is a 416x416 Roboflow export,
   and many are rotation/mosaic augmentations — four rotated tiles packed into
   one frame. GPT is therefore classifying sideways vehicles, which degrades the
   class labels this script exists to produce. Prefer un-augmented source frames.

The generic class is 'Vehicle' (id 1). It MUST stay consistent with
configs/vehicle_7class.yaml — mismatched vocabularies between the auto-labels
and the base dataset are what regressed v5/v6/v7. Verify a batch with
scripts/evaluation/diagnose_labels.py before merging it.
"""

import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import openai

from _paths import CLASS_NAMES, ID2NAME, KAGGLE_GPT, KAGGLE_IMG, KAGGLE_LBL, KAGGLE_PREVIEW, VEHICLE_ID

# ─── CONFIG ──────────────────────────────────────────────────────────
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable not set.\n"
        "Set it in PowerShell before running, for example:\n"
        "  $env:OPENAI_API_KEY = '<your-key>'\n"
        "Or export it in your shell / CI secrets."
    )

GPT_MODEL_ID = "gpt-4o"
MAX_IMAGES   = None      # None = all; or an int for a quick test
DETAIL       = "high"    # high detail crops for better ID


def to_data_url(img_bgr):
    _, buf = cv2.imencode(".jpg", img_bgr)
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"


def draw_box(img, xyxy, label):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def prompt_for(crop):
    """
    Note the deliberate bias toward the generic class. The base dataset resolves
    ambiguous vehicles to 'Vehicle' (78% of its boxes), so a prompt that pushes
    hard for subtypes produces labels that contradict the val set: measured class
    agreement was 0.19, versus 0.93 between the base train and val splits.
    """
    return [
        {
            "type": "text",
            "text": (
                "You are labeling ONE vehicle crop for an object detector.\n"
                "Return ONLY JSON {\"class_id\": int} where:\n"
                " 0 = Bus\n"
                " 1 = Vehicle (GENERIC — use this whenever the subtype is not "
                "unmistakable, including small, blurry, distant, rotated or "
                "partly occluded crops)\n"
                " 2 = Motorcycle\n"
                " 3 = SUV (only if clearly taller/boxier with high ground clearance)\n"
                " 4 = Standard Car (only if clearly a sedan, coupe or hatchback)\n"
                " 5 = Truck (pickup or commercial)\n"
                " 6 = Van (minivan or cargo van)\n"
                "Prefer 1 when in any doubt. A wrong subtype is worse than a "
                "correct generic label."
            ),
        },
        {"type": "image_url", "image_url": {"url": to_data_url(crop), "detail": DETAIL}},
    ]


def main():
    client = openai.Client(api_key=API_KEY)
    KAGGLE_GPT.mkdir(parents=True, exist_ok=True)
    KAGGLE_PREVIEW.mkdir(parents=True, exist_ok=True)

    print(f"Classes: {CLASS_NAMES}")
    print(f"Images : {KAGGLE_IMG}\nLabels : {KAGGLE_GPT}\n")

    images = sorted(KAGGLE_IMG.glob("*.[jp][pn]g"))
    if MAX_IMAGES:
        images = images[:MAX_IMAGES]

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"{img_path.name:>20} → unreadable, skipping")
            continue
        h, w = img.shape[:2]

        txt_path = KAGGLE_GPT / f"{img_path.stem}.txt"
        if txt_path.exists():
            print(f"{img_path.name:>20} → Already processed. Resuming!")
            continue

        kaggle_txt = KAGGLE_LBL / f"{img_path.stem}.txt"
        if not kaggle_txt.exists():
            print(f"{img_path.name:>20} → No Kaggle label found, skipping")
            continue

        # reuse the human boxes verbatim; only the class is inferred
        boxes = []
        for line in kaggle_txt.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = max(0, min(int((xc - bw / 2) * w), w - 1))
            y1 = max(0, min(int((yc - bh / 2) * h), h - 1))
            x2 = max(0, min(int((xc + bw / 2) * w), w - 1))
            y2 = max(0, min(int((yc + bh / 2) * h), h - 1))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))

        if not boxes:
            print(f"{img_path.name:>20} → No valid boxes in Kaggle label, skipping")
            continue

        class_ids = []
        for x1, y1, x2, y2 in boxes:
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                class_ids.append(VEHICLE_ID)
                continue
            try:
                resp = client.chat.completions.create(
                    model=GPT_MODEL_ID,
                    response_format={"type": "json_object"},
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt_for(crop)}],
                    temperature=0,
                )
                cid = int(json.loads(resp.choices[0].message.content)["class_id"])
                class_ids.append(cid if 0 <= cid < len(CLASS_NAMES) else VEHICLE_ID)
            except Exception as e:
                print(f"Error classifying crop via API: {e}")
                class_ids.append(VEHICLE_ID)

        with txt_path.open("w") as f:
            for (x1, y1, x2, y2), cid in zip(boxes, class_ids):
                f.write(f"{cid} {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
                        f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")

        for xyxy, cid in zip(boxes, class_ids):
            draw_box(img, xyxy, ID2NAME[cid])
        cv2.imwrite(str(KAGGLE_PREVIEW / f"{img_path.stem}_preview.jpg"), img)

        print(f"{img_path.name:>20} → {txt_path.name} ({len(boxes)} boxes, preview saved)")

    print(f"\nDONE ✓  Labels → {KAGGLE_GPT}\n         Previews → {KAGGLE_PREVIEW}")
    print("\nNext: scripts/labeling/extract_good_kaggle.py, then verify with")
    print("      scripts/evaluation/diagnose_labels.py BEFORE training on it.")


if __name__ == "__main__":
    main()

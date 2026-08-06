import os, base64, json, pathlib, cv2, yaml, openai
from ultralytics import YOLO

# ─── CONFIG ──────────────────────────────────────────────────────────
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable not set.\n"
        "Set it in PowerShell before running, for example:\n"
        "  $env:OPENAI_API_KEY = '<your-key>'\n"
        "Or export it in your shell / CI secrets."
    )
GPT_MODEL_ID  = "gpt-4o"                     # using the smarter model
YOLO_WEIGHTS  = "yolov8n.pt"                 # detector weights (swap for your own)
CONF_THR      = 0.3                          # YOLO confidence threshold
MAX_IMAGES    = None                         # None = all; or put 5 for quick test
DETAIL        = "high"                       # GPT image detail

# paths relative to this script
ROOT_DIR   = pathlib.Path(__file__).resolve().parent
IMG_DIR    = ROOT_DIR / "images_to_label"
LBL_DIR    = ROOT_DIR / "labels"
PREV_DIR   = ROOT_DIR / "preview"
YAML_FILE  = ROOT_DIR.parent /"YoloTraining"/"data_vehicle_type_detection_v4.yaml"

# ─── read YAML class list ────────────────────────────────────────────
with open(YAML_FILE, "r") as f:
    names = yaml.safe_load(f)["names"]           # ['Bus', 'CAR', ... , 'Van']
id2name = dict(enumerate(names))

# ─── helpers ─────────────────────────────────────────────────────────
def to_data_url(img_bgr):
    _, buf = cv2.imencode(".jpg", img_bgr)
    b64 = base64.b64encode(buf).decode()
    return f"data:image/jpeg;base64,{b64}"

def draw_box(img, xyxy, label):
    x1,y1,x2,y2 = map(int, xyxy)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# ─── init models & dirs ─────────────────────────────────────────────
detector   = YOLO(YOLO_WEIGHTS)
openai.api_key = API_KEY
client     = openai.Client(api_key=API_KEY)

LBL_DIR.mkdir(parents=True, exist_ok=True)
PREV_DIR.mkdir(parents=True, exist_ok=True)

# ─── main loop ──────────────────────────────────────────────────────
images = sorted(IMG_DIR.glob("*.[jp][pn]g"))
if MAX_IMAGES: images = images[:MAX_IMAGES]

for img_path in images:
    img   = cv2.imread(str(img_path))
    h, w  = img.shape[:2]

    # Skip if we already labeled this image
    txt_path = LBL_DIR / f"{img_path.stem}.txt"
    if txt_path.exists():
        print(f"{img_path.name:>20} → Already processed successfully. Resuming!")
        continue

    # YOLO detect (filter to COCO vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck)
    det_res = detector.predict(source=img, conf=CONF_THR, classes=[2, 3, 5, 7], verbose=False, save=False)[0]
    boxes   = det_res.boxes.xyxy.cpu().tolist()      # list of [x1,x2,y1,y2]
    if not boxes:
        print(f"{img_path.name:>20} → No detections, skipping")
        continue

    class_ids = []
    # loop boxes → GPT classify crop
    for xyxy in boxes:
        x1,y1,x2,y2 = map(int, xyxy)
        crop = img[y1:y2, x1:x2]
        msg = [
                {
                "type": "text",
                "text": (
                    "You are classifying ONE vehicle crop. "
                    "Return ONLY JSON {\"class_id\": int} where:\n"
                    " 0 = Bus\n 1 = CAR (use this if you are uncertain!)\n"
                    " 2 = Motorcycle\n 3 = SUV\n"
                    " 4 = Standard_Car (ONLY if clearly a sedan or hatchback)\n"
                    " 5 = Truck\n 6 = Van\n"
                    "If you are unsure or the vehicle doesn’t fit, choose 1."
                ),
                },
                {
                "type": "image_url",
                "image_url": {"url": to_data_url(crop), "detail": DETAIL},
                },
            ]
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL_ID,
                response_format={"type":"json_object"},
                max_tokens=10,
                messages=[{"role":"user","content":msg}],
                temperature=0,
            )
            cls_id = int(json.loads(resp.choices[0].message.content)["class_id"])
            class_ids.append(cls_id)
        except Exception as e:
            print(f"Error classifying crop via API: {e}")
            class_ids.append(1) # fallback to CAR

    # 3️⃣  write YOLO label file
    txt_path = LBL_DIR / f"{img_path.stem}.txt"
    with open(txt_path, "w") as f:
        for (x1,y1,x2,y2), cid in zip(boxes, class_ids):
            xc = ((x1+x2)/2) / w
            yc = ((y1+y2)/2) / h
            bw = (x2-x1) / w
            bh = (y2-y1) / h
            f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    # 4️⃣  preview with boxes + class names
    for xyxy, cid in zip(boxes, class_ids):
        draw_box(img, xyxy, id2name[cid])
    cv2.imwrite(str(PREV_DIR / f"{img_path.stem}_preview.jpg"), img)

    print(f"{img_path.name:>20} → {txt_path.name} ({len(boxes)} boxes, preview saved)")

print("\nDONE ✓  Labels →", LBL_DIR, "\n          Previews →", PREV_DIR)

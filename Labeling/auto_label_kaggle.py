import os, base64, json, pathlib, cv2, yaml, openai

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
MAX_IMAGES    = 500                          # None = all; or put count
DETAIL        = "high"                       # high detail crops for better ID

# paths relative to this script
ROOT_DIR   = pathlib.Path(__file__).resolve().parent
IMG_DIR    = ROOT_DIR / "kaggle_dataset" / "images"
LBL_DIR    = ROOT_DIR / "kaggle_dataset" / "labels"
PREV_DIR   = ROOT_DIR / "kaggle_dataset" / "preview"
YAML_FILE  = ROOT_DIR.parent /"YoloTraining"/"data_vehicle_type_detection_v4.yaml"

# PATH TO KAGGLE CACHE LABELS
KAGGLE_LBL_DIR = pathlib.Path(os.path.expanduser("~/.cache/kagglehub/datasets/ryankraus/traffic-camera-object-detection/versions/1/traffic/train/labels"))

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
openai.api_key = API_KEY
client     = openai.Client(api_key=API_KEY)

LBL_DIR.mkdir(parents=True, exist_ok=True)
PREV_DIR.mkdir(parents=True, exist_ok=True)

# ─── main loop ──────────────────────────────────────────────────────
images = sorted(IMG_DIR.glob("*.[jp][pn]g"))
if MAX_IMAGES: images = images[:MAX_IMAGES]

for img_path in images:
    img   = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w  = img.shape[:2]

    # Skip if we already perfectly labeled this image
    txt_path = LBL_DIR / f"{img_path.stem}.txt"
    if txt_path.exists():
        print(f"{img_path.name:>20} → Already processed successfully. Resuming!")
        continue

    # Look up the human-labeled ground truth box from the Kaggle dataset
    kaggle_txt_path = KAGGLE_LBL_DIR / f"{img_path.stem}.txt"
    if not kaggle_txt_path.exists():
        print(f"{img_path.name:>20} → No Kaggle label found, skipping")
        continue

    boxes = []
    with open(kaggle_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # cls, xc, yc, bw, bh
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)
                
                # clamp coordinates to image dimensions
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
                    
    if not boxes:
        print(f"{img_path.name:>20} → No valid boxes inside Kaggle label, skipping")
        continue

    class_ids = []
    # loop over strictly the perfect crops → GPT classify crop
    for xyxy in boxes:
        x1,y1,x2,y2 = xyxy
        crop = img[y1:y2, x1:x2]
        
        # fallback just in case crop is invalid
        if crop.size == 0:
            class_ids.append(1) # fallback to CAR
            continue

        msg = [
                {
                "type": "text",
                "text": (
                    "You are a computer vision expert identifying vehicle sub-classes. "
                    "Return ONLY JSON {\"class_id\": int} based on this crop:\n"
                    " 0 = Bus\n"
                    " 1 = CAR (generic, but you MUST try to classify into a subclass below first!)\n"
                    " 2 = Motorcycle\n"
                    " 3 = SUV (taller, boxier, larger ground clearance)\n"
                    " 4 = Standard_Car (specifically sedans, coupes, hatchbacks)\n"
                    " 5 = Truck (pickup trucks or commercial)\n"
                    " 6 = Van (minivans or cargo vans)\n"
                    "Look for distinct features. Only fall back to '1' if the crop is completely ambiguous or blurry."
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
            class_ids.append(1) # fallback

    # 3️⃣  write custom label file
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

    print(f"{img_path.name:>20} → {txt_path.name} ({len(boxes)} perfectly cropped boxes, preview saved)")

print("\nDONE ✓  Labels →", LBL_DIR, "\n          Previews →", PREV_DIR)

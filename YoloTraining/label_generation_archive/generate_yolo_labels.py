import os
import scipy.io
from PIL import Image

# Paths
train_dir = 'images/train'
label_dir = 'labels/train'
anno_file = 'devkit/cars_train_annos.mat'

# Make label directory if it doesn't exist
os.makedirs(label_dir, exist_ok=True)

# Load .mat annotation file
annos = scipy.io.loadmat(anno_file)['annotations'][0]

def convert_to_yolo(bbox, img_w, img_h):
    x_center = (bbox[0] + bbox[2]) / 2 / img_w
    y_center = (bbox[1] + bbox[3]) / 2 / img_h
    width = (bbox[2] - bbox[0]) / img_w
    height = (bbox[3] - bbox[1]) / img_h
    return x_center, y_center, width, height

# Loop through all annotations
for anno in annos:
    x1 = int(anno[0].item())
    y1 = int(anno[1].item())
    x2 = int(anno[2].item())
    y2 = int(anno[3].item())
    class_id = int(anno[4].item()) - 1  # YOLO classes are 0-indexed
    filename = anno[5][0]

    img_path = os.path.join(train_dir, filename)
    with Image.open(img_path) as img:
        w, h = img.size

    # Convert bbox to YOLO format
    x_c, y_c, bw, bh = convert_to_yolo([x1, y1, x2, y2], w, h)

    # Save label
    label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {x_c} {y_c} {bw} {bh}\n")

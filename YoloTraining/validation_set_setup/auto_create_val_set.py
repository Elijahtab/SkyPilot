import os
import random
import shutil

train_img_dir = 'images/train'
train_lbl_dir = 'labels/train'
val_img_dir = 'images/val'
val_lbl_dir = 'labels/val'

os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

images = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
val_count = int(len(images) * 0.2)  # 20% split
val_images = random.sample(images, val_count)

for img in val_images:
    label = img.replace('.jpg', '.txt')

    shutil.move(os.path.join(train_img_dir, img), os.path.join(val_img_dir, img))
    shutil.move(os.path.join(train_lbl_dir, label), os.path.join(val_lbl_dir, label))

print(f"Moved {val_count} images and labels to validation set.")
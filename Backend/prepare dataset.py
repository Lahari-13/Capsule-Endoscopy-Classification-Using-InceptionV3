import os
import cv2
import numpy as np
import shutil
import random

RAW = "../dataset/raw/binary_masks"
OUT = "../dataset"

def get_label(mask_path):
    img = cv2.imread(mask_path, 0)
    white_ratio = np.sum(img > 200) / img.size
    return "clear" if white_ratio > 0.6 else "contaminated"

images = os.listdir(RAW)
random.shuffle(images)

n = len(images)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

for i, img in enumerate(images):
    img_path = os.path.join(RAW, img)
    label = get_label(img_path)

    if i < train_end:
        split = "train"
    elif i < val_end:
        split = "val"
    else:
        split = "test"

    dest = os.path.join(OUT, split, label)
    os.makedirs(dest, exist_ok=True)
    shutil.copy(img_path, os.path.join(dest, img))

print("✅ Dataset prepared using ONLY binary images")

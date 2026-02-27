#Used the most popular dataset on Kaggle
#Next step would be to use a dataset from 1)Current Project 2)IEEE DataPort 3)Other sources


'''“For quick prototyping I use KaggleHub, but for full ML pipelines I should use Kaggle API 
so I can control data structure and reproducibility. (Didn't understand) (Not priority for now)”'''

'''Notes about the dataset: Give in the word file'''

from PIL import Image

import os

root = "backend/data/raw"
for cls in os.listdir(root):
    cls_path = os.path.join(root, cls)
    img = Image.open(os.path.join(cls_path, os.listdir(cls_path)[0]))
    print(cls, img.size, img.mode)

# It opens the first image file inside each class folder.

import os

root = "backend/data/raw"
for cls in os.listdir(root):
    print(cls, len(os.listdir(os.path.join(root, cls))))

#find the number of images in each class by listing the files in each class directory and counting them using len(). 

#Image Verify for later (Not a priority for now)

#Deleted new images in bird-drop


import os
import shutil
import random

random.seed(42)

RAW_DIR = "backend/data/raw"
OUTPUT_DIR = "data"
#Gave inccorect output directory

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

for cls in os.listdir(RAW_DIR):
    cls_path = os.path.join(RAW_DIR, cls)

    # Skip anything that is not a folder 
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    n = len(images)
    train_end = int(SPLIT_RATIOS["train"] * n)
    val_end = train_end + int(SPLIT_RATIOS["val"] * n)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, img_list in splits.items():
        for img in img_list:
            src = os.path.join(cls_path, img)
            dst = os.path.join(OUTPUT_DIR, split, cls, img)

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

print("Dataset split completed.")


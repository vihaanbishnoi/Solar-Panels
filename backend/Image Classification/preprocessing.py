#Used the most popular dataset on Kaggle
#Next step would be to use a dataset from 1)Current Project 2)IEEE DataPort 3)Other sources


'''“For quick prototyping I use KaggleHub, but for full ML pipelines I prefer Kaggle API 
so I can control data structure and reproducibility.”'''

'''Notes about the dataset: Give in the word file'''

"""
from PIL import Image
import os

root = "backend/data/raw"
for cls in os.listdir(root):
    cls_path = os.path.join(root, cls)
    img = Image.open(os.path.join(cls_path, os.listdir(cls_path)[0]))
    print(cls, img.size, img.mode)
    

import os

root = "backend/data/raw"
for cls in os.listdir(root):
    print(cls, len(os.listdir(os.path.join(root, cls))))
"""


"""os.listdir(cls_path)[0] gets the name of the first file in the directory for the current class (e.g., "Bird-drop", "Clean", etc.).
os.path.join(cls_path, ...) creates the full path to that first image file.
Image.open(...) opens the image file using the Python Imaging Library (PIL), returning an Image object.
The result is assigned to the variable img

and second block is to find the number of images in each class by listing the files in each class directory and counting them using len(). This helps us understand the distribution of the dataset across different classes."""

#image verify (to check if all images are valid and can be opened without errors)

"""from PIL import Image
import os
root = "backend/data/raw"
for cls in os.listdir(root):
    cls_path = os.path.join(root, cls)
    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)
        try:
            img = Image.open(img_path)
            img.verify()  # Verify that the image can be opened
        except (IOError, SyntaxError) as e:
            print(f"Invalid image: {img_path} - {e}")
"""

#deleted new images in bird-drop

#step 2 would be splitting the dataset into training, validation, and test sets, and then applying data augmentation techniques to increase the diversity of the training data. After that, we can proceed with building and training a machine learning model for image classification.

#I deleted the code for the initial steps

"""
import os
import shutil
import random

random.seed(42)

RAW_DIR = "backend/data/raw"
OUTPUT_DIR = "data"

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

print("Dataset split completed.")"""

#didnt understand the code properly
#homework for next time

from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

from torchvision.datasets import ImageFolder

train_ds = ImageFolder("data/train", transform=train_transforms)
val_ds   = ImageFolder("data/val", transform=val_test_transforms)
test_ds  = ImageFolder("data/test", transform=val_test_transforms)

print(train_ds.classes)

#use jupyter for preprocessing?

import matplotlib.pyplot as plt

img, label = train_ds[0]
plt.imshow(img.permute(1, 2, 0))
plt.title(train_ds.classes[label])
plt.axis("off")

#So many doubts right now
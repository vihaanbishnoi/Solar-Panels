#Used the most popular dataset on Kaggle
#Next step would be to use a dataset from 1)Current Project 2)IEEE DataPort 3)Other sources


'''“For quick prototyping I use KaggleHub, but for full ML pipelines I prefer Kaggle API 
so I can control data structure and reproducibility.”'''

'''Notes about the dataset: Give in the word file'''


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



"""os.listdir(cls_path)[0] gets the name of the first file in the directory for the current class (e.g., "Bird-drop", "Clean", etc.).
os.path.join(cls_path, ...) creates the full path to that first image file.
Image.open(...) opens the image file using the Python Imaging Library (PIL), returning an Image object.
The result is assigned to the variable img

and second block is to find the number of images in each class by listing the files in each class directory and counting them using len(). This helps us understand the distribution of the dataset across different classes."""

#image verify (to check if all images are valid and can be opened without errors)

from PIL import Image
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
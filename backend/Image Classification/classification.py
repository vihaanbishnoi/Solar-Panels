#Used the most popular dataset on Kaggle
#Next step would be to use a dataset from 1)Current Project 2)IEEE DataPort 3)Other sources

import kagglehub

# Download latest version
path = kagglehub.dataset_download("pythonafroz/solar-panel-images")

print("Path to dataset files:", path)

'''“For quick prototyping I use KaggleHub, but for full ML pipelines I prefer Kaggle API 
so I can control data structure and reproducibility.”'''

import os

print(os.listdir(path))

"""Split raw class folders into train/val/test directories."""

import os
import random
import shutil
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "clean"

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}


def main():
    for cls in os.listdir(RAW_DIR):
        cls_path = RAW_DIR / cls
        if not cls_path.is_dir():
            continue

        images = os.listdir(cls_path)
        random.shuffle(images)

        n = len(images)
        train_end = int(SPLIT_RATIOS["train"] * n)
        val_end = train_end + int(SPLIT_RATIOS["val"] * n)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split, img_list in splits.items():
            for img in img_list:
                src = cls_path / img
                dst = OUTPUT_DIR / split / cls / img
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)

    print("Dataset split completed.")


if __name__ == "__main__":
    main()

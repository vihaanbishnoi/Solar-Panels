"""Prepare train/val/test image folders from the raw vision dataset."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

VISION_DIR = Path(__file__).resolve().parent
BACKEND_DIR = VISION_DIR.parent
DATA_DIR = BACKEND_DIR / "data" / "vision"

RAW_DATA_DIR = DATA_DIR / "raw"
SPLIT_DATA_DIR = DATA_DIR / "split"
TRAIN_DATA_DIR = SPLIT_DATA_DIR / "train"
VAL_DATA_DIR = SPLIT_DATA_DIR / "val"
TEST_DATA_DIR = SPLIT_DATA_DIR / "test"


def _clear_and_create(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_split(
    files: list[Path],
    class_name: str,
    destination_root: Path,
) -> int:
    target_dir = destination_root / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, target_dir / src.name)
    return len(files)


def split_dataset(
    raw_dir: Path,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, int]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    _clear_and_create(train_dir)
    _clear_and_create(val_dir)
    _clear_and_create(test_dir)

    rng = random.Random(seed)
    stats = {"train": 0, "val": 0, "test": 0}

    class_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found in: {raw_dir}")

    for class_dir in sorted(class_dirs):
        images = [
            p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            continue

        rng.shuffle(images)
        total = len(images)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        test_count = total - train_count - val_count

        train_files = images[:train_count]
        val_files = images[train_count : train_count + val_count]
        test_files = images[train_count + val_count : train_count + val_count + test_count]

        stats["train"] += _copy_split(train_files, class_dir.name, train_dir)
        stats["val"] += _copy_split(val_files, class_dir.name, val_dir)
        stats["test"] += _copy_split(test_files, class_dir.name, test_dir)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split raw vision dataset into train/val/test.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR, help="Path to raw class folders.")
    parser.add_argument(
        "--split-root",
        type=Path,
        default=SPLIT_DATA_DIR,
        help="Root output folder for train/val/test splits.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = args.split_root / "train"
    val_dir = args.split_root / "val"
    test_dir = args.split_root / "test"

    total_ratio = args.train_ratio + args.val_ratio
    if total_ratio >= 1.0:
        raise ValueError("train-ratio + val-ratio must be less than 1.0.")

    stats = split_dataset(
        raw_dir=args.raw_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print("Dataset split complete.")
    print(f"Raw data   : {args.raw_dir.resolve()}")
    print(f"Train data : {train_dir.resolve()}")
    print(f"Val data   : {val_dir.resolve()}")
    print(f"Test data  : {test_dir.resolve()}")
    print(f"Image counts -> train: {stats['train']}, val: {stats['val']}, test: {stats['test']}")


if __name__ == "__main__":
    main()

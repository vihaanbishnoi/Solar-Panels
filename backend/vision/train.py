"""Training entrypoint with configured default dataset and artifact paths."""

from __future__ import annotations

import argparse
from pathlib import Path

VISION_DIR = Path(__file__).resolve().parent
BACKEND_DIR = VISION_DIR.parent
DATA_DIR = BACKEND_DIR / "data" / "vision"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts" / "vision"

SPLIT_DATA_DIR = DATA_DIR / "split"
TRAIN_DATA_DIR = SPLIT_DATA_DIR / "train"
VAL_DATA_DIR = SPLIT_DATA_DIR / "val"
TEST_DATA_DIR = SPLIT_DATA_DIR / "test"

MODEL_DIR = ARTIFACTS_DIR / "model"
CHECKPOINT_PATH = MODEL_DIR / "solar_model_v1.pth"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vision model for solar panel fault detection.")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DATA_DIR, help="Training images directory.")
    parser.add_argument("--val-dir", type=Path, default=VAL_DATA_DIR, help="Validation images directory.")
    parser.add_argument("--test-dir", type=Path, default=TEST_DATA_DIR, help="Test images directory.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Path to save model checkpoint.",
    )
    parser.add_argument(
        "--class-names-path",
        type=Path,
        default=CLASS_NAMES_PATH,
        help="Path to save class names JSON.",
    )
    return parser.parse_args()


def validate_paths(train_dir: Path, val_dir: Path) -> None:
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")


def main() -> None:
    args = parse_args()
    validate_paths(args.train_dir, args.val_dir)

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    args.class_names_path.parent.mkdir(parents=True, exist_ok=True)

    print("Vision training paths configured.")
    print(f"Train dir        : {args.train_dir.resolve()}")
    print(f"Validation dir   : {args.val_dir.resolve()}")
    print(f"Test dir         : {args.test_dir.resolve()}")
    print(f"Checkpoint path  : {args.checkpoint_path.resolve()}")
    print(f"Class names path : {args.class_names_path.resolve()}")
    print("Next step: plug your training loop below these configured paths.")


if __name__ == "__main__":
    main()

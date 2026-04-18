"""Fine-tuning entrypoint with configured default dataset and checkpoint paths."""

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

MODEL_DIR = ARTIFACTS_DIR / "model"
BASE_CHECKPOINT_PATH = MODEL_DIR / "solar_model_v1.pth"
FINETUNED_CHECKPOINT_PATH = MODEL_DIR / "solar_model_v2.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune existing vision model checkpoint.")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DATA_DIR, help="Training images directory.")
    parser.add_argument("--val-dir", type=Path, default=VAL_DATA_DIR, help="Validation images directory.")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=BASE_CHECKPOINT_PATH,
        help="Path to existing checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=FINETUNED_CHECKPOINT_PATH,
        help="Path to save fine-tuned checkpoint.",
    )
    return parser.parse_args()


def validate_paths(train_dir: Path, val_dir: Path, base_checkpoint: Path) -> None:
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    if not base_checkpoint.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint}")


def main() -> None:
    args = parse_args()
    validate_paths(args.train_dir, args.val_dir, args.base_checkpoint)
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print("Vision fine-tuning paths configured.")
    print(f"Train dir          : {args.train_dir.resolve()}")
    print(f"Validation dir     : {args.val_dir.resolve()}")
    print(f"Base checkpoint    : {args.base_checkpoint.resolve()}")
    print(f"Output checkpoint  : {args.output_checkpoint.resolve()}")
    print("Next step: plug your fine-tuning loop below these configured paths.")


if __name__ == "__main__":
    main()

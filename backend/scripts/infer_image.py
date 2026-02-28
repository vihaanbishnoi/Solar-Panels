"""Run single-image inference with the trained checkpoint."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from backend.src.augmentation import val_test_transforms
from backend.src.data_loader import test_dataset
from backend.src.model import model


def main():
    parser = argparse.ArgumentParser(description="Infer class for one solar panel image")
    parser.add_argument("image_path", type=str, help="Path to input image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(__file__).resolve().parents[1]
    checkpoint_path = base_dir / "checkpoints" / "best_model.pth"

    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.to(device)
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    tensor = val_test_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred_idx = torch.argmax(output, dim=1).item()

    label = test_dataset.classes[pred_idx]
    print(f"Prediction: {label}")


if __name__ == "__main__":
    main()

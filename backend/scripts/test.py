"""Evaluate the trained model on the test split."""

from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from backend.src.data_loader import test_dataset, test_loader
from backend.src.model import model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(__file__).resolve().parents[1]
    checkpoint_path = base_dir / "checkpoints" / "best_model.pth"

    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: checkpoint not found at {checkpoint_path}. Running with current model weights.")

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = test_dataset.classes
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()

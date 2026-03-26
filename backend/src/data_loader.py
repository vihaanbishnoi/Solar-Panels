"""Data loaders for train/val/test splits."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from backend.src import augmentation as tr

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "clean"

train_dataset = ImageFolder(
    root=str(DATA_DIR / "train"),
    transform=tr.train_transforms,
)

val_dataset = ImageFolder(
    root=str(DATA_DIR / "val"),
    transform=tr.val_test_transforms,
)

test_dataset = ImageFolder(
    root=str(DATA_DIR / "test"),
    transform=tr.val_test_transforms,
)

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
)

if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import augmentation as tr

train_dataset = ImageFolder(
    root="backend/data/clean/train",
    transform=tr.train_transforms
)

val_dataset = ImageFolder(
    root="backend/data/clean/val",
    transform=tr.val_test_transforms
)

test_dataset = ImageFolder(
    root="backend/data/clean/test",
    transform=tr.val_test_transforms
)

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=False
    #CPU
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
#num_workers controls how many parallel processes load your data so training doesn't have to wait.

def main():
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)

if __name__ == "__main__":
    main()
#if __name__ == "__main__": ensures main() runs only when the file is executed directly, not when imported.

#Finished here for now

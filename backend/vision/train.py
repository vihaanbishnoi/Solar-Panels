import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

TRAIN_DIR    = "backend\\data\\vision\\dataset\\train"
VAL_DIR      = "backend\\data\\vision\\dataset\\val"
SAVE_PATH    = "backend\\artifacts\\vision\\model\\solar_model_v1.pth"
NUM_CLASSES  = 6
BATCH_SIZE   = 16
EPOCHS       = 25
LR           = 0.001                 
IMAGE_SIZE   = 224
PATIENCE     = 5                      # stop if val loss doesn't improve for 5 epochs

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),           # randomly mirror image
    transforms.RandomVerticalFlip(),             # solar panels can be any orientation
    transforms.RandomRotation(15),               # rotate up to 15 degrees
    transforms.ColorJitter(                      # simulate different lighting conditions
        brightness=0.3,
        contrast=0.3,
        saturation=0.2
    ),
    transforms.ToTensor(),                       # convert to tensor (0-1 range)
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)  # normalize to ImageNet scale
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


def get_class_weights(train_dataset):
    """
    Calculate class weights inversely proportional to class frequency.
    Rare classes (like Physical-Damage with 70 images) get higher weight
    so the model doesn't ignore them.
    """
    class_counts = {}
    for _, label in train_dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1

    total = sum(class_counts.values())
    num_classes = len(class_counts)

    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        # Weight = total / (num_classes * count)
        # This normalises so average weight = 1.0
        weights.append(total / (num_classes * count))

    print("\nClass weights (higher = rarer class, gets more attention):")
    for class_name, idx in sorted(train_dataset.class_to_idx.items(), key=lambda x: x[1]):
        print(f"  {class_name:25s} count={class_counts[idx]:4d}  weight={weights[idx]:.3f}")

    return torch.tensor(weights, dtype=torch.float)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one full pass through the training data."""
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # clear gradients from last batch
        outputs = model(images)        # forward pass: model makes predictions
        loss = criterion(outputs, labels)  # how wrong was it?
        loss.backward()                # calculate which weights caused the error
        optimizer.step()               # adjust those weights

        total_loss += loss.item()
        _, predicted = outputs.max(1)  # pick the class with highest score
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """Run one full pass through the validation data — no weight updates."""
    model.eval()   # turns off dropout and batch norm randomness
    total_loss = 0
    correct    = 0
    total      = 0

    with torch.no_grad():   # don't track gradients — saves memory, speeds up
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    device="cpu"
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)

    print(f"\nClasses found: {train_dataset.classes}")
    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── Calculate class weights from actual counts in your dataset ──
    class_weights = get_class_weights(train_dataset).to(device)

    # ── Load EfficientNet-B0 with ImageNet pretrained weights ──
    # pretrained=True downloads weights trained on 14M images
    # num_classes=6 replaces the final layer to output 6 scores
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)

    # ── Loss function and optimizer ──
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Learning rate scheduler: reduce LR by half if val loss plateaus for 3 epochs
    # This helps the model make finer adjustments as it gets better
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ── Training loop ──
    best_val_loss    = float("inf")
    epochs_no_improve = 0

    print(f"\nStarting training for up to {EPOCHS} epochs...\n")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6} {train_loss:>12.4f} {train_acc:>9.1f}% {val_loss:>10.4f} {val_acc:>8.1f}% {current_lr:>10.6f}")

        # Tell scheduler how validation loss is doing
        scheduler.step(val_loss)

        # Save model only when it's the best we've seen
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"         --> Best model saved to {SAVE_PATH}")
        else:
            epochs_no_improve += 1
            print(f"         --> No improvement ({epochs_no_improve}/{PATIENCE})")

        # Early stopping: if no improvement for PATIENCE epochs, stop
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {SAVE_PATH}")
    print(f"\nNext step: run finetune.py to improve further, or convert.py to export.")


if __name__ == "__main__":
    main()
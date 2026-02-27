import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_loader import train_loader, val_loader
from model import model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# LOSS FUNCTION (no class weights)
criterion = nn.CrossEntropyLoss()
#How does CrossEntropyLoss work?

# OPTIMIZER (only train classifier head)
optimizer = optim.Adam(
    model.fc.parameters(),
    lr=1e-3
)
#How does Adam work?


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()  # training mode

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)          # forward pass
        loss = criterion(outputs, labels)

        loss.backward()                  # backprop
        optimizer.step()                 # update weights

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()  # evaluation mode

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():
    EPOCHS = 10
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, "best_model.pth")
            )
            print("✅ Saved best model")


if __name__ == "__main__":
    print(f"Using device: {device}")
    main()

#81% accuracy on validation set
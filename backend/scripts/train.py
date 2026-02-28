"""Train the classifier head and save the best checkpoint."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from backend.src.data_loader import train_loader, val_loader
from backend.src.model import model


def train_one_epoch(net, dataloader, optimizer, criterion, device):
    net.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), correct / total


def validate(net, dataloader, criterion, device):
    net.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(dataloader), correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.fc.parameters(), lr=1e-3)

    epochs = 10
    best_val_loss = float("inf")

    base_dir = Path(__file__).resolve().parents[1]
    checkpoint_dir = base_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            net, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(net, val_loader, criterion, device)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), checkpoint_dir / "best_model.pth")
            print("Saved best model")


if __name__ == "__main__":
    main()

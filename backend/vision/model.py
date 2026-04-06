"""Model definition for solar panel fault classification."""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

NUM_CLASSES = 6

model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(
    in_features=model.fc.in_features,
    out_features=NUM_CLASSES,
)

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

if __name__ == "__main__":
    print(model)

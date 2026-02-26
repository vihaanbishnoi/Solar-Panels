import torch
import torch.nn as nn
from torchvision import models

model = models.resnet50(pretrained=True)

num_classes = 6

model.fc = nn.Linear(
    in_features=model.fc.in_features,
    out_features=num_classes
)

for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the classifier
for param in model.fc.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(device)
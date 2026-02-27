import torch.nn as nn
from torchvision import models

model = models.resnet50(pretrained=True)
#In the end take a deep look at how this model works

num_classes = 6

#fc = final fully connected classifier layer
#Take 2048 learned features and convert them into 6 class scores

model.fc = nn.Linear(
    in_features=model.fc.in_features,
    out_features=num_classes
)

#Freezes all model weights, No gradients computed, No weight updates during training

for param in model.parameters():
    param.requires_grad = False

#Only the new classifier learns during training
for param in model.fc.parameters():
    param.requires_grad = True

if __name__ == "__main__":
    print(model)
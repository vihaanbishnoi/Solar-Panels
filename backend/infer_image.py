import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Paths
MODEL_PATH = "ml/models/pv_classifier.pth"
IMAGE_PATH = "ml/data/PVEL-CLASSIFY-SMALL/val/bad/img022065.jpg"  # we will change this when running

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load image
img = Image.open(IMAGE_PATH).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output, dim=1).item()

# IMPORTANT: ImageFolder sorts classes alphabetically!
# bad -> Class 0, good -> Class 1
label_map = {0: "DEFECTIVE (BAD)", 1: "NORMAL (GOOD)"}
print("Prediction:", label_map[pred])

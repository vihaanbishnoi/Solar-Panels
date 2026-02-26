from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
#Normalize values?
#Understand this file next iter

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

from torchvision.datasets import ImageFolder

train_ds = ImageFolder("backend/data/clean/train", transform=train_transforms)
val_ds   = ImageFolder("backend/data/clean/val", transform=val_test_transforms)
test_ds  = ImageFolder("backend/data/clean/test", transform=val_test_transforms)

print(train_ds.classes)

#Use Jupyter for preprocessing?

import matplotlib.pyplot as plt

img, label = train_ds[0]
plt.imshow(img.permute(1, 2, 0))
plt.title(train_ds.classes[label])
plt.axis("off")

#What is this shit?
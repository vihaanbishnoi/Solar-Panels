from torchvision import transforms

#ImageNet images are 224x224

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(), #(H, W, C) → (C, H, W) and scales the pixel values to [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        #mean and std are the mean and standard deviation of the ImageNet dataset
    )

])

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

#Should you use Jupyter for preprocessing?


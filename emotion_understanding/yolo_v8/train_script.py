# train_script.py
# ======================= IMPORTS =======================
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from collections import Counter

# ======================= CONFIG =======================
data_dir = './dataset'  # Root folder containing class subfolders (anger, happy, etc.)
labels_csv = os.path.join(data_dir, 'labels.csv')
input_size = 224
batch_size = 100
num_epochs = 200
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================= TRANSFORMS =======================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Define the transform with albumentations
# train_transform = A.Compose([
#     # A.RandomResizedCrop(height=input_size, width=input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
#     A.RandomResizedCrop(size=(input_size, input_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),

#     A.HorizontalFlip(),
#     A.Rotate(limit=15),
#     A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
#     A.ToGray(p=0.1),
#     A.Affine(rotate=(-10, 10), translate_percent=(0.05, 0.05), scale=(0.95, 1.05)),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    normalize,
])


val_transform = A.Compose([
    A.Resize(int(input_size/0.875), int(input_size/0.875)),
    A.CenterCrop(input_size, input_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ======================= DATASET =======================
class AffectNetDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform

        # Create a label-to-index mapping
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.df['label'].unique()))}
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the image path and label
        img_path = self.df.iloc[idx]['pth']
        label_name = self.df.iloc[idx]['label']
        label = self.label2idx[label_name]

        # Load the image
        full_path = os.path.join(self.data_dir, img_path)
        image = Image.open(full_path).convert("RGB")

        # Convert PIL image to NumPy array only if using albumentations
        if self.transform:
            if isinstance(self.transform, A.Compose):
                image = np.array(image)
                image = self.transform(image=image)['image']
            else:
                image = self.transform(image)

        return image, label


# ======================= LOAD CSV & SPLIT =======================
labels_df = pd.read_csv(labels_csv)  # Assumes columns: path,label
train_df, val_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['label'])

train_dataset = AffectNetDataset(csv_file=labels_csv, data_dir=data_dir, transform=train_transform)
val_dataset = AffectNetDataset(csv_file=labels_csv, data_dir=data_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# ======================= MODEL =======================
def get_model(name):
    if name == 'mobilenetv3':
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 8)  # 8 emotions
    elif name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 8)
    elif name == 'efficientnet':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    else:
        raise ValueError("Unsupported model name")
    return model

model = get_model('efficientnet').to(device)

# ======================= TRAINING =======================
def main():
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1) # Uncomment if using StepLR

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        losses = []

        # Training
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            losses.append(loss.item())
            # Backpropagation

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Average loss
            if len(losses) > 0:
                avg_loss = np.mean(losses)
                # print(f"Batch Loss: {avg_loss:.4f}")
                losses = []
                

            total_loss /= len(train_loader)
        # Calculate accuracy

        train_acc = correct / total
        # print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss:.4f} - Train Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        # val_acc = correct / total
        # print(f"Validation Acc: {val_acc:.4f}\n")
    
        # Step the scheduler
        scheduler.step()

        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {correct / total:.4f}")

    # Save the model every 20 epochs
    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), f'efficientnet_model_epoch_{epoch+1}.pth')
        print(f"Model saved as efficientnet_model_epoch_{epoch+1}.pth")
    # Save the final model
    torch.save(model.state_dict(), 'efficientnet_model.pth') 
    print("Model saved as model.pth")

if __name__ == "__main__":
    main()

# emotion_train_test.py
import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

# ================= CONFIG =================
data_dir = './dataset'
csv_path = os.path.join(data_dir, 'labels.csv')
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

input_size = 224
batch_size = 128
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================ TRANSFORMS ================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


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
val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    normalize,
])

# ============== DATASET ===============
class AffectNetFaceDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, detect_face=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.detect_face = detect_face
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.label2idx = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['pth'])
        label = self.label2idx[row['label']]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Face detection
        if self.detect_face:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                image = image[y:y+h, x:x+w]

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return image, label

# ============== DATA LOADING ===============
df = pd.read_csv(csv_path)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])

train_dataset = AffectNetFaceDataset(train_df, data_dir, transform=train_transform)
val_dataset = AffectNetFaceDataset(val_df, data_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ============== MODEL =================
def get_model(name='mobilenetv3'):
    if name == 'mobilenetv3':
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 8)
    elif name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 8)
    elif name == 'efficientnet':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    else:
        raise ValueError("Invalid model name")
    return model

model = get_model('mobilenetv3').to(device)

# ============== TRAIN =================
def train():
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = evaluate()
        scheduler.step()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "mobilenet_opencv.pth")
    print("Training finished and model saved.")

# ============== EVAL =================
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ============== TEST =================
def test_single_image(image_path):
    model.eval()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        image = image[y:y+h, x:x+w]

    image = Image.fromarray(image)
    transform = val_transform
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()

    idx2label = {v: k for k, v in train_dataset.label2idx.items()}
    print(f"Predicted Emotion: {idx2label[pred]}")

# ============== MAIN =================
if __name__ == "__main__":
    train()
    # Example test:
    # test_single_image('./dataset/images/sample.jpg')

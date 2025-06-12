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

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# ================= CONFIG =================
data_dir = '../dataset'
csv_path = os.path.join(data_dir, 'dataset_withoutCon.csv')
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

input_size = 224
batch_size = 16
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================ TRANSFORMS ================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Data augmentation for training
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

# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.ToTensor(),
# ])


# Validation and test transforms
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
# train_df, val_df = train_test_split(df, test_size=0.3, stratify=df['label'])

# Split the dataset into train, validation, and test sets
# 70% train, 20% validation, 10% test
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
test_df, val_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

train_dataset = AffectNetFaceDataset(train_df, data_dir, transform=train_transform)
val_dataset = AffectNetFaceDataset(val_df, data_dir, transform=val_transform)
test_dataset = AffectNetFaceDataset(test_df, data_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ============== MODEL =================
name = 'efficientnet'  # Change this to 'resnet18', 'resnet50', 'efficientnet', or 'efficientnetv2' as needed

def get_model(name):
    num_classes = 7
    dropout_rate = 0.2

    if name == 'mobilenetv3':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    elif name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    elif name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    elif name == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    elif name == 'efficientnetv2':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError("Invalid model name")
    
    return model


model = get_model(name).to(device)

# ============== TRAIN =================
modelName = "efficientnet_HQRAF_drop_withoutCon.pth"

def train():

    # CrossEntropyLoss is used for multi-class classification
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

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
        scheduler.step(val_acc)
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), modelName)
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

# ============== EVALUATE TEST =================
def evaluate_on_test_set():
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    labels_list = sorted(train_df['label'].unique())
    idx2label = {i: label for i, label in enumerate(labels_list)}

    print("\n=== Classification Report on Test Set ===")
    print(classification_report(all_labels, all_preds, target_names=[idx2label[i] for i in range(len(idx2label))]))

    # Optional: Visualize Confusion Matrix
    try:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm_display = np.array([["{0}\n{1:.1f}%".format(int(v), p) for v, p in zip(row_c, row_p)] 
                       for row_c, row_p in zip(cm, cm_normalized)])

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=cm_display, fmt="", cmap="Blues",
            xticklabels=[idx2label[i] for i in range(7)],
            yticklabels=[idx2label[i] for i in range(7)])

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix on Test Set")
        plt.tight_layout()
        plt.savefig("confusion_efficientnet_newdata_drop_withoutCon.png")
        plt.show()
    except ImportError:
        print("Skipping confusion matrix plot (matplotlib or seaborn not installed).")


# ============== MAIN =================
if __name__ == "__main__":
    train()
    # Evaluate the model on the testing set
    evaluate_on_test_set()

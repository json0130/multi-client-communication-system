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

# ================= MIXUP/CUTMIX =================
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    '''Returns cutmixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[index]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# ================= CONFIG =================
data_dir = '../dataset'
csv_path = os.path.join(data_dir, 'dataset_withoutCon.csv')
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

input_size = 224
batch_size = 32
num_epochs = 70
num_classes = 7  # 7 emotions: neutral, happy, sad, angry, disgust, fear, surprise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================ TRANSFORMS ================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Data augmentation for training
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
#     transforms.RandomGrayscale(p=0.1),
#     transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
#     transforms.ToTensor(),
#     normalize,
# ])
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),  # More aggressive crop
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),  # Increased rotation
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Add blur
    transforms.ToTensor(),
    normalize,
])

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
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.df['label'].unique()))}

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
def get_efficientnet_v2_model():
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    
    # Custom classifier head as specified
    # model.classifier = nn.Sequential(
    #     nn.Linear(model.classifier[1].in_features, 512),
    #     nn.BatchNorm1d(512),
    #     nn.ReLU(),
    #     nn.Dropout(0.4),
    #     nn.Linear(512, 256),
    #     nn.ReLU(),
    #     nn.Dropout(0.3),
    #     nn.Linear(256, num_classes)
    # )
    # Slightly larger classifier:
    model.classifier = nn.Sequential(
        nn.Linear(1280, 1024),       # Increased from 512
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.3),             # Reduced dropout
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),         # Add BatchNorm here too
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 7)
    )
        
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Unfreeze top feature layers (last few blocks)
    # EfficientNet V2-S has 7 feature blocks (0-6), unfreeze the last 2-3 blocks
    for i in range(5, 7):  # Unfreeze blocks 5 and 6 (last 2 blocks)
        if hasattr(model.features, str(i)):
            for param in model.features[i].parameters():
                param.requires_grad = True
    
    # Also unfreeze the final convolutional layers
    if hasattr(model.features, '7'):  # Final conv layer
        for param in model.features[7].parameters():
            param.requires_grad = True
    
    return model

model = get_efficientnet_v2_model().to(device)

# Print trainable parameters info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")

# ============== TRAIN =================
modelName = "efficientnetv2_7emotions_optimized2.pth"

def train():
    # CrossEntropyLoss is used for multi-class classification
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Standard training without MixUp/CutMix for now
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Uncomment below for MixUp/CutMix training
            use_mixup = np.random.rand() < 0.5
            if use_mixup:
                images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.4)
            else:
                images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=0.4)
            outputs = model(images)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

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
    labels_list = sorted(train_dataset.df['label'].unique())
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
            xticklabels=[idx2label[i] for i in range(num_classes)],
            yticklabels=[idx2label[i] for i in range(num_classes)])

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix on Test Set - EfficientNet V2 (7 Emotions)")
        plt.tight_layout()
        plt.savefig("confusion_matrix_efficientnetv2_7emotions_optimized2.png")
        plt.show()
    except ImportError:
        print("Skipping confusion matrix plot (matplotlib or seaborn not installed).")


# ============== MAIN =================
if __name__ == "__main__":
    print(f"Training with {num_classes} emotion classes: neutral, happy, sad, angry, disgust, fear, surprise")
    print(f"Available emotions: {sorted(train_dataset.label2idx.keys())}")
    train()
    # Evaluate the model on the testing set
    evaluate_on_test_set()
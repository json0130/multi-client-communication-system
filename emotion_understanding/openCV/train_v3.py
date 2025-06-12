# emotion_train_test_optimized_v2.py
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import warnings
from torch.optim.swa_utils import AveragedModel, SWALR

warnings.filterwarnings('ignore')

# ================= IMPROVED MIXUP/CUTMIX =================
def mixup_data(x, y, alpha=0.4):  # Increased alpha from 0.2
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

def cutmix_data(x, y, alpha=0.4):  # Increased alpha from 0.2
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
num_epochs = 40 
num_classes = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================ OPTIMIZED TRANSFORMS ================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0)),  # Less aggressive cropping
    transforms.RandomHorizontalFlip(p=0.3),  # Reduced probability
    transforms.RandomRotation(10),  # Reduced from 15
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),  # Reduced jitter
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomAffine(5, translate=(0.03, 0.03)),  # Reduced augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # Less aggressive
])

val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============== CORRECT DATASET CLASS ==============
class AffectNetFaceDataset(Dataset):  # Note: inherits from Dataset, not nn.Module
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

        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.detect_face:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
                if len(faces) > 0:
                    areas = [w * h for (x, y, w, h) in faces]
                    largest_face_idx = np.argmax(areas)
                    x, y, w, h = faces[largest_face_idx]
                    padding = int(0.1 * min(w, h))
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(image.shape[1] - x, w + 2 * padding)
                    h = min(image.shape[0] - y, h + 2 * padding)
                    image = image[y:y+h, x:x+w]

            image = Image.fromarray(image)
            if self.transform:
                image = self.transform(image)

            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.df))

# ============== CORRECT MODEL CLASS ==============
class EnhancedEfficientNet(nn.Module):  # Renamed to avoid confusion
    def __init__(self, num_classes=7, dropout_rate=0.3):
        super(EnhancedEfficientNet, self).__init__()
        
        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.attention = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1536),
            nn.BatchNorm1d(1536),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1536, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(768, num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Change from 'gelu' to 'relu' for initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Attention mechanism
        attn_weights = self.attention(features)
        weighted_features = features * attn_weights
        
        output = self.classifier(weighted_features)
        return output

def get_model():
    model = EnhancedEfficientNet(num_classes=num_classes)
    
    # Partial unfreezing from the start
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Unfreeze last 4 blocks initially
    for i in range(4, 8):
        for param in model.backbone.features[i].parameters():
            param.requires_grad = True
    
    return model

# ============== DATA LOADING ==============
df = pd.read_csv(csv_path)
print("Class distribution:")
print(df['label'].value_counts())

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
test_df, val_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# More aggressive class balancing
class_counts = Counter(train_df['label'])
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
class_weights = class_weights ** 0.7  # Flatten the weights
class_weights_dict = dict(zip(np.unique(train_df['label']), class_weights))

sample_weights = [class_weights_dict[label] for label in train_df['label']]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_dataset = AffectNetFaceDataset(train_df, data_dir, transform=train_transform)
val_dataset = AffectNetFaceDataset(val_df, data_dir, transform=val_transform)
test_dataset = AffectNetFaceDataset(test_df, data_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ============== TRAINING ==============
model = get_model().to(device)
swa_model = AveragedModel(model)  # For SWA
swa_start = int(num_epochs * 0.75)  # Last 25% of training

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

modelName = "efficientnetv2_7emotions_enhanced.pth"

# Enhanced loss function
class_weights_tensor = torch.FloatTensor(list(class_weights_dict.values())).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.15)

def train():
    classifier_params = list(model.classifier.parameters()) + list(model.attention.parameters())
    backbone_params = [p for p in model.parameters() if p not in set(classifier_params)]
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 5e-5, 'weight_decay': 1e-4},
        {'params': classifier_params, 'lr': 5e-3, 'weight_decay': 1e-3}
    ])
    
    # SWA scheduler
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
    
    # OneCycleLR for main phase
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[5e-4, 5e-2],  # Higher learning rates
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy='cos'
    )
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Reduced mixup/cutmix probability (20%)
            if np.random.rand() < 0.2:
                if np.random.rand() < 0.5:
                    images, targets_a, targets_b, lam = mixup_data(images, labels)
                    outputs = model(images)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    _, preds = outputs.max(1)
                    correct += (lam * (preds == targets_a).float() + (1 - lam) * (preds == targets_b).float()).sum().item()
                else:
                    images, targets_a, targets_b, lam = cutmix_data(images, labels)
                    outputs = model(images)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    _, preds = outputs.max(1)
                    correct += (lam * (preds == targets_a).float() + (1 - lam) * (preds == targets_b).float()).sum().item()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if epoch < swa_start:
                scheduler.step()
            else:
                swa_scheduler.step()

            total_loss += loss.item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = evaluate()
        
        # SWA update
        if epoch >= swa_start:
            swa_model.update_parameters(model)
        
        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early unfreezing
        if epoch == 5 and best_val_acc > 0.65:
            print("Early unfreezing of more layers...")
            for i in range(2, 4):
                for param in model.backbone.features[i].parameters():
                    param.requires_grad = True
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), modelName)
            patience_counter = 0
            print(f"New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Update bn statistics for SWA
    torch.optim.swa_utils.update_bn(train_loader, swa_model)
    swa_val_acc = evaluate(model=swa_model)
    print(f"SWA Validation Accuracy: {swa_val_acc:.4f}")
    torch.save(swa_model.state_dict(), "swa_"+modelName)

def evaluate(model=None):
    model = model if model else model
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

# ============== EVALUATE TEST =================
def evaluate_on_test_set(model_path=None):
    # Load best model if path provided
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Ensure we use test dataset's label mapping
    label2idx = test_dataset.label2idx
    idx2label = {v: k for k, v in label2idx.items()}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate classification report
    print("\n=== Classification Report on Test Set ===")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=[idx2label[i] for i in range(len(idx2label))],
        digits=4
    ))
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.2)
    
    # Create annotations - show both counts and percentages
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c, p = cm[i, j], cm_percentage[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = f'{c}\n({p:.1f}%)'
    
    # Plot heatmap
    ax = sns.heatmap(
        cm_percentage, 
        annot=annot,
        fmt='',
        cmap='Blues',
        cbar_kws={'label': 'Percentage'},
        xticklabels=[idx2label[i] for i in range(num_classes)],
        yticklabels=[idx2label[i] for i in range(num_classes)],
        vmin=0,
        vmax=100
    )
    
    # Customize plot
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix (Percentage of True Labels)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Save and show
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return metrics for further analysis
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'confusion_matrix': cm,
        'class_names': [idx2label[i] for i in range(num_classes)]
    }

if __name__ == "__main__":
    print(f"Training with {num_classes} emotion classes")
    print(f"Available emotions: {sorted(train_dataset.label2idx.keys())}")
    
    train()  # Your training function
    
    # Evaluate with best model
    results = evaluate_on_test_set(modelName)
    print("Evaluation results:", results)
# emotion_train_improved.py
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
import random

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

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

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ================= CONFIG =================
data_dir = '../dataset'
csv_path = os.path.join(data_dir, 'dataset.csv')
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

input_size = 224
batch_size = 16
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training hyperparameters
initial_lr = 0.001  # Slightly higher initial learning rate
weight_decay = 1e-4
label_smoothing = 0.1
mixup_alpha = 0.2  # Reduced from 0.4 for more stable training
cutmix_alpha = 1.0
augmentation_prob = 0.5  # Reduced from 0.7

# ================ IMPROVED TRANSFORMS ================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Improved data augmentation - less aggressive
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.97, 1.03)),
    # Add some noise for robustness
    transforms.ToTensor(),
    normalize,
])

# Validation and test transforms
val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    normalize,
])

# ============== IMPROVED DATASET ===============
class AffectNetFaceDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, detect_face=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.detect_face = detect_face
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.label2idx = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        
        print(f"Dataset size: {len(self.df)}")
        print(f"Labels: {self.label2idx}")

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

            # Face detection with fallback
            if self.detect_face:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0:
                    # Use the largest face
                    areas = [w * h for (x, y, w, h) in faces]
                    largest_face_idx = np.argmax(areas)
                    x, y, w, h = faces[largest_face_idx]
                    image = image[y:y+h, x:x+w]
                # If no face detected, use the full image

            image = Image.fromarray(image)
            if self.transform:
                image = self.transform(image)

            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            black_image = torch.zeros(3, input_size, input_size)
            return black_image, label

# ============== DATA LOADING ===============
print("Loading dataset...")
df = pd.read_csv(csv_path)
print(f"Total samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Split the dataset: 70% train, 15% validation, 15% test
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

train_dataset = AffectNetFaceDataset(train_df, data_dir, transform=train_transform)
val_dataset = AffectNetFaceDataset(val_df, data_dir, transform=val_transform)
test_dataset = AffectNetFaceDataset(test_df, data_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ============== IMPROVED MODEL =================
def get_improved_model(name='efficientnet'):
    num_classes = len(train_dataset.label2idx)
    dropout_rate = 0.3

    if name == 'mobilenetv3':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(128, num_classes)
        )
    elif name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(256, num_classes)
        )
    elif name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, num_classes)
        )
    elif name == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Get the correct input features for EfficientNet-B0
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate * 0.5),  # Initial dropout
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(256, num_classes)
        )
    elif name == 'efficientnetv2':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Invalid model name: {name}")
    
    return model

# Initialize model
model_name = 'efficientnet'
model = get_improved_model(model_name).to(device)
print(f"Model: {model_name}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============== IMPROVED TRAINING =================
def train():
    model_save_name = f"{model_name}_improved_HQRAF_opencv.pth"
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 15
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training history
    train_losses = []
    train_accs = []
    val_accs = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Apply augmentation techniques
            if np.random.rand() < augmentation_prob:
                if np.random.rand() < 0.5:  # 50% MixUp, 50% CutMix
                    mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                    outputs = model(mixed_images)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    mixed_images, targets_a, targets_b, lam = cutmix_data(images.clone(), labels, alpha=cutmix_alpha)
                    outputs = model(mixed_images)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                
                # For mixed training, approximate accuracy
                _, preds = outputs.max(1)
                correct += (lam * (preds == targets_a).float() + (1 - lam) * (preds == targets_b).float()).sum().item()
                
            else:
                # Regular training
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total += labels.size(0)
            
            # Update progress bar
            if batch_idx % 50 == 0:
                current_acc = correct / total
                progress_bar.set_postfix({
                    'Loss': f'{total_loss/(batch_idx+1):.4f}',
                    'Acc': f'{current_acc:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        
        # Validation phase
        val_acc = evaluate()
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save metrics
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        val_accs.append(val_acc)
        
        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
            }, model_save_name)
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training curves
    plot_training_curves(train_losses, train_accs, val_accs)

def plot_training_curves(train_losses, train_accs, val_accs):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============== EVALUATION =================
def evaluate():
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

# ============== TEST EVALUATION =================
def evaluate_on_test_set():
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    print("Evaluating on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")

    # Classification report
    labels_list = [train_dataset.idx2label[i] for i in range(len(train_dataset.idx2label))]
    print("\n=== Classification Report on Test Set ===")
    print(classification_report(all_labels, all_preds, target_names=labels_list))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, labels_list, test_acc)

def plot_confusion_matrix(cm, labels_list, test_acc):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations
    annotations = np.array([[f"{int(v)}\n{p:.1f}%" for v, p in zip(row_c, row_p)] 
                           for row_c, row_p in zip(cm, cm_normalized)])
    
    sns.heatmap(cm_normalized, annot=annotations, fmt="", cmap="Blues",
                xticklabels=labels_list, yticklabels=labels_list,
                cbar_kws={'label': 'Percentage (%)'})
    
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - Test Accuracy: {test_acc:.4f}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name}_improved.png", dpi=300, bbox_inches='tight')
    plt.show()

# ============== SINGLE IMAGE TESTING =================
def test_single_image(image_path):
    """Test the model on a single image"""
    model.eval()
    
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Face detection
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        image = image[y:y+h, x:x+w]
        print("Face detected and cropped")
    else:
        print("No face detected, using full image")

    # Transform and predict
    image = Image.fromarray(image)
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][pred].item()

    predicted_emotion = train_dataset.idx2label[pred]
    print(f"Predicted Emotion: {predicted_emotion}")
    print(f"Confidence: {confidence:.4f}")
    
    # Show top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities[0], 3)
    print("\nTop 3 predictions:")
    for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
        emotion = train_dataset.idx2label[idx.item()]
        print(f"{i+1}. {emotion}: {prob.item():.4f}")

# ============== MAIN =================
if __name__ == "__main__":
    print("Starting improved emotion recognition training...")
    
    # Train the model
    train()
    
    # Load the best model for evaluation
    model_save_name = f"{model_name}_improved_HQRAF_opencv.pth"
    if os.path.exists(model_save_name):
        print(f"Loading best model from {model_save_name}")
        checkpoint = torch.load(model_save_name, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best validation accuracy was: {checkpoint['best_val_acc']:.4f}")
    
    # Evaluate on test set
    evaluate_on_test_set()
    
    print("Training and evaluation completed!")
    
    # Example usage for single image testing:
    # test_single_image("path/to/your/test/image.jpg")
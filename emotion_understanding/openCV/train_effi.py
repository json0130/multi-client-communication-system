# emotion_train_improved_v2.py
import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import random
from collections import Counter

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

# ================= IMPROVED MIXUP/CUTMIX =================
def mixup_data(x, y, alpha=0.2):
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
batch_size = 32  # Increased batch size
num_epochs = 60  # More epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Improved training hyperparameters
initial_lr = 0.0005  # Lower initial learning rate
weight_decay = 1e-4
label_smoothing = 0.1
mixup_alpha = 0.15  # Reduced mixup
cutmix_alpha = 0.8  # Reduced cutmix
augmentation_prob = 0.3  # Much lower augmentation probability

# ================ IMPROVED TRANSFORMS ================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# More conservative augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
    transforms.RandomGrayscale(p=0.02),
    transforms.ToTensor(),
    normalize,
])

# Test Time Augmentation transforms
tta_transforms = [
    transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ]),
    transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ]),
    transforms.Compose([
        transforms.Resize((int(input_size * 1.1), int(input_size * 1.1))),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ]),
]

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

            # Improved face detection
            if self.detect_face:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=6, 
                    minSize=(50, 50), maxSize=(500, 500)
                )
                if len(faces) > 0:
                    # Use the largest face with some padding
                    areas = [w * h for (x, y, w, h) in faces]
                    largest_face_idx = np.argmax(areas)
                    x, y, w, h = faces[largest_face_idx]
                    
                    # Add padding
                    padding = 0.1
                    pad_x = int(w * padding)
                    pad_y = int(h * padding)
                    x = max(0, x - pad_x)
                    y = max(0, y - pad_y)
                    w = min(image.shape[1] - x, w + 2 * pad_x)
                    h = min(image.shape[0] - y, h + 2 * pad_y)
                    
                    image = image[y:y+h, x:x+w]

            image = Image.fromarray(image)
            if self.transform:
                image = self.transform(image)

            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            black_image = torch.zeros(3, input_size, input_size)
            return black_image, label

# ============== IMPROVED MODEL =================
class ImprovedEfficientNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.4):
        super(ImprovedEfficientNet, self).__init__()
        
        # Use EfficientNet-B1 for better performance
        self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        
        # Get feature dimension
        in_features = self.backbone.classifier[1].in_features
        
        # Improved classifier with attention
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def get_improved_model(num_classes):
    return ImprovedEfficientNet(num_classes, dropout_rate=0.4)

# ============== DATA LOADING WITH BALANCED SAMPLING ===============
print("Loading dataset...")
df = pd.read_csv(csv_path)
print(f"Total samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Split the dataset
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

train_dataset = AffectNetFaceDataset(train_df, data_dir, transform=train_transform)
val_dataset = AffectNetFaceDataset(val_df, data_dir, transform=val_transform)
test_dataset = AffectNetFaceDataset(test_df, data_dir, transform=val_transform)

# Balanced sampling for training
def get_balanced_sampler(dataset, df):
    label_counts = Counter(df['label'])
    total_samples = len(df)
    
    # Calculate weights for each sample
    weights = []
    for _, row in df.iterrows():
        label = row['label']
        weight = total_samples / (len(label_counts) * label_counts[label])
        weights.append(weight)
    
    return WeightedRandomSampler(weights, len(weights), replacement=True)

train_sampler = get_balanced_sampler(train_dataset, train_df)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ============== IMPROVED TRAINING =================
def train():
    model_name = 'efficientnet_b1_improved'
    model_save_name = f"{model_name}_best.pth"
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 20  # Increased patience
    
    # Initialize improved model
    num_classes = len(train_dataset.label2idx)
    model = get_improved_model(num_classes).to(device)
    
    print(f"Model: {model_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    
    # Improved learning rate scheduler - no restarts
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )
    
    # Training history
    train_losses = []
    train_accs = []
    val_accs = []
    
    print("Starting improved training...")
    
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
            
            # Apply augmentation less frequently and only after warmup
            use_augmentation = epoch > 5 and np.random.rand() < augmentation_prob
            
            if use_augmentation:
                if np.random.rand() < 0.6:  # Favor MixUp over CutMix
                    mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                    outputs = model(mixed_images)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    mixed_images, targets_a, targets_b, lam = cutmix_data(images.clone(), labels, alpha=cutmix_alpha)
                    outputs = model(mixed_images)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                
                # Approximate accuracy for mixed training
                _, preds = outputs.max(1)
                correct += (lam * (preds == targets_a).float() + (1 - lam) * (preds == targets_b).float()).sum().item()
                
            else:
                # Regular training
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total += labels.size(0)
            
            # Update progress bar
            if batch_idx % 100 == 0:
                current_acc = correct / total
                progress_bar.set_postfix({
                    'Loss': f'{total_loss/(batch_idx+1):.4f}',
                    'Acc': f'{current_acc:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        
        # Validation phase with TTA
        val_acc = evaluate_with_tta(model, val_loader)
        
        # Learning rate scheduling based on validation accuracy
        scheduler.step(val_acc)
        
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
                'model_name': model_name,
            }, model_save_name)
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training curves
    plot_training_curves(train_losses, train_accs, val_accs, model_name)
    
    return model, model_save_name

# ============== TEST TIME AUGMENTATION =================
def evaluate_with_tta(model, data_loader, use_tta=True):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            if use_tta and len(tta_transforms) > 1:
                # Apply multiple transformations and average predictions
                batch_predictions = []
                for transform in tta_transforms:
                    # Apply transform to each image in the batch
                    transformed_batch = []
                    for img in images:
                        # Convert tensor back to PIL for transformation
                        img_pil = transforms.ToPILImage()(img.cpu())
                        transformed_img = transform(img_pil)
                        transformed_batch.append(transformed_img)
                    
                    transformed_batch = torch.stack(transformed_batch).to(device)
                    outputs = model(transformed_batch)
                    batch_predictions.append(torch.softmax(outputs, dim=1))
                
                # Average predictions
                avg_outputs = torch.mean(torch.stack(batch_predictions), dim=0)
                _, preds = torch.max(avg_outputs, 1)
            else:
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

def plot_training_curves(train_losses, train_accs, val_accs, model_name):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(train_losses, label='Training Loss', linewidth=2)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Training Accuracy', linewidth=2)
    ax2.plot(val_accs, label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add max validation accuracy annotation
    max_val_acc = max(val_accs)
    max_epoch = val_accs.index(max_val_acc)
    ax2.annotate(f'Best Val Acc: {max_val_acc:.4f}', 
                xy=(max_epoch, max_val_acc), 
                xytext=(max_epoch + 5, max_val_acc - 0.02),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============== MAIN =================
if __name__ == "__main__":
    print("Starting improved emotion recognition training...")
    
    # Train the model
    model, model_save_name = train()
    
    # Load the best model for evaluation
    if os.path.exists(model_save_name):
        print(f"Loading best model from {model_save_name}")
        checkpoint = torch.load(model_save_name, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best validation accuracy was: {checkpoint['best_val_acc']:.4f}")
    
    # Evaluate on test set with TTA
    test_acc = evaluate_with_tta(model, test_loader, use_tta=True)
    print(f"Test Accuracy with TTA: {test_acc:.4f}")
    
    print("Training and evaluation completed!")
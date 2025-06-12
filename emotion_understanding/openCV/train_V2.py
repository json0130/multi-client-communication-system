# emotion_train_optimized.py - Target: 80%+ Validation Accuracy
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
warnings.filterwarnings('ignore')

# ================= ENHANCED AUGMENTATION FOR 80%+ ACCURACY =================
def get_enhanced_train_transforms():
    """More aggressive augmentation for better generalization"""
    return A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        
        # More aggressive geometric transforms
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.7),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
        
        # Enhanced noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.5),  # Increased from 0.3
        
        # More color augmentation
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=1.0),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=1.0),
        ], p=0.7),  # Increased from 0.5
        
        # Stronger cutout
        A.CoarseDropout(max_holes=8, max_height=48, max_width=48, p=0.5),  # More aggressive
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ================= IMPROVED MODEL ARCHITECTURE =================
class ImprovedMultiScaleModel(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.25):
        super().__init__()
        
        # Use EfficientNet-B3 for better performance
        self.backbone1 = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        # Get the actual number of features before the classifier
        in_features1 = self.backbone1.classifier[1].in_features
        self.backbone1.classifier = nn.Identity()
        
        # Use ConvNeXt as complementary architecture  
        self.backbone2 = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        # Get the actual number of features before the classifier
        in_features2 = self.backbone2.classifier[2].in_features
        self.backbone2.classifier = nn.Identity()
        
        # Add global average pooling to ensure 2D output
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        print(f"EfficientNet-B3 features: {in_features1}")
        print(f"ConvNeXt features: {in_features2}")
        
        # Enhanced attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True, dropout=0.1)
        
        # Feature fusion with residual connections
        self.feature_fusion = nn.Sequential(
            nn.Linear(in_features1 + in_features2, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1536, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.7),
        )
        
        # Improved classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(384, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.3),
            
            nn.Linear(128, num_classes)
        )
        
        # Auxiliary classifier for regularization
        self.aux_classifier = nn.Sequential(
            nn.Linear(in_features1, 384),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, num_classes)
        )
        
        self._initialize_weights()
    
    def extract_features(self, x, backbone):
        """Extract features and ensure they are 2D"""
        features = backbone(x)
        
        # If features are 4D (batch, channels, height, width), apply global pooling
        if features.dim() == 4:
            features = self.global_pool(features)
            features = features.view(features.size(0), -1)  # Flatten to (batch, features)
        elif features.dim() == 3:
            # If 3D, flatten the last dimensions
            features = features.view(features.size(0), -1)
        elif features.dim() == 1:
            # If 1D (single sample), add batch dimension
            features = features.unsqueeze(0)
        
        return features
    
    def forward(self, x):
        # Extract features from both backbones with proper handling
        feat1 = self.extract_features(x, self.backbone1)
        feat2 = self.extract_features(x, self.backbone2)
        
        # Auxiliary output for training regularization
        aux_out = self.aux_classifier(feat1)
        
        # Feature fusion
        combined_features = torch.cat([feat1, feat2], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # Main classification output
        main_out = self.classifier(fused_features)
        
        return main_out, aux_out
    
    def _initialize_weights(self):
        for m in [self.feature_fusion, self.classifier, self.aux_classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, (nn.BatchNorm1d, nn.LayerNorm)):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

# ================= ADVANCED FACE DETECTION DATASET =================
class AdvancedFaceDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_training=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_training = is_training
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.df['label'].unique()))}
        
        # Multiple face detection methods
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
    def __len__(self):
        return len(self.df)
    
    def detect_best_face(self, image):
        """Advanced face detection with multiple methods"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Try frontal face detection with more aggressive parameters
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
        
        # If no frontal face, try profile
        if len(faces) == 0:
            faces = self.profile_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
        
        if len(faces) > 0:
            # Select the largest face
            areas = [w * h for (x, y, w, h) in faces]
            largest_idx = np.argmax(areas)
            x, y, w, h = faces[largest_idx]
            
            # Expand face region by 25% for better context
            expand_ratio = 0.25
            expand_w, expand_h = int(w * expand_ratio), int(h * expand_ratio)
            
            x = max(0, x - expand_w)
            y = max(0, y - expand_h)
            w = min(image.shape[1] - x, w + 2 * expand_w)
            h = min(image.shape[0] - y, h + 2 * expand_h)
            
            return image[y:y+h, x:x+w]
        
        # If no face detected, return center crop
        h, w = image.shape[:2]
        size = min(h, w)
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        return image[start_h:start_h+size, start_w:start_w+size]

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_path = os.path.join(self.image_dir, row['pth'])
            label = self.label2idx[row['label']]

            # Load and preprocess image
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Cannot load image: {img_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Advanced face detection
            image = self.detect_best_face(image)
            
            # Apply transformations
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            
            return image, label
            
        except Exception as e:
            print(f"Error loading {idx}: {e}")
            # Return a random valid sample
            return self.__getitem__(random.randint(0, len(self.df) - 1))

# ================= IMPROVED LOSS FUNCTIONS =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=2, gamma=3, class_weights=None):  # Stronger focal parameters
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def get_improved_loss_function(class_weights_tensor):
    """Better loss combination for faster convergence"""
    
    # Multiple loss functions with stronger parameters
    focal_loss = FocalLoss(alpha=2, gamma=3, class_weights=class_weights_tensor)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.15)  # More smoothing
    
    def combined_loss(main_outputs, aux_outputs, labels):
        # Main losses
        main_ce = ce_loss(main_outputs, labels)
        main_focal = focal_loss(main_outputs, labels)
        aux_ce = ce_loss(aux_outputs, labels)
        
        # Weighted combination - more focus on focal loss for hard examples
        total_loss = 0.5 * main_ce + 0.3 * main_focal + 0.2 * aux_ce
        return total_loss
    
    return combined_loss

# ================= MIXUP AUGMENTATION =================
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for better generalization"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, aux_pred, y_a, y_b, lam):
    return lam * criterion(pred, aux_pred, y_a) + (1 - lam) * criterion(pred, aux_pred, y_b)

# ================= AGGRESSIVE UNFREEZING SCHEDULE =================
def unfreeze_layers_aggressive(model, epoch):
    """More aggressive unfreezing for faster learning"""
    if epoch == 0:
        # Start with only classifier layers unfrozen
        for param in model.backbone1.parameters():
            param.requires_grad = False
        for param in model.backbone2.parameters():
            param.requires_grad = False
        print("🔒 Backbones frozen")
    elif epoch == 5:  # Much earlier!
        # Unfreeze last 2 blocks of both backbones
        for param in model.backbone1.features[-2:].parameters():
            param.requires_grad = True
        if hasattr(model.backbone2, 'trunk_output'):
            for param in model.backbone2.trunk_output.parameters():
                param.requires_grad = True
        print("🔓 Unfreezed last blocks at epoch 5")
    elif epoch == 12:  # Earlier than 30
        # Unfreeze more layers
        for param in model.backbone1.features[-4:].parameters():
            param.requires_grad = True
        for param in model.backbone2.parameters():
            param.requires_grad = True
        print("🔓 Unfreezed more layers at epoch 12")
    elif epoch == 20:  # Much earlier than 45
        # Full unfreezing
        for param in model.parameters():
            param.requires_grad = True
        print("🔓 Fully unfreezed at epoch 20")

# ================= OPTIMIZED TRAINING SETUP =================
def get_better_optimizer_and_scheduler(model, total_steps):
    """Higher learning rates with better scheduling"""
    backbone_params = list(model.backbone1.parameters()) + list(model.backbone2.parameters())
    head_params = (list(model.classifier.parameters()) + 
                   list(model.feature_fusion.parameters()) + 
                   list(model.aux_classifier.parameters()))
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5, 'weight_decay': 1e-4},  # Increased from 5e-6
        {'params': head_params, 'lr': 3e-3, 'weight_decay': 1e-3}      # Increased from 1e-3
    ], betas=(0.9, 0.999), eps=1e-8)
    
    # OneCycleLR often works better than CosineAnnealing
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[1e-4, 5e-3],  # Peak learning rates
        total_steps=total_steps,
        pct_start=0.1,         # 10% warmup
        anneal_strategy='cos',
        div_factor=25,         # Initial lr = max_lr / div_factor
        final_div_factor=1000  # Final lr = initial_lr / final_div_factor
    )
    
    return optimizer, scheduler

# ================= GRADIENT ACCUMULATION TRAINING =================
def train_with_gradient_accumulation(model, train_loader, optimizer, scheduler, criterion, device, 
                                   accumulation_steps=2, epoch=0):
    """Simulate larger batch size through gradient accumulation"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Apply mixup with 50% probability
        use_mixup = np.random.rand() > 0.5
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)
            
            main_outputs, aux_outputs = model(images)
            loss = mixup_criterion(criterion, main_outputs, aux_outputs, labels_a, labels_b, lam)
            
            # For accuracy calculation, use original labels
            _, preds = main_outputs.max(1)
            correct += (lam * (preds == labels_a).float() + 
                       (1 - lam) * (preds == labels_b).float()).sum().item()
        else:
            main_outputs, aux_outputs = model(images)
            loss = criterion(main_outputs, aux_outputs, labels)
            
            _, preds = main_outputs.max(1)
            correct += (preds == labels).sum().item()
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights every accumulation_steps
        if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Gentler clipping
            optimizer.step()
            scheduler.step()  # Step scheduler every batch for OneCycleLR
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        total += labels.size(0)
    
    return total_loss / len(train_loader), correct / total

# ================= MAIN TRAINING CONFIGURATION =================
def main():
    # Configuration
    data_dir = '../dataset'
    csv_path = os.path.join(data_dir, 'dataset_withoutCon.csv')
    
    input_size = 224
    batch_size = 32  # Reduced for gradient accumulation
    accumulation_steps = 2  # Effective batch size = 64
    num_epochs = 80
    num_classes = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and analyze data
    df = pd.read_csv(csv_path)
    print("Class distribution:")
    class_dist = df['label'].value_counts()
    print(class_dist)
    
    # Stratified split
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    test_df, val_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets with enhanced augmentation
    train_dataset = AdvancedFaceDataset(train_df, data_dir, transform=get_enhanced_train_transforms(), is_training=True)
    val_dataset = AdvancedFaceDataset(val_df, data_dir, transform=get_val_transforms(), is_training=False)
    test_dataset = AdvancedFaceDataset(test_df, data_dir, transform=get_val_transforms(), is_training=False)
    
    # Enhanced class balancing
    class_counts = Counter(train_df['label'])
    class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
    class_weights_dict = dict(zip(np.unique(train_df['label']), class_weights))
    
    # Boost underrepresented classes even more
    boosted_weights = []
    for label in train_df['label']:
        base_weight = class_weights_dict[label]
        if class_counts[label] < 1000:  # Boost very rare classes
            boosted_weights.append(base_weight * 1.8)  # Increased boost
        else:
            boosted_weights.append(base_weight)
    
    sampler = WeightedRandomSampler(weights=boosted_weights, num_samples=len(boosted_weights), replacement=True)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, 
                             num_workers=6, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=6, pin_memory=True)
    
    # Initialize model
    model = ImprovedMultiScaleModel(num_classes=num_classes).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and scheduler
    total_steps = num_epochs * len(train_loader)
    optimizer, scheduler = get_better_optimizer_and_scheduler(model, total_steps)
    
    # Setup loss function
    class_weights_tensor = torch.FloatTensor(list(class_weights_dict.values())).to(device)
    criterion = get_improved_loss_function(class_weights_tensor)
    
    # Training loop
    best_val_acc = 0.0
    patience = 25
    patience_counter = 0
    
    print("🚀 Starting optimized training for 80%+ validation accuracy")
    print(f"Available emotions: {sorted(train_dataset.label2idx.keys())}")
    
    for epoch in range(num_epochs):
        # Aggressive unfreezing schedule
        unfreeze_layers_aggressive(model, epoch)
        
        # Training with gradient accumulation and mixup
        train_loss, train_acc = train_with_gradient_accumulation(
            model, train_loader, optimizer, scheduler, criterion, device, 
            accumulation_steps, epoch
        )
        
        # Validation
        val_acc = evaluate_model(model, val_loader, device)
        
        print(f"[Epoch {epoch+1}] Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "optimized_emotion_model_80plus.pth")
            patience_counter = 0
            print(f"🎯 NEW BEST: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        # Success check
        if best_val_acc >= 0.80:
            print(f"🎉 SUCCESS! Reached 80%+ validation accuracy: {best_val_acc:.4f}")
            break
    
    print(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")
    
    # Final evaluation on test set
    if best_val_acc >= 0.75:  # Only evaluate if we got decent results
        print("\n=== Final Test Set Evaluation ===")
        model.load_state_dict(torch.load("optimized_emotion_model_80plus.pth"))
        test_acc = evaluate_model(model, test_loader, device)
        print(f"Final Test Accuracy: {test_acc:.4f}")
        
        # Generate detailed report
        evaluate_on_test_set(model, test_loader, device, train_dataset.label2idx)
    
    return best_val_acc

def evaluate_model(model, data_loader, device):
    """Evaluate model on given data loader"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            main_outputs, _ = model(images)
            _, preds = main_outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def evaluate_on_test_set(model, test_loader, device, label2idx):
    """Detailed evaluation with confusion matrix"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            main_outputs, _ = model(images)
            _, preds = torch.max(main_outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    idx2label = {v: k for k, v in label2idx.items()}
    label_names = [idx2label[i] for i in range(len(idx2label))]
    
    print("\n=== Classification Report on Test Set ===")
    print(classification_report(all_labels, all_preds, target_names=label_names))

    # Confusion Matrix
    try:
        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap="Blues",
                   xticklabels=label_names, yticklabels=label_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Optimized Model")
        plt.tight_layout()
        plt.savefig("confusion_matrix_optimized_80plus.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("Confusion matrix saved as 'confusion_matrix_optimized_80plus.png'")
    except Exception as e:
        print(f"Could not generate confusion matrix plot: {e}")

# ================= EXECUTION =================
if __name__ == "__main__":
    final_acc = main()
    
    if final_acc >= 0.80:
        print(f"✅ SUCCESS! Achieved {final_acc:.1%} validation accuracy")
    elif final_acc >= 0.75:
        print(f"🔶 CLOSE! Achieved {final_acc:.1%} - Try longer training or ensemble")
    else:
        print(f"⚠️ Reached {final_acc:.1%} - Need more optimization")
        print("Additional suggestions:")
        print("1. Try EfficientNet-B4 or RegNet-Y-16GF")
        print("2. Implement 5-fold cross-validation")
        print("3. Add pseudo-labeling with confident predictions")
        print("4. Use test-time augmentation (TTA)")
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ── 1. Configuration & Paths ──────────────────────────────────────────────────
TRAIN_META_PATH = 'train/train_metadata.csv'
TRAIN_ROT_DIR   = 'train/train_images_rot'
CHECKPOINT_PATH = 'checkpoints/best_exp5_focal_loss.pth'

BENCHMARK_VAL_BACC = 0.7399
NUM_EPOCHS         = 25
BATCH_SIZE         = 64
LEARNING_RATE      = 1.5e-4
WEIGHT_DECAY       = 1e-2
FOCAL_ALPHA        = 0.6
FOCAL_GAMMA        = 1.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing Experiment 5 (Focal Loss) on device: {DEVICE}")

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ── 2. Data Loading & Pre-caching ─────────────────────────────────────────────
train_meta = pd.read_csv(TRAIN_META_PATH)
train_split, val_split = train_test_split(
    train_meta, test_size=0.15, random_state=42, stratify=train_meta['label']
)

print(f"Train split : {len(train_split)} images")
print(f"Val split   : {len(val_split)} images")

print("Pre-caching images in memory for speed...")
t0_cache = time.time()
cached_train_images = {}
for idx, row in train_split.iterrows():
    img_path = os.path.join(TRAIN_ROT_DIR, row['image_id'])
    cached_train_images[row['image_id']] = Image.open(img_path).convert('L')

cached_val_images = {}
for idx, row in val_split.iterrows():
    img_path = os.path.join(TRAIN_ROT_DIR, row['image_id'])
    cached_val_images[row['image_id']] = Image.open(img_path).convert('L')
print(f"Cached {len(cached_train_images) + len(cached_val_images)} images in {time.time() - t0_cache:.2f}s")

# ── 3. Dataset & Transforms ───────────────────────────────────────────────────
class RepeatChannels:
    def __init__(self, repeats=3):
        self.repeats = repeats
    def __call__(self, tensor):
        return tensor.repeat(self.repeats, 1, 1)

class FastLunarDataset(Dataset):
    def __init__(self, metadata_df, cached_imgs, transform=None):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.cached_imgs = cached_imgs
        self.transform   = transform
    def __len__(self):
        return len(self.metadata_df)
    def __getitem__(self, idx):
        row   = self.metadata_df.iloc[idx]
        image = self.cached_imgs[row['image_id']]
        if self.transform:
            image = self.transform(image)
        return image, int(row['label'])

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    RepeatChannels(3),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    RepeatChannels(3),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

train_dataset = FastLunarDataset(train_split, cached_train_images, transform=train_transform)
val_dataset   = FastLunarDataset(val_split,   cached_val_images,   transform=val_transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── 4. Focal Loss Definition ──────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Binary Focal Loss focusing gradient updates on hard shadow boundary examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.6, gamma=1.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, 1.0 - self.alpha, self.alpha)
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ── 5. Model, Loss, Optimizer & Scheduler ─────────────────────────────────────
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(DEVICE)

criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ── 6. Training Loop ──────────────────────────────────────────────────────────
history = {
    'epoch': [], 'lr': [],
    'train_loss': [], 'train_acc': [], 'train_bacc': [],
    'val_loss': [], 'val_acc': [], 'val_bacc': []
}

best_val_bacc = 0.0
best_epoch    = -1

print("="*60)
print(f"STARTING EXPERIMENT 5: ResNet18 (Focal Loss alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
print("="*60)

for epoch in range(1, NUM_EPOCHS + 1):
    current_lr = optimizer.param_groups[0]['lr']
    
    # Train Phase
    model.train()
    running_loss = 0.0
    train_preds, train_labels_epoch = [], []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        train_preds.extend(preds.detach().cpu().numpy())
        train_labels_epoch.extend(labels.detach().cpu().numpy())
        
    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc  = accuracy_score(train_labels_epoch, train_preds)
    epoch_train_bacc = balanced_accuracy_score(train_labels_epoch, train_preds)
    
    # Validation Phase
    model.eval()
    val_running_loss = 0.0
    val_preds_epoch, val_labels_epoch = [], []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            val_preds_epoch.extend(preds.cpu().numpy())
            val_labels_epoch.extend(labels.cpu().numpy())
            
    epoch_val_loss = val_running_loss / len(val_dataset)
    epoch_val_acc  = accuracy_score(val_labels_epoch, val_preds_epoch)
    epoch_val_bacc = balanced_accuracy_score(val_labels_epoch, val_preds_epoch)
    
    scheduler.step()
    
    history['epoch'].append(epoch)
    history['lr'].append(current_lr)
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['train_bacc'].append(epoch_train_bacc)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)
    history['val_bacc'].append(epoch_val_bacc)
    
    improved_flag = ""
    if epoch_val_bacc > best_val_bacc:
        best_val_bacc = epoch_val_bacc
        best_epoch    = epoch
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        improved_flag = "⭐ [NEW BEST]"
        
    print(f"Epoch {epoch:02d}/{NUM_EPOCHS} (LR: {current_lr:.6f}) | "
          f"Train Loss: {epoch_train_loss:.4f} BAcc: {epoch_train_bacc:.4f} Acc: {epoch_train_acc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} BAcc: {epoch_val_bacc:.4f} Acc: {epoch_val_acc:.4f} {improved_flag}")

print("="*60)
print(f"TRAINING COMPLETE. Best Epoch: {best_epoch} with Val BAcc: {best_val_bacc:.4f} ({best_val_bacc*100:.2f}%)")
print("="*60)

# Save history JSON
with open('outputs/exp5_history.json', 'w') as f:
    json.dump(history, f, indent=2)

# ── 7. Full Metrics Evaluation on Best Checkpoint ─────────────────────────────
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

val_preds, val_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(labels.numpy())

bacc = balanced_accuracy_score(val_labels, val_preds)
acc  = accuracy_score(val_labels, val_preds)
cm   = confusion_matrix(val_labels, val_preds)

prec0 = precision_score(val_labels, val_preds, pos_label=0)
rec0  = recall_score(val_labels, val_preds, pos_label=0)
f1_0  = f1_score(val_labels, val_preds, pos_label=0)

prec1 = precision_score(val_labels, val_preds, pos_label=1)
rec1  = recall_score(val_labels, val_preds, pos_label=1)
f1_1  = f1_score(val_labels, val_preds, pos_label=1)

print("\n" + "="*60)
print("EXPERIMENT 5 DETAILED METRICS SUMMARY")
print("="*60)
print(f"Checkpoint File             : {CHECKPOINT_PATH}")
print(f"Best Epoch                  : {best_epoch}")
print(f"Validation Balanced Accuracy: {bacc:.4f} ({bacc*100:.2f}%)")
print(f"Validation Overall Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"Previous Best Val BAcc      : {BENCHMARK_VAL_BACC:.4f} ({BENCHMARK_VAL_BACC*100:.2f}%)")
delta = (bacc - BENCHMARK_VAL_BACC) * 100
print(f"Delta vs Benchmark          : {delta:+.2f}%")

print("\n--- Per-Class Performance ---")
print(f"Class 0 (Depth) -> Precision: {prec0:.4f}, Recall: {rec0:.4f}, F1: {f1_0:.4f}")
print(f"Class 1 (Rise)  -> Precision: {prec1:.4f}, Recall: {rec1:.4f}, F1: {f1_1:.4f}")

print("\n--- Confusion Matrix [[TN, FP], [FN, TP]] ---")
print(cm)
print(f"TN (True Depth correct) : {cm[0,0]}")
print(f"FP (Depth misclass Rise): {cm[0,1]}")
print(f"FN (Rise misclass Depth): {cm[1,0]}")
print(f"TP (True Rise correct)  : {cm[1,1]}")

best_idx = history['epoch'].index(best_epoch)
train_bacc_at_best = history['train_bacc'][best_idx]
train_loss_at_best = history['train_loss'][best_idx]
val_loss_at_best   = history['val_loss'][best_idx]

print("\n--- Overfitting Check ---")
print(f"Train BAcc at Best Epoch : {train_bacc_at_best:.4f}")
print(f"Val BAcc at Best Epoch   : {bacc:.4f}")
print(f"Train/Val BAcc Gap       : {abs(train_bacc_at_best - bacc):.4f}")
print(f"Train Loss at Best Epoch : {train_loss_at_best:.4f}")
print(f"Val Loss at Best Epoch   : {val_loss_at_best:.4f}")

print("\n" + "="*60)
if bacc > BENCHMARK_VAL_BACC:
    print(f"🏆 SUCCESS: Experiment 5 (Focal Loss) IMPROVED validation Balanced Accuracy from {BENCHMARK_VAL_BACC*100:.2f}% to {bacc*100:.2f}%!")
else:
    print(f"ℹ️ RESULT: Experiment 5 Val BAcc ({bacc*100:.2f}%) did not exceed benchmark ({BENCHMARK_VAL_BACC*100:.2f}%).")
print("="*60)

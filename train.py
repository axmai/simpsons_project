import os
import random
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "data/characters_train"   # path to training data folder
MODEL_PATH = "model.pth"          # where to save the trained model
IMG_SIZE   = 64                   # resize images to 64×64 (fast on CPU)
BATCH_SIZE = 32
EPOCHS     = 25
LR         = 1e-3
VAL_SPLIT  = 0.15                 # 15% of training data used for validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ── Data transforms ───────────────────────────────────────────────────────────
# Training: some augmentation to help generalisation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Validation: no augmentation
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
# ── Load dataset ──────────────────────────────────────────────────────────────
# ImageFolder reads class names from subdirectory names automatically
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)

class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Found {len(full_dataset)} images across {num_classes} classes")
print("Classes:", class_names)
# ── Train / validation split ──────────────────────────────────────────────────
val_size   = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

# Apply the no-augmentation transform to the validation split
val_dataset.dataset = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {train_size} | Val samples: {val_size}")
# ── Model definition ──────────────────────────────────────────────────────────
# A simple CNN with 3 conv blocks + a small classifier head.
# Designed to be fast enough to train on CPU in reasonable time.

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # Each conv block: Conv → BatchNorm → ReLU → MaxPool
        self.features = nn.Sequential(
            # Block 1: 3 → 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 64 → 32

            # Block 2: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 32 → 16

            # Block 3: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 16 → 8

            # Block 4: 128 → 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # always 4×4 regardless of input size
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SimpleCNN(num_classes=num_classes).to(device)
print(model)

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal trainable parameters: {total_params:,}")
# ── Loss, optimizer, scheduler ────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# Reduce LR by half if validation loss doesn't improve for 3 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


best_val_loss = float("inf")
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}")
print("-" * 55)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss,   val_acc   = evaluate(model, val_loader, criterion)

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.4f}  {val_loss:>9.4f}  {val_acc:>8.4f}")

    # Save the best model checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "class_names":  class_names,
            "img_size":     IMG_SIZE,
        }, MODEL_PATH)
        print(f"         ✓ Saved best model (val_loss={val_loss:.4f})")

print("\nTraining complete. Best val loss:", round(best_val_loss, 4))
# ── Plot training curves ──────────────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"],   label="Val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history["train_acc"], label="Train")
axes[1].plot(history["val_acc"],   label="Val")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=100)
plt.show()
print("Curves saved to training_curves.png")
# ── Quick sanity check: reload model and confirm it works ─────────────────────
checkpoint = torch.load(MODEL_PATH, map_location=device)
print("Saved at epoch:", checkpoint["epoch"])
print("Classes stored:", checkpoint["class_names"])
print("Model file saved at:", os.path.abspath(MODEL_PATH))
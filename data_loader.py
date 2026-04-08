import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# CPU-friendly data amount (from Data/train only)
MAX_TRAIN_SAMPLES = 2000
VAL_RATIO = 0.15

# DataLoader knobs for Windows CPU
BATCH_SIZE = 16
NUM_WORKERS = 0

# Train transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Eval transform (no random augmentation)
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Build train+val from Data/train only (different transforms, same indices)
train_aug_dataset = datasets.ImageFolder("Data/train", transform=train_transform)
train_eval_dataset = datasets.ImageFolder("Data/train", transform=eval_transform)

all_indices = torch.randperm(len(train_aug_dataset), generator=torch.Generator().manual_seed(SEED)).tolist()

# Optional cap for speed
capped_count = min(MAX_TRAIN_SAMPLES, len(all_indices))
all_indices = all_indices[:capped_count]

val_count = max(1, int(len(all_indices) * VAL_RATIO))
train_count = len(all_indices) - val_count

train_indices = all_indices[:train_count]
val_indices = all_indices[train_count:]

train_data = Subset(train_aug_dataset, train_indices)
val_data = Subset(train_eval_dataset, val_indices)

# Keep full held-out test set
test_data = datasets.ImageFolder("Data/test", transform=eval_transform)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)
test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

print("Train size:", len(train_data))
print("Val size:", len(val_data))
print("Test size:", len(test_data))

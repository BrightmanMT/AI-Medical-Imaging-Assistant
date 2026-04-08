import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import train_loader, val_loader, test_loader
from resnet_model import get_resnet_model

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Device
device = torch.device("cpu")
PNEUMONIA_THRESHOLD = 0.7

# Load model
model = get_resnet_model().to(device)


def extract_labels(dataset):
    """Extract labels from an ImageFolder dataset or a Subset of it."""
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base_dataset = dataset.dataset
        indices = dataset.indices
        if hasattr(base_dataset, "targets"):
            return [base_dataset.targets[i] for i in indices]
        return [base_dataset.samples[i][1] for i in indices]

    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    return [label for _, label in dataset.samples]


# Class-weighted loss (helps imbalance)
train_labels = extract_labels(train_loader.dataset)
class_counts = torch.bincount(torch.tensor(train_labels, dtype=torch.long), minlength=2).float()
class_weights = class_counts.sum() / (len(class_counts) * class_counts.clamp(min=1.0))
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimize only trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-4)

# Reduce LR when validation F1 plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)


def evaluate(loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)

            running_loss += loss.item()
            predicted = (probs[:, 1] >= PNEUMONIA_THRESHOLD).long()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

    eps = 1e-8
    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# Training loop
epochs = 8
best_val_f1 = 0.0
best_state = None
epochs_without_improvement = 0
early_stop_patience = 3

print(f"Train class counts: {class_counts.tolist()}")
print(f"Class weights: {[round(x, 4) for x in class_weights.tolist()]}")
print(f"Pneumonia decision threshold: {PNEUMONIA_THRESHOLD}")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    val_metrics = evaluate(val_loader)
    scheduler.step(val_metrics["f1"])
    current_lr = optimizer.param_groups[0]["lr"]

    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.2f}% | "
        f"Val F1: {val_metrics['f1']:.4f} | "
        f"Val Recall: {val_metrics['recall']:.4f} | "
        f"LR: {current_lr:.6f}"
    )

    if epochs_without_improvement >= early_stop_patience:
        print(
            f"Early stopping at epoch {epoch+1} "
            f"(no val F1 improvement for {early_stop_patience} epochs)."
        )
        break

# Restore best validation model before final test
if best_state is not None:
    model.load_state_dict(best_state)
    torch.save(best_state, "model.pth")

print("Training complete!")
print(f"Best Val F1: {best_val_f1:.4f}")
print("Saved best model to model.pth")

# Final test evaluation
test_metrics = evaluate(test_loader)
print(f"Test Loss: {test_metrics['loss']:.4f}")
print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
print(f"Test Precision: {test_metrics['precision']:.4f}")
print(f"Test Recall: {test_metrics['recall']:.4f}")
print(f"Test F1: {test_metrics['f1']:.4f}")
print(
    f"Test Confusion Matrix: [[TN={test_metrics['tn']}, FP={test_metrics['fp']}], "
    f"[FN={test_metrics['fn']}, TP={test_metrics['tp']}]]"
)

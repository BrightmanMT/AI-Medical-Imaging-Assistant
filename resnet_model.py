import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def get_resnet_model():
    """Build ResNet18 for 2-class classification with robust offline fallback."""
    try:
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        pretrained_loaded = True
    except Exception:
        # Offline fallback when pretrained weights cannot be downloaded.
        model = models.resnet18(weights=None)
        pretrained_loaded = False

    if pretrained_loaded:
        # Freeze backbone, then unfreeze the last residual block for better adaptation.
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Replace final classifier for binary prediction with dropout regularization.
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 2),
    )

    return model

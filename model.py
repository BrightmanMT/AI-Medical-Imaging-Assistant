import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 🧠 First convolution layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        
        # 🧠 Second convolution layer
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        # 🧠 Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # 🧠 Fully connected layers
        self.fc1 = nn.Linear(32 * 56 * 56, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)  # 2 classes

    def forward(self, x):
        # Layer 1
        x = self.pool(F.relu(self.conv1(x)))  
        
        # Layer 2
        x = self.pool(F.relu(self.conv2(x)))  

        # Flatten
        x = x.view(-1, 32 * 56 * 56)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Excitation
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Input shape: (batch, channels, height, width)
        batch, channels, height, width = x.shape
        
        # Squeeze
        squeezed = self.gap(x).view(batch, channels)
        
        # Excitation
        weights = F.relu(self.fc1(squeezed))
        weights = self.sigmoid(self.fc2(weights))
        
        # Rescale
        weights = weights.view(batch, channels, 1, 1)
        output = x * weights.expand_as(x)
        
        return output
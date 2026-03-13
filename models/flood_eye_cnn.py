import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import SimpleAttention, DynamicConv

class FloodEyeCNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=5):
        super().__init__()
        # High-resolution path
        self.hi_res_path = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Low-resolution path
        self.lo_res_path = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.attn = SimpleAttention(128) # Fused channels
        self.dyn  = DynamicConv(128, 64, 3)
        self.out  = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        h = self.hi_res_path(x)
        l = self.lo_res_path(x)
        
        # Cross-path fusion
        fused = torch.cat([h, l], dim=1)
        x = self.attn(fused)
        x = self.dyn(x)
        x = self.out(x)
        
        # Upsample to match input size if needed (already 256 in mock)
        return x

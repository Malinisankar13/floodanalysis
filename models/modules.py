import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key   = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        # Pool spatial dimensions to 32x32 for attention
        pool_h, pool_w = 32, 32
        x_pooled = F.adaptive_avg_pool2d(x, (pool_h, pool_w))
        
        q = self.query(x_pooled).view(B, C, -1)
        k = self.key(x_pooled).view(B, C, -1)
        v = self.value(x_pooled).view(B, C, -1)

        attn = torch.softmax(torch.bmm(q.transpose(1,2), k) * self.scale, dim=-1)
        out_pooled = torch.bmm(attn, v.transpose(1,2)).transpose(1,2).view(B, C, pool_h, pool_w)
        
        # Upsample back to original size
        out = F.interpolate(out_pooled, size=(H, W), mode='bilinear', align_corners=False)
        return out + x

class DynamicConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch * in_ch * k * k, 1)
        )
        self.k = k
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        weights = self.weight_gen(x).view(B, self.out_ch, self.in_ch, self.k, self.k)
        outs = []
        for b in range(B):
            out = F.conv2d(x[b:b+1], weights[b], padding=self.k//2)
            outs.append(out)
        return torch.cat(outs, dim=0)

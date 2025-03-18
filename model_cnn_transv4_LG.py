import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class Generator(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        upsample_block_num = int(math.log(scale_factor, 2))
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=9, padding=4),  # Reduced channels from 64 → 48
            nn.ReLU()
        )
        
        self.transformer_blocks = nn.ModuleList([
            EfficientTransformerBlock(
                dim=48,
                num_heads=8,  # Reduced heads from 32 → 8
                chunk_size=128  # Increased chunk size to process more at once
            ) for _ in range(4)  # Reduced blocks from 6 → 4
        ])
        
        self.block7 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.GroupNorm(4, 48)  # Using GroupNorm instead of BatchNorm
        )
        
        self.block8 = nn.Sequential(*[UpsampleBlock(48) for _ in range(upsample_block_num)])
        
        self.block9 = nn.Conv2d(48, 3, kernel_size=9, padding=4)
        
    def forward(self, x):
        with torch.no_grad() if not self.training else torch.enable_grad():
            block1 = self.block1(x)
            features = block1
            
            for transformer in self.transformer_blocks:
                features = features + transformer(features)
            
            block7 = self.block7(features)
            block8 = self.block8(block1 + block7)
            block9 = self.block9(block8)
            return (torch.tanh(block9) + 1) / 2


class EfficientTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, chunk_size=128):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.dim = dim
        self.scale = (dim // num_heads) ** -0.5
        
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        out = self._regular_attention(x_flat)  # Removed chunked attention for simplicity
        out = out.transpose(1, 2).view(B, C, H, W)
        return out
    
    def _regular_attention(self, x_flat):
        B, N, C = x_flat.shape
        x_ln = self.norm1(x_flat)
        qkv = self.qkv(x_ln).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x_flat = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_flat = self.proj(x_flat)
        return x_flat


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU()  # Changed from PReLU to ReLU
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),  # Reduced channels from 64 → 48
            nn.LeakyReLU(0.2),

            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 48),
            nn.LeakyReLU(0.2),

            nn.Conv2d(48, 96, kernel_size=3, padding=1),  # Reduced channels from 128 → 96
            nn.GroupNorm(4, 96),
            nn.LeakyReLU(0.2),

            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 96),
            nn.LeakyReLU(0.2),

            nn.Conv2d(96, 192, kernel_size=3, padding=1),  # Reduced channels from 256 → 192
            nn.GroupNorm(4, 192),
            nn.LeakyReLU(0.2),

            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 192),
            nn.LeakyReLU(0.2),

            nn.Conv2d(192, 256, kernel_size=3, padding=1),  # Reduced channels from 512 → 256
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 512, kernel_size=1),  # Reduced final layer from 1024 → 512
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))

import math
import torch
from torch import nn
from torch.nn import functional as F

class Generator(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        upsample_block_num = int(math.log(scale_factor, 2))
        
        # Initial feature extraction
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        # Transformer-based processing blocks
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(
                dim=64,
                num_heads=8,
                ffn_expansion=2,
                bias=False
            ) for _ in range(6)
        ])
        
        # Final conv before upsampling
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Upsampling with attention-enhanced blocks
        self.upsample_blocks = nn.Sequential(*[
            UpsampleBlockWithAttention(64, 2) 
            for _ in range(upsample_block_num)
        ])
        
        self.final_conv = nn.Conv2d(64, 3, kernel_size=9, padding=4)
        
    def forward(self, x):
        # Initial features
        x1 = self.block1(x)
        
        # Transformer processing
        x = self.transformer_blocks(x1)
        
        # Residual connection
        x = self.block7(x) + x1
        
        # Upsampling
        x = self.upsample_blocks(x)
        x = self.final_conv(x)
        
        return (torch.tanh(x) + 1) / 2


class TransformerBlock(nn.Module):
    """Efficient Transformer Block using depth-wise convolutions and local attention."""
    def __init__(self, dim, num_heads, ffn_expansion=2, bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads=num_heads, bias=bias)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion, bias)
        
    def forward(self, x):
        # Reshape for Transformer operations
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Attention with residual
        x = x + self.attn(self.norm1(x_flat)).reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # FFN with residual
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x = x + self.ffn(self.norm2(x_flat)).reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return x


class EfficientAttention(nn.Module):
    """Efficient multi-head self-attention with depth-wise convolution."""
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, groups=dim*3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        qkv = self.dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, C//self.num_heads, H*W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, C//self.num_heads, H*W)
        v = v.view(B, self.num_heads, C//self.num_heads, H*W).permute(0, 1, 3, 2)
        
        # Scaled dot-product attention
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        x = self.proj(x)
        return x.permute(0, 2, 3, 1).reshape(B, N, C)


class FeedForward(nn.Module):
    """Feed-forward network with depth-wise convolution."""
    def __init__(self, dim, expansion=2, bias=False):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, bias=bias),
        )
        
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.proj(x)
        return x.permute(0, 2, 3, 1).reshape(B, N, C)


class UpsampleBlockWithAttention(nn.Module):
    """Upsampling block with spatial attention."""
    def __init__(self, in_channels, up_scale):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale**2, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.attention = SpatialAttention(in_channels)
        self.act = nn.PReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.act(x)
        x = self.attention(x) * x
        return x


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.GELU(),
            nn.Conv2d(channels//8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.conv(x)


# Keep Discriminator same as original
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))
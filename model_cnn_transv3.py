import math
import torch
from torch import nn
from torch.nn import functional as F

class Generator(nn.Module):
    def __init__(self, scale_factor, base_channels=64, num_transformer_blocks=6):
        super().__init__()
        self.scale_factor = scale_factor
        upsample_block_num = int(math.log(scale_factor, 2))
        
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, padding=3),
            nn.InstanceNorm2d(base_channels),
            nn.PReLU()
        )
        
        self.channel_attention = ChannelAttention(base_channels)
        
        # Modified transformer blocks with adaptive window size
        self.transformer_blocks = nn.ModuleList([
            AdaptiveTransformerBlock(
                dim=base_channels,
                num_heads=8,
                ffn_expansion=4,
                use_bias=False,
                dropout=0.1
            ) for _ in range(num_transformer_blocks)
        ])
        
        self.upsampling_blocks = nn.ModuleList([
            ProgressiveUpsampleBlock(
                base_channels,
                scale_factor=2,
                use_residual=True
            ) for _ in range(upsample_block_num)
        ])
        
        self.refinement = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.PReLU(),
            nn.Conv2d(base_channels, 3, kernel_size=7, padding=3)
        )
        
    def forward(self, x):
        features = self.feature_extraction(x)
        features = self.channel_attention(features)
        identity = features
        
        for transformer in self.transformer_blocks:
            features = transformer(features) + features
            
        features = features + identity
        
        for upsample in self.upsampling_blocks:
            features = upsample(features)
        
        output = self.refinement(features)
        return torch.tanh(output)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        out = avg_out + max_out
        return x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3)


class AdaptiveTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion, use_bias, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.attn = AdaptiveSelfAttention(
            dim=dim,
            num_heads=num_heads,
            use_bias=use_bias,
            dropout=dropout
        )
        
        self.ffn = ImprovedFeedForward(
            dim=dim,
            expansion=ffn_expansion,
            dropout=dropout
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Attention
        x_flat = x_flat + self.attn(self.norm1(x_flat))
        
        # FFN
        x_flat = x_flat + self.ffn(self.norm2(x_flat))
        
        # Restore spatial dimensions
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        return x


class AdaptiveSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, use_bias, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=use_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class ImprovedFeedForward(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class ProgressiveUpsampleBlock(nn.Module):
    def __init__(self, channels, scale_factor=2, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        
        self.conv1 = nn.Conv2d(channels, channels * scale_factor**2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.activation = nn.PReLU()
        
        if use_residual:
            self.residual_conv = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(channels, channels, kernel_size=1)
            )
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.activation(x)
        
        if self.use_residual:
            x = x + self.residual_conv(identity)
        
        return x
    

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
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
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        self.transformer_blocks = nn.ModuleList([
            EfficientTransformerBlock(
                dim=64,
                num_heads=32,
                chunk_size=64  # Reduced chunk size
            ) for _ in range(6)
        ])
        
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        block8 = [UpsampleBlock(64) for _ in range(upsample_block_num)]
        self.block8 = nn.Sequential(*block8)
        
        self.block9 = nn.Conv2d(64, 3, kernel_size=9, padding=4)
        
    def forward(self, x):
        if self.training:
            return self._forward_train(x)
        return self._forward_eval(x)
    
    def _forward_train(self, x):
        block1 = self.block1(x)
        features = block1
        
        for transformer in self.transformer_blocks:
            features = features + checkpoint(transformer, features)
        
        block7 = self.block7(features)
        block8 = self.block8(block1 + block7)
        block9 = self.block9(block8)
        return (torch.tanh(block9) + 1) / 2
    
    def _forward_eval(self, x):
        block1 = self.block1(x)
        features = block1
        
        for transformer in self.transformer_blocks:
            features = features + transformer(features)
        
        block7 = self.block7(features)
        block8 = self.block8(block1 + block7)
        block9 = self.block9(block8)
        return (torch.tanh(block9) + 1) / 2


class EfficientTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, chunk_size=64):
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
        N = H * W
        
        # Reshape to sequence
        x_flat = x.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Use chunked attention for large spatial dimensions
        if N > self.chunk_size * self.chunk_size and not self.training:
            out = self._chunked_attention(x_flat, H, W)
        else:
            out = self._regular_attention(x_flat)
            
        # Restore spatial dimensions
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
    
    def _chunked_attention(self, x_flat, H, W):
        B, N, C = x_flat.shape
        
        # Calculate chunk dimensions
        chunk_size = min(self.chunk_size, H)
        num_chunks_h = math.ceil(H / chunk_size)
        num_chunks_w = math.ceil(W / chunk_size)
        
        # Process chunks
        chunks_out = []
        for h in range(num_chunks_h):
            for w in range(num_chunks_w):
                h_start = h * chunk_size
                w_start = w * chunk_size
                h_end = min((h + 1) * chunk_size, H)
                w_end = min((w + 1) * chunk_size, W)
                
                # Extract chunk indices
                chunk_indices = []
                for i in range(h_start, h_end):
                    chunk_indices.extend(range(i * W + w_start, i * W + w_end))
                
                # Process chunk
                if chunk_indices:
                    chunk = x_flat[:, chunk_indices, :]
                    chunk_out = self._regular_attention(chunk)
                    chunks_out.append((chunk_indices, chunk_out))
        
        # Reconstruct output
        output = torch.zeros_like(x_flat)
        for indices, chunk_out in chunks_out:
            output[:, indices, :] = chunk_out
        
        return output


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
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
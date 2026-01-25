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
        
        # FIXED: Reduced from 6 to 4 transformer blocks
        # FIXED: Reduced num_heads from 32 to 4 (8x memory reduction)
        self.transformer_blocks = nn.ModuleList([
            EfficientTransformerBlock(
                dim=64,
                num_heads=4,  # Reduced from 32
                chunk_size=32  # Reduced from 64 for better memory efficiency
            ) for _ in range(4)  # Reduced from 6
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
            features = features + checkpoint(transformer, features, use_reentrant=False)
        
        block7 = self.block7(features)
        block8 = self.block8(block1 + block7)
        block9 = self.block9(block8)
        return (torch.tanh(block9) + 1) / 2
    
    def _forward_eval(self, x):
        block1 = self.block1(x)
        features = block1
        
        # Process transformer blocks one at a time and clear cache
        for transformer in self.transformer_blocks:
            with torch.no_grad():
                features = features + transformer(features)
                # Clear CUDA cache after each transformer block
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        block7 = self.block7(features)
        block8 = self.block8(block1 + block7)
        block9 = self.block9(block8)
        return (torch.tanh(block9) + 1) / 2


class EfficientTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, chunk_size=32):
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
        
        # FIXED: Always use chunked attention for large inputs to save memory
        # Threshold reduced to be more aggressive
        if N > (self.chunk_size * self.chunk_size // 2):
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
        
        # Use scaled dot-product attention (more memory efficient)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x_flat = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_flat = self.proj(x_flat)
        
        return x_flat
    
    def _chunked_attention(self, x_flat, H, W):
        """Memory-efficient chunked attention processing"""
        B, N, C = x_flat.shape
        
        # Calculate chunk dimensions
        chunk_size = min(self.chunk_size, H, W)
        num_chunks_h = math.ceil(H / chunk_size)
        num_chunks_w = math.ceil(W / chunk_size)
        
        # Reshape to spatial grid for easier chunking
        x_spatial = x_flat.view(B, H, W, C)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        output_spatial = output.view(B, H, W, C)
        
        # Process chunks with overlap to maintain context
        overlap = chunk_size // 4  # 25% overlap
        
        for h in range(num_chunks_h):
            for w in range(num_chunks_w):
                h_start = max(0, h * chunk_size - overlap)
                w_start = max(0, w * chunk_size - overlap)
                h_end = min(H, (h + 1) * chunk_size + overlap)
                w_end = min(W, (w + 1) * chunk_size + overlap)
                
                # Extract chunk
                chunk = x_spatial[:, h_start:h_end, w_start:w_end, :].contiguous()
                chunk_flat = chunk.view(B, -1, C)
                
                # Process chunk
                chunk_out = self._regular_attention(chunk_flat)
                
                # Calculate actual write region (without overlap)
                write_h_start = h * chunk_size
                write_w_start = w * chunk_size
                write_h_end = min(H, (h + 1) * chunk_size)
                write_w_end = min(W, (w + 1) * chunk_size)
                
                # Calculate offset in chunk
                offset_h = write_h_start - h_start
                offset_w = write_w_start - w_start
                chunk_h = write_h_end - write_h_start
                chunk_w = write_w_end - write_w_start
                
                # Extract the non-overlapping part
                chunk_out_spatial = chunk_out.view(B, h_end - h_start, w_end - w_start, C)
                chunk_out_core = chunk_out_spatial[:, offset_h:offset_h+chunk_h, offset_w:offset_w+chunk_w, :]
                
                # Write to output
                output_spatial[:, write_h_start:write_h_end, write_w_start:write_w_end, :] = chunk_out_core
                
                # Clear intermediate tensors
                del chunk, chunk_flat, chunk_out, chunk_out_spatial, chunk_out_core
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
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

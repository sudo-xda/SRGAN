import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# (Optional) small perf wins on Ampere+
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    # PyTorch 2.x: prefer flash/mem-efficient SDPA when available
    import torch.backends.cuda as cuda_backends
    if hasattr(cuda_backends, "sdp_kernel"):
        cuda_backends.sdp_kernel.enable_flash_sdp(True)
        cuda_backends.sdp_kernel.enable_mem_efficient_sdp(True)
        cuda_backends.sdp_kernel.enable_math_sdp(True)
except Exception:
    pass


class Generator(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        upsample_block_num = int(math.log(scale_factor, 2))

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # Fewer heads; eval uses windowed attention to avoid OOM
        self.transformer_blocks = nn.ModuleList([
            EfficientTransformerBlock(dim=64, num_heads=8, chunk_size=32)
            for _ in range(6)
        ])

        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.block8 = nn.Sequential(*[UpsampleBlock(64) for _ in range(upsample_block_num)])
        self.block9 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        if self.training:
            return self._forward_train(x)
        return self._forward_eval(x)

    def _forward_train(self, x):
        block1 = self.block1(x)
        features = block1
        for transformer in self.transformer_blocks:
            # use_reentrant=False avoids future error in PyTorch 2.5+
            features = features + checkpoint(lambda t: transformer(t), features, use_reentrant=False)
        block7 = self.block7(features)
        block8 = self.block8(block1 + block7)
        block9 = self.block9(block8)
        return (torch.tanh(block9) + 1) / 2

    def _forward_eval(self, x):
        # AMP cuts validation memory a lot
        with torch.cuda.amp.autocast():
            block1 = self.block1(x)
            features = block1
            for transformer in self.transformer_blocks:
                features = features + transformer(features)  # eval path uses windowed SDPA inside the block
            block7 = self.block7(features)
            block8 = self.block8(block1 + block7)
            block9 = self.block9(block8)
            out = (torch.tanh(block9) + 1) / 2
        return out


class EfficientTransformerBlock(nn.Module):
    """
    Memory-friendly transformer block:
    - Uses SDPA (scaled_dot_product_attention) for attention.
    - In EVAL mode, switches to windowed (chunked) attention with tiles of size chunk_size x chunk_size.
    """
    def __init__(self, dim, num_heads=8, chunk_size=32):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W

        # BCHW -> BNC
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]

        # Train usually uses small patches -> global is fine.
        # Eval on 512x512 -> enforce windowed attention to avoid OOM.
        if (not self.training) and (N > self.chunk_size * self.chunk_size):
            out = self._chunked_attention(x_flat, H, W)
        else:
            out = self._global_attention(x_flat)

        # BNC -> BCHW
        out = out.transpose(1, 2).view(B, C, H, W)
        return out

    def _split_heads(self, x):
        # x: [B, N, C] -> [B, h, N, d]
        B, N, C = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        # x: [B, h, N, d] -> [B, N, C]
        B, h, N, d = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, N, h * d)

    def _qkv(self, x_flat):
        x_ln = self.norm1(x_flat)
        qkv = self.qkv(x_ln)              # [B, N, 3C]
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split_heads(q)          # [B, h, N, d]
        k = self._split_heads(k)
        v = self._split_heads(v)
        return q, k, v

    def _global_attention(self, x_flat):
        # SDPA does the scaling internally; dropout=0 in eval
        q, k, v = self._qkv(x_flat)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = self._merge_heads(attn_out)  # [B, N, C]
        out = self.proj(out)
        return out

    def _chunked_attention(self, x_flat, H, W):
        """
        Apply self-attention independently within non-overlapping windows of size (chunk_size x chunk_size).
        This keeps attention matrices small (<= (cs*cs)^2) and prevents OOM at 512x512.
        """
        B, N, C = x_flat.shape
        cs = min(self.chunk_size, H, W)
        num_chunks_h = math.ceil(H / cs)
        num_chunks_w = math.ceil(W / cs)

        # Precompute QKV once: reshape to [B, H, W, C] for easy slicing
        x_ln = self.norm1(x_flat)
        qkv = self.qkv(x_ln)
        q_all, k_all, v_all = qkv.chunk(3, dim=-1)

        q_all = q_all.view(B, H, W, C)
        k_all = k_all.view(B, H, W, C)
        v_all = v_all.view(B, H, W, C)

        out = torch.empty_like(x_flat)

        for hi in range(num_chunks_h):
            h_start = hi * cs
            h_end = min((hi + 1) * cs, H)
            for wi in range(num_chunks_w):
                w_start = wi * cs
                w_end = min((wi + 1) * cs, W)

                # [B, hs, ws, C] -> [B, Nt, C]
                hs, ws = (h_end - h_start), (w_end - w_start)
                Nt = hs * ws

                q_tile = q_all[:, h_start:h_end, w_start:w_end, :].contiguous().view(B, Nt, C)
                k_tile = k_all[:, h_start:h_end, w_start:w_end, :].contiguous().view(B, Nt, C)
                v_tile = v_all[:, h_start:h_end, w_start:w_end, :].contiguous().view(B, Nt, C)

                # split heads: [B, h, Nt, d]
                q = q_tile.view(B, Nt, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                k = k_tile.view(B, Nt, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                v = v_tile.view(B, Nt, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

                # SDPA within the window
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
                tile_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, Nt, C)  # [B, Nt, C]

                # write back: [B, Nt, C] -> [B, hs, ws, C] -> into out view
                tile_out_2d = tile_out.view(B, hs, ws, C)
                out_view = out.view(B, H, W, C)
                out_view[:, h_start:h_end, w_start:w_end, :] = tile_out_2d
                out = out_view.view(B, N, C)

        out = self.proj(out)
        return out


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
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1), nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        b = x.size(0)
        return torch.sigmoid(self.net(x).view(b))

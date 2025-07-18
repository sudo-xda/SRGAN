import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.fc(x)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 x B x heads x N x C'

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=dim * 4)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B H W C

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows = rearrange(x, 'b (h ws1) (w ws2) c -> (b h w) (ws1 ws2) c',
                              ws1=self.window_size, ws2=self.window_size)
        attn_windows = self.attn(self.norm1(x_windows))
        x_windows = x_windows + attn_windows

        x = rearrange(x_windows, '(b h w) (ws1 ws2) c -> b (h ws1) (w ws2) c',
                      h=H // self.window_size, w=W // self.window_size,
                      ws1=self.window_size, ws2=self.window_size)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 3, 1, 2)  # B C H W
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.prelu(res)
        res = self.conv2(res)
        return x + res


class Generator(nn.Module):
    def __init__(self, upscale_factor=4, in_channels=1, base_channels=64, num_blocks=6):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.body = nn.Sequential(
            *[SwinBlock(dim=base_channels, window_size=8, num_heads=4, shift_size=0 if i % 2 == 0 else 4)
              for i in range(num_blocks // 2)],
            *[ResidualBlock(base_channels) for _ in range(num_blocks // 2)]
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        self.tail = nn.Conv2d(base_channels, in_channels, kernel_size=9, padding=4)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.upsample(x)
        return self.tail(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.main(x)

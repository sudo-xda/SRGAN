import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


# ---------- Wavelet Transform Module ----------
class WaveletTransform(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        self.wave = wave

    def forward(self, x):
        # x: [B, C, H, W], apply wavelet on each channel separately
        edge_maps = []
        for b in range(x.size(0)):
            batch_edges = []
            for c in range(x.size(1)):
                x_np = x[b, c].detach().cpu().numpy().clip(0, 1)
                coeffs2 = pywt.dwt2(x_np, self.wave)
                #coeffs2 = pywt.dwt2(x[b, c].detach().cpu().numpy(), self.wave)
                LL, (LH, HL, HH) = coeffs2
                edge = torch.tensor((LH + HL + HH), dtype=torch.float32)
                batch_edges.append(edge.unsqueeze(0))
            edge_maps.append(torch.cat(batch_edges, dim=0).unsqueeze(0))
        return torch.cat(edge_maps, dim=0).to(x.device)


# ---------- Residual Block (No BatchNorm) ----------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        self.wavelet = WaveletTransform()

    def forward(self, x):
        edge = self.wavelet(x)
        edge = F.interpolate(edge, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x + self.block(x) + edge  # add wavelet residual


# ---------- Generator (No BatchNorm) ----------
class Generator(nn.Module):
    def __init__(self, in_channels=1, num_res_blocks=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 9, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])

        self.mid_conv = nn.Conv2d(64, 64, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        self.output = nn.Conv2d(64, in_channels, 9, padding=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.res_blocks(x1)
        x3 = self.mid_conv(x2)
        x = x1 + x3
        x = self.upsample(x)
        return self.output(x)


# ---------- Discriminator (No BatchNorm) ----------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        #return torch.sigmoid(self.net(x).view(batch_size))
        return self.net(x).view(batch_size)

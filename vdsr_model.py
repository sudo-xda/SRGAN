import torch
import torch.nn as nn
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class VDSR(nn.Module):
    def __init__(self, num_channels=1, depth=20):
        super(VDSR, self).__init__()

        layers = []
        layers.append(nn.Conv2d(num_channels, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(64, num_channels, kernel_size=3, padding=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.network(x)
        return x + residual  # residual learning


class MedicalSRDataset(Dataset):
    def __init__(self, root_dir, scale=4):
        self.paths = []
        for dp, _, fns in os.walk(root_dir):
            for f in fns:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    self.paths.append(os.path.join(dp, f))

        self.transform = transforms.ToTensor()
        self.scale = scale

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        hr = self.transform(img)

        lr = F.interpolate(
            hr.unsqueeze(0),
            scale_factor=1/self.scale,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)

        lr = F.interpolate(
            lr.unsqueeze(0),
            scale_factor=self.scale,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)

        return lr, hr

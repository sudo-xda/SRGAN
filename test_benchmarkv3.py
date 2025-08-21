import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import pytorch_ssim

from model_cnn_transv3_LG import Generator

# ---------------------- #
# Dataset Preparation
# ---------------------- #
class TestDataset(Dataset):
    def __init__(self, hr_root, upscale_factor):
        super(TestDataset, self).__init__()
        self.hr_root = hr_root
        self.hr_filenames = [os.path.join(dp, f) for dp, dn, filenames in os.walk(hr_root) for f in filenames if f.endswith(('png', 'jpg', 'jpeg'))]
        self.upscale_factor = upscale_factor
        self.lr_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.hr_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr_image_path = self.hr_filenames[index]
        hr_image = Image.open(hr_image_path).convert('RGB')

        # Generate LR image by downscaling
        w, h = hr_image.size
        lr_image = hr_image.resize((w // self.upscale_factor, h // self.upscale_factor), Image.BICUBIC)

        return os.path.relpath(hr_image_path, self.hr_root), self.lr_transform(lr_image), self.hr_transform(hr_image)

    def __len__(self):
        return len(self.hr_filenames)

# ---------------------- #
# Main Test Script
# ---------------------- #
def main():
    parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--model_name', default='Flicker2K_Hybrid__netG_epoch_4_100.pth', type=str, help='generator model file name')
    parser.add_argument('--data_dir', default='data/test', type=str, help='directory with HR images organized by dataset')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name
    DATA_DIR = opt.data_dir

    results = {}
    model = Generator(UPSCALE_FACTOR).eval()
    if torch.cuda.is_available():
        model = model.cpu()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

    test_set = TestDataset(DATA_DIR, upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

    out_path = f'benchmark_results/SRF_{UPSCALE_FACTOR}/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for image_relpath, lr_image, hr_image in test_bar:
        image_relpath = image_relpath[0]
        dataset_name = image_relpath.split(os.sep)[0]
        image_name = os.path.basename(image_relpath)

        # Prepare directories
        save_dir = os.path.join(out_path, dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        lr_image = Variable(lr_image, volatile=True)
        hr_image = Variable(hr_image, volatile=True)
        if torch.cuda.is_available():
            lr_image = lr_image.cpu()
            hr_image = hr_image.cpu()

        # Forward pass
        torch.cuda.empty_cache()

        sr_image = model(lr_image)
        torch.cuda.empty_cache()


        # Calculate PSNR and SSIM
        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

        # Save images: bicubic (upsample LR), SR output, HR ground truth
        bicubic_img = torch.nn.functional.interpolate(lr_image, scale_factor=UPSCALE_FACTOR, mode='bicubic', align_corners=False)

        display_images = torch.cat((
            bicubic_img.cpu().clamp(0,1),
            sr_image.cpu().clamp(0,1),
            hr_image.cpu().clamp(0,1)
        ), dim=3)  # concatenate horizontally

        save_name = os.path.join(save_dir, f'{image_name.split(".")[0]}_psnr_{psnr:.4f}_ssim_{ssim:.4f}.png')
        utils.save_image(display_images, save_name)

        if dataset_name not in results:
            results[dataset_name] = {'psnr': [], 'ssim': []}
        results[dataset_name]['psnr'].append(psnr)
        results[dataset_name]['ssim'].append(ssim)

    # Save stats
    out_stat_path = 'statistics/'
    os.makedirs(out_stat_path, exist_ok=True)

    saved_results = {'psnr': [], 'ssim': []}
    for dataset, metrics in results.items():
        avg_psnr = np.mean(metrics['psnr']) if metrics['psnr'] else 'No data'
        avg_ssim = np.mean(metrics['ssim']) if metrics['ssim'] else 'No data'
        saved_results['psnr'].append(avg_psnr)
        saved_results['ssim'].append(avg_ssim)

    df = pd.DataFrame(saved_results, index=results.keys())
    df.to_csv(os.path.join(out_stat_path, f'srf_{UPSCALE_FACTOR}_test_results.csv'), index_label='DataSet')

if __name__ == '__main__':
    main()

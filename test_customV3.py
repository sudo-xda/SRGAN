import argparse
import os
import time
import csv
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from fvcore.nn import FlopCountAnalysis

#from model import Generator
from model_cnn_transv4_LG import Generator

# Argument Parser
parser = argparse.ArgumentParser(description='Super Resolution Processing with Interpolation Comparison')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--test_folder', type=str, help='folder containing high-resolution images')
parser.add_argument('--output_folder', type=str, help='folder to save GT, LR, and SR images')
parser.add_argument('--model_name', default='/CT_HYBRIDV4_4B_netG_epoch_4_99.pth', type=str, help='generator model epoch name')
# parser.add_argument('--model_name', default='/CT_SRGAN_DHT__netG_epoch_4_95.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

# Parameters
UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
TEST_FOLDER = opt.test_folder
OUTPUT_FOLDER = opt.output_folder
MODEL_NAME = opt.model_name

# Define Output Subfolders
GT_FOLDER = os.path.join(OUTPUT_FOLDER, "GT")
LR_FOLDER = os.path.join(OUTPUT_FOLDER, "LR")
SR_FOLDER = os.path.join(OUTPUT_FOLDER, "SR_SRGAN")
NEAREST_FOLDER = os.path.join(OUTPUT_FOLDER, "SR_Nearest")
BILINEAR_FOLDER = os.path.join(OUTPUT_FOLDER, "SR_Bilinear")
BICUBIC_FOLDER = os.path.join(OUTPUT_FOLDER, "SR_Bicubic")
LANCZOS_FOLDER = os.path.join(OUTPUT_FOLDER, "SR_Lanczos")

# Create Output Folders if they do not exist
os.makedirs(GT_FOLDER, exist_ok=True)
os.makedirs(LR_FOLDER, exist_ok=True)
os.makedirs(SR_FOLDER, exist_ok=True)
os.makedirs(NEAREST_FOLDER, exist_ok=True)
os.makedirs(BILINEAR_FOLDER, exist_ok=True)
os.makedirs(BICUBIC_FOLDER, exist_ok=True)
os.makedirs(LANCZOS_FOLDER, exist_ok=True)

# Load Generator Model
model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME), strict=False)
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=torch.device('cpu')))

# Calculate FLOPs for SRGAN model (one-time calculation)
dummy_input = torch.randn(1, 3, 64, 64)  # Example LR input
if TEST_MODE:
    dummy_input = dummy_input.cuda()
flops_analysis = FlopCountAnalysis(model, dummy_input)
srgan_flops = flops_analysis.total()

# Interpolation methods mapping
interpolation_methods = {
    'Nearest': Image.NEAREST,
    'Bilinear': Image.BILINEAR,
    'Bicubic': Image.BICUBIC,
    'Lanczos': Image.LANCZOS
}

# Storage for metrics
results = []
method_metrics = defaultdict(lambda: {'psnr': [], 'ssim': [], 'time': []})

def calculate_metrics(gt_image, sr_image):
    """Calculate PSNR and SSIM between GT and SR images"""
    # Convert PIL images to numpy arrays
    gt_array = np.array(gt_image)
    sr_array = np.array(sr_image)
    
    # Ensure images have the same dimensions
    if gt_array.shape != sr_array.shape:
        # Resize SR to match GT if needed
        sr_image = sr_image.resize(gt_image.size, Image.BICUBIC)
        sr_array = np.array(sr_image)
    
    # Calculate PSNR
    psnr_value = psnr(gt_array, sr_array, data_range=255)
    
    # Calculate SSIM
    if len(gt_array.shape) == 3:  # Color image
        ssim_value = ssim(gt_array, sr_array, multichannel=True, channel_axis=2, data_range=255)
    else:  # Grayscale
        ssim_value = ssim(gt_array, sr_array, data_range=255)
    
    return psnr_value, ssim_value

# Process Each Image in the Test Folder
print("=" * 80)
print("Starting Super-Resolution Comparison")
print("=" * 80)

for image_name in os.listdir(TEST_FOLDER):
    image_path = os.path.join(TEST_FOLDER, image_name)
    if not os.path.isfile(image_path):
        continue

    print(f"\nProcessing: {image_name}")
    print("-" * 80)
    
    # Load HR Image (GT)
    hr_image = Image.open(image_path)
    hr_image.save(os.path.join(GT_FOLDER, image_name))  # Save GT Image

    # Create LR Image by Downsampling
    lr_image = hr_image.resize(
        (hr_image.width // UPSCALE_FACTOR, hr_image.height // UPSCALE_FACTOR),
        Image.BICUBIC
    )
    lr_image.save(os.path.join(LR_FOLDER, image_name))  # Save LR Image
    
    image_results = {'image': image_name}
    
    # 1. SRGAN Super-Resolution
    lr_tensor = Variable(ToTensor()(lr_image)).unsqueeze(0)
    if TEST_MODE:
        lr_tensor = lr_tensor.cuda()

    start = time.time()
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    elapsed = time.time() - start
    
    sr_image = ToPILImage()(sr_tensor[0].data.cpu())
    sr_image.save(os.path.join(SR_FOLDER, image_name))
    
    psnr_val, ssim_val = calculate_metrics(hr_image, sr_image)
    image_results['SRGAN_PSNR'] = psnr_val
    image_results['SRGAN_SSIM'] = ssim_val
    image_results['SRGAN_Time'] = elapsed
    image_results['SRGAN_FLOPs'] = srgan_flops
    
    method_metrics['SRGAN']['psnr'].append(psnr_val)
    method_metrics['SRGAN']['ssim'].append(ssim_val)
    method_metrics['SRGAN']['time'].append(elapsed)
    
    print(f"  SRGAN       - Time: {elapsed:.4f}s | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | FLOPs: {srgan_flops:,}")
    
    # 2. Interpolation-based methods
    target_size = (hr_image.width, hr_image.height)
    
    for method_name, interpolation_type in interpolation_methods.items():
        start = time.time()
        interp_image = lr_image.resize(target_size, interpolation_type)
        elapsed = time.time() - start
        
        # Save interpolated image
        folder_map = {
            'Nearest': NEAREST_FOLDER,
            'Bilinear': BILINEAR_FOLDER,
            'Bicubic': BICUBIC_FOLDER,
            'Lanczos': LANCZOS_FOLDER
        }
        interp_image.save(os.path.join(folder_map[method_name], image_name))
        
        psnr_val, ssim_val = calculate_metrics(hr_image, interp_image)
        
        image_results[f'{method_name}_PSNR'] = psnr_val
        image_results[f'{method_name}_SSIM'] = ssim_val
        image_results[f'{method_name}_Time'] = elapsed
        image_results[f'{method_name}_FLOPs'] = 0  # Interpolation has negligible FLOPs
        
        method_metrics[method_name]['psnr'].append(psnr_val)
        method_metrics[method_name]['ssim'].append(ssim_val)
        method_metrics[method_name]['time'].append(elapsed)
        
        print(f"  {method_name:11s} - Time: {elapsed:.4f}s | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | FLOPs: 0")
    
    results.append(image_results)

# Calculate and Display Average Metrics
print("\n" + "=" * 80)
print("AVERAGE METRICS ACROSS ALL IMAGES")
print("=" * 80)

summary_results = []
for method in ['SRGAN', 'Nearest', 'Bilinear', 'Bicubic', 'Lanczos']:
    avg_psnr = np.mean(method_metrics[method]['psnr'])
    avg_ssim = np.mean(method_metrics[method]['ssim'])
    avg_time = np.mean(method_metrics[method]['time'])
    flops = srgan_flops if method == 'SRGAN' else 0
    
    summary_results.append({
        'Method': method,
        'Avg_PSNR': avg_psnr,
        'Avg_SSIM': avg_ssim,
        'Avg_Time': avg_time,
        'FLOPs': flops
    })
    
    print(f"{method:11s} - Avg PSNR: {avg_psnr:.2f} dB | Avg SSIM: {avg_ssim:.4f} | Avg Time: {avg_time:.4f}s | FLOPs: {flops:,}")

# Save detailed results to CSV
csv_path = os.path.join(OUTPUT_FOLDER, "detailed_results.csv")
if results:
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved to: {csv_path}")

# Save summary results to CSV
summary_csv_path = os.path.join(OUTPUT_FOLDER, "summary_results.csv")
with open(summary_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['Method', 'Avg_PSNR', 'Avg_SSIM', 'Avg_Time', 'FLOPs']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_results)
print(f"Summary results saved to: {summary_csv_path}")

print("\n" + "=" * 80)
print("Processing complete. All images and results saved in:", OUTPUT_FOLDER)
print("=" * 80)

# Example command:
# python test_customV3.py --test_folder "/home/dst/Desktop/GAN/SRGAN/Test_set" --output_folder "/home/dst/Desktop/GAN/SRGAN/output"
# python test_customV3.py --test_folder "/home/dst/Desktop/GAN/SRGAN/Test_set"     --output_folder "/home/dst/Desktop/GAN/SRGAN/output"
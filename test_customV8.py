import argparse
import os
import time
import csv
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from fvcore.nn import FlopCountAnalysis
import cv2

from model import Generator as GeneratorV1
from model_cnn_transv3_LG import Generator as GeneratorV2
from model_srcnn import SRCNN
from vdsr_model import VDSR
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
wgan_module = import_module('model_W-GAN')
GeneratorWGAN = wgan_module.Generator

# Argument Parser
parser = argparse.ArgumentParser(description='Super Resolution Processing with Four Models (PSNR/SSIM)')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--test_folder', type=str, help='folder containing high-resolution images')
parser.add_argument('--output_folder', type=str, help='folder to save GT, LR, and SR images')
parser.add_argument('--model1_name', default='/CT_HYBRIDV4_4B_netG_epoch_4_99.pth', type=str, help='first generator model')
parser.add_argument('--model2_name', default='/CT_SRGAN_DHT__netG_epoch_4_95.pth', type=str, help='second generator model')
parser.add_argument('--srcnn_name', default='srcnn_ct_best.pth', type=str, help='SRCNN model weights')
parser.add_argument('--vdsr_name', default='vdsr_CT.pth', type=str, help='VDSR model weights')
parser.add_argument('--wgan_name', default='WGAN_epoch_090.pth', type=str, help='W-GAN model weights')
parser.add_argument('--model1_label', default='HYBRIDV4', type=str, help='label for first model')
parser.add_argument('--model2_label', default='SRGAN_DHT', type=str, help='label for second model')
parser.add_argument('--srcnn_label', default='SRCNN', type=str, help='label for SRCNN model')
parser.add_argument('--vdsr_label', default='VDSR', type=str, help='label for VDSR model')
parser.add_argument('--wgan_label', default='WGAN', type=str, help='label for W-GAN model')
opt = parser.parse_args()

# Parameters
UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
TEST_FOLDER = opt.test_folder
OUTPUT_FOLDER = opt.output_folder
MODEL1_NAME = opt.model1_name
MODEL2_NAME = opt.model2_name
SRCNN_NAME = opt.srcnn_name
VDSR_NAME = opt.vdsr_name
WGAN_NAME = opt.wgan_name
MODEL1_LABEL = opt.model1_label
MODEL2_LABEL = opt.model2_label
SRCNN_LABEL = opt.srcnn_label
VDSR_LABEL = opt.vdsr_label
WGAN_LABEL = opt.wgan_label

# Define Output Subfolders
GT_FOLDER = os.path.join(OUTPUT_FOLDER, "GT")
LR_FOLDER = os.path.join(OUTPUT_FOLDER, "LR")
MODEL1_FOLDER = os.path.join(OUTPUT_FOLDER, f"SR_{MODEL1_LABEL}")
MODEL2_FOLDER = os.path.join(OUTPUT_FOLDER, f"SR_{MODEL2_LABEL}")
SRCNN_FOLDER = os.path.join(OUTPUT_FOLDER, f"SR_{SRCNN_LABEL}")
VDSR_FOLDER = os.path.join(OUTPUT_FOLDER, f"SR_{VDSR_LABEL}")
WGAN_FOLDER = os.path.join(OUTPUT_FOLDER, f"SR_{WGAN_LABEL}")
NEAREST_FOLDER = os.path.join(OUTPUT_FOLDER, "SR_Nearest")
BILINEAR_FOLDER = os.path.join(OUTPUT_FOLDER, "SR_Bilinear")
BICUBIC_FOLDER = os.path.join(OUTPUT_FOLDER, "SR_Bicubic")
LANCZOS_FOLDER = os.path.join(OUTPUT_FOLDER, "SR_Lanczos")
COMPARISON_FOLDER = os.path.join(OUTPUT_FOLDER, "Comparisons")

# Create Output Folders if they do not exist
os.makedirs(GT_FOLDER, exist_ok=True)
os.makedirs(LR_FOLDER, exist_ok=True)
os.makedirs(MODEL1_FOLDER, exist_ok=True)
os.makedirs(MODEL2_FOLDER, exist_ok=True)
os.makedirs(SRCNN_FOLDER, exist_ok=True)
os.makedirs(VDSR_FOLDER, exist_ok=True)
os.makedirs(WGAN_FOLDER, exist_ok=True)
os.makedirs(NEAREST_FOLDER, exist_ok=True)
os.makedirs(BILINEAR_FOLDER, exist_ok=True)
os.makedirs(BICUBIC_FOLDER, exist_ok=True)
os.makedirs(LANCZOS_FOLDER, exist_ok=True)
os.makedirs(COMPARISON_FOLDER, exist_ok=True)

# Load Generator Models
print("Loading Model 1:", MODEL1_LABEL, "(using GeneratorV2)")
model1 = GeneratorV2(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model1.cuda()
    model1.load_state_dict(torch.load('epochs/' + MODEL1_NAME), strict=False)
else:
    model1.load_state_dict(torch.load('epochs/' + MODEL1_NAME, map_location=torch.device('cpu')), strict=False)

print("Loading Model 2:", MODEL2_LABEL, "(using GeneratorV1)")
model2 = GeneratorV1(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model2.cuda()
    model2.load_state_dict(torch.load('epochs/' + MODEL2_NAME), strict=False)
else:
    model2.load_state_dict(torch.load('epochs/' + MODEL2_NAME, map_location=torch.device('cpu')), strict=False)

print("Loading Model 3:", SRCNN_LABEL, "(using SRCNN)")
model_srcnn = SRCNN().eval()
if TEST_MODE:
    model_srcnn.cuda()
    checkpoint_srcnn = torch.load(SRCNN_NAME)
else:
    checkpoint_srcnn = torch.load(SRCNN_NAME, map_location=torch.device('cpu'))

if 'state_dict' in checkpoint_srcnn:
    model_srcnn.load_state_dict(checkpoint_srcnn['state_dict'])
else:
    model_srcnn.load_state_dict(checkpoint_srcnn)

print("Loading Model 4:", VDSR_LABEL, "(using VDSR)")
model_vdsr = VDSR().eval()
if TEST_MODE:
    model_vdsr.cuda()
    checkpoint_vdsr = torch.load(VDSR_NAME)
else:
    checkpoint_vdsr = torch.load(VDSR_NAME, map_location=torch.device('cpu'))

if 'state_dict' in checkpoint_vdsr:
    state_dict = checkpoint_vdsr['state_dict']
else:
    state_dict = checkpoint_vdsr

if any(key.startswith('net.') for key in state_dict.keys()):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('net.', 'network.')
        new_state_dict[new_key] = value
    model_vdsr.load_state_dict(new_state_dict)
else:
    model_vdsr.load_state_dict(state_dict)

# Model 5: W-GAN (Wavelet-based GAN)
print("Loading Model 5:", WGAN_LABEL, "(using W-GAN)")
model_wgan = GeneratorWGAN(in_channels=1, num_res_blocks=16).eval()
if TEST_MODE:
    model_wgan.cuda()
    checkpoint_wgan = torch.load(WGAN_NAME)
else:
    checkpoint_wgan = torch.load(WGAN_NAME, map_location=torch.device('cpu'))

# Handle different checkpoint formats
if 'generator_state_dict' in checkpoint_wgan:
    model_wgan.load_state_dict(checkpoint_wgan['generator_state_dict'])
elif 'state_dict' in checkpoint_wgan:
    model_wgan.load_state_dict(checkpoint_wgan['state_dict'])
else:
    model_wgan.load_state_dict(checkpoint_wgan)

# Calculate FLOPs for all models
dummy_input_rgb = torch.randn(1, 3, 64, 64)
dummy_input_y = torch.randn(1, 1, 64, 64)

if TEST_MODE:
    dummy_input_rgb = dummy_input_rgb.cuda()
    dummy_input_y = dummy_input_y.cuda()

flops_analysis1 = FlopCountAnalysis(model1, dummy_input_rgb)
model1_flops = flops_analysis1.total()

flops_analysis2 = FlopCountAnalysis(model2, dummy_input_rgb)
model2_flops = flops_analysis2.total()

flops_analysis_srcnn = FlopCountAnalysis(model_srcnn, dummy_input_y)
srcnn_flops = flops_analysis_srcnn.total()

flops_analysis_vdsr = FlopCountAnalysis(model_vdsr, dummy_input_y)
vdsr_flops = flops_analysis_vdsr.total()

flops_analysis_wgan = FlopCountAnalysis(model_wgan, dummy_input_y)
wgan_flops = flops_analysis_wgan.total()

print(f"{MODEL1_LABEL} FLOPs: {model1_flops:,}")
print(f"{MODEL2_LABEL} FLOPs: {model2_flops:,}")
print(f"{SRCNN_LABEL} FLOPs: {srcnn_flops:,}")
print(f"{VDSR_LABEL} FLOPs: {vdsr_flops:,}")
print(f"{WGAN_LABEL} FLOPs: {wgan_flops:,}")

# Interpolation methods mapping
interpolation_methods = {
    'Nearest': Image.NEAREST,
    'Bilinear': Image.BILINEAR,
    'Bicubic': Image.BICUBIC,
    'Lanczos': Image.LANCZOS
}

# Storage for metrics (PSNR and SSIM only)
results = []
method_metrics = defaultdict(lambda: {'psnr': [], 'ssim': [], 'time': []})

def bgr2ycbcr(img, only_y=True):
    """Convert BGR image to YCbCr"""
    if only_y:
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

def ycbcr2bgr(img):
    """Convert YCbCr image to BGR"""
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

def calculate_metrics(gt_image, sr_image):
    """Calculate PSNR and SSIM metrics only"""
    gt_array = np.array(gt_image)
    sr_array = np.array(sr_image)
    
    if gt_array.shape != sr_array.shape:
        sr_image = sr_image.resize(gt_image.size, Image.BICUBIC)
        sr_array = np.array(sr_image)
    
    metrics = {}
    metrics['psnr'] = psnr(gt_array, sr_array, data_range=255)
    
    if len(gt_array.shape) == 3:
        metrics['ssim'] = ssim(gt_array, sr_array, multichannel=True, channel_axis=2, data_range=255)
    else:
        metrics['ssim'] = ssim(gt_array, sr_array, data_range=255)
    
    return metrics

def add_label_to_image(image, label, metrics_text=""):
    """Add a label and metrics to an image"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_metrics = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font_title = ImageFont.load_default()
        font_metrics = ImageFont.load_default()
    
    padding = 4
    
    text_bbox = draw.textbbox((0, 0), label, font=font_title)
    text_height = text_bbox[3] - text_bbox[1]
    draw.rectangle([(0, 0), (img_copy.width, text_height + 2*padding)], fill=(0, 0, 0, 180))
    draw.text((padding, padding), label, fill=(255, 255, 255), font=font_title)
    
    if metrics_text:
        metrics_bbox = draw.textbbox((0, 0), metrics_text, font=font_metrics)
        metrics_height = metrics_bbox[3] - metrics_bbox[1]
        y_pos = img_copy.height - metrics_height - 2*padding
        draw.rectangle([(0, y_pos), (img_copy.width, img_copy.height)], fill=(0, 0, 0, 180))
        draw.text((padding, y_pos), metrics_text, fill=(255, 255, 255), font=font_metrics)
    
    return img_copy

def create_comparison_grid(images_dict, image_name):
    """Create a side-by-side comparison grid of all methods"""
    method_order = ['GT', 'LR', MODEL1_LABEL, MODEL2_LABEL, SRCNN_LABEL, VDSR_LABEL, WGAN_LABEL, 'Bicubic', 'Lanczos']
    available_methods = [m for m in method_order if m in images_dict]
    
    if not available_methods:
        return None
    
    first_img = images_dict[available_methods[0]]['image']
    img_width, img_height = first_img.size
    
    cols = 4
    rows = (len(available_methods) + cols - 1) // cols
    
    grid_width = img_width * cols
    grid_height = img_height * rows
    grid = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    
    for idx, method in enumerate(available_methods):
        row = idx // cols
        col = idx % cols
        
        img_data = images_dict[method]
        img = img_data['image']
        metrics = img_data.get('metrics', '')
        
        labeled_img = add_label_to_image(img, method, metrics)
        
        x_offset = col * img_width
        y_offset = row * img_height
        grid.paste(labeled_img, (x_offset, y_offset))
    
    return grid

# Process Each Image in the Test Folder
print("=" * 100)
print("Starting Super-Resolution Comparison with Five Models (PSNR/SSIM Only)")
print("=" * 100)

for image_name in os.listdir(TEST_FOLDER):
    image_path = os.path.join(TEST_FOLDER, image_name)
    if not os.path.isfile(image_path):
        continue

    print(f"\nProcessing: {image_name}")
    print("-" * 100)
    
    # Load HR Image (GT)
    hr_image = Image.open(image_path)
    hr_image.save(os.path.join(GT_FOLDER, image_name))

    # Create LR Image by Downsampling
    lr_image = hr_image.resize(
        (hr_image.width // UPSCALE_FACTOR, hr_image.height // UPSCALE_FACTOR),
        Image.BICUBIC
    )
    lr_image.save(os.path.join(LR_FOLDER, image_name))
    
    # Prepare LR tensor for RGB models
    lr_tensor = Variable(ToTensor()(lr_image)).unsqueeze(0)
    if TEST_MODE:
        lr_tensor = lr_tensor.cuda()
    
    image_results = {'image': image_name}
    comparison_images = {}
    
    # Add GT and LR to comparison
    comparison_images['GT'] = {'image': hr_image, 'metrics': 'Ground Truth'}
    lr_upscaled = lr_image.resize((hr_image.width, hr_image.height), Image.BICUBIC)
    comparison_images['LR'] = {'image': lr_upscaled, 'metrics': 'Low Resolution'}
    
    # 1. Model 1 (HYBRIDV4)
    start = time.time()
    with torch.no_grad():
        sr_tensor1 = model1(lr_tensor)
    elapsed1 = time.time() - start
    
    sr_image1 = ToPILImage()(sr_tensor1[0].data.cpu())
    sr_image1.save(os.path.join(MODEL1_FOLDER, image_name))
    
    metrics1 = calculate_metrics(hr_image, sr_image1)
    
    image_results[f'{MODEL1_LABEL}_PSNR'] = metrics1['psnr']
    image_results[f'{MODEL1_LABEL}_SSIM'] = metrics1['ssim']
    image_results[f'{MODEL1_LABEL}_Time'] = elapsed1
    image_results[f'{MODEL1_LABEL}_FLOPs'] = model1_flops
    
    method_metrics[MODEL1_LABEL]['psnr'].append(metrics1['psnr'])
    method_metrics[MODEL1_LABEL]['ssim'].append(metrics1['ssim'])
    method_metrics[MODEL1_LABEL]['time'].append(elapsed1)
    
    comparison_images[MODEL1_LABEL] = {
        'image': sr_image1,
        'metrics': f'PSNR: {metrics1["psnr"]:.2f} | SSIM: {metrics1["ssim"]:.4f}'
    }
    
    print(f"  {MODEL1_LABEL:12s} - PSNR: {metrics1['psnr']:.2f} dB | SSIM: {metrics1['ssim']:.4f} | Time: {elapsed1:.4f}s")
    
    # 2. Model 2 (SRGAN_DHT)
    start = time.time()
    with torch.no_grad():
        sr_tensor2 = model2(lr_tensor)
    elapsed2 = time.time() - start
    
    sr_image2 = ToPILImage()(sr_tensor2[0].data.cpu())
    sr_image2.save(os.path.join(MODEL2_FOLDER, image_name))
    
    metrics2 = calculate_metrics(hr_image, sr_image2)
    
    image_results[f'{MODEL2_LABEL}_PSNR'] = metrics2['psnr']
    image_results[f'{MODEL2_LABEL}_SSIM'] = metrics2['ssim']
    image_results[f'{MODEL2_LABEL}_Time'] = elapsed2
    image_results[f'{MODEL2_LABEL}_FLOPs'] = model2_flops
    
    method_metrics[MODEL2_LABEL]['psnr'].append(metrics2['psnr'])
    method_metrics[MODEL2_LABEL]['ssim'].append(metrics2['ssim'])
    method_metrics[MODEL2_LABEL]['time'].append(elapsed2)
    
    comparison_images[MODEL2_LABEL] = {
        'image': sr_image2,
        'metrics': f'PSNR: {metrics2["psnr"]:.2f} | SSIM: {metrics2["ssim"]:.4f}'
    }
    
    print(f"  {MODEL2_LABEL:12s} - PSNR: {metrics2['psnr']:.2f} dB | SSIM: {metrics2['ssim']:.4f} | Time: {elapsed2:.4f}s")
    
    # 3. SRCNN
    start = time.time()
    
    hr_cv = cv2.cvtColor(np.array(hr_image), cv2.COLOR_RGB2BGR)
    lr_cv = cv2.cvtColor(np.array(lr_image), cv2.COLOR_RGB2BGR)
    lr_bicubic = cv2.resize(lr_cv, (hr_image.width, hr_image.height), interpolation=cv2.INTER_CUBIC)
    
    lr_ycbcr = bgr2ycbcr(lr_bicubic, only_y=False)
    lr_y = lr_ycbcr[:, :, 0].astype(np.float32) / 255.0
    
    lr_y_tensor = torch.from_numpy(lr_y).unsqueeze(0).unsqueeze(0)
    if TEST_MODE:
        lr_y_tensor = lr_y_tensor.cuda()
    
    with torch.no_grad():
        sr_y_tensor = model_srcnn(lr_y_tensor).clamp_(0, 1.0)
    
    sr_y = sr_y_tensor.squeeze().cpu().numpy()
    sr_y = (sr_y * 255.0).astype(np.uint8)
    
    sr_ycbcr = lr_ycbcr.copy()
    sr_ycbcr[:, :, 0] = sr_y
    sr_bgr = ycbcr2bgr(sr_ycbcr)
    sr_rgb = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB)
    
    elapsed_srcnn = time.time() - start
    
    sr_image_srcnn = Image.fromarray(sr_rgb)
    sr_image_srcnn.save(os.path.join(SRCNN_FOLDER, image_name))
    
    metrics_srcnn = calculate_metrics(hr_image, sr_image_srcnn)
    
    image_results[f'{SRCNN_LABEL}_PSNR'] = metrics_srcnn['psnr']
    image_results[f'{SRCNN_LABEL}_SSIM'] = metrics_srcnn['ssim']
    image_results[f'{SRCNN_LABEL}_Time'] = elapsed_srcnn
    image_results[f'{SRCNN_LABEL}_FLOPs'] = srcnn_flops
    
    method_metrics[SRCNN_LABEL]['psnr'].append(metrics_srcnn['psnr'])
    method_metrics[SRCNN_LABEL]['ssim'].append(metrics_srcnn['ssim'])
    method_metrics[SRCNN_LABEL]['time'].append(elapsed_srcnn)
    
    comparison_images[SRCNN_LABEL] = {
        'image': sr_image_srcnn,
        'metrics': f'PSNR: {metrics_srcnn["psnr"]:.2f} | SSIM: {metrics_srcnn["ssim"]:.4f}'
    }
    
    print(f"  {SRCNN_LABEL:12s} - PSNR: {metrics_srcnn['psnr']:.2f} dB | SSIM: {metrics_srcnn['ssim']:.4f} | Time: {elapsed_srcnn:.4f}s")
    
    # 4. VDSR
    start = time.time()
    
    lr_y_vdsr = lr_ycbcr[:, :, 0].astype(np.float32) / 255.0
    
    lr_y_tensor_vdsr = torch.from_numpy(lr_y_vdsr).unsqueeze(0).unsqueeze(0)
    if TEST_MODE:
        lr_y_tensor_vdsr = lr_y_tensor_vdsr.cuda()
    
    with torch.no_grad():
        sr_y_tensor_vdsr = model_vdsr(lr_y_tensor_vdsr).clamp_(0, 1.0)
    
    sr_y_vdsr = sr_y_tensor_vdsr.squeeze().cpu().numpy()
    sr_y_vdsr = (sr_y_vdsr * 255.0).astype(np.uint8)
    
    sr_ycbcr_vdsr = lr_ycbcr.copy()
    sr_ycbcr_vdsr[:, :, 0] = sr_y_vdsr
    sr_bgr_vdsr = ycbcr2bgr(sr_ycbcr_vdsr)
    sr_rgb_vdsr = cv2.cvtColor(sr_bgr_vdsr, cv2.COLOR_BGR2RGB)
    
    elapsed_vdsr = time.time() - start
    
    sr_image_vdsr = Image.fromarray(sr_rgb_vdsr)
    sr_image_vdsr.save(os.path.join(VDSR_FOLDER, image_name))
    
    metrics_vdsr = calculate_metrics(hr_image, sr_image_vdsr)
    
    image_results[f'{VDSR_LABEL}_PSNR'] = metrics_vdsr['psnr']
    image_results[f'{VDSR_LABEL}_SSIM'] = metrics_vdsr['ssim']
    image_results[f'{VDSR_LABEL}_Time'] = elapsed_vdsr
    image_results[f'{VDSR_LABEL}_FLOPs'] = vdsr_flops
    
    method_metrics[VDSR_LABEL]['psnr'].append(metrics_vdsr['psnr'])
    method_metrics[VDSR_LABEL]['ssim'].append(metrics_vdsr['ssim'])
    method_metrics[VDSR_LABEL]['time'].append(elapsed_vdsr)
    
    comparison_images[VDSR_LABEL] = {
        'image': sr_image_vdsr,
        'metrics': f'PSNR: {metrics_vdsr["psnr"]:.2f} | SSIM: {metrics_vdsr["ssim"]:.4f}'
    }
    
    print(f"  {VDSR_LABEL:12s} - PSNR: {metrics_vdsr['psnr']:.2f} dB | SSIM: {metrics_vdsr['ssim']:.4f} | Time: {elapsed_vdsr:.4f}s")
    
    # 5. W-GAN (Wavelet-based GAN)
    start = time.time()
    
    # W-GAN works on grayscale medical images (CT scans)
    lr_gray = lr_image.convert('L')
    lr_gray_array = np.array(lr_gray).astype(np.float32) / 255.0
    
    lr_gray_tensor = torch.from_numpy(lr_gray_array).unsqueeze(0).unsqueeze(0)
    if TEST_MODE:
        lr_gray_tensor = lr_gray_tensor.cuda()
    
    with torch.no_grad():
        sr_gray_tensor = model_wgan(lr_gray_tensor).clamp_(0, 1.0)
    
    sr_gray = sr_gray_tensor.squeeze().cpu().numpy()
    sr_gray = (sr_gray * 255.0).astype(np.uint8)
    
    # Save as grayscale
    sr_image_wgan_gray = Image.fromarray(sr_gray, mode='L')
    sr_image_wgan_gray.save(os.path.join(WGAN_FOLDER, image_name))
    
    # Convert to RGB for metric calculation (to match GT dimensions)
    sr_image_wgan = sr_image_wgan_gray.convert('RGB')
    
    elapsed_wgan = time.time() - start
    
    metrics_wgan = calculate_metrics(hr_image, sr_image_wgan)
    
    image_results[f'{WGAN_LABEL}_PSNR'] = metrics_wgan['psnr']
    image_results[f'{WGAN_LABEL}_SSIM'] = metrics_wgan['ssim']
    image_results[f'{WGAN_LABEL}_Time'] = elapsed_wgan
    image_results[f'{WGAN_LABEL}_FLOPs'] = wgan_flops
    
    method_metrics[WGAN_LABEL]['psnr'].append(metrics_wgan['psnr'])
    method_metrics[WGAN_LABEL]['ssim'].append(metrics_wgan['ssim'])
    method_metrics[WGAN_LABEL]['time'].append(elapsed_wgan)
    
    comparison_images[WGAN_LABEL] = {
        'image': sr_image_wgan,
        'metrics': f'PSNR: {metrics_wgan["psnr"]:.2f} | SSIM: {metrics_wgan["ssim"]:.4f}'
    }
    
    print(f"  {WGAN_LABEL:12s} - PSNR: {metrics_wgan['psnr']:.2f} dB | SSIM: {metrics_wgan['ssim']:.4f} | Time: {elapsed_wgan:.4f}s")
    
    # 6. Interpolation-based methods
    target_size = (hr_image.width, hr_image.height)
    
    for method_name, interpolation_type in interpolation_methods.items():
        start = time.time()
        interp_image = lr_image.resize(target_size, interpolation_type)
        elapsed = time.time() - start
        
        folder_map = {
            'Nearest': NEAREST_FOLDER,
            'Bilinear': BILINEAR_FOLDER,
            'Bicubic': BICUBIC_FOLDER,
            'Lanczos': LANCZOS_FOLDER
        }
        interp_image.save(os.path.join(folder_map[method_name], image_name))
        
        metrics_interp = calculate_metrics(hr_image, interp_image)
        
        image_results[f'{method_name}_PSNR'] = metrics_interp['psnr']
        image_results[f'{method_name}_SSIM'] = metrics_interp['ssim']
        image_results[f'{method_name}_Time'] = elapsed
        image_results[f'{method_name}_FLOPs'] = 0
        
        method_metrics[method_name]['psnr'].append(metrics_interp['psnr'])
        method_metrics[method_name]['ssim'].append(metrics_interp['ssim'])
        method_metrics[method_name]['time'].append(elapsed)
        
        if method_name in ['Bicubic', 'Lanczos']:
            comparison_images[method_name] = {
                'image': interp_image,
                'metrics': f'PSNR: {metrics_interp["psnr"]:.2f} | SSIM: {metrics_interp["ssim"]:.4f}'
            }
        
        print(f"  {method_name:12s} - PSNR: {metrics_interp['psnr']:.2f} dB | SSIM: {metrics_interp['ssim']:.4f} | Time: {elapsed:.4f}s")
    
    # Create and save comparison grid
    comparison_grid = create_comparison_grid(comparison_images, image_name)
    if comparison_grid:
        comparison_path = os.path.join(COMPARISON_FOLDER, f"comparison_{image_name}")
        comparison_grid.save(comparison_path)
        print(f"  ✓ Comparison grid saved")
    
    results.append(image_results)

# Calculate and Display Average Metrics
print("\n" + "=" * 100)
print("AVERAGE METRICS ACROSS ALL IMAGES (PSNR/SSIM Only)")
print("=" * 100)

summary_results = []
all_methods = [MODEL1_LABEL, MODEL2_LABEL, SRCNN_LABEL, VDSR_LABEL, WGAN_LABEL, 'Nearest', 'Bilinear', 'Bicubic', 'Lanczos']

print(f"\n{'Method':<12} | {'Avg PSNR':<10} | {'Avg SSIM':<10} | {'Avg Time':<10} | {'FLOPs':<15}")
print("-" * 70)

for method in all_methods:
    if method_metrics[method]['psnr']:
        avg_psnr = np.mean(method_metrics[method]['psnr'])
        avg_ssim = np.mean(method_metrics[method]['ssim'])
        avg_time = np.mean(method_metrics[method]['time'])
        
        if method == MODEL1_LABEL:
            flops = model1_flops
        elif method == MODEL2_LABEL:
            flops = model2_flops
        elif method == SRCNN_LABEL:
            flops = srcnn_flops
        elif method == VDSR_LABEL:
            flops = vdsr_flops
        elif method == WGAN_LABEL:
            flops = wgan_flops
        else:
            flops = 0
        
        summary = {
            'Method': method,
            'Avg_PSNR': avg_psnr,
            'Avg_SSIM': avg_ssim,
            'Avg_Time': avg_time,
            'FLOPs': flops
        }
        summary_results.append(summary)
        
        print(f"{method:<12} | {avg_psnr:<10.2f} | {avg_ssim:<10.4f} | {avg_time:<10.4f} | {flops:<15,}")

# Save detailed results to CSV
csv_path = os.path.join(OUTPUT_FOLDER, "detailed_results_v8.csv")
if results:
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Detailed results saved to: {csv_path}")

# Save summary results to CSV
summary_csv_path = os.path.join(OUTPUT_FOLDER, "summary_results_v8.csv")
with open(summary_csv_path, 'w', newline='') as csvfile:
    if summary_results:
        fieldnames = ['Method', 'Avg_PSNR', 'Avg_SSIM', 'Avg_Time', 'FLOPs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_results)
print(f"✓ Summary results saved to: {summary_csv_path}")

print("\n" + "=" * 100)
print("Processing complete!")
print(f"  - All images saved in: {OUTPUT_FOLDER}")
print(f"  - Side-by-side comparisons: {COMPARISON_FOLDER}")
print(f"  - Metrics: PSNR, SSIM, Time, FLOPs only")
print(f"  - Models compared: {MODEL1_LABEL}, {MODEL2_LABEL}, {SRCNN_LABEL}, {VDSR_LABEL}, {WGAN_LABEL} + 4 interpolation methods")
print("=" * 100)

# Example command:
# python test_customV8.py --test_folder "/home/dst/Desktop/GAN/SRGAN/Test_set" --output_folder "/home/dst/Desktop/GAN/SRGAN/output"

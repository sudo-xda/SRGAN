import argparse
import os
import time
import csv
from collections import defaultdict

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from fvcore.nn import FlopCountAnalysis

from model import Generator as GeneratorV1
from model_cnn_transv3_LG import Generator as GeneratorV2

# Argument Parser
parser = argparse.ArgumentParser(description='Super Resolution Processing with Dual Model Comparison')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--test_folder', type=str, help='folder containing high-resolution images')
parser.add_argument('--output_folder', type=str, help='folder to save GT, LR, and SR images')
parser.add_argument('--model1_name', default='/CT_HYBRIDV4_4B_netG_epoch_4_99.pth', type=str, help='first generator model')
parser.add_argument('--model2_name', default='/CT_SRGAN_DHT__netG_epoch_4_95.pth', type=str, help='second generator model')
parser.add_argument('--model1_label', default='HYBRIDV4', type=str, help='label for first model')
parser.add_argument('--model2_label', default='SRGAN_DHT', type=str, help='label for second model')
opt = parser.parse_args()

# Parameters
UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
TEST_FOLDER = opt.test_folder
OUTPUT_FOLDER = opt.output_folder
MODEL1_NAME = opt.model1_name
MODEL2_NAME = opt.model2_name
MODEL1_LABEL = opt.model1_label
MODEL2_LABEL = opt.model2_label

# Define Output Subfolders
GT_FOLDER = os.path.join(OUTPUT_FOLDER, "GT")
LR_FOLDER = os.path.join(OUTPUT_FOLDER, "LR")
MODEL1_FOLDER = os.path.join(OUTPUT_FOLDER, f"SR_{MODEL1_LABEL}")
MODEL2_FOLDER = os.path.join(OUTPUT_FOLDER, f"SR_{MODEL2_LABEL}")
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
os.makedirs(NEAREST_FOLDER, exist_ok=True)
os.makedirs(BILINEAR_FOLDER, exist_ok=True)
os.makedirs(BICUBIC_FOLDER, exist_ok=True)
os.makedirs(LANCZOS_FOLDER, exist_ok=True)
os.makedirs(COMPARISON_FOLDER, exist_ok=True)

# Load Generator Models
# Model 1: HYBRIDV4 uses GeneratorV2 (model_cnn_transv3_LG)
print("Loading Model 1:", MODEL1_LABEL, "(using GeneratorV2)")
model1 = GeneratorV2(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model1.cuda()
    model1.load_state_dict(torch.load('epochs/' + MODEL1_NAME), strict=False)
else:
    model1.load_state_dict(torch.load('epochs/' + MODEL1_NAME, map_location=torch.device('cpu')), strict=False)

# Model 2: SRGAN_DHT uses GeneratorV1 (model)
print("Loading Model 2:", MODEL2_LABEL, "(using GeneratorV1)")
model2 = GeneratorV1(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model2.cuda()
    model2.load_state_dict(torch.load('epochs/' + MODEL2_NAME), strict=False)
else:
    model2.load_state_dict(torch.load('epochs/' + MODEL2_NAME, map_location=torch.device('cpu')), strict=False)

# Calculate FLOPs for both models (one-time calculation)
dummy_input = torch.randn(1, 3, 64, 64)  # Example LR input
if TEST_MODE:
    dummy_input = dummy_input.cuda()

flops_analysis1 = FlopCountAnalysis(model1, dummy_input)
model1_flops = flops_analysis1.total()

flops_analysis2 = FlopCountAnalysis(model2, dummy_input)
model2_flops = flops_analysis2.total()

print(f"{MODEL1_LABEL} FLOPs: {model1_flops:,}")
print(f"{MODEL2_LABEL} FLOPs: {model2_flops:,}")

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

def add_label_to_image(image, label, metrics_text=""):
    """Add a label and metrics to an image"""
    # Create a copy to avoid modifying original
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_metrics = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_title = ImageFont.load_default()
        font_metrics = ImageFont.load_default()
    
    # Add label at the top
    text_bbox = draw.textbbox((0, 0), label, font=font_title)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Draw background rectangle for better visibility
    padding = 5
    draw.rectangle([(0, 0), (img_copy.width, text_height + 2*padding)], fill=(0, 0, 0, 180))
    draw.text((padding, padding), label, fill=(255, 255, 255), font=font_title)
    
    # Add metrics if provided
    if metrics_text:
        metrics_bbox = draw.textbbox((0, 0), metrics_text, font=font_metrics)
        metrics_height = metrics_bbox[3] - metrics_bbox[1]
        y_pos = img_copy.height - metrics_height - 2*padding
        draw.rectangle([(0, y_pos), (img_copy.width, img_copy.height)], fill=(0, 0, 0, 180))
        draw.text((padding, y_pos), metrics_text, fill=(255, 255, 255), font=font_metrics)
    
    return img_copy

def create_comparison_grid(images_dict, image_name):
    """Create a side-by-side comparison grid of all methods"""
    # Order of methods to display
    method_order = ['GT', 'LR', MODEL1_LABEL, MODEL2_LABEL, 'Bicubic', 'Lanczos']
    
    # Filter to only include available methods
    available_methods = [m for m in method_order if m in images_dict]
    
    if not available_methods:
        return None
    
    # Get dimensions from first image
    first_img = images_dict[available_methods[0]]['image']
    img_width, img_height = first_img.size
    
    # Calculate grid dimensions (2 rows x 3 columns)
    cols = 3
    rows = (len(available_methods) + cols - 1) // cols
    
    # Create blank canvas
    grid_width = img_width * cols
    grid_height = img_height * rows
    grid = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    
    # Place images in grid
    for idx, method in enumerate(available_methods):
        row = idx // cols
        col = idx % cols
        
        img_data = images_dict[method]
        img = img_data['image']
        metrics = img_data.get('metrics', '')
        
        # Add label and metrics to image
        labeled_img = add_label_to_image(img, method, metrics)
        
        # Paste into grid
        x_offset = col * img_width
        y_offset = row * img_height
        grid.paste(labeled_img, (x_offset, y_offset))
    
    return grid

# Process Each Image in the Test Folder
print("=" * 80)
print("Starting Super-Resolution Comparison with Dual Models")
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
    
    # Prepare LR tensor for models
    lr_tensor = Variable(ToTensor()(lr_image)).unsqueeze(0)
    if TEST_MODE:
        lr_tensor = lr_tensor.cuda()
    
    image_results = {'image': image_name}
    comparison_images = {}
    
    # Add GT and LR to comparison
    comparison_images['GT'] = {'image': hr_image, 'metrics': 'Ground Truth'}
    lr_upscaled = lr_image.resize((hr_image.width, hr_image.height), Image.BICUBIC)
    comparison_images['LR'] = {'image': lr_upscaled, 'metrics': 'Low Resolution'}
    
    # 1. Model 1 Super-Resolution
    start = time.time()
    with torch.no_grad():
        sr_tensor1 = model1(lr_tensor)
    elapsed1 = time.time() - start
    
    sr_image1 = ToPILImage()(sr_tensor1[0].data.cpu())
    sr_image1.save(os.path.join(MODEL1_FOLDER, image_name))
    
    psnr_val1, ssim_val1 = calculate_metrics(hr_image, sr_image1)
    image_results[f'{MODEL1_LABEL}_PSNR'] = psnr_val1
    image_results[f'{MODEL1_LABEL}_SSIM'] = ssim_val1
    image_results[f'{MODEL1_LABEL}_Time'] = elapsed1
    image_results[f'{MODEL1_LABEL}_FLOPs'] = model1_flops
    
    method_metrics[MODEL1_LABEL]['psnr'].append(psnr_val1)
    method_metrics[MODEL1_LABEL]['ssim'].append(ssim_val1)
    method_metrics[MODEL1_LABEL]['time'].append(elapsed1)
    
    comparison_images[MODEL1_LABEL] = {
        'image': sr_image1,
        'metrics': f'PSNR: {psnr_val1:.2f} dB | SSIM: {ssim_val1:.4f}'
    }
    
    print(f"  {MODEL1_LABEL:12s} - Time: {elapsed1:.4f}s | PSNR: {psnr_val1:.2f} dB | SSIM: {ssim_val1:.4f} | FLOPs: {model1_flops:,}")
    
    # 2. Model 2 Super-Resolution
    start = time.time()
    with torch.no_grad():
        sr_tensor2 = model2(lr_tensor)
    elapsed2 = time.time() - start
    
    sr_image2 = ToPILImage()(sr_tensor2[0].data.cpu())
    sr_image2.save(os.path.join(MODEL2_FOLDER, image_name))
    
    psnr_val2, ssim_val2 = calculate_metrics(hr_image, sr_image2)
    image_results[f'{MODEL2_LABEL}_PSNR'] = psnr_val2
    image_results[f'{MODEL2_LABEL}_SSIM'] = ssim_val2
    image_results[f'{MODEL2_LABEL}_Time'] = elapsed2
    image_results[f'{MODEL2_LABEL}_FLOPs'] = model2_flops
    
    method_metrics[MODEL2_LABEL]['psnr'].append(psnr_val2)
    method_metrics[MODEL2_LABEL]['ssim'].append(ssim_val2)
    method_metrics[MODEL2_LABEL]['time'].append(elapsed2)
    
    comparison_images[MODEL2_LABEL] = {
        'image': sr_image2,
        'metrics': f'PSNR: {psnr_val2:.2f} dB | SSIM: {ssim_val2:.4f}'
    }
    
    print(f"  {MODEL2_LABEL:12s} - Time: {elapsed2:.4f}s | PSNR: {psnr_val2:.2f} dB | SSIM: {ssim_val2:.4f} | FLOPs: {model2_flops:,}")
    
    # 3. Interpolation-based methods
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
        
        # Add to comparison (only Bicubic and Lanczos for cleaner grid)
        if method_name in ['Bicubic', 'Lanczos']:
            comparison_images[method_name] = {
                'image': interp_image,
                'metrics': f'PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}'
            }
        
        print(f"  {method_name:12s} - Time: {elapsed:.4f}s | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | FLOPs: 0")
    
    # Create and save comparison grid
    comparison_grid = create_comparison_grid(comparison_images, image_name)
    if comparison_grid:
        comparison_path = os.path.join(COMPARISON_FOLDER, f"comparison_{image_name}")
        comparison_grid.save(comparison_path)
        print(f"  Comparison grid saved: comparison_{image_name}")
    
    results.append(image_results)

# Calculate and Display Average Metrics
print("\n" + "=" * 80)
print("AVERAGE METRICS ACROSS ALL IMAGES")
print("=" * 80)

summary_results = []
all_methods = [MODEL1_LABEL, MODEL2_LABEL, 'Nearest', 'Bilinear', 'Bicubic', 'Lanczos']

for method in all_methods:
    if method_metrics[method]['psnr']:  # Check if method has data
        avg_psnr = np.mean(method_metrics[method]['psnr'])
        avg_ssim = np.mean(method_metrics[method]['ssim'])
        avg_time = np.mean(method_metrics[method]['time'])
        
        if method == MODEL1_LABEL:
            flops = model1_flops
        elif method == MODEL2_LABEL:
            flops = model2_flops
        else:
            flops = 0
        
        summary_results.append({
            'Method': method,
            'Avg_PSNR': avg_psnr,
            'Avg_SSIM': avg_ssim,
            'Avg_Time': avg_time,
            'FLOPs': flops
        })
        
        print(f"{method:12s} - Avg PSNR: {avg_psnr:.2f} dB | Avg SSIM: {avg_ssim:.4f} | Avg Time: {avg_time:.4f}s | FLOPs: {flops:,}")

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
print("Side-by-side comparisons saved in:", COMPARISON_FOLDER)
print("=" * 80)

# Example command:
# python test_customV4.py --test_folder "/home/dst/Desktop/GAN/SRGAN/Test_set" --output_folder "/home/dst/Desktop/GAN/SRGAN/output"

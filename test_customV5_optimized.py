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
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
from scipy.stats import entropy
from scipy.ndimage import laplace
from fvcore.nn import FlopCountAnalysis
import cv2

from model import Generator as GeneratorV1
from model_cnn_transv3_LG_optimized import Generator as GeneratorV2

# Argument Parser
parser = argparse.ArgumentParser(description='Super Resolution Processing with Comprehensive Metrics (Memory Optimized)')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--test_folder', type=str, help='folder containing high-resolution images')
parser.add_argument('--output_folder', type=str, help='folder to save GT, LR, and SR images')
parser.add_argument('--model1_name', default='/CT_HYBRIDV4_4B_netG_epoch_4_99.pth', type=str, help='first generator model')
parser.add_argument('--model2_name', default='/CT_SRGAN_DHT__netG_epoch_4_95.pth', type=str, help='second generator model')
parser.add_argument('--model1_label', default='HYBRIDV4', type=str, help='label for first model')
parser.add_argument('--model2_label', default='SRGAN_DHT', type=str, help='label for second model')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision for inference')
parser.add_argument('--tile_size', default=512, type=int, help='tile size for processing large images')
parser.add_argument('--tile_overlap', default=32, type=int, help='overlap between tiles')
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
USE_AMP = opt.use_amp
TILE_SIZE = opt.tile_size
TILE_OVERLAP = opt.tile_overlap

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

print("=" * 100)
print("MEMORY OPTIMIZATION SETTINGS")
print("=" * 100)
print(f"  - Mixed Precision (AMP): {'ENABLED' if USE_AMP else 'DISABLED'}")
print(f"  - Tile Size: {TILE_SIZE}x{TILE_SIZE}")
print(f"  - Tile Overlap: {TILE_OVERLAP}px")
if TEST_MODE and torch.cuda.is_available():
    print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("=" * 100)

# Load Generator Models
# Model 1: HYBRIDV4 uses GeneratorV2 (model_cnn_transv3_LG_optimized)
print("\nLoading Model 1:", MODEL1_LABEL, "(using GeneratorV2 - Optimized)")
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

# Clear dummy input
del dummy_input, flops_analysis1, flops_analysis2
if TEST_MODE:
    torch.cuda.empty_cache()

# Interpolation methods mapping
interpolation_methods = {
    'Nearest': Image.NEAREST,
    'Bilinear': Image.BILINEAR,
    'Bicubic': Image.BICUBIC,
    'Lanczos': Image.LANCZOS
}

# Storage for metrics
results = []
metric_names = ['psnr', 'ssim', 'mse', 'rmse', 'mae', 'nrmse', 'uqi', 'ergas', 'scc', 
                'vif', 'sharpness', 'entropy', 'edge_strength', 'contrast', 'time']
method_metrics = defaultdict(lambda: {metric: [] for metric in metric_names})

def calculate_uqi(gt_array, sr_array):
    """Calculate Universal Quality Index (UQI)"""
    try:
        # Normalize to 0-1 range
        gt_norm = gt_array.astype(np.float64) / 255.0
        sr_norm = sr_array.astype(np.float64) / 255.0
        
        # Calculate means
        mean_gt = np.mean(gt_norm)
        mean_sr = np.mean(sr_norm)
        
        # Calculate variances
        var_gt = np.var(gt_norm)
        var_sr = np.var(sr_norm)
        
        # Calculate covariance
        cov = np.mean((gt_norm - mean_gt) * (sr_norm - mean_sr))
        
        # Calculate UQI
        numerator = 4 * cov * mean_gt * mean_sr
        denominator = (var_gt + var_sr) * (mean_gt**2 + mean_sr**2)
        
        if denominator == 0:
            return 0.0
        
        uqi = numerator / denominator
        return uqi
    except:
        return 0.0

def calculate_ergas(gt_array, sr_array, scale=4):
    """Calculate ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse)"""
    try:
        # Convert to float
        gt_float = gt_array.astype(np.float64)
        sr_float = sr_array.astype(np.float64)
        
        # Calculate per-channel RMSE
        if len(gt_array.shape) == 3:
            sum_squared_relative_error = 0
            for i in range(gt_array.shape[2]):
                mean_gt = np.mean(gt_float[:, :, i])
                if mean_gt == 0:
                    continue
                mse_channel = np.mean((gt_float[:, :, i] - sr_float[:, :, i]) ** 2)
                sum_squared_relative_error += mse_channel / (mean_gt ** 2)
            
            ergas = 100 * scale * np.sqrt(sum_squared_relative_error / gt_array.shape[2])
        else:
            mean_gt = np.mean(gt_float)
            if mean_gt == 0:
                return 0.0
            mse_val = np.mean((gt_float - sr_float) ** 2)
            ergas = 100 * scale * np.sqrt(mse_val / (mean_gt ** 2))
        
        return ergas
    except:
        return 0.0

def calculate_scc(gt_array, sr_array):
    """Calculate Spatial Correlation Coefficient"""
    try:
        gt_flat = gt_array.flatten().astype(np.float64)
        sr_flat = sr_array.flatten().astype(np.float64)
        
        correlation = np.corrcoef(gt_flat, sr_flat)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    except:
        return 0.0

def calculate_vif(gt_array, sr_array):
    """Calculate Visual Information Fidelity (VIF) - Simplified version"""
    try:
        # Convert to grayscale if color
        if len(gt_array.shape) == 3:
            gt_gray = cv2.cvtColor(gt_array, cv2.COLOR_RGB2GRAY)
            sr_gray = cv2.cvtColor(sr_array, cv2.COLOR_RGB2GRAY)
        else:
            gt_gray = gt_array
            sr_gray = sr_array
        
        # Normalize
        gt_norm = gt_gray.astype(np.float64) / 255.0
        sr_norm = sr_gray.astype(np.float64) / 255.0
        
        # Calculate local statistics
        sigma_nsq = 2
        
        # Simple VIF approximation using correlation
        correlation = np.corrcoef(gt_norm.flatten(), sr_norm.flatten())[0, 1]
        vif_approx = max(0, correlation)
        
        return vif_approx
    except:
        return 0.0

def calculate_sharpness(image_array):
    """Calculate image sharpness using Laplacian variance"""
    try:
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    except:
        return 0.0

def calculate_entropy_metric(image_array):
    """Calculate image entropy"""
    try:
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Calculate histogram
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        
        # Remove zeros
        hist = hist[hist > 0]
        
        # Calculate entropy
        ent = -np.sum(hist * np.log2(hist))
        return ent
    except:
        return 0.0

def calculate_edge_strength(image_array):
    """Calculate average edge strength using Sobel operator"""
    try:
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_strength = np.mean(magnitude)
        
        return edge_strength
    except:
        return 0.0

def calculate_contrast(image_array):
    """Calculate RMS contrast"""
    try:
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # RMS contrast
        contrast = np.std(gray)
        return contrast
    except:
        return 0.0

def calculate_comprehensive_metrics(gt_image, sr_image):
    """Calculate comprehensive set of image quality metrics"""
    # Convert PIL images to numpy arrays
    gt_array = np.array(gt_image)
    sr_array = np.array(sr_image)
    
    # Ensure images have the same dimensions
    if gt_array.shape != sr_array.shape:
        sr_image = sr_image.resize(gt_image.size, Image.BICUBIC)
        sr_array = np.array(sr_image)
    
    metrics = {}
    
    # 1. PSNR (Peak Signal-to-Noise Ratio) - Higher is better
    metrics['psnr'] = psnr(gt_array, sr_array, data_range=255)
    
    # 2. SSIM (Structural Similarity Index) - Higher is better (0-1)
    if len(gt_array.shape) == 3:
        metrics['ssim'] = ssim(gt_array, sr_array, multichannel=True, channel_axis=2, data_range=255)
    else:
        metrics['ssim'] = ssim(gt_array, sr_array, data_range=255)
    
    # 3. MSE (Mean Squared Error) - Lower is better
    metrics['mse'] = mse(gt_array, sr_array)
    
    # 4. RMSE (Root Mean Squared Error) - Lower is better
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # 5. MAE (Mean Absolute Error) - Lower is better
    metrics['mae'] = np.mean(np.abs(gt_array.astype(np.float64) - sr_array.astype(np.float64)))
    
    # 6. NRMSE (Normalized Root Mean Squared Error) - Lower is better
    metrics['nrmse'] = nrmse(gt_array, sr_array, normalization='mean')
    
    # 7. UQI (Universal Quality Index) - Higher is better (-1 to 1)
    metrics['uqi'] = calculate_uqi(gt_array, sr_array)
    
    # 8. ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse) - Lower is better
    metrics['ergas'] = calculate_ergas(gt_array, sr_array, scale=UPSCALE_FACTOR)
    
    # 9. SCC (Spatial Correlation Coefficient) - Higher is better (0-1)
    metrics['scc'] = calculate_scc(gt_array, sr_array)
    
    # 10. VIF (Visual Information Fidelity) - Higher is better
    metrics['vif'] = calculate_vif(gt_array, sr_array)
    
    # 11. Sharpness (Laplacian variance) - Higher is better
    metrics['sharpness'] = calculate_sharpness(sr_array)
    
    # 12. Entropy - Higher is better (more information)
    metrics['entropy'] = calculate_entropy_metric(sr_array)
    
    # 13. Edge Strength - Higher is better
    metrics['edge_strength'] = calculate_edge_strength(sr_array)
    
    # 14. Contrast - Higher is better
    metrics['contrast'] = calculate_contrast(sr_array)
    
    return metrics

def add_label_to_image(image, label, metrics_text=""):
    """Add a label and metrics to an image"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_metrics = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_title = ImageFont.load_default()
        font_metrics = ImageFont.load_default()
    
    padding = 5
    
    # Add label at the top
    text_bbox = draw.textbbox((0, 0), label, font=font_title)
    text_height = text_bbox[3] - text_bbox[1]
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
    method_order = ['GT', 'LR', MODEL1_LABEL, MODEL2_LABEL, 'Bicubic', 'Lanczos']
    available_methods = [m for m in method_order if m in images_dict]
    
    if not available_methods:
        return None
    
    first_img = images_dict[available_methods[0]]['image']
    img_width, img_height = first_img.size
    
    cols = 3
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

def process_with_tiling(model, lr_image, tile_size, overlap):
    """Process large images using tiling to reduce memory usage"""
    lr_w, lr_h = lr_image.size
    
    # If image is small enough, process normally
    if lr_w <= tile_size and lr_h <= tile_size:
        lr_tensor = Variable(ToTensor()(lr_image)).unsqueeze(0)
        if TEST_MODE:
            lr_tensor = lr_tensor.cuda()
        
        with torch.no_grad():
            if USE_AMP and TEST_MODE:
                with torch.cuda.amp.autocast():
                    sr_tensor = model(lr_tensor)
            else:
                sr_tensor = model(lr_tensor)
        
        sr_image = ToPILImage()(sr_tensor[0].data.cpu())
        
        # Clear memory
        del lr_tensor, sr_tensor
        if TEST_MODE:
            torch.cuda.empty_cache()
        
        return sr_image
    
    # Process with tiling
    sr_w = lr_w * UPSCALE_FACTOR
    sr_h = lr_h * UPSCALE_FACTOR
    sr_image = Image.new('RGB', (sr_w, sr_h))
    
    # Calculate tile positions
    stride = tile_size - overlap
    tiles_x = (lr_w - overlap) // stride + (1 if (lr_w - overlap) % stride != 0 else 0)
    tiles_y = (lr_h - overlap) // stride + (1 if (lr_h - overlap) % stride != 0 else 0)
    
    print(f"    Processing with {tiles_x}x{tiles_y} tiles...")
    
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            # Calculate tile boundaries
            x_start = tx * stride
            y_start = ty * stride
            x_end = min(x_start + tile_size, lr_w)
            y_end = min(y_start + tile_size, lr_h)
            
            # Extract tile
            tile = lr_image.crop((x_start, y_start, x_end, y_end))
            
            # Process tile
            tile_tensor = Variable(ToTensor()(tile)).unsqueeze(0)
            if TEST_MODE:
                tile_tensor = tile_tensor.cuda()
            
            with torch.no_grad():
                if USE_AMP and TEST_MODE:
                    with torch.cuda.amp.autocast():
                        sr_tile_tensor = model(tile_tensor)
                else:
                    sr_tile_tensor = model(tile_tensor)
            
            sr_tile = ToPILImage()(sr_tile_tensor[0].data.cpu())
            
            # Calculate paste position
            paste_x = x_start * UPSCALE_FACTOR
            paste_y = y_start * UPSCALE_FACTOR
            
            # Handle overlap blending
            if tx > 0 or ty > 0:
                # Crop overlap region
                crop_left = overlap * UPSCALE_FACTOR if tx > 0 else 0
                crop_top = overlap * UPSCALE_FACTOR if ty > 0 else 0
                sr_tile = sr_tile.crop((crop_left, crop_top, sr_tile.width, sr_tile.height))
                paste_x += crop_left
                paste_y += crop_top
            
            # Paste tile
            sr_image.paste(sr_tile, (paste_x, paste_y))
            
            # Clear memory
            del tile, tile_tensor, sr_tile_tensor, sr_tile
            if TEST_MODE:
                torch.cuda.empty_cache()
    
    return sr_image

# Process Each Image in the Test Folder
print("=" * 100)
print("Starting Super-Resolution Comparison with Comprehensive Metrics (Memory Optimized)")
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
    
    image_results = {'image': image_name}
    comparison_images = {}
    
    # Add GT and LR to comparison
    comparison_images['GT'] = {'image': hr_image, 'metrics': 'Ground Truth'}
    lr_upscaled = lr_image.resize((hr_image.width, hr_image.height), Image.BICUBIC)
    comparison_images['LR'] = {'image': lr_upscaled, 'metrics': 'Low Resolution'}
    
    # 1. Model 1 Super-Resolution (with tiling and AMP)
    print(f"  Processing {MODEL1_LABEL}...")
    start = time.time()
    sr_image1 = process_with_tiling(model1, lr_image, TILE_SIZE // UPSCALE_FACTOR, TILE_OVERLAP // UPSCALE_FACTOR)
    elapsed1 = time.time() - start
    
    sr_image1.save(os.path.join(MODEL1_FOLDER, image_name))
    
    metrics1 = calculate_comprehensive_metrics(hr_image, sr_image1)
    metrics1['time'] = elapsed1
    
    for metric_name, metric_value in metrics1.items():
        image_results[f'{MODEL1_LABEL}_{metric_name.upper()}'] = metric_value
        method_metrics[MODEL1_LABEL][metric_name].append(metric_value)
    
    image_results[f'{MODEL1_LABEL}_FLOPs'] = model1_flops
    
    comparison_images[MODEL1_LABEL] = {
        'image': sr_image1,
        'metrics': f'PSNR: {metrics1["psnr"]:.2f} | SSIM: {metrics1["ssim"]:.4f}'
    }
    
    print(f"  {MODEL1_LABEL:12s} - PSNR: {metrics1['psnr']:.2f} | SSIM: {metrics1['ssim']:.4f} | "
          f"MSE: {metrics1['mse']:.2f} | MAE: {metrics1['mae']:.2f} | UQI: {metrics1['uqi']:.4f} | "
          f"Time: {elapsed1:.4f}s")
    
    # Clear memory
    del sr_image1
    if TEST_MODE:
        torch.cuda.empty_cache()
    
    # 2. Model 2 Super-Resolution (with tiling and AMP)
    print(f"  Processing {MODEL2_LABEL}...")
    start = time.time()
    sr_image2 = process_with_tiling(model2, lr_image, TILE_SIZE // UPSCALE_FACTOR, TILE_OVERLAP // UPSCALE_FACTOR)
    elapsed2 = time.time() - start
    
    sr_image2.save(os.path.join(MODEL2_FOLDER, image_name))
    
    metrics2 = calculate_comprehensive_metrics(hr_image, sr_image2)
    metrics2['time'] = elapsed2
    
    for metric_name, metric_value in metrics2.items():
        image_results[f'{MODEL2_LABEL}_{metric_name.upper()}'] = metric_value
        method_metrics[MODEL2_LABEL][metric_name].append(metric_value)
    
    image_results[f'{MODEL2_LABEL}_FLOPs'] = model2_flops
    
    comparison_images[MODEL2_LABEL] = {
        'image': sr_image2,
        'metrics': f'PSNR: {metrics2["psnr"]:.2f} | SSIM: {metrics2["ssim"]:.4f}'
    }
    
    print(f"  {MODEL2_LABEL:12s} - PSNR: {metrics2['psnr']:.2f} | SSIM: {metrics2['ssim']:.4f} | "
          f"MSE: {metrics2['mse']:.2f} | MAE: {metrics2['mae']:.2f} | UQI: {metrics2['uqi']:.4f} | "
          f"Time: {elapsed2:.4f}s")
    
    # Clear memory
    del sr_image2
    if TEST_MODE:
        torch.cuda.empty_cache()
    
    # 3. Interpolation-based methods
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
        
        metrics_interp = calculate_comprehensive_metrics(hr_image, interp_image)
        metrics_interp['time'] = elapsed
        
        for metric_name_key, metric_value in metrics_interp.items():
            image_results[f'{method_name}_{metric_name_key.upper()}'] = metric_value
            method_metrics[method_name][metric_name_key].append(metric_value)
        
        image_results[f'{method_name}_FLOPs'] = 0
        
        if method_name in ['Bicubic', 'Lanczos']:
            comparison_images[method_name] = {
                'image': interp_image,
                'metrics': f'PSNR: {metrics_interp["psnr"]:.2f} | SSIM: {metrics_interp["ssim"]:.4f}'
            }
        
        print(f"  {method_name:12s} - PSNR: {metrics_interp['psnr']:.2f} | SSIM: {metrics_interp['ssim']:.4f} | "
              f"MSE: {metrics_interp['mse']:.2f} | MAE: {metrics_interp['mae']:.2f} | UQI: {metrics_interp['uqi']:.4f} | "
              f"Time: {elapsed:.4f}s")
    
    # Create and save comparison grid
    comparison_grid = create_comparison_grid(comparison_images, image_name)
    if comparison_grid:
        comparison_path = os.path.join(COMPARISON_FOLDER, f"comparison_{image_name}")
        comparison_grid.save(comparison_path)
        print(f"  ✓ Comparison grid saved")
    
    results.append(image_results)
    
    # Clear memory after each image
    if TEST_MODE:
        torch.cuda.empty_cache()

# Calculate and Display Average Metrics
print("\n" + "=" * 100)
print("COMPREHENSIVE AVERAGE METRICS ACROSS ALL IMAGES")
print("=" * 100)

summary_results = []
all_methods = [MODEL1_LABEL, MODEL2_LABEL, 'Nearest', 'Bilinear', 'Bicubic', 'Lanczos']

print(f"\n{'Method':<12} | {'PSNR':<8} | {'SSIM':<8} | {'MSE':<10} | {'MAE':<8} | {'UQI':<8} | "
      f"{'ERGAS':<8} | {'VIF':<8} | {'Sharp':<10} | {'Time':<8} | {'FLOPs':<12}")
print("-" * 130)

for method in all_methods:
    if method_metrics[method]['psnr']:
        summary = {'Method': method}
        
        for metric in metric_names:
            if method_metrics[method][metric]:
                avg_value = np.mean(method_metrics[method][metric])
                summary[f'Avg_{metric.upper()}'] = avg_value
        
        if method == MODEL1_LABEL:
            summary['FLOPs'] = model1_flops
        elif method == MODEL2_LABEL:
            summary['FLOPs'] = model2_flops
        else:
            summary['FLOPs'] = 0
        
        summary_results.append(summary)
        
        print(f"{method:<12} | {summary.get('Avg_PSNR', 0):<8.2f} | {summary.get('Avg_SSIM', 0):<8.4f} | "
              f"{summary.get('Avg_MSE', 0):<10.2f} | {summary.get('Avg_MAE', 0):<8.2f} | "
              f"{summary.get('Avg_UQI', 0):<8.4f} | {summary.get('Avg_ERGAS', 0):<8.2f} | "
              f"{summary.get('Avg_VIF', 0):<8.4f} | {summary.get('Avg_SHARPNESS', 0):<10.2f} | "
              f"{summary.get('Avg_TIME', 0):<8.4f} | {summary['FLOPs']:<12,}")

# Save detailed results to CSV
csv_path = os.path.join(OUTPUT_FOLDER, "detailed_results_comprehensive_optimized.csv")
if results:
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Detailed results saved to: {csv_path}")

# Save summary results to CSV
summary_csv_path = os.path.join(OUTPUT_FOLDER, "summary_results_comprehensive_optimized.csv")
with open(summary_csv_path, 'w', newline='') as csvfile:
    if summary_results:
        fieldnames = summary_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_results)
print(f"✓ Summary results saved to: {summary_csv_path}")

print("\n" + "=" * 100)
print("Processing complete!")
print(f"  - All images saved in: {OUTPUT_FOLDER}")
print(f"  - Side-by-side comparisons: {COMPARISON_FOLDER}")
print(f"  - Comprehensive metrics: {csv_path}")
print("=" * 100)

# Example commands:
# Basic usage (CPU):
# python test_customV5_optimized.py --test_folder "/home/dst/Desktop/GAN/SRGAN/Test_set" --output_folder "/home/dst/Desktop/GAN/SRGAN/output"
#
# GPU with mixed precision (recommended for RTX 4080):
# python test_customV5_optimized.py --test_mode GPU --use_amp --test_folder "/home/dst/Desktop/GAN/SRGAN/Test_set" --output_folder "/home/dst/Desktop/GAN/SRGAN/output"
#
# GPU with custom tile size for very large images:
# python test_customV5_optimized.py --test_mode GPU --use_amp --tile_size 256 --tile_overlap 16 --test_folder "/home/dst/Desktop/GAN/SRGAN/Test_set" --output_folder "/home/dst/Desktop/GAN/SRGAN/output"

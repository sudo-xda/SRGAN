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
from model_cnn_transv3_LG import Generator as GeneratorV2
from model_srcnn import SRCNN
from vdsr_model import VDSR

# Argument Parser
parser = argparse.ArgumentParser(description='Super Resolution Processing with Four Models + Comprehensive Metrics')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--test_folder', type=str, help='folder containing high-resolution images')
parser.add_argument('--output_folder', type=str, help='folder to save GT, LR, and SR images')
parser.add_argument('--model1_name', default='/CT_HYBRIDV4_4B_netG_epoch_4_99.pth', type=str, help='first generator model')
parser.add_argument('--model2_name', default='/CT_SRGAN_DHT__netG_epoch_4_95.pth', type=str, help='second generator model')
parser.add_argument('--srcnn_name', default='srcnn_ct_best.pth', type=str, help='SRCNN model weights')
parser.add_argument('--vdsr_name', default='vdsr_CT.pth', type=str, help='VDSR model weights')
parser.add_argument('--model1_label', default='HYBRIDV4', type=str, help='label for first model')
parser.add_argument('--model2_label', default='SRGAN_DHT', type=str, help='label for second model')
parser.add_argument('--srcnn_label', default='SRCNN', type=str, help='label for SRCNN model')
parser.add_argument('--vdsr_label', default='VDSR', type=str, help='label for VDSR model')
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
MODEL1_LABEL = opt.model1_label
MODEL2_LABEL = opt.model2_label
SRCNN_LABEL = opt.srcnn_label
VDSR_LABEL = opt.vdsr_label

# Define Output Subfolders
GT_FOLDER = os.path.join(OUTPUT_FOLDER, "GT")
LR_FOLDER = os.path.join(OUTPUT_FOLDER, "LR")
MODEL1_FOLDER = os.path.join(OUTPUT_FOLDER, f"SR_{MODEL1_LABEL}")
MODEL2_FOLDER = os.path.join(OUTPUT_FOLDER, f"SR_{MODEL2_LABEL}")
SRCNN_FOLDER = os.path.join(OUTPUT_FOLDER, f"SR_{SRCNN_LABEL}")
VDSR_FOLDER = os.path.join(OUTPUT_FOLDER, f"SR_{VDSR_LABEL}")
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

# Model 3: SRCNN
print("Loading Model 3:", SRCNN_LABEL, "(using SRCNN)")
model_srcnn = SRCNN().eval()
if TEST_MODE:
    model_srcnn.cuda()
    checkpoint_srcnn = torch.load(SRCNN_NAME)
else:
    checkpoint_srcnn = torch.load(SRCNN_NAME, map_location=torch.device('cpu'))

# Load SRCNN weights
if 'state_dict' in checkpoint_srcnn:
    model_srcnn.load_state_dict(checkpoint_srcnn['state_dict'])
else:
    model_srcnn.load_state_dict(checkpoint_srcnn)

# Model 4: VDSR
print("Loading Model 4:", VDSR_LABEL, "(using VDSR)")
model_vdsr = VDSR().eval()
if TEST_MODE:
    model_vdsr.cuda()
    checkpoint_vdsr = torch.load(VDSR_NAME)
else:
    checkpoint_vdsr = torch.load(VDSR_NAME, map_location=torch.device('cpu'))

# Load VDSR weights (handle key name mismatch: 'net' -> 'network')
if 'state_dict' in checkpoint_vdsr:
    state_dict = checkpoint_vdsr['state_dict']
else:
    state_dict = checkpoint_vdsr

# Remap keys if needed (net.X.Y -> network.X.Y)
if any(key.startswith('net.') for key in state_dict.keys()):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('net.', 'network.')
        new_state_dict[new_key] = value
    model_vdsr.load_state_dict(new_state_dict)
else:
    model_vdsr.load_state_dict(state_dict)

# Calculate FLOPs for all models (one-time calculation)
dummy_input_rgb = torch.randn(1, 3, 64, 64)  # RGB input for SRGAN models
dummy_input_y = torch.randn(1, 1, 64, 64)    # Y channel input for SRCNN/VDSR

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

print(f"{MODEL1_LABEL} FLOPs: {model1_flops:,}")
print(f"{MODEL2_LABEL} FLOPs: {model2_flops:,}")
print(f"{SRCNN_LABEL} FLOPs: {srcnn_flops:,}")
print(f"{VDSR_LABEL} FLOPs: {vdsr_flops:,}")

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

def bgr2ycbcr(img, only_y=True):
    """Convert BGR image to YCbCr"""
    if only_y:
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

def ycbcr2bgr(img):
    """Convert YCbCr image to BGR"""
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

def calculate_uqi(gt_array, sr_array):
    """Calculate Universal Quality Index (UQI)"""
    try:
        gt_norm = gt_array.astype(np.float64) / 255.0
        sr_norm = sr_array.astype(np.float64) / 255.0
        
        mean_gt = np.mean(gt_norm)
        mean_sr = np.mean(sr_norm)
        var_gt = np.var(gt_norm)
        var_sr = np.var(sr_norm)
        cov = np.mean((gt_norm - mean_gt) * (sr_norm - mean_sr))
        
        numerator = 4 * cov * mean_gt * mean_sr
        denominator = (var_gt + var_sr) * (mean_gt**2 + mean_sr**2)
        
        if denominator == 0:
            return 0.0
        
        uqi = numerator / denominator
        return uqi
    except:
        return 0.0

def calculate_ergas(gt_array, sr_array, scale=4):
    """Calculate ERGAS"""
    try:
        gt_float = gt_array.astype(np.float64)
        sr_float = sr_array.astype(np.float64)
        
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
        if len(gt_array.shape) == 3:
            gt_gray = cv2.cvtColor(gt_array, cv2.COLOR_RGB2GRAY)
            sr_gray = cv2.cvtColor(sr_array, cv2.COLOR_RGB2GRAY)
        else:
            gt_gray = gt_array
            sr_gray = sr_array
        
        gt_norm = gt_gray.astype(np.float64) / 255.0
        sr_norm = sr_gray.astype(np.float64) / 255.0
        
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
        
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
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
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
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
        
        contrast = np.std(gray)
        return contrast
    except:
        return 0.0

def calculate_comprehensive_metrics(gt_image, sr_image):
    """Calculate comprehensive set of image quality metrics"""
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
    
    metrics['mse'] = mse(gt_array, sr_array)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = np.mean(np.abs(gt_array.astype(np.float64) - sr_array.astype(np.float64)))
    metrics['nrmse'] = nrmse(gt_array, sr_array, normalization='mean')
    metrics['uqi'] = calculate_uqi(gt_array, sr_array)
    metrics['ergas'] = calculate_ergas(gt_array, sr_array, scale=UPSCALE_FACTOR)
    metrics['scc'] = calculate_scc(gt_array, sr_array)
    metrics['vif'] = calculate_vif(gt_array, sr_array)
    metrics['sharpness'] = calculate_sharpness(sr_array)
    metrics['entropy'] = calculate_entropy_metric(sr_array)
    metrics['edge_strength'] = calculate_edge_strength(sr_array)
    metrics['contrast'] = calculate_contrast(sr_array)
    
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
    method_order = ['GT', 'LR', MODEL1_LABEL, MODEL2_LABEL, SRCNN_LABEL, VDSR_LABEL, 'Bicubic', 'Lanczos']
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
print("=" * 120)
print("Starting Super-Resolution Comparison with Four Models + Comprehensive Metrics")
print("=" * 120)

for image_name in os.listdir(TEST_FOLDER):
    image_path = os.path.join(TEST_FOLDER, image_name)
    if not os.path.isfile(image_path):
        continue

    print(f"\nProcessing: {image_name}")
    print("-" * 120)
    
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
    
    # 1. Model 1 (HYBRIDV4) Super-Resolution
    start = time.time()
    with torch.no_grad():
        sr_tensor1 = model1(lr_tensor)
    elapsed1 = time.time() - start
    
    sr_image1 = ToPILImage()(sr_tensor1[0].data.cpu())
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
          f"MSE: {metrics1['mse']:.2f} | MAE: {metrics1['mae']:.2f} | Time: {elapsed1:.4f}s")
    
    # 2. Model 2 (SRGAN_DHT) Super-Resolution
    start = time.time()
    with torch.no_grad():
        sr_tensor2 = model2(lr_tensor)
    elapsed2 = time.time() - start
    
    sr_image2 = ToPILImage()(sr_tensor2[0].data.cpu())
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
          f"MSE: {metrics2['mse']:.2f} | MAE: {metrics2['mae']:.2f} | Time: {elapsed2:.4f}s")
    
    # 3. SRCNN Super-Resolution (Y channel only)
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
    
    metrics_srcnn = calculate_comprehensive_metrics(hr_image, sr_image_srcnn)
    metrics_srcnn['time'] = elapsed_srcnn
    
    for metric_name, metric_value in metrics_srcnn.items():
        image_results[f'{SRCNN_LABEL}_{metric_name.upper()}'] = metric_value
        method_metrics[SRCNN_LABEL][metric_name].append(metric_value)
    
    image_results[f'{SRCNN_LABEL}_FLOPs'] = srcnn_flops
    
    comparison_images[SRCNN_LABEL] = {
        'image': sr_image_srcnn,
        'metrics': f'PSNR: {metrics_srcnn["psnr"]:.2f} | SSIM: {metrics_srcnn["ssim"]:.4f}'
    }
    
    print(f"  {SRCNN_LABEL:12s} - PSNR: {metrics_srcnn['psnr']:.2f} | SSIM: {metrics_srcnn['ssim']:.4f} | "
          f"MSE: {metrics_srcnn['mse']:.2f} | MAE: {metrics_srcnn['mae']:.2f} | Time: {elapsed_srcnn:.4f}s")
    
    # 4. VDSR Super-Resolution (Y channel only)
    start = time.time()
    
    # VDSR also uses bicubic upscaled image as input
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
    
    metrics_vdsr = calculate_comprehensive_metrics(hr_image, sr_image_vdsr)
    metrics_vdsr['time'] = elapsed_vdsr
    
    for metric_name, metric_value in metrics_vdsr.items():
        image_results[f'{VDSR_LABEL}_{metric_name.upper()}'] = metric_value
        method_metrics[VDSR_LABEL][metric_name].append(metric_value)
    
    image_results[f'{VDSR_LABEL}_FLOPs'] = vdsr_flops
    
    comparison_images[VDSR_LABEL] = {
        'image': sr_image_vdsr,
        'metrics': f'PSNR: {metrics_vdsr["psnr"]:.2f} | SSIM: {metrics_vdsr["ssim"]:.4f}'
    }
    
    print(f"  {VDSR_LABEL:12s} - PSNR: {metrics_vdsr['psnr']:.2f} | SSIM: {metrics_vdsr['ssim']:.4f} | "
          f"MSE: {metrics_vdsr['mse']:.2f} | MAE: {metrics_vdsr['mae']:.2f} | Time: {elapsed_vdsr:.4f}s")
    
    # 5. Interpolation-based methods
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
              f"MSE: {metrics_interp['mse']:.2f} | MAE: {metrics_interp['mae']:.2f} | Time: {elapsed:.4f}s")
    
    # Create and save comparison grid
    comparison_grid = create_comparison_grid(comparison_images, image_name)
    if comparison_grid:
        comparison_path = os.path.join(COMPARISON_FOLDER, f"comparison_{image_name}")
        comparison_grid.save(comparison_path)
        print(f"  ✓ Comparison grid saved")
    
    results.append(image_results)

# Calculate and Display Average Metrics
print("\n" + "=" * 120)
print("COMPREHENSIVE AVERAGE METRICS ACROSS ALL IMAGES")
print("=" * 120)

summary_results = []
all_methods = [MODEL1_LABEL, MODEL2_LABEL, SRCNN_LABEL, VDSR_LABEL, 'Nearest', 'Bilinear', 'Bicubic', 'Lanczos']

print(f"\n{'Method':<12} | {'PSNR':<8} | {'SSIM':<8} | {'MSE':<10} | {'MAE':<8} | {'UQI':<8} | "
      f"{'ERGAS':<8} | {'VIF':<8} | {'Sharp':<10} | {'Time':<8} | {'FLOPs':<12}")
print("-" * 150)

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
        elif method == SRCNN_LABEL:
            summary['FLOPs'] = srcnn_flops
        elif method == VDSR_LABEL:
            summary['FLOPs'] = vdsr_flops
        else:
            summary['FLOPs'] = 0
        
        summary_results.append(summary)
        
        print(f"{method:<12} | {summary.get('Avg_PSNR', 0):<8.2f} | {summary.get('Avg_SSIM', 0):<8.4f} | "
              f"{summary.get('Avg_MSE', 0):<10.2f} | {summary.get('Avg_MAE', 0):<8.2f} | "
              f"{summary.get('Avg_UQI', 0):<8.4f} | {summary.get('Avg_ERGAS', 0):<8.2f} | "
              f"{summary.get('Avg_VIF', 0):<8.4f} | {summary.get('Avg_SHARPNESS', 0):<10.2f} | "
              f"{summary.get('Avg_TIME', 0):<8.4f} | {summary['FLOPs']:<12,}")

# Save detailed results to CSV
csv_path = os.path.join(OUTPUT_FOLDER, "detailed_results_v7.csv")
if results:
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Detailed results saved to: {csv_path}")

# Save summary results to CSV
summary_csv_path = os.path.join(OUTPUT_FOLDER, "summary_results_v7.csv")
with open(summary_csv_path, 'w', newline='') as csvfile:
    if summary_results:
        fieldnames = summary_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_results)
print(f"✓ Summary results saved to: {summary_csv_path}")

print("\n" + "=" * 120)
print("Processing complete!")
print(f"  - All images saved in: {OUTPUT_FOLDER}")
print(f"  - Side-by-side comparisons: {COMPARISON_FOLDER}")
print(f"  - Comprehensive metrics: {csv_path}")
print(f"  - Models compared: {MODEL1_LABEL}, {MODEL2_LABEL}, {SRCNN_LABEL}, {VDSR_LABEL} + 4 interpolation methods")
print("=" * 120)

# Example command:
# python test_customV7.py --test_folder "/home/dst/Desktop/GAN/SRGAN/Test_set" --output_folder "/home/dst/Desktop/GAN/SRGAN/output"

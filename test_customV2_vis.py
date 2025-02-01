import argparse
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Resize
import torchvision.utils as vutils

from model import Generator

# Argument Parser
parser = argparse.ArgumentParser(description='Super Resolution Processing')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--test_folder', type=str, help='folder containing high-resolution images')
parser.add_argument('--output_folder', type=str, help='folder to save GT, LR, SR images')
parser.add_argument('--model_name', default='netG_BN_epoch_4  _100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

# Parameters
UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
TEST_FOLDER = opt.test_folder
OUTPUT_FOLDER = opt.output_folder
MODEL_NAME = opt.model_name


GT_FOLDER = os.path.join(OUTPUT_FOLDER, "GT")
LR_FOLDER = os.path.join(OUTPUT_FOLDER, "LR")
SR_FOLDER = os.path.join(OUTPUT_FOLDER, "SR")
VIS_FOLDER = os.path.join(OUTPUT_FOLDER, "vis")  


os.makedirs(GT_FOLDER, exist_ok=True)
os.makedirs(LR_FOLDER, exist_ok=True)
os.makedirs(SR_FOLDER, exist_ok=True)
os.makedirs(VIS_FOLDER, exist_ok=True)


model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=torch.device('cpu')))


activation = {}  

def hook_fn(module, input, output):
    """Hook function to save layer activations."""
    activation[module] = output


for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Conv2d):  # Add hooks only to Conv2d layers
        layer.register_forward_hook(hook_fn)

# Process Each Image in the Test Folder
for image_name in os.listdir(TEST_FOLDER):
    image_path = os.path.join(TEST_FOLDER, image_name)
    if not os.path.isfile(image_path):
        continue

    # Load HR Image (GT)
    hr_image = Image.open(image_path)
    hr_image.save(os.path.join(GT_FOLDER, image_name))  # Save GT Image

    # Create LR Image by Downsampling
    lr_image = hr_image.resize(
        (hr_image.width // UPSCALE_FACTOR, hr_image.height // UPSCALE_FACTOR),
        Image.BICUBIC
    )
    lr_image.save(os.path.join(LR_FOLDER, image_name))  # Save LR Image

    # Upsample LR Image (Back to HR)
    lr_tensor = Variable(ToTensor()(lr_image)).unsqueeze(0)
    if TEST_MODE:
        lr_tensor = lr_tensor.cuda()

    start = time.time()
    sr_tensor = model(lr_tensor)  # Super-Resolution Output
    elapsed = time.time() - start
    print(f"Processed {image_name} in {elapsed:.4f}s")

    # Save SR Image
    sr_image = ToPILImage()(sr_tensor[0].data.cpu())
    sr_image.save(os.path.join(SR_FOLDER, image_name))  # Save SR Image

    # Save intermediate layer visualizations
    vis_image_path = os.path.join(VIS_FOLDER, image_name.split('.')[0])  # Subfolder for the image
    os.makedirs(vis_image_path, exist_ok=True)

    for idx, (layer, activation_data) in enumerate(activation.items()):
        # Get the activation data
        act = activation_data[0].data.cpu()  # Get the first batch activation (shape: [C, H, W])

        # Save each channel (feature map) separately
        for channel_idx in range(act.shape[0]):
            channel_act = act[channel_idx].unsqueeze(0)  # Shape: [1, H, W]
            grid = vutils.make_grid(channel_act, normalize=True, scale_each=True)
            vutils.save_image(grid, os.path.join(vis_image_path, f"{idx}_{layer.__class__.__name__}_channel_{channel_idx}.png"))


print("Processing complete. GT, LR, SR images and visualizations are saved in:", OUTPUT_FOLDER)

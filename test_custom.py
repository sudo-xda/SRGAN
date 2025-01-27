import argparse
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Super Resolution on a Folder of Images')
parser.add_argument('--upscale_factor', default=2, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--test_folder', type=str, help='folder containing test low-resolution images')
parser.add_argument('--output_folder', type=str, help='folder to save super-resolved images')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

# Parameters
UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
TEST_FOLDER = opt.test_folder
OUTPUT_FOLDER = opt.output_folder
MODEL_NAME = opt.model_name

# Load Generator Model
model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=torch.device('cpu')))

# Ensure output folder exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Process each image in the test folder
for image_name in os.listdir(TEST_FOLDER):
    image_path = os.path.join(TEST_FOLDER, image_name)
    if not os.path.isfile(image_path):
        continue

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()

    # Super-resolve the image
    start = time.time()
    out = model(image)
    elapsed = time.time() - start
    print(f"Processed {image_name} in {elapsed:.4f}s")

    # Save the output image
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save(os.path.join(OUTPUT_FOLDER, f'srf_{UPSCALE_FACTOR}_{image_name}'))

print("Processing complete. Super-resolved images are saved in:", OUTPUT_FOLDER)

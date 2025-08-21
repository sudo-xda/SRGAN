import os
import shutil
from PIL import Image

upscale_factor = 4
source_root = 'dataset'
destination_hr_root = f'data/test/SRF_{upscale_factor}/target'
destination_lr_root = f'data/test/SRF_{upscale_factor}'

os.makedirs(destination_hr_root, exist_ok=True)
os.makedirs(destination_lr_root, exist_ok=True)

# Function to check if file is an image
def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp'])

# Move HR images and generate LR images
for dataset_name in os.listdir(source_root):
    dataset_path = os.path.join(source_root, dataset_name, 'target')
    if not os.path.isdir(dataset_path):
        continue

    lr_folder = os.path.join(destination_lr_root, f'{dataset_name}_lr')
    os.makedirs(lr_folder, exist_ok=True)

    for filename in os.listdir(dataset_path):
        if is_image_file(filename):
            hr_image_path = os.path.join(dataset_path, filename)
            # Move HR image to common HR folder
            shutil.copy(hr_image_path, os.path.join(destination_hr_root, filename))

            # Create LR version
            hr_image = Image.open(hr_image_path).convert('RGB')
            w, h = hr_image.size
            lr_image = hr_image.resize((w // upscale_factor, h // upscale_factor), Image.BICUBIC)
            lr_image.save(os.path.join(lr_folder, filename))

print(f"[INFO] Finished moving HR images and creating LR images for all datasets!")

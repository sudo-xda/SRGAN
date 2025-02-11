import os
import cv2
import pywt
import numpy as np

# Define the paths
input_dir = '/home/dst/Desktop/GAN/SRGAN_old/data/Flickr2K/Flickr2K_HR'
output_dir = '/home/dst/Desktop/GAN/SRGAN/data/wavlet'

# Create subfolders for each subband (LL, LH, HL, HH)
ll_dir = os.path.join(output_dir, 'LL')
lh_dir = os.path.join(output_dir, 'LH')
hl_dir = os.path.join(output_dir, 'HL')
hh_dir = os.path.join(output_dir, 'HH')

# Make sure output subdirectories exist
os.makedirs(ll_dir, exist_ok=True)
os.makedirs(lh_dir, exist_ok=True)
os.makedirs(hl_dir, exist_ok=True)
os.makedirs(hh_dir, exist_ok=True)

# Function to perform wavelet transform and save subbands in respective folders
def wavelet_transform_and_save(image, filename):
    # Perform 2D discrete wavelet transform
    coeffs2 = pywt.dwt2(image, 'db2')  # You can change 'haar' to other wavelets like 'db1', 'db2', etc.
    LL, (LH, HL, HH) = coeffs2
    
    # Save the subbands in respective directories
    cv2.imwrite(os.path.join(ll_dir, f"{filename}_LL.png"), LL)
    cv2.imwrite(os.path.join(lh_dir, f"{filename}_LH.png"), LH)
    cv2.imwrite(os.path.join(hl_dir, f"{filename}_HL.png"), HL)
    cv2.imwrite(os.path.join(hh_dir, f"{filename}_HH.png"), HH)

# Loop over all images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Process image files only
        image_path = os.path.join(input_dir, filename)
        
        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale for wavelet transform
        
        if image is not None:
            print(f"Processing {filename}...")
            wavelet_transform_and_save(image, filename.split('.')[0])  # Remove file extension for naming
        else:
            print(f"Error reading {filename}")

import os
import cv2
import numpy as np

# Function to load and check the range of DWT components (limited to 5-10 images)
def check_dwt_range(folder_path, num_images=5):
    # List of image filenames in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".jpg")]
    
    # Limit to the first 'num_images' images
    selected_images = image_files[:min(num_images, len(image_files))]
    
    for filename in selected_images:
        image_path = os.path.join(folder_path, filename)
        
        # Load the image (assuming it's saved in grayscale format)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            min_val = np.min(img)
            max_val = np.max(img)
            mean_val = np.mean(img)
            print(f"Image: {filename}")
            print(f"Min: {min_val}, Max: {max_val}, Mean: {mean_val}")
        else:
            print(f"Error loading {filename}")
# Example usage: Replace with the actual folder paths
folder_ll = "/home/dst/Desktop/GAN/SRGAN/output_images/SR"  # Path to the LL component folder
# folder_hl = "/home/dst/Desktop/GAN/SRGAN/data/wavlet/HL_val"  # Path to the HL component folder
# folder_lh = "/home/dst/Desktop/GAN/SRGAN/data/wavlet/LH_val"  # Path to the LH component folder
#folder_hh = "/home/dst/Desktop/GAN/SRGAN/data/wavlet/HH_val"  # Path to the HH component folder

print("Checking range for LL component:")
check_dwt_range(folder_ll, num_images=10)

# print("\nChecking range for HL component:")
# check_dwt_range(folder_hl, num_images=10)

# print("\nChecking range for LH component:")
# check_dwt_range(folder_lh, num_images=10)

# print("\nChecking range for HH component:")
# check_dwt_range(folder_hh, num_images=10)

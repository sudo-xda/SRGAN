import os
from PIL import Image

def resize_and_save_images(src_folder, dest_folder, size=(512, 512)):
    """
    Resize all images in src_folder to the given size and save them to dest_folder.
    
    Args:
        src_folder (str): Path to the source folder containing images.
        dest_folder (str): Path to the destination folder where resized images will be saved.
        size (tuple): Target size for resizing (width, height).
    """

    if not os.path.exists(src_folder):
        raise ValueError(f"Source folder '{src_folder}' does not exist.")

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

    count = 0
    for file_name in all_files:
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)

        try:
            with Image.open(src_path) as img:
                # Convert to RGB if grayscale or RGBA
                img = img.convert("RGB")
                img_resized = img.resize(size, Image.BICUBIC)  # Use BICUBIC for smoother results
                img_resized.save(dest_path, "JPEG", quality=95)
                count += 1
        except Exception as e:
            print(f"Skipping {file_name}: {e}")

    print(f"Resized and saved {count} images to '{dest_folder}'.")


# Example usage
src_folder = "/home/dst/Desktop/GAN/SRGAN_old/data/CHEST-XRAY-BIG-val"
dest_folder = "/home/dst/Desktop/GAN/SRGAN_old/data/CHEST-XRAY-BIG-512-val"
resize_and_save_images(src_folder, dest_folder, size=(512, 512))

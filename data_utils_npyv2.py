import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
from os import listdir
from os.path import join
import PIL.Image as Image

# Utility function to check if the file is a .npy file
def is_npy_file(filename):
    return filename.endswith('.npy')

# Function to calculate a valid crop size, ensuring it is divisible by the upscale factor
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


# TrainDatasetFromFolder class for loading and transforming .npy files for training
class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_npy_file(x)]
        self.upscale_factor = upscale_factor  # Store upscale_factor in the class
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = Compose([ToTensor()])
        self.lr_transform = Compose([ToTensor()])

    def __getitem__(self, index):
        hr_image = np.load(self.image_filenames[index])  # Load the high-res image as a NumPy array
        hr_image = hr_image.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Convert high-res image to PIL Image and then to RGB
        hr_image = Image.fromarray((hr_image * 255).astype(np.uint8))  # Convert to PIL Image
        hr_image = hr_image.convert('RGB')  # Ensure it's in RGB format
        
        # Apply ToTensor transformation for high-resolution image
        hr_image = ToTensor()(hr_image)

        # Simulate the low-resolution image by downsampling
        lr_image = hr_image[::self.upscale_factor, ::self.upscale_factor]  # Downsample using the upscale_factor
        
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


# ValDatasetFromFolder class for loading and transforming .npy files for validation
class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_npy_file(x)]

    def __getitem__(self, index):
        hr_image = np.load(self.image_filenames[index])  # Load the high-res image as a NumPy array
        hr_image = hr_image.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Convert high-res image to PIL Image and then to RGB
        hr_image = Image.fromarray((hr_image * 255).astype(np.uint8))  # Convert to PIL Image
        hr_image = hr_image.convert('RGB')  # Ensure it's in RGB format

        # Apply ToTensor transformation for high-resolution image
        hr_image = ToTensor()(hr_image)

        w, h = hr_image.shape[1], hr_image.shape[2]
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)

        # Simulate the low-resolution image by downsampling
        lr_image = hr_image[::self.upscale_factor, ::self.upscale_factor]  # Downsample
        lr_image = ToTensor()(lr_image)  # Apply ToTensor transformation for low-resolution image

        lr_restore_img = lr_image  # Or any other restoration process
        
        return lr_image, lr_restore_img, hr_image

    def __len__(self):
        return len(self.image_filenames)


# TestDatasetFromFolder class for loading and transforming .npy files for testing
class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_npy_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_npy_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]  # Extract image name for matching
        lr_image = np.load(self.lr_filenames[index])  # Load the low-res image
        hr_image = np.load(self.hr_filenames[index])  # Load the high-res image

        lr_image = lr_image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        hr_image = hr_image.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Convert to PIL Image and then to RGB
        lr_image = Image.fromarray((lr_image * 255).astype(np.uint8))  # Convert to PIL Image
        lr_image = lr_image.convert('RGB')  # Ensure it's in RGB format

        hr_image = Image.fromarray((hr_image * 255).astype(np.uint8))  # Convert to PIL Image
        hr_image = hr_image.convert('RGB')  # Ensure it's in RGB format

        # Upscale the low-res image to match the high-res image size
        hr_scale = np.array(hr_image)  # Adjust for upscale factor
        hr_restore_img = hr_scale  # Or any other method to restore images

        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)


# Display transform to prepare images for visualization
def display_transform():
    return Compose([
        ToTensor(),  # Convert to tensor
        Resize(400),  # Resize for display
        CenterCrop(400),  # Crop to a centered 400x400 image
    ])

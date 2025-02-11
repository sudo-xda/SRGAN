import os
import random
import shutil

def create_validation_dataset(hr_folder, val_folder, percentage):

    if not os.path.exists(hr_folder):
        raise ValueError(f"HR folder '{hr_folder}' does not exist.")
    
 
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
    
  
    all_files = [f for f in os.listdir(hr_folder) if os.path.isfile(os.path.join(hr_folder, f))]
    

    num_val_files = int(len(all_files) * (percentage / 100))
    if num_val_files == 0:
        raise ValueError("Percentage too small to move any files.")
    

    val_files = random.sample(all_files, num_val_files)
    
 
    for file_name in val_files:
        src_path = os.path.join(hr_folder, file_name)
        dest_path = os.path.join(val_folder, file_name)
        shutil.move(src_path, dest_path)
    
    print(f"Moved {len(val_files)} files to the validation folder.")


hr_folder = "/home/dst/Desktop/GAN/SRGAN/data/wavlet/LH"
val_folder = "/home/dst/Desktop/GAN/SRGAN/data/wavlet/LH_val"
percentage = 20  

create_validation_dataset(hr_folder, val_folder, percentage)

import os
import shutil
import numpy as np

# Paths to the folder with images and the destination folders
source_folder = "/opt/data/osooli/mydata/train2017/"  # Path to the folder with all images
train_folder = "/opt/data/osooli/mydata/coco/train/"
val_folder = "/opt/data/osooli/mydata/coco/val/"
test_folder = "/opt/data/osooli/mydata/coco/test/"

# Create destination folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get all image files in the source folder
all_images = [f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Shuffle the list of images
np.random.seed(42)  # For reproducibility
np.random.shuffle(all_images)

# Split the data
train_images = all_images[:50000]
val_images = all_images[50000:52000]
test_images = all_images[52000:]

# Function to copy images to the destination folder
def copy_images(image_list, dest_folder):
    for img in image_list:
        shutil.copy(os.path.join(source_folder, img), os.path.join(dest_folder, img))

# Copy the images to their respective folders
copy_images(train_images, train_folder)
copy_images(val_images, val_folder)
copy_images(test_images, test_folder)

# Print results
print(f"Train set: {len(train_images)} images")
print(f"Validation set: {len(val_images)} images")
print(f"Test set: {len(test_images)} images")

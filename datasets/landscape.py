import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class PairedImageDataset(Dataset):
    def __init__(self, gray_dir, color_dir, transform=None):
        self.gray_dir = gray_dir
        self.color_dir = color_dir
        self.transform = transform
        
        # Get list of files in each directory
        self.gray_images = sorted([f for f in os.listdir(gray_dir) if os.path.isfile(os.path.join(gray_dir, f))])
        self.color_images = sorted([f for f in os.listdir(color_dir) if os.path.isfile(os.path.join(color_dir, f))])
        
        # Ensure the number of gray and color images match
        assert len(self.gray_images) == len(self.color_images), "Mismatched number of gray and color images."

    def __len__(self):
        return len(self.gray_images)

    def __getitem__(self, idx):
        gray_img_path = os.path.join(self.gray_dir, self.gray_images[idx])
        color_img_path = os.path.join(self.color_dir, self.color_images[idx])
        
        gray_image = Image.open(gray_img_path).convert("L")   # Convert to grayscale
        color_image = Image.open(color_img_path).convert("RGB") # Convert to RGB

        if self.transform:
            gray_image = self.transform(gray_image)
            color_image = self.transform(color_image)
        gray_image = gray_image.repeat( 3,1,1)

        return color_image,gray_image   # Return the pair (input, target)
# Image transformation (if needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize if necessary
    transforms.ToTensor(),          # Convert to tensor
])
train_dataset = PairedImageDataset(gray_dir='/opt/data/osooli/landscape Images/splits/gray/train', 
                                    color_dir='/opt/data/osooli/landscape Images/splits/color/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataset = PairedImageDataset(gray_dir='/opt/data/osooli/landscape Images/splits/gray/val', 
                                    color_dir='/opt/data/osooli/landscape Images/splits/color/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)


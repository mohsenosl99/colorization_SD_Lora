import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RGBAndGrayDataset(Dataset):
    def __init__(self, root_dir, transform=None, transform_gray =None):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.dataset = os.listdir(root_dir)
        self.transform = transform
        self.transform_gray = transform_gray
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,self.dataset[idx])
        image = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format

        # Convert to grayscale
        # gray_image = image.convert('L')
        if self.transform_gray:

            gray_image = self.transform_gray(image)
            # gray_image = np.repeat(gray_image, 3, 0)


        if self.transform:
            image = self.transform(image)

        return image, gray_image

if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader

    transform = transforms.Compose([
        transforms.Resize(size=(512, 512), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        # transforms.CenterCrop(size=(512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    # Define the transform pipeline for grayscale images
    transform_gray = transforms.Compose([
        transforms.Resize(size=512, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(size=(512, 512)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with 1 channel
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))  # Normalize for 1 channel
    ])
    train_paths = '/opt/data/osooli/mydata/train'

    train_dataset = RGBAndGrayDataset(root_dir=train_paths, transform=transform,transform_gray = transform_gray)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_paths = '/opt/data/osooli/mydata/val'

    val_dataset = RGBAndGrayDataset(root_dir=val_paths, transform=transform,transform_gray = transform_gray)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    for batch in val_loader:
        print(batch[0].shape)
        print(batch[1].shape)
        break
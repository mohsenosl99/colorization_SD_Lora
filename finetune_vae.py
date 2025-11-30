from torchvision import transforms
from datasets.dataset import RGBAndGrayDataset
from torch.utils.data import Dataset, DataLoader
from diffusers import  StableDiffusionInstructPix2PixPipeline,AutoencoderTiny

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn, optim
from diffusers import StableDiffusionImg2ImgPipeline

transform = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
])
# Define the transform pipeline for grayscale images
transform_gray = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with 1 channel
    transforms.ToTensor()])
train_paths = '/opt/data/mydata/train'

train_dataset = RGBAndGrayDataset(root_dir=train_paths, transform=transform,transform_gray = transform_gray)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_paths = '/opt/data/mydata/val'

val_dataset = RGBAndGrayDataset(root_dir=val_paths, transform=transform,transform_gray = transform_gray)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=torch.float16,cache_dir="/opt/data/mydata/cache"
)


device = "cuda"

model_id_or_path = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16,cache_dir="/opt/data/mydata/cache")

rgb_model = pipe.vae
gray_model = pipe.vae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gray_model.encoder.requires_grad_(True)
gray_model.decoder.requires_grad_(False)
rgb_model.encoder.requires_grad_(False)
rgb_model.decoder.requires_grad_(False)

mse_loss = nn.MSELoss()
optimizer = optim.Adam(gray_model.parameters(), lr=3e-5)
gray_model.to(device)
rgb_model.to(device)
best_val_loss = float("inf")
train_losses = []
val_losses = []
epochs = 100

for epoch in range(epochs):
    gray_model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
        
        images, gray = batch
        images = images.to(device).float()
        gray = gray.to(device).float()

        # Get CLIP output (teacher model)
        with torch.no_grad():
            rgb_output = rgb_model.encoder(images)

        # Get Vision Transformer output (student model)
        gray_output = gray_model.encoder(gray)
        # Compute MSE loss
        loss = mse_loss(gray_output, rgb_output)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation loop
    gray_model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}"):
            images, gray = batch
            images = images.to(device)
            gray = gray.to(device)

            # Get CLIP output (teacher model)
            rgb_output = rgb_model.encoder(images)
            # Get Vision Transformer output (student model)
            gray_output = gray_model.encoder(gray)

            # Compute MSE loss
            loss = mse_loss(gray_output, rgb_output)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save the model with the best validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(gray_model.state_dict(), "/opt/data/mydata/checkpoints/tinyautoencoder.pt")
        print(f"Model saved with validation loss: {best_val_loss:.4f}")

# Plotting the losses
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Losses_head4")
plt.savefig('plot/instructpix2pix.jpg')
print("Training Complete!")

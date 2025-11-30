from PIL import Image
import requests
import os

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoProcessor, CLIPVisionModel
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.dataset import RGBAndGrayDataset

from diffusers import StableDiffusionInstructPix2PixPipeline, AutoencoderTiny

import matplotlib.pyplot as plt
from tqdm import tqdm


# --------------------------------------------------
# Transform Definitions
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])

transform_gray = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(size=(224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])


# --------------------------------------------------
# Dataset & Dataloaders
# --------------------------------------------------

train_paths = '/opt/data/mydata/train'
val_paths = '/opt/data/mydata/val'

train_dataset = RGBAndGrayDataset(root_dir=train_paths, transform=transform, transform_gray=transform_gray)
val_dataset = RGBAndGrayDataset(root_dir=val_paths, transform=transform, transform_gray=transform_gray)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)


# --------------------------------------------------
# Model Setup
# --------------------------------------------------

device = 'cuda'

vit_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/opt/data/mydata/cache")
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/opt/data/mydata/cache")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/opt/data/mydata/cache")

vit_model.to(device)
model.to(device)

# Freeze teacher model
for p in model.parameters():
    p.requires_grad = False

mse_loss = nn.MSELoss()
optimizer = optim.Adam(vit_model.parameters(), lr=3e-4)


# --------------------------------------------------
# Training Loop
# --------------------------------------------------

epochs = 100
best_val_loss = float("inf")

train_losses = []
val_losses = []

for epoch in range(epochs):

    # ------------------------------
    # Training
    # ------------------------------
    vit_model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
        images, gray = batch
        images = images.to(device)
        gray = gray.to(device)

        # Teacher output
        with torch.no_grad():
            clip_output = model(images)['last_hidden_state']

        # Student output
        vit_output = vit_model(gray)['last_hidden_state']

        # Loss
        loss = mse_loss(vit_output, clip_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)


    # ------------------------------
    # Validation
    # ------------------------------
    vit_model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}"):
            images, gray = batch
            images = images.to(device)
            gray = gray.to(device)

            clip_output = model(images)['last_hidden_state']
            vit_output = vit_model(gray)['last_hidden_state']

            loss = mse_loss(vit_output, clip_output)

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


    # ------------------------------
    # Save Best Model
    # ------------------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(vit_model.state_dict(), "/opt/data/mydata/checkpoints/clip.pt")
        print(f"Model saved with validation loss: {best_val_loss:.4f}")


# --------------------------------------------------
# Plot Loss Curve
# --------------------------------------------------

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("CLIP")
plt.savefig('plot/Clip.jpg')

print("Training Complete!")

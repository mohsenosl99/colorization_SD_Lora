# ---------------------------
# Imports
# ---------------------------
import torch
import numpy as np
import kornia.color as kc
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionImg2ImgPipeline,
)

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModel,
)

from datasets.dataset import RGBAndGrayDataset
from losses_func import *
from peft import LoraConfig

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# VGG16 Perceptual Model
# ---------------------------
weights_path = "/opt/data/mydata/cache/vgg16-397923af.pth"

vgg_model = vgg16()
state_dict = torch.load(weights_path, map_location=device)
vgg_model.load_state_dict(state_dict)

vgg_model = vgg_model.features.eval().to(device)

return_nodes = {
    "3": "relu1_2",
    "8": "relu2_2",
    "15": "relu3_3",
}

feature_extractor = create_feature_extractor(vgg_model, return_nodes)

for param in feature_extractor.parameters():
    param.requires_grad = False

# ---------------------------
# Transforms
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
])

transform_gray = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# ---------------------------
# Dataset & Loader
# ---------------------------
train_paths = '/opt/data/mydata/coco/train'
val_paths = '/opt/data/Coco/val2017'

train_dataset = RGBAndGrayDataset(
    root_dir=train_paths,
    transform=transform,
    transform_gray=transform_gray
)

val_dataset = RGBAndGrayDataset(
    root_dir=val_paths,
    transform=transform,
    transform_gray=transform_gray
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)

# ---------------------------
# Load Stable Diffusion Models
# ---------------------------
model_id_or_path = "timbrooks/instruct-pix2pix"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path,
    torch_dtype=torch.float32,
    cache_dir="/opt/data/mydata/cache"
)

noise_scheduler = DDPMScheduler.from_pretrained(
    model_id_or_path,
    subfolder="scheduler",
    cache_dir="/opt/data/mydata/cache"
)

tokenizer = CLIPTokenizer.from_pretrained(
    model_id_or_path,
    subfolder="tokenizer",
    cache_dir="/opt/data/mydata/cache"
)

text_encoder = CLIPTextModel.from_pretrained(
    model_id_or_path,
    subfolder="text_encoder",
    cache_dir="/opt/data/mydata/cache"
).to(device)

vae = AutoencoderKL.from_pretrained(
    model_id_or_path,
    subfolder="vae",
    cache_dir="/opt/data/mydata/cache"
).to(device)

image_encoder = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir="/opt/data/mydata/cache"
)

image_encoder.load_state_dict(torch.load("/opt/data/mydata/checkpoints/clip_train2.pt"))
vae.load_state_dict(torch.load("/opt/data/mydata/checkpoints/vae_decoder_LAB3_focal_loss2.pt"))

unet = UNet2DConditionModel.from_pretrained(
    model_id_or_path,
    subfolder="unet",
    cache_dir="/opt/data//mydata/cache"
)

# ---------------------------
# LoRA Setup
# ---------------------------
def load_lora_model(unet, device, diffusion_model_learning_rate):
    for param in unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)

    trainable_params = filter(lambda p: p.requires_grad, unet.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=diffusion_model_learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    return unet, optimizer


model_params = {
    "lora": {"lr": 4e-5, "steps": 50},
    "baseline": {"lr": 1e-5, "steps": 300},
    "SVD": {"lr": 1e-3, "steps": 300},
}

lr = model_params["lora"]["lr"]

optimizer_cls = torch.optim.AdamW
optimizer = optimizer_cls(
    unet.parameters(),
    lr=5e-05,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    eps=1e-08,
)

unet, optimizer = load_lora_model(unet, device, lr)

unet.load_state_dict(
    torch.load('/opt/data/mydata/checkpoints/unet_lora_pix2pix_with_decoder_LAB3_focal_loss2.pt')
)

# ---------------------------
# Trainable Params Table
# ---------------------------
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        params = parameter.numel()
        table.add_row([name, params])
        total_params += params

    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


# ---------------------------
# Text Tokens
# ---------------------------
empty_token = tokenizer(
    ["Colorize the photo"],
    padding="max_length",
    truncation=True,
    return_tensors="pt"
).input_ids[:, :50].to(device)

empty_encoding = text_encoder(empty_token, return_dict=False)[0].to(device)

# ---------------------------
# Criteria
# ---------------------------
mse_loss = nn.MSELoss()
text_encoder.requires_grad_(False)

best_val_loss = float('inf')
train_losses, val_losses = [], []

# ---------------------------
# Training Loop
# ---------------------------
EPOCHS = 20
image_encoder.to(device)

for epoch in tqdm(range(EPOCHS)):

    train_loss = 0.0
    vae.train()
    unet.train()
    image_encoder.train()

    for batch in tqdm(train_loader):
        images, gray = batch
        images, gray = images.to(device).float(), gray.to(device).float()

        lab_images = kc.rgb_to_lab(images)
        L_channel, a_b_channel = lab_images[:, :1], lab_images[:, 1:3]
        L_channel = L_channel.repeat(1, 3, 1, 1)

        # -------------------
        # Encode → Noise → UNet Predict
        # -------------------
        latents = vae.encode(L_channel).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(999, 1000, (latents.size(0),), device=device)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = empty_encoding.repeat(len(L_channel), 1, 1)
        image_embedding = image_encoder(L_channel)["last_hidden_state"]

        concatenated_noisy_latents = torch.cat(
            [noisy_latents, vae.encode(L_channel).latent_dist.sample()],
            dim=1
        )

        model_pred = unet(
            concatenated_noisy_latents,
            timesteps,
            torch.cat([image_embedding, encoder_hidden_states], dim=1),
            return_dict=False
        )[0]

        print(model_pred.shape)
        print(torch.cat([image_embedding, encoder_hidden_states], dim=1).shape)
        print(concatenated_noisy_latents.shape)
        exit()

        # ------------------------------------------------
        # Loss Computation
        # ------------------------------------------------
        loss = mse_loss(model_pred.float(), noise.float())

        alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        beta_prod_t = (1 - alpha_prod_t)

        pred_original_sample = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
        fac = beta_prod_t.sqrt()

        current_estimate = pred_original_sample * fac + latents * (1 - fac)
        current_estimate = vae.decode(current_estimate / vae.config.scaling_factor).sample
        current_estimate = torch.clamp(current_estimate, 0.0, 255.0)

        loss_perceptual = perceptual_loss(current_estimate, images, feature_extractor)

        pred_lab = kc.rgb_to_lab(current_estimate)
        pred_ab = pred_lab[:, 1:3]

        ab_loss = mse_loss(pred_ab, a_b_channel)

        loss = 0.01 * ab_loss + 2 * loss_perceptual

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f}")

    # (Your visualization code preserved exactly)
    pred_lab_image = torch.cat([L_channel[:, :1], pred_ab], dim=1)[0].cpu().numpy()
    lab_min = np.array([-50, -128, -128]).reshape(3, 1, 1)
    lab_max = np.array([100, 127, 127]).reshape(3, 1, 1)
    pred_lab_image = np.clip(pred_lab_image, lab_min, lab_max)

    pred_rgb = kc.lab_to_rgb(torch.tensor(pred_lab_image).unsqueeze(0)).squeeze(0).numpy()
    pred_rgb = np.clip(pred_rgb, 0, 1)

    plt.imshow(pred_rgb.transpose(1, 2, 0))
    plt.savefig(f'output2/train_{epoch}.jpg')

    # -------------------
    # Validation Loop
    # -------------------
    vae.eval()
    unet.eval()
    image_encoder.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            images, gray = batch
            images, gray = images.to(device).float(), gray.to(device).float()

            lab_images = kc.rgb_to_lab(images)
            L_channel, a_b_channel = lab_images[:, :1], lab_images[:, 1:3]
            L_channel = L_channel.repeat(1, 3, 1, 1)

            latents = vae.encode(L_channel).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(999, 1000, (latents.size(0),), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = empty_encoding.repeat(len(L_channel), 1, 1)
            image_embedding = image_encoder(L_channel)["last_hidden_state"]

            concatenated_noisy_latents = torch.cat(
                [noisy_latents, vae.encode(L_channel).latent_dist.mode()],
                dim=1
            )

            model_pred = unet(
                concatenated_noisy_latents,
                timesteps,
                torch.cat([image_embedding, encoder_hidden_states], dim=1),
                return_dict=False
            )[0]

            loss = mse_loss(model_pred.float(), noise.float())

            alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            beta_prod_t = (1 - alpha_prod_t)

            pred_original_sample = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
            fac = beta_prod_t.sqrt()

            current_estimate = pred_original_sample * fac + latents * (1 - fac)
            current_estimate = vae.decode(current_estimate / vae.config.scaling_factor).sample
            current_estimate = torch.clamp(current_estimate, 0.0, 255.0)

            loss_perceptual = perceptual_loss(current_estimate, images, feature_extractor)
            pred_lab = kc.rgb_to_lab(current_estimate)
            pred_ab = pred_lab[:, 1:3]

            ab_loss = mse_loss(pred_ab, a_b_channel)

            loss = 0.01 * ab_loss + 2 * loss_perceptual
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] - Validation Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(unet.state_dict(), "/opt/data/mydata/checkpoints/unet_lora_pix2pix_with_decoder_LAB3_focal_loss22.pt")
        torch.save(vae.state_dict(), "/opt/data/mydata/checkpoints/vae_decoder_LAB3_focal_loss22.pt")
        torch.save(image_encoder.state_dict(), "/opt/data/mydata/checkpoints/clip_train22.pt")
        print(f"Saved Best Model (Val Loss: {best_val_loss:.4f})")
    else:
        torch.save(unet.state_dict(), "/opt/data/mydata/checkpoints/unet_lora_pix2pix_with_decoder_LAB3_last_focal_loss22.pt")
        torch.save(vae.state_dict(), "/opt/data/mydata/checkpoints/vae_decoder_LAB3_focal_loss22_last.pt")
        torch.save(image_encoder.state_dict(), "/opt/data/mydata/checkpoints/clip_last22.pt")

    # Visualization
    pred_lab_image = torch.cat([L_channel[:, :1], pred_ab], dim=1)[0].cpu().numpy()
    pred_lab_image = np.clip(pred_lab_image, lab_min, lab_max)

    pred_rgb = kc.lab_to_rgb(torch.tensor(pred_lab_image).unsqueeze(0)).squeeze(0).numpy()
    pred_rgb = np.clip(pred_rgb, 0, 1)

    plt.imshow(pred_rgb.transpose(1, 2, 0))
    plt.savefig(f'output2/{epoch}.jpg')


# ---------------------------
# Plot Training Curve
# ---------------------------
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.savefig("plot/unet.jpg")

print("Training Complete!")

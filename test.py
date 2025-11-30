# from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
# from diffusers.optimization import get_scheduler
# from diffusers.training_utils import EMAModel
# from diffusers.utils import check_min_version, deprecate, is_wandb_available
# from diffusers.utils.import_utils import is_xformers_available
# from diffusers.utils.torch_utils import is_compiled_module
# from torchvision import transforms
# from datasets.dataset import RGBAndGrayDataset
# from torch.utils.data import Dataset, DataLoader
# from diffusers import  StableDiffusionInstructPix2PixPipeline,StableDiffusionImg2ImgPipeline
# import torch
# import kornia.color as kc

# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from torch import nn, optim
# from diffusers import StableDiffusionImageVariationPipeline
# from transformers import CLIPTextModel, CLIPTokenizer
# from transformers import AutoProcessor, CLIPVisionModel

# from peft import LoraConfig, LoraModel 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_id_or_path = "timbrooks/instruct-pix2pix"
# transform = transforms.Compose([
#     transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
#     transforms.ToTensor()])
# # Define the transform pipeline for grayscale images
# transform_gray = transforms.Compose([
#     transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
#     transforms.Grayscale(num_output_channels=3),  # Convert to grayscale with 1 channel
#     transforms.ToTensor()
# ])
# train_paths = '/opt/data/osooli/mydata/train'
# # train_paths = '/opt/data/osooli/landscape Images/splits/color/train'
# # from diffusers import  StableDiffusionInstructPix2PixPipeline,StableDiffusionImg2ImgPipeline

# # model_id_or_path = "runwayml/stable-diffusion-v1-5"

# # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float32,cache_dir="/opt/data/osooli/mydata/cache")

# train_dataset = RGBAndGrayDataset(root_dir=train_paths, transform=transform,transform_gray = transform_gray)
# train_loader = DataLoader(train_dataset, batch_size = 2, shuffle = True)
# val_paths = '/opt/data/osooli/Coco/val2017'
# val_dataset = RGBAndGrayDataset(root_dir=val_paths, transform=transform,transform_gray = transform_gray)
# val_loader = DataLoader(val_dataset, batch_size = 2, shuffle = True)
# # pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
# #      "timbrooks/instruct-pix2pix", torch_dtype=torch.float32,cache_dir="/opt/data/osooli/mydata/cache")
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float32,cache_dir="/opt/data/osooli/mydata/cache")
# noise_scheduler = DDPMScheduler.from_pretrained(model_id_or_path, subfolder="scheduler",cache_dir="/opt/data/osooli/mydata/cache")

# tokenizer = CLIPTokenizer.from_pretrained(
#     model_id_or_path, subfolder="tokenizer",cache_dir="/opt/data/osooli/mydata/cache"
# )
# text_encoder = CLIPTextModel.from_pretrained(
#     model_id_or_path, subfolder="text_encoder",cache_dir="/opt/data/osooli/mydata/cache"
# )
# text_encoder.to(device)
# vae = AutoencoderKL.from_pretrained(
#     model_id_or_path, subfolder="vae",cache_dir="/opt/data/osooli/mydata/cache"
# )
# image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32",cache_dir="/opt/data/osooli/mydata/cache")
# image_encoder.load_state_dict(torch.load("/opt/data/osooli/mydata/checkpoints/clip_train22.pt"))

# vae.load_state_dict(torch.load("/opt/data/osooli/mydata/checkpoints/vae_decoder_LAB3_focal_loss22.pt"))
# vae.cuda()
# unet = UNet2DConditionModel.from_pretrained(
#     model_id_or_path, subfolder="unet",cache_dir="/opt/data/osooli/mydata/cache"
# )
# def load_lora_model(unet, device, diffusion_model_learning_rate):
 
#     for param in unet.parameters():
#         param.requires_grad_(False)
    
#     unet_lora_config = LoraConfig(
#         r=16,
#         lora_alpha=16,
#         init_lora_weights="gaussian",
#         target_modules=["to_k", "to_q", "to_v", "to_out.0"],
#     )

#     unet.add_adapter(unet_lora_config)
#     lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

#     optimizer = torch.optim.AdamW(
#         lora_layers,
#         lr=diffusion_model_learning_rate,
#     )
#     return unet, optimizer

# model_params = {
#     "lora": {"lr": 4e-5
#     , "steps": 50}, #1e-4
#     "baseline": {"lr": 1e-5, "steps": 300},
#     "SVD": {"lr": 1e-3, "steps": 300}
# }
# lr = model_params['lora']["lr"]

# optimizer_cls = torch.optim.AdamW
# optimizer = optimizer_cls(
#     unet.parameters(),
#     lr=5e-05,
#     betas=(0.9, 0.999),
#     weight_decay=1e-2,
#     eps=1e-08,
# )
# unet, optimizer = load_lora_model(unet, device, lr) 
# unet.load_state_dict(torch.load('/opt/data/osooli/mydata/checkpoints/unet_lora_pix2pix_with_decoder_LAB3_focal_loss22.pt'))
    
# unet.cuda()
# empty_token    = tokenizer(["Colorize the photo"], padding="max_length", truncation=True, return_tensors="pt").input_ids
# empty_token    = empty_token[:,:50].to(device)
# empty_encoding = text_encoder(empty_token, return_dict=False)[0]
# empty_encoding = empty_encoding.to(device)
# unet.cuda()
# mse_loss = nn.MSELoss()
# # vae.requires_grad_(False)
# text_encoder.requires_grad_(False)
# best_val_loss = float('inf')
# train_losses = []
# val_losses = []
# EPOCHS = 100
# # Training loop
# # unet.requires_grad_(False).to(device).eval()
# vae.encoder.requires_grad_(False).to(device).eval()
# image_encoder.requires_grad_(False).to(device).eval()
# # unet, optimizer = load_lora_model(unet, device, lr) 
# import matplotlib.pyplot as plt
# import kornia.color as kc  # For Lab to RGB conversion
# import numpy as np
# import torch


# vae.eval()  # Set the model to evaluation mode
# unet.eval()  # Set the model to evaluation mode
# val_loss = 0.0
# from PIL import  Image
# count = 0
# val_paths = '/opt/data/osooli/Coco/val2017'
# # val_paths = '/opt/data/osooli/mydata/test_clip'
# val_dataset = RGBAndGrayDataset(root_dir=val_paths, transform=transform,transform_gray = transform_gray)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
# # gray,L_channel,pred_ab,images = test_val(val_loader)
# import os
# os.makedirs('/opt/data/osooli/mydata/evaluation/color_new',exist_ok=True)
# os.makedirs('/opt/data/osooli/mydata/evaluation/real_new',exist_ok=True)
# with torch.no_grad():
#     for batch in tqdm((val_loader)):
#         images, gray = batch
#         images, gray = images.to(device).float(), gray.to(device).float()
#         lab_images = kc.rgb_to_lab(images)

#         L_channel,a_b_channel = lab_images[:,0:1,:,:] , lab_images[:,1:3,:,:] 

#         L_channel = L_channel.repeat(1,3,1,1)



#         # Encode grayscale images into latent space
#         latents = vae.encode(L_channel).latent_dist.sample()
#         latents = latents * vae.config.scaling_factor
        
#         # Sample random noise and timesteps
#         noise = torch.randn_like(latents)
#         timesteps = torch.randint(999, 1000, (latents.shape[0],), device=latents.device)
#         # Add noise to the latents
#         noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
#         # Encode the instruction text and expand for batch
#         encoder_hidden_states = empty_encoding.repeat(len(L_channel), 1, 1)
#         image_embedding = image_encoder(L_channel)['last_hidden_state']

#         # Predict the noise using the UNet model
#         concatenated_noisy_latents = torch.cat([noisy_latents, vae.encode(L_channel).latent_dist.mode()], dim=1)
#         model_pred = unet(concatenated_noisy_latents, timesteps, torch.cat([image_embedding,encoder_hidden_states],dim =1),return_dict=False)[0]

#         # Calculate the loss (MSE between predicted and actual noise)
#         loss = mse_loss(model_pred.float(), noise.float())
#         # loss = torch.tensor(0.0, device=device, requires_grad=True)
#         # current_estimate = noisy_latents - 1 * model_pred      # Simplified denoising step
#         # alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps]
#         # beta_prod_t = 1 - alpha_prod_t
#         alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
#         beta_prod_t = (1 - alpha_prod_t)

#         # compute predicted original sample from predicted noise also called
#         # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
#         pred_original_sample = (noisy_latents - beta_prod_t ** (0.5) * model_pred) / alpha_prod_t ** (0.5)

#         fac = torch.sqrt(beta_prod_t)
#         current_estimate = pred_original_sample * (fac) + latents * (1 - fac)

#         current_estimate = vae.decode(current_estimate / vae.config.scaling_factor).sample
#         current_estimate = torch.clamp(current_estimate, min=0.0, max=255.0)


#         pred = current_estimate

#         pred_lab = kc.rgb_to_lab(pred)

#         pred_ab = pred_lab[:, 1:3, :, :] 
        
#         for i in range(4):
#             pred_lab_image = torch.cat([L_channel[:,0:1,:],pred_ab],dim = 1)[i].detach().cpu().numpy()  # Shape: (3, 224, 224)

#             lab_min = np.array([-50, -128, -128]).reshape(3, 1, 1)  # Reshape to match pred_lab_image
#             lab_max = np.array([100, 127, 127]).reshape(3, 1, 1)

#             # Clamp values to the valid Lab range
#             pred_lab_image = np.clip(pred_lab_image, lab_min, lab_max)

#             # Convert Lab to RGB
#             pred_lab_image_rgb = kc.lab_to_rgb(torch.tensor(pred_lab_image).unsqueeze(0)).squeeze(0).numpy()

#             # Ensure RGB values are in the range [0, 1]
#             pred_lab_image_rgb = np.clip(pred_lab_image_rgb, 0, 1) * 255
#             image = Image.fromarray(np.array(pred_lab_image_rgb).astype(np.uint8).transpose(1,2,0))
#             real = images[i].permute(1,2,0).cpu().detach().numpy() * 255
#             real = Image.fromarray((real).astype(np.uint8))
        
#             image.save(f'/opt/data/osooli/mydata/evaluation/color_6/color_{count}.jpg')
#             real.save(f'/opt/data/osooli/mydata/evaluation/real_6/real_{count}.jpg')
#             count+=1
        
            


from torchvision import transforms
from datasets.landscape import train_loader,val_loader
from torch.utils.data import Dataset, DataLoader
from diffusers import  StableDiffusionInstructPix2PixPipeline,AutoencoderTiny,UNet2DConditionModel,StableDiffusionImg2ImgPipeline
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPVisionModel
from datasets.dataset import RGBAndGrayDataset
from peft import LoraConfig, LoraModel 

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn, optim
transform = transforms.Compose([
    transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor()])
# Define the transform pipeline for grayscale images
transform_gray = transforms.Compose([
    transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.Grayscale(num_output_channels=3),  # Convert to grayscale with 1 channel
    transforms.ToTensor()
])
train_paths = '/opt/data/osooli/mydata/train'

train_dataset = RGBAndGrayDataset(root_dir=train_paths, transform=transform,transform_gray = transform_gray)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_paths = '/opt/data/osooli/mydata/test2'

val_dataset = RGBAndGrayDataset(root_dir=val_paths, transform=transform,transform_gray = transform_gray)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=torch.float32,cache_dir="/opt/data/osooli/mydata/cache"
)

vit_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32",cache_dir="/opt/data/osooli/mydata/cache")
device = 'cuda'
gray_model = pipe.vae
gray_model.load_state_dict(torch.load("/opt/data/osooli/mydata/checkpoints/pix2pix_instruct_224.pt"))
gray_model.to(device)
with torch.no_grad():
    for batch in tqdm(val_loader):
        images, gray = batch
        
        images = images.to(device).float()
        gray = gray.to(device).float()

        # latents = gray_model.encode(gray).latent_dist.sample()
        # latents = latents * 0.18215
        # latents = 1 / 0.18215 * latents
        # pred_images = gray_model.decode(latents).sample
        # pred_images = pred_images.clamp(0, 255)
        break
pipe.safety_checker = None
pipe.requires_safety_checker = False
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=torch.float32,cache_dir="/opt/data/osooli/mydata/cache",vae = gray_model
)
pipe.image_encoder = vit_model.load_state_dict(torch.load("/opt/data/osooli/mydata/checkpoints/clip.pt"))


# pipe.vae = gray_model
pipe.to(device)
prompt = "colorize the photo"
image = pipe(prompt=prompt, image=gray[0].cuda(),strength=0.2).images[0]
image_array = gray[0].permute(1, 2, 0).cpu().numpy()

# If the image is in float format (0.0 to 1.0), multiply by 255 to convert to the 0-255 range
gray_img = (image_array - image_array.min()) / (image_array.max() - image_array.min())  # Normalize
plt.imshow(gray_img, cmap='gray')  # Assuming gray images (single channel)
plt.axis('off')
plt.savefig('plot/gray_img.jpg')
plt.show()
plt.imshow(image)  # Assuming gray images (single channel)
plt.axis('off')
plt.savefig('plot/image_array.jpg')

plt.show()
image = pipe(prompt=prompt, image=gray[1].cuda(),strength=0.2).images[0]
image_array = gray[1].permute(1, 2, 0).cpu().numpy()

# If the image is in float format (0.0 to 1.0), multiply by 255 to convert to the 0-255 range
gray_img = (image_array - image_array.min()) / (image_array.max() - image_array.min())  # Normalize
plt.imshow(gray_img, cmap='gray')  # Assuming gray images (single channel)
plt.axis('off')
plt.savefig('plot/gray_img2.jpg')
plt.show()
plt.imshow(image)  # Assuming gray images (single channel)
plt.axis('off')
plt.savefig('plot/image_array2.jpg')

plt.show()

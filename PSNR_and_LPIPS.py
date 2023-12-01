import torch
import lpips
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2 
from math import log10, sqrt 

# Function to load and preprocess an image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

# Function to compute PSNR
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def PSNR(original, compressed): 
	mse = np.mean((original - compressed) ** 2) 
	if(mse == 0): # MSE is zero means no noise is present in the signal . 
				# Therefore PSNR have no importance. 
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse)) 
	return psnr 

# Load images
img1 = load_image('inverted_Prelu_no_regularizer.png')
img2 = load_image('CelebA_ex1.jpg')

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='alex')  # Using AlexNet

# Compute LPIPS
lpips_score = lpips_model(img1, img2)

# Compute PSNR
psnr_score = PSNR(img1, img2)

print(f"LPIPS Score: {lpips_score.item()}")
print(f"PSNR Score: {psnr_score.item()}")

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from invert_functions_noBN import *
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
import time
from invert_functions_noBN import *
from utilities import total_variation_4d
#from models_noBN import ResNet
from PIL import Image
from torch.autograd import Variable
#from invert_functions_noBN import conv_inverted
from Image_Inversion_test import invert
from utilities import extract_resnet_info,extract_number,plot_channels,extract_conv_details,plot_image, extract_prelu_details
from ResNet18_func import ResNet18
from IR_152_PReLU import IR_64,IR152_PReLU
import os
from torch.nn import  MaxPool2d
from celebA_init import CelebADataset
from torchvision.utils import save_image
from sklearn.metrics import accuracy_score
from PIL import Image, ImageEnhance
from IR_152_PReLU_inverted import *

### Might need this
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
torch.manual_seed(0)

# Function to check if the reconstruction file exists and load it
# def load_or_create_and_reconstruct(model, noise_filename, recon_filename):
#     # Load or generate the noise tensor N
#     if os.path.isfile(noise_filename):
#         with open(noise_filename, 'rb') as f:
#             N = pickle.load(f)
#     else:
#         N = torch.randn(1, 3, 224, 224)  # Generate new noise tensor
#         with open(noise_filename, 'wb') as f:
#             pickle.dump(N, f)

#     # Load or reconstruct N_recons
#     if os.path.isfile(recon_filename):
#         with open(recon_filename, 'rb') as f:
#             N_recons = pickle.load(f)
#     else:
#         N_recons = invert_model_layers(model, N, resblocks=True, maxpool=False, fc_layer=False, output_layer=False, first_layer=True)
#         with open(recon_filename, 'wb') as f:
#             pickle.dump(N_recons, f)

#     return N, N_recons



# Function to check if the reconstruction file exists and load it
def load_or_reconstruct(model, noise, filename):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            N_recons = pickle.load(f)
    else:
        N_recons = invert_model_layers(model, noise, resblocks=True, maxpool=False, fc_layer=False, output_layer=False, first_layer=True)
        with open(filename, 'wb') as f:
            pickle.dump(N_recons, f)
    return N_recons

# Define the convolution operation W_hat
class ConvolutionalOperation(nn.Module):
    def __init__(self, weights_path=None):
        super(ConvolutionalOperation, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1)
        if weights_path and os.path.isfile(weights_path):
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        return self.conv(x)

    def save_weights(self, weights_path):
        torch.save(self.state_dict(), weights_path)

# Make a model and load pretrained weights
model = IR152_PReLU()
#model.load_pretrained_weights(checkpoint_path='best_model_checkpoint_Prelu_152_IR_224.pth')

# Assuming N is the noise tensor of shape (10, 224, 224, 3)
N_recons_list = []
N_list = []







for i in range(10):  # Loop over 10 tensors
    # Initialize N as a noise tensor for a single image of shape [1, 3, 224, 224]
    N = torch.randn(1, 3, 224, 224)
    # Filename for storing each reconstruction
    reconstructions_filename = f'N_recons_{i}.pkl'
    # Load or reconstruct the noise tensor N
    N_recons = load_or_reconstruct(model, N, reconstructions_filename)
    N_recons_list.append(N_recons)
    N_list.append(N)
    

# Store the reconstructions
#reconstructions_filename = 'N_recons.pkl'
#_recons = load_or_reconstruct(model, N, reconstructions_filename)

N = torch.cat(N_list, dim=0)
N_recons = torch.cat(N_recons_list, dim=0)

#import pdb;pdb.set_trace()

# Instantiate the ConvolutionalOperation with an optional path to pre-trained weights
weights_path = 'W_hat_weights.pth'
W_hat = ConvolutionalOperation(weights_path=weights_path)

# Optimizer and loss function
optimizer = optim.Adam(W_hat.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    N_hat = W_hat(N_recons)
    loss = loss_function(N_hat, N)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:  # Print the loss every 100 epochs
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Save weights after training
    W_hat.save_weights(weights_path)







################## Now adjusting the image ###############################

def adjust_image (orig_img = 'orig_3.png', recons_img = 'recons_3.png',W_hat = None):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor()           # Convert the image to a PyTorch tensor
    ])
    
    # Read the image
    image = Image.open(orig_img).convert('RGB')
    image_tensor = transform(image)
    orignal_img = image_tensor.unsqueeze(0)
    
    ####
    
    image = Image.open(recons_img).convert('RGB')
    image_tensor = transform(image)
    recons_img = image_tensor.unsqueeze(0)
    
    
    #import pdb;pdb.set_trace()
    
    corrected_image = W_hat(recons_img)
    
    
    plot_image(orignal_img)
    plot_image(recons_img)
    plot_image(corrected_image)





adjust_image (orig_img = 'orig_3.png', recons_img = 'recons_3.png',W_hat = W_hat)



## invert_model_layers(model, N, resblocks=True, maxpool=False, fc_layer=False, output_layer=False, first_layer=True)

### Store all these Ns and N_recons as tensors.

## 

## Store them as pickels so that I dont have to learn them again and run the resconstruction only if not present

### Then learn the convolution W

### N = W_hat(N_recons) here W_hat is a convolution operation 

### for  1000 epochs  run the N = W_hat(N_recons) as loss = MSE_loss(N,N_recons)
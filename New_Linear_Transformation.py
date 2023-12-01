import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_for_inversion import FaceNet64,IR152
import os
from torch.nn import  MaxPool2d
from celebA_init import CelebADataset
from torchvision.utils import save_image
from sklearn.metrics import accuracy_score
from PIL import Image, ImageEnhance
from IR_152_PReLU_inverted import *
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
torch.manual_seed(0)

# Function to check if the reconstruction file exists and load it
def load_or_create_and_reconstruct(model, noise_filename, recon_filename):
    # Load or generate the noise tensor N
    if os.path.isfile(noise_filename):
        with open(noise_filename, 'rb') as f:
            N = pickle.load(f)
    else:
        N = torch.randn(1, 3, 224, 224)  # Generate new noise tensor
        with open(noise_filename, 'wb') as f:
            pickle.dump(N, f)

    # Load or reconstruct N_recons
    if os.path.isfile(recon_filename):
        with open(recon_filename, 'rb') as f:
            N_recons = pickle.load(f)
    else:
        N_recons = invert_model_layers(model, N, resblocks=True, maxpool=False, fc_layer=False, output_layer=False, first_layer=True)
        with open(recon_filename, 'wb') as f:
            pickle.dump(N_recons, f)

    return N, N_recons


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
model = IR152()
checkpoint = torch.load('best_model_checkpoint_Prelu_152_IR_224_new.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

#model.load_pretrained_weights(checkpoint_path='best_model_checkpoint_Prelu_152_IR_224.pth')

N_list = []
N_recons_list = []
for i in range(10):
    noise_filename = f'N_original_new{i}.pkl'
    reconstructions_filename = f'N_recons_new{i}.pkl'
    N, N_recons = load_or_create_and_reconstruct(model, noise_filename, reconstructions_filename)
    N_list.append(N)
    N_recons_list.append(N_recons)

N = torch.cat(N_list, dim=0)
N_recons = torch.cat(N_recons_list, dim=0)
# Store the reconstructions
#reconstructions_filename = 'N_recons.pkl'
#_recons = load_or_reconstruct(model, N, reconstructions_filename)


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








# def adjust_image (orig_img = 'orig_3.png', recons_img = 'recons_3.png',W_hat = None):
    
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize the image to 224x224
#         transforms.ToTensor()           # Convert the image to a PyTorch tensor
#     ])
    
#     # Read the image
#     image = Image.open(orig_img).convert('RGB')
#     image_tensor = transform(image)
#     orignal_img = image_tensor.unsqueeze(0)
    
#     ####
    
#     image = Image.open(recons_img).convert('RGB')
#     image_tensor = transform(image)
#     recons_img = image_tensor.unsqueeze(0)
    
    
#     #import pdb;pdb.set_trace()
    
#     corrected_image = W_hat(recons_img)
    
    
#     plot_image(orignal_img)
#     plot_image(recons_img)
#     plot_image(corrected_image)


## adjust_image (orig_img = 'orig_3.png', recons_img = 'recons_3.png',W_hat = W_hat)


################## Processing the whole folder #########################


def adjust_image(recons_img_path, W_hat):
    transform = transforms.Compose([
        transforms.ToTensor()           # Convert the image to a PyTorch tensor
    ])

    # Read and transform the image
    image = Image.open(recons_img_path).convert('RGB')
    image_tensor = transform(image)
    recons_img_tensor = image_tensor.unsqueeze(0)

    # Correct the image using W_hat
    corrected_image_tensor = W_hat(recons_img_tensor)

    # Convert tensor back to PIL Image for saving
    corrected_image = transforms.ToPILImage()(corrected_image_tensor.squeeze())

    return corrected_image

def process_and_save_images(input_folder, output_folder, W_hat):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):  # or any other file extension you expect
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            corrected_image = adjust_image(input_path, W_hat)
            corrected_image.save(output_path)

# Example usage
input_folder = 'recovered_images_brighter'
output_folder = 'calibrated_images_celebA_IR152'
process_and_save_images(input_folder, output_folder, W_hat)






## invert_model_layers(model, N, resblocks=True, maxpool=False, fc_layer=False, output_layer=False, first_layer=True)

### Store all these Ns and N_recons as tensors.

## 

## Store them as pickels so that I dont have to learn them again and run the resconstruction only if not present

### Then learn the convolution W

### N = W_hat(N_recons) here W_hat is a convolution operation 

### for  1000 epochs  run the N = W_hat(N_recons) as loss = MSE_loss(N,N_recons)
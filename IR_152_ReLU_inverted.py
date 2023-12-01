import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#from pygranso.pygranso import pygranso
#from pygranso.pygransoStruct import pygransoStruct
import time
from invert_functions_noBN import *
from utilities import total_variation_4d
#from models_noBN import ResNet
from PIL import Image
from torch.autograd import Variable
#from invert_functions_noBN import conv_inverted
from Image_Inversion_test import invert
from utilities import extract_resnet_info,extract_number,plot_channels,extract_conv_details,plot_image
from ResNet18_func import ResNet18
from models_for_inversion import IR_64,IR152#,EvolveFace
import os
from torch.nn import  MaxPool2d
from Orignal_IR_152 import IR152_orignal


#function to decay the learning rate

def increase_rate(p):
    p*=1.1
    return p




def decay_lr(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor

class FP(nn.Module):
    def __init__(self, params:dict):
        super(FP, self).__init__()
        # Define a 2D convolutional layer and initialize its weights with the pretrained weights
        self.conv_layer = nn.Conv2d(in_channels=params['in_channels'], out_channels= params['out_channels'], kernel_size= params['kernel_size'],
                                    stride=params['stride'], padding=params['padding'], bias= params['bias'])
        if params['weight'] is not None:
            self.conv_layer.weight = nn.Parameter(params['weight'])

    def forward(self, x):
        conv_result = self.conv_layer(x)
        
        return conv_result
    
# class MaxPool(nn.Module):
#     def __init__(self, kernel_size=3, stride=2, padding=1):
#         super(MaxPool, self).__init__()
#         # Define a 2D max-pooling layer with specified kernel size, stride, and padding
#         self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

#     def forward(self, x):
#         maxpool = self.maxpool(x)
#         return maxpool
        
#        return maxpool

def Maxpool(maxpool_details:dict):
    
    maxpool_layer = nn.MaxPool2d(
    kernel_size=maxpool_details['kernel_size'],
    stride=maxpool_details['stride'],
    padding=maxpool_details['padding'],
    dilation=maxpool_details['dilation'],
    return_indices=maxpool_details['return_indices'],
    ceil_mode=maxpool_details['ceil_mode']
    
)
    return maxpool_layer

def conv_inverted(conv_out,conv_in,conv1_params:dict,maxpool=True):
    
    cuda = None
    
    # Step 2: Initialize variables
    x = Variable((1e-3 * torch.randn(*conv_in.size()).cuda() if cuda else 
        1e-3 * torch.randn(*conv_in.size())), requires_grad=True)
    
    
    # Step 3: Choose an optimizer
    optimizer = torch.optim.Adam([x], lr=0.01)
    device = 'cpu'
    W1 = FP(conv1_params).to(device)


    
    # Training loop
    num_epochs = 1000
   # mu = 1e3  # Penalty term for the inequality constraint
    
    alpha = 6
    beta = 2
    lamda_alpha = 1e-5
    lamda_beta = 1e-5
    y = conv_out
    
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        alpha_norm  = alpha_prior(x, alpha=alpha)
        tv_x = tv_norm(x, beta=beta)
        
        # Define the loss
       
        
        
        #loss = (y - F.relu(W1(x)) ) .norm()**2 +  lamda_beta * tv_x + lamda_alpha * alpha_norm
        fp_pass = F.relu(W1(x))
        
        if(maxpool):
            Maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            fp_pass = Maxpool(fp_pass)
        
        loss = norm_loss(fp_pass,y) +  lamda_beta * tv_x + lamda_alpha * alpha_norm
        # Penalty for the inequality constraint

        #constraint_ineq = W1(x) - F.relu(W1(x)) 
        #loss += mu * constraint_ineq.norm()**2
        
        # Compute gradients
        loss.backward()
        
        # Update variables
        optimizer.step()
        
        # Print every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    print(f'Solution x: {x.detach().numpy()}')
    
    return x

def basic_block_inverted(res_block_out,res_block_in,conv1_params:dict,conv2_params:dict,conv3_params:dict,shortcut ='MaxPool'):
    
    y = res_block_out
    
    cuda = False
    device = 'cpu'
    
    ### Creating the convolution layers 
    W1 = FP(conv1_params).to(device)
    W2 = FP(conv2_params).to(device)
    
    
    
    # Step 2: Initialize variables
    x = Variable((1e-3 * torch.randn(*res_block_in.size()).cuda() if cuda else 
        1e-3 * torch.randn(*res_block_in.size())), requires_grad=True)
    
    p_sample = W1(x) ### Just used to get the shape of p and n
    
    p  = Variable((1e-3 * torch.randn(*p_sample.size()).cuda() if cuda else 
        1e-3 * torch.randn(*p_sample.size())), requires_grad=True)
    
    n = Variable((1e-3 * torch.randn(*p_sample.size()).cuda() if cuda else 
       1e-3 * torch.randn(*p_sample.size())), requires_grad=True)
    
    # Step 3: Choose an optimizer
   # decay_factor = 1e-3
    optimizer = torch.optim.Adam([x, p, n], lr=0.01)
    
    
   
    
    
    #############
    if(shortcut == 'Conv'):
        #import pdb;pdb.set_trace()
        W3 = FP(conv3_params)  ### 
        
    elif(shortcut == 'Maxpool'):
          W3 = Maxpool(conv3_params)
    else:
       pass
    
    
    # elif(shortcut == 'Maxpool'):
    #     W3 = Maxpool(conv3_params)
    
    
    # Training loop
    num_epochs = 1000
    
    # Reality __-> should change
    lambda_1 = 1e2  # Regularization term for the first constraint
    lambda_2 = 1e2  # Regularization term for the second constraint
    
    mu_1 = 1e2
    mu_2 = 1e2
    
    #decay_iter = 100
    
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        
        # Step 4: Define the loss
        #loss = ( y - W3(x) - W2(p)).norm()**2
        
        if(shortcut):
            #simport pdb;pdb.set_trace()
            loss = ( y - W3(x) - W2(p)).norm()**2
        else:
            #import pdb;pdb.set_trace()
            loss = (y - x - W2(p)).norm()**2
        #loss = x.norm()**2  # This is ||x||^2
        ## Check solutions where constraint are met and store that solution
        
        
        # Add the constraint terms to the loss
        constraint_1 = lambda_1 * ( W1(x) - p + n).norm()**2
        loss += constraint_1

        constraint_2 = lambda_2 * (torch.matmul(n.view(n.size(0), -1), p.view(p.size(0), -1).T).squeeze())**2
        loss += constraint_2
        
        
        
        
        
        # Add barrier term for n > 0
        #barrier_n = -mu * torch.sum(torch.log(n))
        #loss += barrier_n

        # Add barrier term for p > 0
        #barrier_p = -mu * torch.sum(torch.log(p))
        #loss += barrier_p
        
        # Compute gradients
        loss.backward()
        
        # Update variables
        optimizer.step()
        
        
        with torch.no_grad(): # make sure no gradient is computed during projection
            p.data = torch.clamp(p.data, min=0)
            n.data = torch.clamp(n.data, min=0)
        #if (epoch+1) % decay_iter == 0:
        #    decay_lr(optimizer, decay_factor)
        
        # Print every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    
    return x.detach()





def resblock_inverted(resnet18_info,resnet):
    #inverted_outputs = {}

    # Process layers in reverse order
    for layer_name in reversed(list(resnet18_info.keys())):
        layer_info = resnet18_info[layer_name]
        #layer_output = {}

        print(f"Processing {layer_name}...")  # Print the layer name
        res_block_out = resnet.activations[layer_name][1]['output']
        # Process blocks in reverse order within each layer
        
        
        for block_name in reversed(list(layer_info.keys())):
            block_info = layer_info[block_name]

            print(f"  Inverting {block_name}...")  # Print the block name
            
            conv1_params = block_info['conv1']
            conv2_params = block_info['conv2']
            
            #print("conv1", conv1_params.keys())
            #print("conv2", conv2_params.keys())
            #import pdb;pdb.set_trace()
            # Check if downsample details are present
            downsample_params = block_info.get('downsample', None)
            
           
            
            if  (downsample_params):
              if 'weight' in  downsample_params:  # This is Conv2d
                  shortcut = 'Conv'
                  #res_block_out = original_block_out
              else:         # This is MaxPool2d
                  shortcut = 'Maxpool'
            else:
                shortcut = 0
                
                
            print(downsample_params)
            
            #   print("downsample", downsample_params.keys())
            # For demonstration purposes, using placeholders for block output and input
            # Replace with your method to obtain these tensors
           # import pdb;pdb.set_trace()
            
            res_block_in = resnet.activations[layer_name][extract_number(block_name) -  1 ]['input']
            orignal_block_out = resnet.activations[layer_name][extract_number(block_name) -  1 ]['output']
            
            print(res_block_out.shape)
            print(res_block_in.shape)
            print(orignal_block_out.shape)

            inverted_input = basic_block_inverted(res_block_out, res_block_in, conv1_params, conv2_params, 
                                                   downsample_params,shortcut)
            plot_channels(inverted_input,rec = True)
            plot_channels(orignal_block_out,rec = False)
            
            
            res_block_out = inverted_input
            #layer_output[block_name] = inverted_output
        
        #inverted_outputs[layer_name] = layer_output
    orignal_input  = res_block_in
    return inverted_input,orignal_input #None #inverted_outputs





def main():
    # Load the pretrained ResNet-18 model
    #resnet= ResNet18(pretrained=True)
    
    ir_model = IR_64() #EvolveFace() #IR152()#IR_64()
    
    ################### Forward Pass #############3
    
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Specify the filename of the image
    image_filename = 'CelebA_ex1.jpg'

    # Construct the full path to the image
    image_path = os.path.join(current_directory, image_filename)
    

    image = Image.open(image_path).convert('RGB')


    # Assuming you've resized the image using PIL
    resized_image = image.resize((224, 224)) # Using 56 by 56 images

    # Convert to tensor
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(resized_image)


    input_image = image_tensor.unsqueeze(0)

    # # Ensure the model is in evaluation mode
    ir_model.eval()

    ## Perform a forward pass
    with torch.no_grad():
        outputs  =  ir_model(input_image)

    # # #  Here the Input to the 4 resblocks is recovered:
        
    resblocks = True
    first_layer = True
    maxpool = False
    lastlayer = False
    
    
    if(lastlayer):
        pass
    
    
    
        
    if(resblocks):
        resnet_info = extract_resnet_info(ir_model)
        #import pdb;pdb.set_trace()
        inverted_input,orignal_input = resblock_inverted(resnet_info,ir_model)
        print(torch.norm(inverted_input - orignal_input))
    
    if(first_layer):
        
        if(resblocks):
            first_layer = inverted_input
        else:
            first_layer = ir_model.activations['after_conv1'] 
        
        #conv_weight = 
        #import pdb;pdb.set_trace()
        conv1_params = extract_conv_details(ir_model.conv1) 
        
        #print(conv1_params)
        input_recovered = conv_inverted(first_layer,input_image,conv1_params,maxpool = maxpool)
        
        #plot_image(input_recovered)
        plot_image(input_recovered,save_path ='inverted_relu_stride=2.png')
        
        plot_image(input_image,save_path ='orignal.png')
    plot_channels(inverted_input,rec = True)
    plot_channels(orignal_input,rec = False)
    
    
    

        

if __name__ == '__main__':
    main()


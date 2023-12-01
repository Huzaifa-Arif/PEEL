#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 15:41:07 2023

@author: huzaifaarif
"""

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
#from IR_152_PReLU import IR_64,IR152_PReLU
import os
from torch.nn import  MaxPool2d
from celebA_init import CelebADataset
from torchvision.utils import save_image
from sklearn.metrics import accuracy_score
from PIL import Image, ImageEnhance
from models_for_inversion import IR152,FaceNet64
#function to decay the learning rate
torch.manual_seed(0)


def linear_inverted(linear_in,linear_out,model,layer = 'fc'):
    
   
    
    cuda = None
    
    # Step 2: Initialize variables
    x = Variable((1e-3 * torch.randn(*linear_in.size()).cuda() if cuda else 
        1e-3 * torch.randn(*linear_in.size())), requires_grad=True)
    
    
    # Step 3: Choose an optimizer
    optimizer = torch.optim.Adam([x], lr=0.01)
    device = 'cpu'
    
    
    if layer == 'fc':
        # Initialize a Linear layer with the model's fc_layer weights and biases
        
         
        W1 = model.fc_layer.weight.data.clone()
        bias = model.fc_layer.bias.data.clone()
        fc_recovered= solve_optimization_problem_final_layer(W1_weights=W1,bias = bias, y=linear_out, input=linear_in)
        recovery =  fc_recovered
        #import pdb;pdb.set_trace()
        
        # W1 = nn.Linear(model.feat_dim, model.num_classes)
        # W1.weight.data = model.fc_layer.weight.clone()
        # W1.bias.data = model.fc_layer.bias.clone()
        
        #import pdb;pdb.set_trace()
    
    elif layer == 'output':
        # Initialize a Sequential model including a Flatten operation and a Linear layer
        #import pdb;pdb.set_trace()
        # stride = 2
        # f = stride * 14
        # w1 = nn.Linear(model.feat_dim * f * f, model.feat_dim)
        # w1.weight.data = model.output_layer[1].weight.clone()
        # w1.bias.data = model.output_layer[1].bias.clone()
        
        W1 =  model.output_layer[1].weight.clone()
        bias = model.output_layer[1].bias.clone()
        #import pdb;pdb.set_trace()
        x = linear_in
        linear_in = linear_in.view(linear_in.size(0), -1)
        fc_recovered= solve_optimization_problem_final_layer(W1_weights=W1,bias = bias, y=linear_out, input=linear_in)
        res_input_recovered = fc_recovered.view(*x.shape)
        recovery = res_input_recovered
       # print(torch.norm(res_input_recovered - x))
        
       # W1 = nn.Sequential(nn.Flatten(), w1)
        #W1 = w1
     



    
    #  #Training loop
    # num_epochs = 2000
    # mu = 1.0  # Initial penalty term for the inequality constraint
    # mu_increase_factor = 1.01  # Factor by which to increase mu each epoch or periodically
    
    # y = linear_out
    
    # for epoch in range(num_epochs):
    #     # Zero gradients
    #     optimizer.zero_grad()
    
    #     # Forward pass depending on the layer type
    #     if layer == 'output':
    #         flat = nn.Flatten()
    #         fp_pass = W1(flat(x))
    #     else:
    #         fp_pass = W1(x)
    
    #     # Calculate the norm of x
    #     norm_x = x.norm()**2
        
    #     # Calculate the discrepancy term
    #     discrepancy = (y - fp_pass).norm()**2
        
    #     # Define the initial loss
    #     loss = norm_x + mu * discrepancy
    
    #     # Compute gradients
    #     loss.backward()
    
    #     # Update variables
    #     optimizer.step()
    
    #     # Update the penalty term mu, making it adaptive
    #     mu *= mu_increase_factor
    
    #     # Print every 100 epochs
    #     if epoch % 100 == 0:
    #         print(f'Epoch {epoch}, Loss: {loss.item()}, Penalty: {mu}')
    
    # print(f'Solution x: {x.detach().numpy()}')
     
    return x




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

def conv_inverted(conv_out,conv_in,conv1_params:dict,prelu1_params:dict,maxpool=True):
    
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
    
    prelu_layer = nn.PReLU()
    #prelu_layer.init = prelu1_params['init']
    prelu_layer.num_parameters = prelu1_params['num_parameters']
    prelu_layer.weight  = prelu1_params['weight']
    
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        #alpha_norm  = alpha_prior(x, alpha=alpha)
        #tv_x = tv_norm(x, beta=beta)
        
        # Define the loss
       
        
        #prelu_layer = nn.PReLU()
        #loss = (y - F.relu(W1(x)) ) .norm()**2 +  lamda_beta * tv_x + lamda_alpha * alpha_norm
        #fp_pass = F.nn.relu(W1(x))
        fp_pass =  prelu_layer(W1(x))
        
        if(maxpool):
            Maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            fp_pass = Maxpool(fp_pass)
        
        #loss = norm_loss(fp_pass,y) +  lamda_beta * tv_x + lamda_alpha * alpha_norm
        loss = torch.norm(prelu_layer(W1(x) - y)) **2
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

def basic_block_inverted(res_block_out,res_block_in,conv1_params:dict,prelu1_params:dict,conv2_params:dict,conv3_params:dict,shortcut ='MaxPool',layer_name = 'layer1',epochs = 2000):
    
    y = res_block_out
    
    cuda = False
    device = 'cpu'
    
    ### Creating the convolution layers 
    W1 = FP(conv1_params).to(device)
    W2 = FP(conv2_params).to(device)
    
    
    prelu_layer = nn.PReLU()
    #prelu_layer.init = prelu1_params['init']
    prelu_layer.num_parameters = prelu1_params['num_parameters']
    prelu_layer.weight  = prelu1_params['weight']  
    

    #prelu_layer.weight.data = p_relu['weight']
    
    # Step 2: Initialize variables
    x = Variable((1e-3 * torch.randn(*res_block_in.size()).cuda() if cuda else 
        1e-3 * torch.randn(*res_block_in.size())), requires_grad=True)
    
    optimizer = torch.optim.Adam([x], lr=0.01)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Adjust the step_size and gamma as needed
   
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    

    #############
    if(shortcut == 'Conv'):
        #import pdb;pdb.set_trace()
        print("Conv Called")
        W3 = FP(conv3_params)  ###
        
        #import pdb;pdb.set_trace()
    elif(shortcut == 'Maxpool'):
        print("MaxPool called....")
        W3 = Maxpool(conv3_params)
    
    
    # Training loop
    
    #block_num = extract_number(block_name)
    
    # if(block_num <= 12):
    #     num_epochs = 2000
    # else:
    #     num_epochs = 1000
    
    num_epochs = epochs
        
    #print(num_epochs)
    
    # Reality __-> should change
    lambda_1 = 1e2  # Regularization term for the first cozznstraint
    lambda_2 = 1e2  # Regularization term for the second constraint
    
    mu_1 = 1e2
    mu_2 = 1e2
    
    # Decay iteration
    # decay_iter = 100
    
    # alpha = 6
    # beta = 2
    # lamda_alpha = 1e-5
    # lamda_beta = 1e-5
    
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
    
        # Step 4: Define the loss
        # alpha_norm  = alpha_prior(x, alpha=alpha)
        # tv_x = tv_norm(x, beta=beta)
        
        # if(layer_name == 'layer1'):
        #    # print("CorrectLayer1")
        #     if shortcut:
        #         fp_pass =  W3(x) - W2(prelu_layer(W1(x)))
        #     else:
        #         fp_pass = x - W2(prelu_layer(W1(x)))
            
        #     loss = norm_loss(fp_pass,y) +  lamda_beta * tv_x + lamda_alpha * alpha_norm
        # else:
            #print("CorrectLayerotherthan1")
        if shortcut:
            loss = (y - W3(x) - W2(prelu_layer(W1(x)))).norm()**2
        else:
            loss = (y - x - W2(prelu_layer(W1(x)))).norm()**2
    
        # Compute gradients
        loss.backward()
    
        # Update variables
        optimizer.step()
        
        #scheduler.step()
    
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
        # Check if the loss is less than 1e-5, and if so, break out of the loop
        #if  loss.item() < 1e-5:
        #    break
        
        # Adjust learning rate based on the loss
        #scheduler.step(loss.item())

    
    return x.detach()





def resblock_inverted(resnet18_info,resnet,res_block_out = None):
    #inverted_outputs = {}

    # Process layers in reverse order
    for layer_name in reversed(list(resnet18_info.keys())):
        layer_info = resnet18_info[layer_name]
        #layer_output = {}
        
        if(layer_name == 'layer1'):
            epochs  = 5000
        else:
            epochs  = 2000
        print(f"Processing {layer_name}...")  # Print the layer name
        res_block_out = resnet.activations[layer_name][1]['output']
        # Process blocks in reverse order within each layer
        
        if(1):#layer_name == 'layer1'
            for block_name in reversed(list(layer_info.keys())):
                block_info = layer_info[block_name]
    
                print(f"  Inverting {block_name}...")  # Print the block name
                
                conv1_params = block_info['conv1']
                prelu1_params = block_info['prelu1']  
                conv2_params = block_info['conv2']
               # prelu2_params = block_info['prelu2']
                
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
                    
                    
               # print(downsample_params)
                
                #   print("downsample", downsample_params.keys())
                # For demonstration purposes, using placeholders for block output and input
                # Replace with your method to obtain these tensors
               # import pdb;pdb.set_trace()
                
                res_block_in = resnet.activations[layer_name][extract_number(block_name) -  1 ]['input']
                orignal_block_out = resnet.activations[layer_name][extract_number(block_name) -  1 ]['output']
                
                #print(res_block_out.shape)
                #print(res_block_in.shape)
                #print(orignal_block_out.shape)
                
                
                #res_block_out = orignal_block_out
              
                inverted_input = basic_block_inverted(res_block_out, res_block_in, conv1_params, prelu1_params,
                                                  conv2_params,downsample_params, shortcut, layer_name,epochs)
               # plot_channels(inverted_input,rec = True)
               # plot_channels(orignal_block_out,rec = False)
                
                
                res_block_out = inverted_input
                #layer_output[block_name] = inverted_output
            
                 #inverted_outputs[layer_name] = layer_output
                orignal_input  = res_block_in
    return inverted_input,orignal_input #None #inverted_outputs





def invert_model_layers(model, input_image, resblocks=True, maxpool=False, fc_layer=False, output_layer=False, first_layer=True):
    
    ### We invert the backbone of the face recognition model here
    ir_model = model.feature
    
    # Ensure the model is in evaluation mode
    ir_model.eval()

    # Perform a forward pass
    with torch.no_grad():
        outputs = ir_model(input_image)
    
    
    
    
    ### NOTE:    In this codebase the backbone inversion is seperated into residual blocks and initial convolution operation
    
    
    ############################ BACKBONE INVERTED ######################################
    
    # Inversion for residual blocks
    if resblocks:
        resnet_info = extract_resnet_info(ir_model)
        inverted_input, original_input = resblock_inverted(resnet_info, ir_model)
        print(torch.norm(inverted_input - original_input))
    
    # Inversion for first layer
    if first_layer:
        if resblocks:
            first_layer_output = inverted_input
        else:
            first_layer_output = ir_model.activations['after_conv1']
        
        conv1_params = extract_conv_details(ir_model.conv1)
        prelu_params =  extract_prelu_details(ir_model.prelu1)
        input_recovered = conv_inverted(first_layer_output, input_image, conv1_params,prelu_params, maxpool=maxpool)
        

    return input_recovered





def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_images_from_folder(folder):
    images = []
    img_to_tensor = transforms.ToTensor()
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        img = img_to_tensor(img)
       # import pdb;pdb.set_trace()
        images.append(img.unsqueeze(0))  # Add batch dimension
    return images




def main():


    
    ### Pretrained_model and its details
    checkpoint = torch.load('best_model_checkpoint_Prelu_152_IR_224_new.pth',map_location=torch.device('cpu') )
    train_loader = checkpoint['trainloader']
    
    
    # Create a new DataLoader with the same dataset but with a batch size of 1
    celebAloader = torch.utils.data.DataLoader(dataset=train_loader.dataset , batch_size=5, 
                                                   num_workers=train_loader.num_workers, 
                                                   collate_fn=train_loader.collate_fn, 
                                                   pin_memory=train_loader.pin_memory, 
                                                   drop_last=train_loader.drop_last)
    

    recovered_images_dir = 'recovered_images_celebA_IR152_stride_1' # 'recovered_images_celebA_FaceNet64' # 
    input_images_dir = 'input_images_celebA_IR152_stride_1' #input_images_celebA_FaceNet64' #
    
    ensure_dir(recovered_images_dir)
    ensure_dir(input_images_dir)
    
    
    model =  IR152() #FaceNet64()
   # model.load_state_dict(checkpoint['model_state_dict'])
    all_labels = []
    for i, (image,labels) in enumerate(celebAloader):

       if(i>=4):
           rec_img = invert_model_layers(model, image, resblocks=True, maxpool=False, fc_layer=False, output_layer=False, first_layer=True)
           batch_size = 1
           for b in range(batch_size):
               save_image(image[b], os.path.join(input_images_dir, f'input_image_{i}{b}_{labels[b]}.png'))
               plot_image(image[b])
               #print('labels',labels[b])
               #plot
               save_image(rec_img[b], os.path.join(recovered_images_dir, f'recovered_image_{i}{b}_{labels[b]}.png'))
       
        
        #all_labels.append(labels)
        #print(all_labels)
       # plot_image(image)
        # Later extend to more labels
       if(i==99): 
           break
    #print("Unique Values:",len(all_labels))      
    #print("Unique Values:",len(list(set(all_labels))))     
            

if __name__ == '__main__':
    main()







































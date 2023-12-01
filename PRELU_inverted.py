
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
from IR_152_PReLU import IR_64,IR152_PReLU
import os
from torch.nn import  MaxPool2d

#function to decay the learning rate

def increase_rate(p):
    p*=1.1
    return p




def decay_lr(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor

class BasicBlock(nn.Module):
    expansion = 1  # No change needed if you're using BasicBlock

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.activations = {}
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.prelu = nn.PReLU() 
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        self.activations['input'] = x
        out = self.conv1(x)
        out = self.prelu(out)
        out = self.conv2(out)
        out += self.downsample(x)
        self.activations['output'] = out
        return out, self.activations

class Shallow_Layer(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(Shallow_Layer, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu = nn.PReLU()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        activations = {}
        activations['input'] = x
        
        # First Convolution
        x = self.prelu(self.conv1(x))
        activations['after_conv1'] = x
        
        # Layer 1
        x, layer1_acts = self._forward_layer(x, self.layer1)
        activations['layer1'] = layer1_acts

        
        return x, activations

    def _forward_layer(self, x, layer):
        layer_activations = []
        for block in layer:
            x, block_act = block(x)
            layer_activations.append(block_act)
        return x, layer_activations

    # def load_weights(self, resnet):
    #     # Load weights for the first convolutional layer
    #     self.conv1.weight.data = resnet.conv1.weight.data.clone()
    #     # Load weights for Layer 1
    #     for i, block in enumerate(self.layer1):
    #         block.conv1.weight.data = resnet.layer1[i].conv1.weight.data.clone()
    #         block.conv2.weight.data = resnet.layer1[i].conv2.weight.data.clone()
    #         if hasattr(block, 'downsample') and hasattr(resnet.layer1[i], 'downsample'):
    #             block.downsample[0].weight.data = resnet.layer1[i].downsample[0].weight.data.clone()










class FP(nn.Module):
    def __init__(self, params:dict):
        super(FP, self).__init__()
        # Define a 2D convolutional layer and initialize its weights with the pretrained weights
        self.conv_layer = nn.Conv2d(in_channels=params['in_channels'], out_channels= params['out_channels'], kernel_size= params['kernel_size'],
                                    stride=params['stride'], padding=params['padding'], bias= params['bias'])
        if params['weight'] is not None:
            self.conv_layer.weight = nn.Parameter(params['weight']) # conv1_params = extract_conv_details(ir_model.conv1) 

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



def shallow_inversion(res_block_out,res_block_in,model):
    
    y = res_block_out
    
    cuda = False
    device = 'cpu'
    num_blocks = [3]
    W = Shallow_Layer(BasicBlock, num_blocks)
    
    shallow_model_dict = W.state_dict()

    for name, param in model.named_parameters():
        if name in shallow_model_dict:
            shallow_model_dict[name].copy_(param.data)
        else:
        # If the layer is not in shallow_model, that means it's a part of deeper layers
        # which do not exist in shallow_model, so you do nothing.
            pass
   # W.load_weights(model)
    W.load_state_dict(shallow_model_dict)
    ## Creating the Shallow model
    
    
    
    #prelu_layer.weight.data = p_relu['weight']
    
    # Step 2: Initialize variables
    x = Variable((1e-3 * torch.randn(*res_block_in.size()).cuda() if cuda else 
        1e-3 * torch.randn(*res_block_in.size())), requires_grad=True)
    
    optimizer = torch.optim.Adam([x], lr=0.01)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Adjust the step_size and gamma as needed
   
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    

    
    # Training loop
    
    #block_num = extract_number(block_name)
    
    # if(block_num <= 12):
    #     num_epochs = 2000
    # else:
    #     num_epochs = 1000
    
    num_epochs = 1000
        
    #print(num_epochs)
    
    
    # Decay iteration
    # decay_iter = 100
    
    alpha = 6
    beta = 2
    lamda_alpha = 1e-5
    lamda_beta = 1e-5
    
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
    
        # Step 4: Define the loss
        alpha_norm  = alpha_prior(x, alpha=alpha)
        tv_x = tv_norm(x, beta=beta)
        fp_pass,_ = W(x)
        
        #import pdb;pdb.set_trace()
        loss = norm_loss(fp_pass,y) +  lamda_beta * tv_x + lamda_alpha * alpha_norm

    
        # Compute gradients
        loss.backward()
    
        # Update variables
        optimizer.step()
        
        #scheduler.step()
    
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
        # Check if the loss is less than 1e-5, and if so, break out of the loop
        #if  loss.item() < 1e-5:
        #    break
        
        # Adjust learning rate based on the loss
        #scheduler.step(loss.item())

    
    return x.detach()

    
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
        W1 = nn.Linear(model.feat_dim, model.num_classes)
        W1.weight.data = model.fc_layer.weight.clone()
        W1.bias.data = model.fc_layer.bias.clone()
        
        #import pdb;pdb.set_trace()
    
    elif layer == 'output':
        # Initialize a Sequential model including a Flatten operation and a Linear layer
        #import pdb;pdb.set_trace()
        stride = 2
        f = stride * 14
        w1 = nn.Linear(model.feat_dim * f * f, model.feat_dim)
        w1.weight.data = model.output_layer[1].weight.clone()
        w1.bias.data = model.output_layer[1].bias.clone()
        
       # W1 = nn.Sequential(nn.Flatten(), w1)
        W1 = w1
     



    
    # Training loop
    num_epochs = 1
   # mu = 1e3  # Penalty term for the inequality constraint
    

    y = linear_out
    
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
      
        # Define the loss
       
        

        #loss = (y - F.relu(W1(x)) ) .norm()**2 +  lamda_beta * tv_x + lamda_alpha * alpha_norm
        #fp_pass = F.nn.relu(W1(x))
       
        
        
        if(layer == 'output'):
           # import pdb;pdb.set_trace()
          #  flat = nn.Flatten()
          # fp_pass = flat(x)
            #import pdb;pdb.set_trace()
            loss = (y - W1(x)).norm()**2
        else:
            fp_pass = W1(x)
            loss = (y - fp_pass ).norm()**2
        #loss = torch.norm(prelu_layer(W1(x) - y)) **2
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
    num_epochs = 500
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
       
        
        prelu_layer = nn.PReLU()
        #loss = (y - F.relu(W1(x)) ) .norm()**2 +  lamda_beta * tv_x + lamda_alpha * alpha_norm
        #fp_pass = F.nn.relu(W1(x))
        fp_pass =  prelu_layer(W1(x))
        
        if(maxpool):
            Maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            fp_pass = Maxpool(fp_pass)
        
        loss = norm_loss(fp_pass,y) +  lamda_beta * tv_x + lamda_alpha * alpha_norm
        #loss = torch.norm(prelu_layer(W1(x) - y)) **2
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
    
    
    prelu_layer = nn.PReLU()
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
    
    num_epochs = 1000
        
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





def resblock_inverted(model_info,resnet,res_out):
    #inverted_outputs = {}

    # Process layers in reverse order
    for layer_name in reversed(list(model_info.keys())):
        if(layer_name == 'layer1'):
            pass
        
        else:
            layer_info = model_info[layer_name]
            #layer_output = {}

            print(f"Processing {layer_name}...")  # Print the layer name
            res_block_out = res_out #resnet.activations[layer_name][1]['output']
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
                
                #print(res_block_out.shape)
               # print(res_block_in.shape)
                #print(orignal_block_out.shape)
    
                inverted_input = basic_block_inverted(res_block_out, res_block_in, conv1_params, conv2_params, 
                                                       downsample_params,shortcut)
                #plot_channels(inverted_input,rec = True)
                #plot_channels(orignal_block_out,rec = False)
                
                
                res_block_out = inverted_input
                #layer_output[block_name] = inverted_output
            
            #inverted_outputs[layer_name] = layer_output
        orignal_input  = res_block_in
    return inverted_input,orignal_input #None #inverted_outputs





def main():
    # Load the pretrained ResNet-18 model
    #resnet= ResNet18(pretrained=True)
    
    ir_model = IR152_PReLU()
    
    ################### Forward Pass #############3
    
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Specify the filename of the image
    image_filename = 'CelebA_ex1.jpg'

    # Construct the full path to the image
    image_path = os.path.join(current_directory, image_filename)
    

    image = Image.open(image_path).convert('RGB')


    # Assuming you've resized the image using PIL
    resized_image = image.resize((224, 224))

    # Convert to tensor
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(resized_image)


    input_image = image_tensor.unsqueeze(0)

    # # Ensure the model is in evaluation mode
    ir_model.eval()

    ## Perform a forward pass
    with torch.no_grad():
        outputs  =  ir_model(input_image)

    # #  Here the Input to the 4 resblocks is recovered:
        
    resblocks = False
    shallow_layer = False
    maxpool = False
    lastlayer = False
    fc_layer = True
    output_layer = True
    
    if(fc_layer):
       
        in_lin1 = linear_inverted(outputs[0],ir_model.activations['fc'],ir_model,layer = 'fc')
        print(torch.norm(outputs[0] - in_lin1))

    
    if(output_layer):
        feat = ir_model.activations['feature']
        feat = feat.view(feat.size(0), -1)
        #import pdb;pdb.set_trace()
        in_lin2 = linear_inverted(feat,in_lin1,ir_model,layer = 'output')
        #import pdb;pdb.set_trace()
        in_lin2 = in_lin2.view(*ir_model.activations['feature'].shape)
        print(torch.norm(ir_model.activations['feature']- in_lin2))
        
    if(resblocks):
        resnet_info = extract_resnet_info(ir_model)
        #import pdb;pdb.set_trace()
        inverted_input,orignal_input = resblock_inverted(resnet_info,ir_model,in_lin_2)
        print(torch.norm(inverted_input - orignal_input))
    
    if(shallow_layer):
        if(resblocks):
            first_layer = inverted_input
        else:
            #import pdb;pdb.set_trace()
            first_layer = ir_model.activations['layer1'][2]['output']
        
        input_recovered = shallow_inversion(first_layer, input_image ,ir_model)
        
        #plot_image(input_recovered)
        plot_image(input_recovered,save_path ='inverted_Prelu_stride=1,image_inversion.png')
        
        plot_image(input_image,save_path ='orignal.png')
    #plot_channels(inverted_input,rec = True)
    #plot_channels(orignal_input,rec = False)
    
    
    

        

if __name__ == '__main__':
    main()


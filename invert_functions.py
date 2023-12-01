#from matplotlib.font_manager import X11FontDirectories
import torch
import time
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
#from pygranso.pygranso import pygranso
#from pygranso.pygransoStruct import pygransoStruct
import numpy as np
from torch.nn import MaxUnpool2d
# Define the basic building blocks: BasicBlock and Bottleneck

class fc_in(nn.Module):
    def __init__(self,linear_weights=None,bias = None,block_expansion = 1,num_classes = 1000):
        super(fc_in,self).__init__()
        self.fc = nn.Linear(5 * block_expansion, num_classes).double()

        if linear_weights is not None:
            # Initialize conv_layer weights with pretrained weights
            self.fc.weight = nn.Parameter(linear_weights)
            self.fc.bias = nn.Parameter(bias)

    def forward(self, x):
        linear_result = self.fc(x)
        return linear_result
    
    
    
class FP(nn.Module):
    def __init__(self, num_channels=5,bn_weights = None, conv_weights=None):
        super(FP,self).__init__()

        # Define a 2D convolutional layer
        self.conv_layer = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=False).double()
        self.bn1 = nn.BatchNorm2d(num_channels)

        if conv_weights is not None:
            # Initialize conv_layer weights with pretrained weights
            self.conv_layer.weight = nn.Parameter(conv_weights)
        if bn_weights is not None:
            # Initialize bn1 layer weights with pretrained weights
            self.bn1.weight.data = bn_weights['weight'].clone().double()
            self.bn1.bias.data = bn_weights['bias'].clone().double()
            self.bn1.running_mean = bn_weights['running_mean'].clone().double()
            self.bn1.running_var = bn_weights['running_var'].clone().double()

    def forward(self, x):
        conv_result = self.conv_layer(x)
        #x = self.bn1(conv_result)
        return conv_result
    
def solve_optimization_problem_conv_layer(W1_weights=None,BN_weights = None, y=None, input=None):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input.double().to(device)
    y = y.double().to(device)

    if W1_weights is not None :
        W1 = FP(conv_weights=W1_weights.double(),bn_weights = BN_weights).to(device)

    else:
        W1 = FP(pretrained_weights=None).to(device)


    # Define the optimization variables
    var_in = {"x": input_tensor.shape}

    def user_fn(X_struct):
        x = X_struct.x

        # Define the objective function
        f = torch.norm( y - F.relu(W1(x)))**2

        # Inequality constraints (element-wise)
        ci = pygransoStruct()
        ci.c1 = W1(x) - F.relu(W1(x))


        # Equality constraint
        # ce = pygransoStruct()
        # ce.c1 = W1(x) - p + n
        # ce.c2 = torch.matmul(n.view(n.size(0), -1), p.view(p.size(0), -1).T).squeeze()
        # ce.c3 = y - p

        ce = None
        #ce.c3 = y - x - W2(p)

        return [f, ci, ce]

    # Combine the user-defined functions
    comb_fn = lambda X_struct: user_fn(X_struct)

    # Define options and initial guess
    opts = pygransoStruct()
    opts.maxit = 150
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opts.torch_device = device

    x0 = torch.randn(*input_tensor.shape, device = device,dtype=torch.double)
    opts.x0 = x0.view(-1, 1)
    opts.print_frequency = 10

    # Solve the optimization problem
    soln = pygranso(var_spec=var_in, combined_fn=comb_fn, user_opts=opts)

    # Get the optimal values
    optimal_x = soln.final.x[:x0.numel()].view(*input_tensor.shape)



    return optimal_x 


def solve_optimization_problem_resblock(W1_weights = None, W2_weights = None,BN1_weights = None,BN2_weights = None,res_block_in = None,res_block_out = None):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = res_block_in.double().to(device)

    in_channels = input_tensor.shape[1]
    out_channels = res_block_out.shape[1]

    y = res_block_out.double().to(device)

    if W1_weights is not None and W2_weights is not None:
        W1 = FP(conv_weights=W1_weights.double(),bn_weights = BN1_weights).to(device)
        W2 = FP(conv_weights=W2_weights.double(),bn_weights = BN2_weights).to(device)

    else:
        W1 = FP(conv_weights=None,bn_weights = None).to(device)
        W2 = FP(conv_weights=None,bn_weights = None).to(device)


    # Get the shape of the intermediate convolutional result
    p_shape = F.relu(W1(input_tensor)).shape

    # Define the optimization variables
    var_in = {"x": input_tensor.shape, "p": p_shape, "n": p_shape}

    def user_fn(X_struct):
        x = X_struct.x
        p = X_struct.p
        n = X_struct.n

        # Define the objective function
        f = torch.norm( y - x - W2(p))**2

        # Inequality constraints (element-wise)
        ci = pygransoStruct()
        ci.c1 = -p
        ci.c2 = -n

        # Equality constraint
        ce = pygransoStruct()
        ce.c1 = W1(x) - p + n
        ce.c2 = torch.matmul(n.view(n.size(0), -1), p.view(p.size(0), -1).T).squeeze()
        #ce.c3 = y - x - W2(p)

        return [f, ci, ce]

    # Combine the user-defined functions
    comb_fn = lambda X_struct: user_fn(X_struct)

    # Define options and initial guess
    opts = pygransoStruct()
    opts.maxit = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opts.torch_device = device
    x0 = torch.randn(*input_tensor.shape, device = device,dtype=torch.double)
    p0 = torch.zeros(*p_shape,device = device, dtype=torch.double)
    n0 = torch.zeros(*p_shape,device = device, dtype=torch.double)
    opts.x0 = torch.cat((x0.view(-1, 1), p0.view(-1, 1), n0.view(-1, 1)), dim=0)
    opts.print_frequency = 10

    start_time = time.time()
    # Solve the optimization problem
    soln = pygranso(var_spec=var_in, combined_fn=comb_fn, user_opts=opts)
    end_time = time.time()
    # Get the optimal values
    optimal_x = soln.final.x[:x0.numel()].view(*input_tensor.shape)
    optimal_p = soln.final.x[x0.numel():x0.numel()+p0.numel()].view(*p_shape)
    optimal_n = soln.final.x[x0.numel()+p0.numel():].view(*p_shape)

    execution_time = end_time - start_time

    return optimal_x, optimal_p, optimal_n, execution_time
    
def solve_optimization_problem_final_layer(W1_weights=None,bias = None, y=None, input=None):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input.double().to(device)
    y = y.double().to(device)

    if W1_weights is not None :
        W1 = fc_in(linear_weights=W1_weights.double(),bias = bias.double()).to(device)

    else:
        W1 = fc_in(linear_weights=None).to(device)


    # Define the optimization variables
    var_in = {"x": input_tensor.shape}

    def user_fn(X_struct):
        x = X_struct.x

        # Define the objective function
        f = torch.norm(x)**2

        # Inequality constraints (element-wise)
        # ci = pygransoStruct()
        # ci.c1 = W1(x) - F.relu(W1(x))
        ci = None


        # Equality constraint
        ce = pygransoStruct()
        ce.c1 = W1(x) - y
        # ce.c2 = torch.matmul(n.view(n.size(0), -1), p.view(p.size(0), -1).T).squeeze()
        # ce.c3 = y - p

        #ce.c3 = y - x - W2(p)

        return [f, ci, ce]

    # Combine the user-defined functions
    comb_fn = lambda X_struct: user_fn(X_struct)

    # Define options and initial guess
    opts = pygransoStruct()
    opts.maxit = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opts.torch_device = device
    x0 = torch.randn(*input_tensor.shape, device = device,dtype=torch.double)
    opts.x0 = x0.view(-1, 1)
    opts.print_frequency = 10

    # Solve the optimization problem
    soln = pygranso(var_spec=var_in, combined_fn=comb_fn, user_opts=opts)

    # Get the optimal values
    optimal_x = soln.final.x[:x0.numel()].view(*input_tensor.shape)



    return optimal_x




class InverseLinear(nn.Module):
    def __init__(self, fc_layer):
        super(InverseLinear, self).__init__()
        self.fc_layer = fc_layer

    def forward(self, y):
        # Calculate pseudo-inverse
        #import pdb;pdb.set_trace()
        W_pseudo_inv = torch.pinverse(self.fc_layer.weight.clone())
        y_scaled = (y - self.fc_layer.bias).T

        #print(W_pseudo_inv.shape)
        #print(y_scaled.shape)
        return torch.mm(W_pseudo_inv,y_scaled)





class InverseAdaptiveAvgPool2d(nn.Module):
    def __init__(self, original_size):
        super(InverseAdaptiveAvgPool2d, self).__init__()
        self.original_size = original_size

    def forward(self, input):
        return input.expand((-1, -1, self.original_size[0], self.original_size[1]))



class MaxUnpool2d(torch.nn.Module):
    def __init__(self):
        super(MaxUnpool2d, self).__init__()

    def forward(self, input, indices, output_size):
        # Perform the max unpooling operation
        output = F.max_unpool2d(input, indices, kernel_size=1, stride=1, padding=0, output_size=output_size)
        
        # Replace zeros with the maximum value from the pooled input
        max_replace = 0
        if(max_replace):
            # Bilinear upscale the pooled result
            bilinear_upscale = F.interpolate(input, size=output_size[2:], mode='bilinear', align_corners=False)
    
            # Create a mask of the same size as the input and set to zero
            mask = torch.zeros_like(input)
            mask.view(-1)[indices.view(-1)] = 1
    
            # Bilinear upscale this mask
            mask_upscale = F.interpolate(mask, size=output_size[2:], mode='bilinear', align_corners=False)
    
            # Multiply to get the locations of the original max values
            max_values_upscale = bilinear_upscale * mask_upscale
    
            # Subtract this from the bilinear result to get the interpolated values
            interpolated_values = bilinear_upscale - max_values_upscale
    
            # Combine
            output = max_values_upscale + interpolated_values
            
        return output
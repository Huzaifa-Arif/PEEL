import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from torch.autograd import Variable
import numpy as np
import time
from utilities import total_variation_4d,tensor_p_norm_powered


######## Some new functions (May need to be shifted) #######

def norm_loss(input, target):
    return torch.div(alpha_prior(input - target, alpha=2.), alpha_prior(target, alpha=2.))


def alpha_prior(x, alpha=2.):
    return torch.abs(x.view(-1)**alpha).sum()


def tv_norm(x, beta=2.):
    assert(x.size(0) == 1)
    img = x[0]
    dy = img - img # set size of derivative and set border = 0
    dx = img - img
    dy[:,1:,:] = -img[:,:-1,:] + img[:,1:,:]
    dx[:,:,1:] = -img[:,:,:-1] + img[:,:,1:]
    return ((dx.pow(2) + dy.pow(2)).pow(beta/2.)).sum()


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


class FC(nn.Module):
    def __init__(self, pretrained_weights = None):
        super(FC, self).__init__()
        # Define a 2D convolutional layer and initialize its weights with the pretrained weights
        self.conv_layer = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=1, padding=1, bias=False).double()
        self.relu = nn.ReLU(inplace=True)

        if pretrained_weights is not None:
            self.conv_layer.weight = nn.Parameter(pretrained_weights)

    def forward(self, x):
        conv_result = self.conv_layer(x)
        relu_result  = self.relu(conv_result)
        return relu_result
    
    
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
    opts.maxit = 10000
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
    
def solve_optimization_problem_conv_layer(W1_weights=None, y=None, input=None):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input.double().to(device)
    y = y.double().to(device)

    if W1_weights is not None :
        W1 = FP(pretrained_weights=W1_weights.double()).to(device)

    else:
        W1 = FP(pretrained_weights=None).to(device)


    # Define the optimization variables
    var_in = {"x": input_tensor.shape}

    def user_fn(X_struct):
        x = X_struct.x

        # Define the objective function
        tv_x = total_variation_4d(x,beta =2)
        alpha_norm = tensor_p_norm_powered(x, alpha = 6)
        lamda_alpha = 1
        lamda_beta = 1
        
        f = torch.norm( y - F.relu(W1(x)))**2 + lamda_beta * tv_x + lamda_alpha * alpha_norm

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

    

def solve_optimization_problem_conv_layer_2(W1_weights=None, y=None, input=None):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input.double().to(device)
    y = y.double().to(device)

    if W1_weights is not None :
        W1 = FP(pretrained_weights=W1_weights.double()).to(device)

    else:
        W1 = FP(pretrained_weights=None).to(device)


    # Define the optimization variables
    p_shape = F.relu(W1(input_tensor)).shape
    var_in = {"x": input_tensor.shape, "p": p_shape, "n": p_shape}

    def user_fn(X_struct):
        x = X_struct.x
        p = X_struct.p
        n = X_struct.n


        # Define the objective function
        f = torch.norm(x) **2
        # f = torch.norm( y - F.relu(W1(x)))**2

        # Inequality constraints (element-wise)
        ci = pygransoStruct()
        ci.c1 = W1(x) - F.relu(W1(x))


        # Equality constraint
        ce = pygransoStruct()
        ce.c1 = W1(x) - p + n
        ce.c2 = torch.matmul(n.view(n.size(0), -1), p.view(p.size(0), -1).T).squeeze()
        ce.c3 = y - p

        #ce = None
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
    
    return optimal_x, optimal_p, optimal_n,execution_time   




def solve_optimization_problem_resnets(W1_weights = None, W2_weights = None,res_block_in = None,res_block_out = None):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = res_block_in.double().to(device)
    y = res_block_out.double().to(device)
    if W1_weights is not None and W2_weights is not None:
        W1 = FP(pretrained_weights=W1_weights.double()).to(device)
        W2 = FP(pretrained_weights=W2_weights.double()).to(device)

    else:
        W1 = FP(pretrained_weights=None).to(device)
        W2 = FP(pretrained_weights=None).to(device)
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
    opts.maxit = 20
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

    return optimal_x, optimal_p, optimal_n,execution_time    

def resnet_inverted(res_block_out,res_block_in,conv1_params:dict,conv2_params:dict,conv3_params:dict,shortcut = 0):
    
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
    if(shortcut):
        #import pdb;pdb.set_trace()
        W3 = FP(conv3_params)  ### 
    
    
    # Training loop
    num_epochs = 1000
    
    # Reality __-> should change
    lambda_1 = 1e3  # Regularization term for the first constraint
    lambda_2 = 1e3  # Regularization term for the second constraint

    #decay_iter = 100
    
    mu_1 = 1e3
    mu_2 = 1e3
    
    def increase_rate(p):
        p*=1.1
    return p
    
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        
        
        # Step 4: Define the loss
        if(shortcut):
            #simport pdb;pdb.set_trace()
            loss = ( y - W3(x) - W2(p)).norm()**2
        else:
            loss = (y - x - W2(p)).norm()**2
        #loss = x.norm()**2  # This is ||x||^2
        ## Check solutions where constraint are met and store that solution
        
        
        # Add the constraint terms to the loss
        constraint_1 = lambda_1 * ( W1(x) - p + n).norm()**2
        loss += constraint_1

        constraint_2 = lambda_2 * (torch.matmul(n.view(n.size(0), -1), p.view(p.size(0), -1).T).squeeze())**2
        loss += constraint_2
        
        
        # constraint_3 = mu_1* torch.relu(n)
        # loss+=constraint_3
        
        # constraint_4 = mu_2*torch.rel(p)
        # loss+=constraint_4
        
        # lambda_1= increase_rate(lambda_1)
        # lambda_2= increase_rate(lambda_2)
        # mu_1 = increase_rate(mu_1)
        # mu_2 = increase_rate(mu_2)
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
        
        #if (epoch+1) % decay_iter == 0:
        #    decay_lr(optimizer, decay_factor)
        
        # Print every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    
    return x.detach()
def conv_inverted(conv_out,conv_in,conv1_params:dict):
    
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
       
        
        
        #loss = (y - F.relu(W1(x)) ) .norm()**2 +  lamda_beta * tv_x + lamda_alpha * alpha_norm
        fp_pass = F.relu(W1(x))
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
import torch
import torch.nn as nn
from IR_152_func import IR_152_64
from IR_152_utilities import extract_backbone64_info
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from invert_functions_noBN import *
from utilities import plot_channels,plot_image
import os
from torch.optim import lr_scheduler

def extract_number(s):
    """Extract the number from a string."""
    return int(''.join(filter(str.isdigit, s)))

def bottleneckIR_inverted(block_out, block_in, res_layer_params, shortcut_params=None):
    """
    Function to invert the bottleneck_IR block.
    
    This is just a placeholder. You need to provide the actual inversion logic 
    similar to `basic_block_inverted`.
    """
    
    pass 
    # TODO: Provide the inversion logic for bottleneck_IR block
    #return inverted_input




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
    

def first_layer_inverted(conv_out,conv_in,conv1_params:dict,p_relU:dict):
    
    cuda = None
    
    # Step 2: Initialize variables
    x = Variable((1e-3 * torch.randn(*conv_in.size()).cuda() if cuda else 
        1e-3 * torch.randn(*conv_in.size())), requires_grad=True)
    
    
    # Step 3: Choose an optimizer
    optimizer = torch.optim.Adam([x], lr=0.01)
    device = 'cpu'
    W1 = FP(conv1_params).to(device)
    
    #### Putting the desired weights #####
    
    prelu_layer = nn.PReLU(num_parameters=len(p_relU['weight']))

    # Set the weights of the PReLU layer to the desired_weights
    prelu_layer.weight.data = p_relU['weight']


    
    # Training loop
    num_epochs = 1000
    y = conv_out
   # mu = 1e3  # Penalty term for the inequality constraint
    alpha = 6
    beta = 2
    lamda_alpha = 1e-5
    lamda_beta = 1e-5
    
    for epoch in range(num_epochs):
            
      # Zero gradients
      optimizer.zero_grad()
      
      alpha_norm  = alpha_prior(x, alpha=alpha)
      tv_x = tv_norm(x, beta=beta)
      
      
      
      # Define the loss
      fp_pass = prelu_layer(W1(x))
      loss = norm_loss(fp_pass,y) +  lamda_beta * tv_x + lamda_alpha * alpha_norm
      #loss = torch.norm(fp_pass - y)
      
        
      # Compute gradients
      loss.backward()
        
      # Update variables
      optimizer.step()
        
       # Print every 100 epochs
      if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    
      if loss.item() < 1e-5:
         break
    
    print(f'Solution x: {x.detach().numpy()}')
    
    return x


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









def bottleneckIR_inverted(block_name,res_block_out,res_block_in,conv1_params:dict,conv2_params:dict,conv3_params:dict,shortcut:str,p_relu:dict):
    
    
       
    
    y = res_block_out
    
    cuda = False
    device = 'cpu'
    
    ### Creating the convolution layers 
    W1 = FP(conv1_params).to(device)
    W2 = FP(conv2_params).to(device)
    
    
    prelu_layer = nn.PReLU(num_parameters=len(p_relu['weight']))
    prelu_layer.weight.data = p_relu['weight']
    
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
    
    block_num = extract_number(block_name)
    
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
    
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
    
        # Step 4: Define the loss
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
        if  loss.item() < 1e-5:
            break
        
        # Adjust learning rate based on the loss
        #scheduler.step(loss.item())

    
    return x.detach()







def IR_152_inverted(backbone64_info, backbone_model):
    #inverted_outputs = {}

    # Process layers in reverse order
    for layer_name in reversed(list(backbone64_info.keys())):
        
        print(f"Processing {layer_name}...")  # Print the layer name

        if(layer_name == 'body'):
            layer_info = backbone64_info[layer_name]
            #import pdb;pdb.set_trace()
            res_block_out = backbone_model.activations[layer_name][extract_number('block50') - 1]['output']

            # Process blocks in reverse order within each layer
            for block_name in reversed(list(layer_info.keys())):
                block_info = layer_info[block_name]
                
                
                
                conv1_params = block_info['res_layer']['conv1']
                conv2_params = block_info['res_layer']['conv3']
                p_relu_params = block_info['res_layer']['prelu2']
    
                print(f"  Inverting {block_name}...")  # Print the block name
                
                shortcut_params = block_info.get('shortcut', None)
                
                #import pdb;pdb.set_trace()
                
                
                res_block_in = backbone_model.activations[layer_name][extract_number(block_name) - 1]['input']
                original_block_out = backbone_model.activations[layer_name][extract_number(block_name) - 1]['output']
                
                
                if shortcut_params:
                    if 'weight' in shortcut_params:  # This is Conv2d
                        shortcut = 'Conv'
                        #res_block_out = original_block_out
                    else:         # This is MaxPool2d
                        shortcut = 'Maxpool'
        
                
                
                
                
                
                print(res_block_out.shape)
                print(res_block_in.shape)
                print(original_block_out.shape)
                #import pdb;pdb.set_trace()
                inverted_input = bottleneckIR_inverted(block_name,res_block_out, res_block_in,conv1_params,conv2_params, shortcut_params, shortcut, p_relu_params)
                
                
                plot_channels(inverted_input,rec = True)
               # plot_channels(original_block_out,rec = False)
                
                res_block_out = inverted_input
    #         # layer_output[block_name] = inverted_output
        if(layer_name == 'input_layer'): 
            
            plot_channels(inverted_input,rec = True)
            plot_channels(backbone_model.activations['output'],rec = False)
          
            layer_info = backbone64_info[layer_name]
            
           # import pdb;pdb.set_trace()
            
            p_relu_params = layer_info['prelu']
            conv_params = layer_info['conv1']
            
            init_layer_output = inverted_input  #backbone_model.activations['output']
            init_layer_input = backbone_model.activations['input']
            
            
           # 
            
            inverted_input = first_layer_inverted(init_layer_output,init_layer_input,conv_params,p_relu_params)
            orignal_input =  init_layer_input 
            
           # 
            
            
            
            
    #     # inverted_outputs[layer_name] = layer_output
    # original_input = res_block_in
    return inverted_input, orignal_input  # None  # inverted_outputs
    #return None




    
def main():
    ir_model = IR_152_64((64,64))
    
    ################### Forward Pass #############3
    
    #image_path = '/Users/huzaifaarif/Desktop/IBM_25th_Oct_2023/IBM_17th_Oct_2023/CelebA_ex1.jpg'
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
        outputs  = ir_model(input_image)

   #  #  Here the Input to the backbone is recovered:
        
   #  backbone = True
   #  first_layer = False
    
    
   #  if(backbone):
   #      model_info = extract_backbone64_info(ir_model)
   #      inverted_input,orignal_input = IR_152_inverted(model_info,ir_model)
        
   #      plot_image(inverted_input,save_path ='inverted4.png')
   #      #plot_image(orignal_input,save_path = 'orignal4.png')
   #      #print(model_info)
   #      #print(torch.norm(inverted_input - orignal_input))

   # ###  The final layer in both cases. 
   #  if(first_layer):
   #      pass
   #  ## Get all the activations from each layer.
   #  #IR_152_inverted(model_info ,ir_model)
   #  #print(model)
    
    
if __name__ == '__main__':
    main()

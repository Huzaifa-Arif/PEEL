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
from utilities import total_variation_4d,extract_conv_details
from models_noBN import ResNet
from PIL import Image
from torch.autograd import Variable
from invert_functions_noBN import resnet_inverted
from Image_Inversion_test import invert
# Define the ResidualBlock class
#torch.manual_seed(8674529339715958261)
#current_seed = torch.initial_seed()
#print("Current seed:",  torch.seed() )





def reconstruct(seed):
    
    print("Results for seed:",seed)
    recons_error = 0
    torch.manual_seed(seed)
    # Hyperparameters
    num_epochs = 2
    batch_size = 64
    learning_rate = 0.001
    Scaling  = 32 #
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((Scaling, Scaling)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    
    # # Initialize ResNet model
    resnet = ResNet()
    
    loading = False
    if(loading):
        # Load the trained weights
        resnet.load_state_dict(torch.load('resnet.pth'))
    
    
    
    # # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=learning_rate)   
    
    
    

    
       
    
    ### Doing Forward Pass on one image
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1,shuffle=False)
    
    with torch.no_grad():
        for images, labels in test_loader:
            features,outputs = resnet(images)
            break
    
    
    ############### Manual Image ######################3
    # image_path = 'CelebA_ex1.jpg'

    # image = Image.open(image_path).convert('RGB')


    # # Assuming you've resized the image using PIL
    # resized_image = image.resize((8, 8))

    # # Convert to tensor
    # to_tensor = transforms.ToTensor()
    # image_tensor = to_tensor(resized_image)


    # input_image = image_tensor.unsqueeze(0)

    # # # Ensure the model is in evaluation mode
    # resnet.eval()

    # # # Perform a forward pass
    # with torch.no_grad():
    #     features,outputs  = resnet(input_image)
       
       
    #Display the processed image

    # img_tensor = image_tensor * 0.5 + 0.5  # unnormalize
    # #print(img_tensor.shape)
    # p_image = img_tensor.detach().cpu().numpy()  # Convert tensor to numpy array
    # plt.imshow(np.transpose(p_image, (1, 2, 0)))  # Transpose the image tensor (C, H, W) -> (H, W, C)
    # plt.show()
    
    
    
    ##################### All the features ###############33
    
   # import pdb;pdb.set_trace()
    
    for key, value in features.items():
        print(f"{key} shape: {value.shape}")
    
    
    
    
    resblocks = True
    init_layer = False
    avg_pool = False
    final_layer = True
    
    # image = features["input"].squeeze(0).permute(1, 2, 0).numpy()
    
    # # Visualization using matplotlib
    # plt.imshow(image)
    # plt.axis('off')  # To hide axis values
    # plt.show()
    
    #import sys
   # sys.exit()
    
    
    if(final_layer):
        W1 = resnet.fc.weight.data.clone()
        bias = resnet.fc.bias.data.clone()
        fc_recovered= solve_optimization_problem_final_layer(W1_weights=W1,bias = bias, y=features['output'], input=features['fc_in'])
        fc_loss = torch.norm(fc_recovered - features['fc_in'])
        #recons_error = fc_loss
        print("Leakage in first layer:", fc_loss)
        
    if(avg_pool):
        print("Not doing anything here")
        print(torch.norm(features['avg_pool_out'] - features['avg_pool_in']))
        
    
    if(resblocks):
    
        #res_block4_weights = resnet.res_block4.state_dict()
        conv1_params =extract_conv_details(resnet.res_block4.conv1)
        conv2_params =extract_conv_details(resnet.res_block4.conv2)
        
        #W1_weights= res_block4_weights['conv1.weight']
        #W2_weights = res_block4_weights['conv2.weight']
        
        
        res_block_out =  features['res_block_out_4']  #fc_recovered.view(1, 5, 8, 8) # features['res_block_in_4'] 
        res_block_in =  features['res_block_in_4'] 
        
        
        
        resblock_recovered = resnet_inverted(res_block_out,res_block_in,conv1_params,conv2_params,None,0)
        
        
        #res_block3_weights = resnet.res_block3.state_dict()
        #W1_weights= res_block3_weights['conv1.weight']
        #W2_weights = res_block3_weights['conv2.weight']
        conv1_params =extract_conv_details(resnet.res_block3.conv1)
        conv2_params =extract_conv_details(resnet.res_block3.conv2)
        
        res_block_out =  resblock_recovered #fc_recovered.view(1, 5, 8, 8) # features['res_block_in_4'] 
        res_block_in =  features['res_block_in_3']
        
        
        
        resblock_recovered = resnet_inverted(res_block_out,res_block_in,conv1_params,conv2_params,None,0)
        
        
        
        # res_block2_weights = resnet.res_block2.state_dict()
        # W1_weights= res_block2_weights['conv1.weight']
        # W2_weights = res_block2_weights['conv2.weight']
        
        conv1_params =extract_conv_details(resnet.res_block2.conv1)
        conv2_params =extract_conv_details(resnet.res_block2.conv2)
        
        res_block_out =  resblock_recovered  #fc_recovered.view(1, 5, 8, 8) # features['res_block_in_4'] 
        res_block_in =  features['res_block_in_2'] 
        
        
        #import pdb;pdb.set_trace()
        
        resblock_recovered = resnet_inverted(res_block_out,res_block_in,conv1_params,conv2_params,None,0)
        
        
        
        conv1_params =extract_conv_details(resnet.res_block1.conv1)
        conv2_params =extract_conv_details(resnet.res_block1.conv2)
        
        res_block_out =  resblock_recovered  #fc_recovered.view(1, 5, 8, 8) # features['res_block_in_4'] 
        res_block_in =  features['res_block_in_1'] 
        
        
        #import pdb;pdb.set_trace()
        
        resblock_recovered = resnet_inverted(res_block_out,res_block_in,conv1_params,conv2_params,None,0)
        
        
        
        
        # # Step 2: Initialize variables
        # x = Variable((1e-3 * torch.randn(*res_block_in.size()).cuda() if cuda else 
        #     1e-3 * torch.randn(*res_block_in.size())), requires_grad=True)
        
        
        # p  = Variable((1e-3 * torch.randn(*res_block_in.size()).cuda() if cuda else 
        #     1e-3 * torch.randn(*res_block_in.size())), requires_grad=True)
        
        # n = Variable((1e-3 * torch.randn(*res_block_in.size()).cuda() if cuda else 
        #    1e-3 * torch.randn(*res_block_in.size())), requires_grad=True)
        
        # # Step 3: Choose an optimizer
        # optimizer = torch.optim.Adam([x, p, n], lr=0.01)
        # device = 'cpu'
        
        # ### Creating the convolution layers 
        # W1 = FP(pretrained_weights=W1_weights).to(device)
        # W2 = FP(pretrained_weights=W2_weights).to(device)
        
        # # Training loop
        # num_epochs = 2000
        # lambda_1 = 1e3  # Regularization term for the first constraint
        # lambda_2 = 1e3  # Regularization term for the second constraint
        
        # mu = 1e3  # Penalty term for the inequality constraints
        
        # for epoch in range(num_epochs):
        #     # Zero gradients
        #     optimizer.zero_grad()
            
            
            
        #     # Step 4: Define the loss
        #     loss = ( y - x - W2(p)).norm()**2
        #     #loss = x.norm()**2  # This is ||x||^2
            
        #     # Add the constraint terms to the loss
        #     constraint_1 = lambda_1 * ( W1(x) - p + n).norm()**2
        #     loss += constraint_1
    
        #     constraint_2 = lambda_2 * (torch.matmul(n.view(n.size(0), -1), p.view(p.size(0), -1).T).squeeze())**2
        #     loss += constraint_2
            
        #     # Compute gradients
        #     loss.backward()
            
        #     # Update variables
        #     optimizer.step()
            
        #     # Print every 100 epochs
        #     if epoch % 100 == 0:
        #         print(f'Epoch {epoch}, Loss: {loss.item()}')
        
        # print(f'Solution x: {x.detach().numpy()}')
        
        # print(torch.norm(x - res_block_in))
        
        
        
        
        #optimal_x, optimal_p, optimal_n,execution_time = solve_optimization_problem_resnets(W1, W2,res_block_in,res_block_out)
        
        
        # res_block3_weights = resnet.res_block3.state_dict()
        # W1= res_block3_weights['conv1.weight']
        # W2 = res_block3_weights['conv2.weight']
        
        # res_block_in =  features['res_block_in_3'] 
        # res_block_out = optimal_x #features['res_block_in_4'] 
        
        # optimal_x, optimal_p, optimal_n,execution_time = solve_optimization_problem_resnets(W1, W2,res_block_in,res_block_out)
        
        
        # res_block2_weights = resnet.res_block2.state_dict()
        # W1= res_block2_weights['conv1.weight']
        # W2 = res_block2_weights['conv2.weight']
        
        # res_block_in =  features['res_block_in_2'] 
        # res_block_out =  optimal_x #features['res_block_in_3'] 
        
        # optimal_x, optimal_p, optimal_n,execution_time = solve_optimization_problem_resnets(W1, W2,res_block_in,res_block_out)
        
        
        # res_block1_weights = resnet.res_block1.state_dict()
        # W1= res_block1_weights['conv1.weight']
        # W2 = res_block1_weights['conv2.weight']
        
        # res_block_in =  features['res_block_in_1'] 
        # res_block_out =  optimal_x #features['res_block_in_2'] 
        
        # optimal_x, optimal_p, optimal_n,execution_time = solve_optimization_problem_resnets(W1, W2,res_block_in,res_block_out)
    
    
    
        ####################################################################################
        
    
    
        # print("Optimal x:", optimal_x)
        # print("Optimal p:", optimal_p)
        # print("Optimal n:", optimal_n)
        # print("Device :", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # print("Execution Time:", execution_time, "seconds")
    
    
        ################ Reconstructed Res blocks #########3
        
        optimal_x = resblock_recovered
    
    
        # Assuming image_tensor is your multi-channel image tensor
        image_tensor = optimal_x.detach().numpy()
    
        # Visualize individual channels as grayscale images
        num_channels = 5
        fig, axes = plt.subplots(1, num_channels, figsize=(12, 4))
    
        for i in range(num_channels):
            channel_image = image_tensor[:,i, :, :]
            print(channel_image.squeeze(0).shape)
            axes[i].imshow(channel_image.squeeze(0), cmap='plasma')
            axes[i].set_title(f'Reconstructed_Ch {i + 1}')
    
        plt.show()
    
    
        # Assuming image_tensor is your multi-channel image tensor
        image_tensor = features['res_block_in_4']  #res_block_in
        
        # Visualize individual channels as grayscale images
        num_channels = 5 
        fig, axes = plt.subplots(1, num_channels, figsize=(12, 4))
        
        for i in range(num_channels):
            channel_image = image_tensor[:,i, :, :]
            print(channel_image.squeeze(0).shape)
            axes[i].imshow(channel_image.squeeze(0), cmap='plasma')
            axes[i].set_title(f'Orignal_Ch {i + 1}')
        
        plt.show()
    
         
        print("Seed number: ",seed )
        print("Error in reconstruction",torch.norm(features['res_block_in_4'] -  optimal_x.to('cpu')))
        recons_error = torch.norm(features['res_block_in_4'] -  optimal_x.to('cpu'))
    
    ################### Initial Layer ##############
    
    #Take the parameters bn and conv
    if(init_layer):
        y = optimal_x #features['conv']  #optimal_x #features['conv']  #optimal_x 
        x = features["input"]
        
        #import pdb;pdb.set_trace()
        
        conv1_params = extract_conv_details(resnet.conv1) 
        
        #print(conv1_params)
        orignal_image_rec = conv_inverted(y,x,conv1_params)
        #orignal_image_rec = invert(image = features['input'],network = resnet,size = 224,layer ='relu', activation = None)
        
        
        #orignal_image_rec = solve_optimization_problem_conv_layer(conv1_weights, y, x)
        #orignal_image_rec, optimal_p, optimal_n,execution_time =  solve_optimization_problem_conv_layer_2(conv1_weights, y, x)
        
        
        print("The error from the orignal image: ",torch.norm(orignal_image_rec - x))
        
    
        ####
        image = orignal_image_rec.detach().squeeze(0).permute(1, 2, 0).numpy()
        
        # Visualization using matplotlib
        plt.imshow(image)
        plt.axis('off')  # To hide axis values
        plt.show()
        
        
        image = features["input"].squeeze(0).permute(1, 2, 0).numpy()
        
        # Visualization using matplotlib
        plt.imshow(image)
        plt.axis('off')  # To hide axis values
        plt.show()
    return recons_error

def main():
    print("This is the main function!")
    seed = [0,1,2,3,4,5]
    reconstruction_error = []
    
    for s in seed:
        reconstruction_error.append(reconstruct(s))
    # Compute the mean
    mean_error = sum(reconstruction_error) / len(reconstruction_error)
    
    # Compute the standard deviation
    variance = sum([(e - mean_error) ** 2 for e in reconstruction_error]) / len(reconstruction_error)
    std_error = variance ** 0.5
    
    print(f"Mean Reconstruction Error: {mean_error}")
    print(f"Standard Deviation of Reconstruction Error: {std_error}")

        

if __name__ == '__main__':
    main()






from matplotlib.font_manager import X11FontDirectories
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
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
import numpy as np
from model import resnet18
from PIL import Image
from invert_functions import solve_optimization_problem_final_layer
from invert_functions import solve_optimization_problem_conv_layer
from invert_functions import solve_optimization_problem_resblock
from utilities import  get_weights_resblocks,get_weights_resblocks_noBN
from invert_functions import InverseLinear
from invert_functions import InverseAdaptiveAvgPool2d
from invert_functions import MaxUnpool2d
from utilities import plot_channels
########## Loading Dataset ############3

torch.manual_seed(8674529339715958261) #for 8 by 8
#current_seed = torch.initial_seed()
#print("Current seed:",  torch.seed() )


# Define transformations for data augmentation and normalization
batch_size = 64
learning_rate = 0.001
Scaling  = 8
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

#############################################33


######## model 

net = resnet18()

train = True

########################

# # Define loss function and optimizer

if(train == True):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    # # Training loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    
    for epoch in range(1):  # Adjust the number of epochs as needed
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad()
            features = net(inputs)
            outputs = features["output"]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
    
    print("Finished Training")


    # net.eval()
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, labels = data
    #         images, labels = images.to(device), labels.to(device)
    #         features = net(inputs)
    #         #fc_in,batch_in,batch_out,max_pool_in, max_pool_out, avg_pool_in, avg_pool_out, outputs= net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    
    # print(f"Accuracy on the test set: {100 * correct / total:.2f}%")


#########################3

#### Doing Forward Pass on one image
net.eval()
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1,shuffle=False)

with torch.no_grad():
    for images, labels in test_loader:
        features = net(images)
        break
    
    
    
    
    
    
    
####################### Manual Image ##############3


# image_path = '/home/arifh/Desktop/IBM_2023/CAT.jpeg'

# image = Image.open(image_path).convert('RGB')


# # Assuming you've resized the image using PIL
# resized_image = image.resize((8, 8))

# # Convert to tensor
# to_tensor = transforms.ToTensor()
# image_tensor = to_tensor(resized_image)


# input_image = image_tensor.unsqueeze(0)

# # # Ensure the model is in evaluation mode
# net.eval()

# # # Perform a forward pass
# with torch.no_grad():
#     features = net(input_image)
    
    
# Display the processed image

# img_tensor = image_tensor * 0.5 + 0.5  # unnormalize
# #print(img_tensor.shape)
# p_image = img_tensor.detach().cpu().numpy()  # Convert tensor to numpy array
# plt.imshow(np.transpose(p_image, (1, 2, 0)))  # Transpose the image tensor (C, H, W) -> (H, W, C)
# plt.show()


##################### All the features ###############33

for key, value in features.items():
    print(f"{key} shape: {value.shape}")
    
    
    
########################### Final Layer ###############3


# W1 = net.fc.weight.data.clone()
# bias = net.fc.bias.data.clone()

# ## Approach 1
# # x_size = features["fc_in"].shape[1]
# # y_size = features["output"].shape[1]

# # # Define the linear layer
# # fc = nn.Linear(x_size,y_size)
# # inverse_fc = InverseLinear(fc)

# # # Try to recover x
# # fc_recovered = inverse_fc(features["output"]).T

# # print(torch.norm(fc_recovered-features["fc_in"]))


# #print("Recovered input tensor shape:", recovered_input.shape)
# #torch.norm(recovered_input - fc_in)
# # plt.imshow(fc_recovered.cpu(), cmap='viridis', aspect='auto')
# # plt.colorbar()
# # plt.show()
# # plt.imshow(features["fc_in"].cpu(), cmap='viridis', aspect='auto')
# # plt.colorbar()
# # plt.show()


# ### Approach 2

# fc_recovered= solve_optimization_problem_final_layer(W1_weights=W1,bias = bias, y=features["output"], input=features["fc_in"])
# #fc_loss = torch.norm(fc_recovered - features["fc_in"])
# #print("Leakage in first layer:", fc_loss)

# #import pdb;pdb.set_trace()

# ######################## Average Pooling #######33

# x = features["res_4"]
# y = fc_recovered.unsqueeze(-1).unsqueeze(-1)#features["avg_pool_out"] ## 

# inverse_pool = InverseAdaptiveAvgPool2d(x[0][0].size())

# # # Recover input
# adaptive_recovered = inverse_pool(y)

#print(x.shape)
#print(adaptive_recovered.shape)
res_block = True
lastlayer = False
maxpool = False
###################### Res Blocks Inversion ###########
if(res_block):
    residual_block_weights = get_weights_resblocks_noBN(net)
    
    
    
    ## Block 4
    
    W1= residual_block_weights["residual_block_4"]["conv1_weights"]
    W2 = residual_block_weights["residual_block_4"]["conv2_weights"]
   # BN1 = residual_block_weights["residual_block_4"]["conv1_bn_weights"]
   # BN2 = residual_block_weights["residual_block_4"]["conv2_bn_weights"]
    
    res_out = features['res_4'] #adaptive_recovered ##features['res_4'] ##
    
    res_in = features['res_3']
   # optimal_b4, optimal_p, optimal_n,execution_time = solve_optimization_problem_resblock(W1, W2,BN1,BN2,res_in,res_out)
    optimal_b4, optimal_p, optimal_n,execution_time = solve_optimization_problem_resblock(W1, W2,None,None,res_in,res_out)
    
    print("Block 4",execution_time)
    print("Leakage in the fourth block",torch.norm(res_in - optimal_b4))
    
    
    
    ### Block 3
    
    res_out = optimal_b4
    res_in = features['res_2']
    
    W1= residual_block_weights["residual_block_3"]["conv1_weights"]
    W2 = residual_block_weights["residual_block_3"]["conv2_weights"]
   # BN1 = residual_block_weights["residual_block_3"]["conv1_bn_weights"]
    #BN2 = residual_block_weights["residual_block_3"]["conv2_bn_weights"]
    
    
    #optimal_b3, optimal_p, optimal_n,execution_time = solve_optimization_problem_resblock(W1, W2,BN1,BN2,res_in,res_out)
    optimal_b3, optimal_p, optimal_n,execution_time = solve_optimization_problem_resblock(W1, W2,None,None,res_in,res_out)
    
    print("Block 3",execution_time)
    print("Leakage in the third block",torch.norm(res_in - optimal_b3))
    
    
    ### Block 2
    
    res_out = optimal_b3
    res_in = features['res_1']
    
    W1= residual_block_weights["residual_block_2"]["conv1_weights"]
    W2 = residual_block_weights["residual_block_2"]["conv2_weights"]
   # BN1 = residual_block_weights["residual_block_2"]["conv1_bn_weights"]
   # BN2 = residual_block_weights["residual_block_2"]["conv2_bn_weights"]
    
    
   # optimal_b2, optimal_p, optimal_n,execution_time = solve_optimization_problem_resblock(W1, W2,BN1,BN2,res_in,res_out)
    optimal_b2, optimal_p, optimal_n,execution_time = solve_optimization_problem_resblock(W1, W2,None,None,res_in,res_out)
    print("Leakage in the second block",torch.norm(res_in - optimal_b2))
    print("Block 2",execution_time)
    
    
    
    
    # ### Block 1
    
    res_out = optimal_b2
    res_in = features['max_pool_out']
    
    W1= residual_block_weights["residual_block_1"]["conv1_weights"]
    W2 = residual_block_weights["residual_block_1"]["conv2_weights"]
    #BN1 = residual_block_weights["residual_block_1"]["conv1_bn_weights"]
   # BN2 = residual_block_weights["residual_block_1"]["conv2_bn_weights"]
    
    
   # optimal_b1, optimal_p, optimal_n,execution_time = solve_optimization_problem_resblock(W1, W2,BN1,BN2,res_in,res_out)
    optimal_b1, optimal_p, optimal_n,execution_time = solve_optimization_problem_resblock(W1, W2,None,None,res_in,res_out)
    print("Leakage in the second block",torch.norm(res_in - optimal_b1))
    print("Block 1",execution_time)
    
    saved_tensors = {
        'optimal_b1': optimal_b1,
        'optimal_b2': optimal_b2,
        'optimal_b3': optimal_b3,
        'optimal_b4': optimal_b4,
        'max_pool_out': features['max_pool_out'],
        'res_4': features['res_4'],
        'res_3': features['res_3'],
        'res_2': features['res_2'],
        'res_1': features['res_1']
    }
    
    torch.save(saved_tensors, '/home/arifh/Desktop/IBM_2023/saved_tensors.pth')


############ Plotting Results ##############3

#print("Leakage in the first block",torch.norm(res_in - optimal_x))


#plot_channels(features['max_pool_out'],rec = False)
#plot_channels(optimal_x,rec = True)

# ############### Max Pooling Layer #############33

if(maxpool):

    unpool = MaxUnpool2d()
    indices = features['max_pool_indics']
    indices = indices.long()
    y = features["max_pool_out"].float() #optimal_x.float() #features["max_pool_out"].float()
    x = features["max_pool_in"]
    # Recover input
    maxpool_recovered = unpool(y, indices, x.size())
    
    
    
    
    # # print(x)
    # # print(y)
    # # print(maxpool_recovered)
    
    #print(torch.norm(maxpool_recovered -x))


# ############ First Layer ###############

#### Reversing the first conv_relu layer

#Take the parameters bn and conv
if(lastlayer):
    conv1_weights = net.conv1.weight.clone()
    
    
    
    # conv1_bn_weights = {
    #             "weight": net.bn1.weight.data.clone(),
    #             "bias": net.bn1.bias.data.clone(),
    #             "running_mean": net.bn1.running_mean.clone(),
    #             "running_var": net.bn1.running_var.clone()
    #         }
    ## feed it through
    
    if(maxpool):
        y = maxpool_recovered
    elif (res_block):
        y = optimal_b1
    else:
        y = features["max_pool_in"] #features["max_pool_in"]  #maxpool_recovered ## features["max_pool_in"] 
    
    x = features["input"]
    
    #print(y.shape)
    #print(input.shape)
    
    
    orignal_image_rec = solve_optimization_problem_conv_layer(conv1_weights, y, x)
    
    
    print("The error from the orignal image: ",torch.norm(orignal_image_rec - x))
    
    
    
    
    
    ####
    image = orignal_image_rec.squeeze(0).permute(1, 2, 0).numpy()
    
    # Visualization using matplotlib
    plt.imshow(image)
    plt.axis('off')  # To hide axis values
    plt.show()
    
    
    image = features["input"].squeeze(0).permute(1, 2, 0).numpy()
    
    # Visualization using matplotlib
    plt.imshow(image)
    plt.axis('off')  # To hide axis values
    plt.show()

# #### 






from invert_functions import *
import matplotlib.pyplot as plt
import re
def get_weights_resblocks(net):
    residual_block_weights = {
    "residual_block_1": {
        "conv1_weights": net.layer1[0].conv1.weight.data.clone(),
        "conv1_bn_weights": {
            "weight": net.layer1[0].bn1.weight.data.clone(),
            "bias": net.layer1[0].bn1.bias.data.clone(),
            "running_mean": net.layer1[0].bn1.running_mean.clone(),
            "running_var": net.layer1[0].bn1.running_var.clone()
        },
        "conv2_weights": net.layer1[0].conv2.weight.data.clone(),
        "conv2_bn_weights": {
            "weight": net.layer1[0].bn2.weight.data.clone(),
            "bias": net.layer1[0].bn2.bias.data.clone(),
            "running_mean": net.layer1[0].bn2.running_mean.clone(),
            "running_var": net.layer1[0].bn2.running_var.clone()
        },
    },
    "residual_block_2": {
        "conv1_weights": net.layer2[0].conv1.weight.data.clone(),
        "conv1_bn_weights": {
            "weight": net.layer2[0].bn1.weight.data.clone(),
            "bias": net.layer2[0].bn1.bias.data.clone(),
            "running_mean": net.layer2[0].bn1.running_mean.clone(),
            "running_var": net.layer2[0].bn1.running_var.clone()
        },
        "conv2_weights": net.layer2[0].conv2.weight.data.clone(),
        "conv2_bn_weights": {
            "weight": net.layer2[0].bn2.weight.data.clone(),
            "bias": net.layer2[0].bn2.bias.data.clone(),
            "running_mean": net.layer2[0].bn2.running_mean.clone(),
            "running_var": net.layer2[0].bn2.running_var.clone()
        },
    },
    "residual_block_3": {
        "conv1_weights": net.layer3[0].conv1.weight.data.clone(),
        "conv1_bn_weights": {
            "weight": net.layer3[0].bn1.weight.data.clone(),
            "bias": net.layer3[0].bn1.bias.data.clone(),
            "running_mean": net.layer3[0].bn1.running_mean.clone(),
            "running_var": net.layer3[0].bn1.running_var.clone()
        },
        "conv2_weights": net.layer3[0].conv2.weight.data.clone(),
        "conv2_bn_weights": {
            "weight": net.layer3[0].bn2.weight.data.clone(),
            "bias": net.layer3[0].bn2.bias.data.clone(),
            "running_mean": net.layer3[0].bn2.running_mean.clone(),
            "running_var": net.layer3[0].bn2.running_var.clone()
        },
    },
    "residual_block_4": {
        "conv1_weights": net.layer4[0].conv1.weight.data.clone(),
        "conv1_bn_weights": {
            "weight": net.layer4[0].bn1.weight.data.clone(),
            "bias": net.layer4[0].bn1.bias.data.clone(),
            "running_mean": net.layer4[0].bn1.running_mean.clone(),
            "running_var": net.layer4[0].bn1.running_var.clone()
        },
        "conv2_weights": net.layer4[0].conv2.weight.data.clone(),
        "conv2_bn_weights": {
            "weight": net.layer4[0].bn2.weight.data.clone(),
            "bias": net.layer4[0].bn2.bias.data.clone(),
            "running_mean": net.layer4[0].bn2.running_mean.clone(),
            "running_var": net.layer4[0].bn2.running_var.clone()
        },
    }
}
    
    return  residual_block_weights

def get_weights_resblocks_noBN(net):
    residual_block_weights = {
        "residual_block_1": {
            "conv1_weights": net.layer1[0].conv1.weight.data.clone(),
            "conv2_weights": net.layer1[0].conv2.weight.data.clone(),
        },
        "residual_block_2": {
            "conv1_weights": net.layer2[0].conv1.weight.data.clone(),
            "conv2_weights": net.layer2[0].conv2.weight.data.clone(),
        },
        "residual_block_3": {
            "conv1_weights": net.layer3[0].conv1.weight.data.clone(),
            "conv2_weights": net.layer3[0].conv2.weight.data.clone(),
        },
        "residual_block_4": {
            "conv1_weights": net.layer4[0].conv1.weight.data.clone(),
            "conv2_weights": net.layer4[0].conv2.weight.data.clone(),
        }
    }
    
    return residual_block_weights

def extract_number(s):
    result = re.search(r'\d+', s)
    return int(result.group()) if result else None

def plot_channels(x,rec = False):
    # Assuming image_tensor is your multi-channel image tensor
    image_tensor = x.to('cpu')

    # Visualize individual channels as grayscale images
    num_channels = 5 #image_tensor.shape[1]
    fig, axes = plt.subplots(1, num_channels, figsize=(12, 4))

    for i in range(num_channels):
        channel_image = image_tensor[:,i, :, :]
        print(channel_image.squeeze(0).shape)
        axes[i].imshow(channel_image.squeeze(0), cmap='plasma')
        if(rec == False):
            axes[i].set_title(f'Orignal_Ch {i + 1}')
        else:
            axes[i].set_title(f'Reconstructed_Ch {i + 1}')

    plt.show()

def total_variation_4d(x,beta =1):
    # Compute differences along width (W) and height (H) dimensions
    diff_w = x[:, :, 1:, :-1] - x[:, :, :-1, :-1]
    diff_h = x[:, :, :-1, 1:] - x[:, :, :-1, :-1]

    # Sum over the square root of the sum of squares of differences
    tv = torch.sum((diff_w**2 + diff_h**2)**(beta/2))

    return tv

def tensor_p_norm_powered(tensor, alpha=2):
    """Compute the p-norm of a tensor raised to the power p."""
    # Flatten the tensor
    tensor_flat = tensor.view(-1)
    # Compute the sum of absolute values raised to the power p
    return torch.sum(torch.abs(tensor_flat) ** alpha)




def extract_conv_details(conv):
    """Helper function to extract convolution details."""
    details = {
        'weight': conv.weight.clone(),
        'in_channels': conv.in_channels,
        'out_channels': conv.out_channels,
        'kernel_size': conv.kernel_size,
        'stride': conv.stride,
        'padding': conv.padding,
        'dilation': conv.dilation,
        'groups': conv.groups,
        'bias': bool(conv.bias)
    }
    return details

def extract_maxpool_details(maxpool):
    """Helper function to extract MaxPool2d details."""
    details = {
        'kernel_size': maxpool.kernel_size,
        'stride': maxpool.stride,
        'padding': maxpool.padding,
        'dilation': maxpool.dilation,
        'return_indices': maxpool.return_indices,
        'ceil_mode': maxpool.ceil_mode
    }
    return details



# def extract_block_info(block):
#     block_info = {
#         'conv1': extract_conv_details(block.conv1),
#         'conv2': extract_conv_details(block.conv2)
#     }
    
#     # Check for downsample and extract its details
#     # import pdb; pdb.set_trace()
    
    
    
#     if block.downsample:
        
#         import pdb;pdb.set_trace()
#         if isinstance(block.downsample, nn.Conv2d):
#             downsample_conv = block.downsample[0]
#             block_info['downsample'] = extract_conv_details(downsample_conv)
#         elif isinstance(block.downsample, nn.MaxPool2d):
#             block_info['downsample'] = extract_maxpool_details(block.downsample)
#             print(block_info['downsample'])
    
        
    return block_info
def extract_prelu_details(prelu_layer):
    # Extract PReLU details (number of channels)
    prelu_details = {
        'num_parameters': prelu_layer.num_parameters,
        'weight':prelu_layer.weight
        #'init': prelu_layer.init
    }
    return prelu_details

def extract_block_info(block):
    #import pdb;pdb.set_trace()
    block_info = {
        'conv1': extract_conv_details(block.conv1),
        #'prelu1': extract_prelu_details(block.prelu),  # Assuming prelu1 is the PReLU after conv1
        'conv2': extract_conv_details(block.conv2),
        
    }
    
    # Check for downsample and extract its details
    if block.downsample:
        downsample_conv = block.downsample[0]
        block_info['downsample'] = extract_conv_details(downsample_conv)
        
    return block_info

def extract_resnet_info(model):
    resnet_info = {}

    # Iterate through main layers of ResNet-18
    for layer_name, layer in model.named_children():
        print(layer_name)
        if "layer" in layer_name:  # Check if it's one of the main layers
            layer_info = {}
            
            # Iterate through blocks in the layer
            for i, block in enumerate(layer):
                block_name = f"block{i+1}"
                layer_info[block_name] = extract_block_info(block)

            resnet_info[layer_name] = layer_info
    #import pdb;pdb.set_trace()

    return resnet_info


def transfer_weights(model, pretrained_model):
    # Create a dictionary of the custom model's state dict
    model_dict = model.state_dict()

    # Filter out the keys from the pretrained model that are not in the custom model
    pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict}

    # Update the custom model's dict
    model_dict.update(pretrained_dict)

    # Load the updated dict into the custom model
    model.load_state_dict(model_dict)

    return model

def plot_image(image, save_path=None):
    """
    Plots and optionally saves the given image.

    Args:
        image (Tensor): The input image tensor.
        save_path (str, optional): The path to save the image. If None, the image is not saved.

    Returns:
        None
    """
    # Convert the tensor to a NumPy array and adjust dimensions and channels
    image = image.detach().squeeze(0).permute(1, 2, 0).numpy()

    # Visualization using matplotlib
    plt.imshow(image)
    plt.axis('off')  # To hide axis values
   # plt.show()

    if save_path:
        # Save the image to the specified path
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, format='png')
        plt.show()
    else:
        plt.show()
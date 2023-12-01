from ResNet18_func import ResNet18

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

def extract_block_info(block):
    block_info = {
        'conv1': extract_conv_details(block.conv1),
        'conv2': extract_conv_details(block.conv2)
    }
   # import pdb;pdb.set_trace()
    # Check for downsample and extract its details
    if block.downsample:
        downsample_conv = block.downsample[0]
        block_info['downsample'] = extract_conv_details(downsample_conv)
        
    return block_info

def extract_resnet_info(model):
    resnet_info = {}

    # Iterate through main layers of ResNet-18
    for layer_name, layer in model.named_children():
        if "layer" in layer_name:  # Check if it's one of the main layers
            layer_info = {}
            
            # Iterate through blocks in the layer
            for i, block in enumerate(layer):
                block_name = f"block{i+1}"
                layer_info[block_name] = extract_block_info(block)

            resnet_info[layer_name] = layer_info

    return resnet_info

# Load the pretrained ResNet-18 model
resnet= ResNet18()

# Extract information
resnet_info = extract_resnet_info(resnet)

import torch
import torch.nn as nn

############# Exctraction Code #######
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

def extract_prelu_details(prelu):
    """Helper function to extract PReLU details."""
    details = {
        'weight': prelu.weight.clone()
    }
    return details

def extract_bottleneckIR_info(block):
    """Extract details from bottleneck_IR block."""
    block_info = {}

    # Extract shortcut layer details
    shortcut = block.shortcut_layer[0] if isinstance(block.shortcut_layer, nn.Sequential) else block.shortcut_layer
    if isinstance(shortcut, nn.Conv2d):
        block_info['shortcut'] = extract_conv_details(shortcut)
    elif isinstance(shortcut, nn.MaxPool2d):
        #import pdb;pdb.set_trace()
        block_info['shortcut'] = extract_maxpool_details(shortcut)
    
    # Extract residual layer details
    res_layer = block.res_layer
    block_info['res_layer'] = {}
    for i, layer in enumerate(res_layer):
        if isinstance(layer, nn.Conv2d):
            block_info['res_layer'][f'conv{i+1}'] = extract_conv_details(layer)
        elif isinstance(layer, nn.PReLU):
            block_info['res_layer'][f'prelu{i+1}'] = extract_prelu_details(layer)

    return block_info

def extract_backbone64_info(model):
    """Extract details from Backbone64 model."""
    backbone_info = {}
    
    # Input layer details
    backbone_info['input_layer'] = {}
    for i, layer in enumerate(model.input_layer):
        if isinstance(layer, nn.Conv2d):
            backbone_info['input_layer'][f'conv{i+1}'] = extract_conv_details(layer)
        elif isinstance(layer, nn.PReLU):
            backbone_info['input_layer'][f'prelu'] = extract_prelu_details(layer)
    
    # Body layer details
    backbone_info['body'] = {}
    for i, block in enumerate(model.body):
        block_name = f"block{i+1}"
        backbone_info['body'][block_name] = extract_bottleneckIR_info(block)

    return backbone_info


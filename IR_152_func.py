import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple


# class bottleneck_IR(Module):
#     def __init__(self, in_channel, depth, stride):
#         super(bottleneck_IR, self).__init__()
#         self.activations = {}
        
#         if in_channel == depth:
#             self.shortcut_layer = MaxPool2d(1, stride)
#         else:
#             self.shortcut_layer = Sequential(
#                 Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
#         self.res_layer = Sequential(
#             BatchNorm2d(in_channel),
#             Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
#             Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))
 
#     def forward(self, x):
#         self.activations['input'] = x
#         shortcut = self.shortcut_layer(x)
#         res = self.res_layer(x)
#         self.activations['output'] = res + shortcut
#         return res + shortcut, self.activations
    
    
class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        self.activations = {}
        #print("In_channel:", in_channel)
       # print("Depth:", depth)
        
        if in_channel == depth:
             self.shortcut_layer = MaxPool2d(1, stride)
        else:
         self.shortcut_layer = Conv2d(in_channel, depth, (1, 1), stride, bias=False)
         
        self.res_layer = Sequential(
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False))
        
    def forward(self, x):
        self.activations['input'] = x
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        #print(res.shape)
        #print(x.shape)
        if(res.shape == x.shape):
            self.activations['output'] = res + x
        else:   
            self.activations['output'] = res + shortcut #x #shortcut
        return res + shortcut, self.activations
        
    
class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
 
 
def get_block(in_channel, depth, num_units, stride=2):
 
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]
 
 
def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
 
    return blocks
 
 
class Backbone64(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone64, self).__init__()
        self.activations = {}
        assert input_size[0] in [64, 112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            print("This module is not supported at the moment")
            
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      #BatchNorm2d(64),
                                      PReLU(64))
        
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
 
        self._initialize_weights()
 
    def forward(self, x):
        
        #x = self.input_layer(x)
        #x = self.body(x)
        #print("here")
        self.activations['input'] = x
        self.activations['output'] = self.input_layer(x)
        x = self.activations['output']
       # import pdb;pdb.set_trace()
        print('ConvLayer input:',self.activations['input'].shape)
        print('ConvLayer output:',self.activations['output'].shape)
        block_activations = []
        for i, module in enumerate(self.body):
            x, block_act = module(x)
            block_activations.append(block_act)
            print('Block input:',block_act['input'].shape)
            print('Block output:',block_act['output'].shape)
        self.activations['body'] = block_activations
       
        #x = self.output_layer(x)
 
        return x
 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    


def IR_152_64(input_size):
    """Constructs a ir-152 model.
    """
    model = Backbone64(input_size, 152, 'ir')
 
    return model




    
# def main():
#     import pdb;pdb.set_trace()
#     model = IR_152_64((64,64))
    
    
    
# if __name__ == '__main__':
#     main()

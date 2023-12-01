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
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        #out = self.bn1(out)
        out_1  = out
        out = self.relu(out)
        out = self.conv2(out)
        #out = self.bn2(out)
        out += self.shortcut(residual)
        #out = self.relu(out)
        return out_1, out

# Define the ResNet class
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        #kernel_size=7, stride=2, padding=3, bias=False
        #self.conv1 = nn.Conv2d(3, 5, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.res_block1 = ResidualBlock(64, 64)
        self.res_block2 = ResidualBlock(64,64)
        self.res_block3 = ResidualBlock(64, 64)
        self.res_block4 = ResidualBlock(64,64)
        
        
        self.avg = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(4096, num_classes) # 4096

    def forward(self, x):
        features={}
        
        features['input'] = x
        out = self.conv1(x)
        out = self.relu(out)
        features['conv'] = out
        
        res_block_in = out
        features['res_block_in_1']  = res_block_in
        out1,out = self.res_block1(out)
        features['res_block_in_2']  = out
        res_block_out = out
        out2,out = self.res_block2(out)
        features['res_block_in_3']  = out
        out2,out = self.res_block3(out)
        features['res_block_in_4']  = out
        out2,out = self.res_block4(out)
        features['res_block_out_4']  = out
        
        
        pool_in = out
        features['avg_pool_in'] = pool_in
        #out = nn.functional.avg_pool2d(out, out.size()[1])
        out = self.avg(pool_in)
        #out = nn.functional.avg_pool2d(out,kernel_size=1, stride=1)
        features['avg_pool_out'] = out
        #import pdb;pdb.set_trace()
        pool_out =out
        out = out.view(out.size(0), -1)
        features['fc_in'] = out 
       # import pdb;pdb.set_trace()
        out = self.fc(out)
        features['output'] = out
        return features,out


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
from torch.nn import MaxUnpool2d
# Define the basic building blocks: BasicBlock and Bottleneck


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
       # out = self.bn2(out)
        out += identity
        #out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 5 # 64


         ## x   ###
        self.conv1 = nn.Conv2d(3, 5, kernel_size=3, stride=1, padding = 1, bias=False)  ### nn.Conv2d(3, 5, kernel_size=7, stride=2, padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(5)
        self.relu = nn.ReLU(inplace=True)

        ## y
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding= 0 ,return_indices=True)

        #import pdb;pdb.set_trace()
        self.layer1 = self._make_layer(block, 5, num_blocks[0]) ## Run experiments on batch size 1 with all the channels ? --> B * 64 * 112 * 112
        self.layer2 = self._make_layer(block, 5, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 5, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 5, num_blocks[3], stride=1)

        # Input to the average pooling :  64 X 512 X n X n (N x C X H X W) -->

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #--> upsampling  method specify

        # 64 * 512 * 1 * 1

        print(block.expansion)
        self.fc = nn.Linear(5 * block.expansion, num_classes)




    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = {} # Dictionary to store all intermediate outputs

        features["input"] = x
        out = self.conv1(x)
        #features["batch_in"] = out
        #out = self.bn1(out)
       # features["batch_out"] = out
        out = self.relu(out)
        features["max_pool_in"] = out
        max_pool_ind,out = self.maxpool(out)

        #import pdb;pdb.set_trace()

        features['max_pool_indics'] = max_pool_ind
        #features['max_pool_indics'].float()
        features["max_pool_out"] = out

        out = self.layer1(out.float())
        features["res_1"] = out
        out = self.layer2(out)
        features["res_2"] = out
        out = self.layer3(out)
        features["res_3"] = out
        out = self.layer4(out)
        features["res_4"] = out
        out = self.avgpool(out)
        features["avg_pool_out"] = out
        out = out.view(out.size(0), -1)
        features["fc_in"] = out
        out = self.fc(out)  #64 * 512 ---> 64 * 1000
        features["output"] = out

        return features
# Create a ResNet-18 model
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])



# You can then train the model and use it for various tasks like image classification.
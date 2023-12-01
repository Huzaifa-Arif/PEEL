import torch
import torch.nn as nn
import torchvision.models as models
from utilities import transfer_weights

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.activations = {}
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.downsample= nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        self.activations['input'] = x
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.downsample(x)
        self.activations['output'] = out
        return out, self.activations

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.activations = {}
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.conv1(x))
        #x = self.maxpool(x)
        self.activations['after_conv1'] = x
        x, self.activations['layer1'] = self._forward_layer(x, self.layer1)
        x, self.activations['layer2'] = self._forward_layer(x, self.layer2)
        x, self.activations['layer3'] = self._forward_layer(x, self.layer3)
        x, self.activations['layer4'] = self._forward_layer(x, self.layer4)
        x = self.avgpool(x)
        #import pdb;pdb.set_trace()
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _forward_layer(self, x, layer):
        block_activations = []
        for block in layer:
            x, block_act = block(x)
            block_activations.append(block_act)
        return x, block_activations

def ResNet18(pretrained = False,resblock_length = 18):
    
    if(resblock_length == 18):
        model = ResNet(BasicBlock, [2, 2, 2, 2])
    
    if(resblock_length == 34):
          model = ResNet(BasicBlock, [3,4,6,3])
          #model = ResNet(BasicBlock, [4, 4, 4, 4])
    
    if(resblock_length == 50):
        model = ResNet(BasicBlock, [6, 8, 12, 6])
    
    if(resblock_length == 152):
        model = ResNet(BasicBlock, [3, 8, 36, 3])
    
    if(pretrained == True):
        # Load the pretrained ResNet18 model
        pretrained_resnet18 = models.resnet18(pretrained=True)

        # Instantiate your custom ResNet18 model
        #custom_resnet18 = ResNet18()

        # Transfer weights from pretrained model to custom model
        model= transfer_weights(model, pretrained_resnet18)
    
    
    
    return model









# # Example
model = ResNet18(pretrained = True)
# output = model(input_tensor)
# print(model.activations['after_conv1'])  # Activations after the initial convolution
# print(model.activations['layer1'][0]['input'])  # Input activations of the first block
# print(model)

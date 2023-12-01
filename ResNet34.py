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
        print(x.shape)
        x = torch.relu(self.conv1(x))
        print(x.shape)
        x = self.maxpool(x)
        print(x.shape)
        self.activations['after_conv1'] = x
        x, self.activations['layer1'] = self._forward_layer(x, self.layer1)
        print(x.shape)
        x, self.activations['layer2'] = self._forward_layer(x, self.layer2)
        print(x.shape)
        x, self.activations['layer3'] = self._forward_layer(x, self.layer3)
        print(x.shape)
        x, self.activations['layer4'] = self._forward_layer(x, self.layer4)
        print(x.shape)
        x = self.avgpool(x)
        print(x.shape)
        #import pdb;pdb.set_trace()
        x = torch.flatten(x, 1)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        return x

    def _forward_layer(self, x, layer):
        block_activations = []
        for block in layer:
            x, block_act = block(x)
            block_activations.append(block_act)
        return x, block_activations

def ResNet34(pretrained = False):
    
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    
    if(pretrained == True):
        # Load the pretrained ResNet18 model
        pretrained_resnet34 = models.resnet34(pretrained=True)

        # Instantiate your custom ResNet18 model
        #custom_resnet18 = ResNet18()

        # Transfer weights from pretrained model to custom model
        model= transfer_weights(model, pretrained_resnet34)
    
    
    
    return model

# Example
model = ResNet34()
x = torch.randn([1,3,224,224])
print(model(x))
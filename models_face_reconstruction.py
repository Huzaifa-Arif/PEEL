import torch
import torch.nn as nn
import torchvision.models as models
from utilities import transfer_weights
from torch.nn import  MaxPool2d

class BasicBlock(nn.Module):
    expansion = 1 # potential change

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.activations = {}
        
        # potential change
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
        # import pdb;pdb.set_trace()
        out += self.downsample(x)
        self.activations['output'] = out
        return out, self.activations
    
class BasicBlock_maxpool(nn.Module):
     expansion = 1 # potential change

     def __init__(self, in_planes, planes, stride=2):
         super(BasicBlock_maxpool, self).__init__()
         self.activations = {}
         
         # potential change
         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        
         
         self.downsample= nn.Sequential()
         if stride != 1 or in_planes != self.expansion*planes:
             self.downsample = nn.Sequential(
                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
             )
         
         else :
             self.downsample = nn.Sequential(MaxPool2d(1, stride))
             #print("Maxpool")

             

     def forward(self, x):
         self.activations['input'] = x
         out = torch.relu(self.conv1(x))
         out = self.conv2(out)
        # import pdb;pdb.set_trace()
         out += self.downsample(x)
         self.activations['output'] = out
         return out, self.activations
    



class IR_152(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(IR_152, self).__init__()
        self.in_planes = 64
        self.activations = {}
        
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
       


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        self.activations['input'] = x
      #  print("Input Shape:",x.shape)
        # First Convolution
        x = torch.relu(self.conv1(x))
       # print("After Conv1:", x.shape)
        self.activations['after_conv1'] = x
        
        # Layer 1
        x, self.activations['layer1'] = self._forward_layer(x, self.layer1)
        #print("After Layer 1:", x.shape)
        
        # Layer 2
        x, self.activations['layer2'] = self._forward_layer(x, self.layer2)
        #print("After Layer 2:", x.shape)
        
        # Layer 3
        x, self.activations['layer3'] = self._forward_layer(x, self.layer3)
        #print("After Layer 3:", x.shape)
        
        # Layer 4
        x, self.activations['layer4'] = self._forward_layer(x, self.layer4)
        #print("After Layer 4:", x.shape)
    
        return x



    def _forward_layer(self, x, layer):
        block_activations = []
        for block in layer:
            x, block_act = block(x)
            block_activations.append(block_act)
        return x, block_activations

def IR_64(pretrained = False): 
    model = IR_152(BasicBlock, [3, 8, 36, 3])
    
    
    # if(pretrained == True):
    #     # Load the pretrained ResNet18 model
    #     pretrained_resnet18 = models.resnet18(pretrained=True)

    #     # Instantiate your custom ResNet18 model
    #     #custom_resnet18 = ResNet18()

    #     # Transfer weights from pretrained model to custom model
    #     model= transfer_weights(model, pretrained_resnet18)
    
    
    
    return model




######## Complete model #####

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class IR152(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR152, self).__init__()
        self.feature = IR_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 *4 * 4, 512)  # Generalize the input size calculation
            )
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        #print(x.shape)
        feat = self.feature(x)
        
        print(feat.shape)
        #import pdb;pdb.set_trace()
        feat = self.output_layer(feat)
        print(feat.shape)
        feat = feat.view(feat.size(0), -1)
        print(feat.shape)
        out = self.fc_layer(feat)
        print(out.shape)
        return feat, out
    def calculate_flatten_size(self, input_shape):
     # Calculate the size needed for the Flatten layer based on the input shape
     size = 1
     for dim in input_shape:
         size *= dim
     return size


class EvolveFace(nn.Module):
    def __init__(self, num_of_classes=1000, IR152=True):
        super(EvolveFace, self).__init__()
        if IR152:
            model = IR_64((64,64))
        else:
            pass
        self.model = model
        self.feat_dim = 512
        self.num_classes = num_of_classes
        self.output_layer = nn.Sequential(
                                     
                                        Flatten(),
                                        nn.Linear(512 *14 * 14, 512),
                                       )

        self.fc_layer = nn.Sequential(
            nn.Linear(self.feat_dim, self.num_classes),)


    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden

    def forward(self,x):
        print(x.shape)
        feature = self.model(x)
        print(feature.shape)
        feature = self.output_layer(feature)
        print(feature.shape)
        feature = feature.view(feature.size(0), -1)
        print(feature.shape)
        out, iden = self.classifier(feature)
        print(out.shape)
        print(iden.shape)

        return  out




# # Example
#model = IR_152(pretrained = True)
#model = IR_152()

# output = model(input_tensor)
# print(model.activations['after_conv1'])  # Activations after the initial convolution
# print(model.activations['layer1'][0]['input'])  # Input activations of the first block
# print(model)

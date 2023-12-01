import torch
import torch.nn as nn
import torchvision.models as models
from utilities import transfer_weights
from torch.nn import  MaxPool2d




class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    
    
class BasicBlock(nn.Module):
    expansion = 1 # potential change

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.activations = {}

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.downsample= nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        self.activations['input'] = x
        out = self.prelu(self.conv1(x))
        out = self.conv2(out)
        out += self.downsample(x)
        self.activations['output'] = out
        return out, self.activations
    



class IR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(IR, self).__init__()
        self.in_planes = 64
        self.activations = {}
        
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu1 = nn.PReLU(self.in_planes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
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
       # print("Input Shape:",x.shape)
        # First Convolution
        #out = torch. torch.nn.PReLU(self.conv1(x))
        #prelu = prelu = torch.nn.PReLU().to(x.device) # Create a PReLU activation instance
        x = self.prelu1(self.conv1(x))   # Apply PReLU activation to the output of conv1

       # x = torch.nn.PReLU(self.conv1(x))
        #x = torch.relu(self.conv1(x))
        #print("After Conv1:", x.shape)
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

def IR_64(backbone = 'IR152'):
    
    if(backbone == 'IR152'):
        model = IR(BasicBlock, [3, 8, 36, 3])
    else:
        model = IR(BasicBlock, [3, 4, 14, 3])
        
    return model





class IR152(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR152, self).__init__()
        self.feature = IR_64(backbone='IR152')
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 *14* 14, 512)  # Generalize the input size calculation
            )
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        #print(x.shape)
        self.activations = {}
        feat = self.feature(x)
        
        #print(feat.shape)
        self.activations['feature'] = feat
        #import pdb;pdb.set_trace()
        #print(feat.shape)
        feat = self.output_layer(feat)
        self.activations ['output'] = feat
        #print(feat.shape)
        feat = feat.view(feat.size(0), -1)
        #print(feat.shape)
        out = self.fc_layer(feat)
        self.activations ['fc'] =  out
        #print(out.shape)
        return feat, out
    def load_pretrained_weights(self, checkpoint_path):
       # Load the pre-trained weights from the checkpoint, if available
       try:
           checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
         #  import pdb;pdb.set_trace()
           self.load_state_dict(checkpoint['model_state_dict'])
           print("Pretrained weights loaded successfully from", checkpoint_path)
       except FileNotFoundError:
           print(f"Checkpoint file {checkpoint_path} not found.")
       except KeyError as e:
           print(f"Key error in the checkpoint file: {e}")
       except Exception as e:
           print("An error occurred while loading the weights:", e)



class FaceNet64(nn.Module):
    def __init__(self, num_classes=1000):
        super(FaceNet64, self).__init__()
        self.feature = IR_64(backbone='IR50')
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 *14* 14, 512)  # Generalize the input size calculation
            )
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        #print(x.shape)
        self.activations = {}
        feat = self.feature(x)
        
        #print(feat.shape)
        self.activations['feature'] = feat
        #import pdb;pdb.set_trace()
        print(feat.shape)
        feat = self.output_layer(feat)
        self.activations ['output'] = feat
        print(feat.shape)
        feat = feat.view(feat.size(0), -1)
        print(feat.shape)
        out = self.fc_layer(feat)
        self.activations ['fc'] =  out
        print(out.shape)
        return feat, out
    def load_pretrained_weights(self, checkpoint_path):
       # Load the pre-trained weights from the checkpoint, if available
       try:
           checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
         #  import pdb;pdb.set_trace()
           self.load_state_dict(checkpoint['model_state_dict'])
           print("Pretrained weights loaded successfully from", checkpoint_path)
       except FileNotFoundError:
           print(f"Checkpoint file {checkpoint_path} not found.")
       except KeyError as e:
           print(f"Key error in the checkpoint file: {e}")
       except Exception as e:
           print("An error occurred while loading the weights:", e)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
# Define the ResidualBlock class
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
kernels = 8
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, kernels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(kernels)
        self.relu = nn.ReLU(inplace=True)
        self.res_block1 = ResidualBlock(kernels, kernels)
        self.res_block2 = ResidualBlock(kernels, kernels)
        self.fc = nn.Linear(kernels, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        res_block_in = out
        out1,out = self.res_block1(out)
        res_block_out = out
        out2,out = self.res_block2(out)
        out = nn.functional.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return res_block_in,res_block_out,out1,out2,out

def linearizing_data(X):
    
    # Reshape the tensor to a vector
    vec_x = X.reshape(-1)
    print("Vectorized Tensor Shape:", vec_x.shape)
    return vec_x
def plot_color_coded_matrix(matrix):
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title("Color-coded Matrix")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()


def convolution_matrix(K, M, D, l,n,W1):
   

    # Create vector kernel with non-zero values
    #kernel = torch.arange(1, l)  # Length l - 1
    #import pdb;pdb.set_trace()
    # Initialize the large matrix with zeros
    large_matrix = torch.zeros(K * n, M * D)
    #print(large_matrix.shape)
    #exit()

    K_idx =  0
    # Populate the large matrix with block matrices
    for i in range(0, K * n, n):
        M_idx = 0
        for j in range(0, M * D, D):
            block_matrix = torch.zeros(n, D)
            #print(K_idx,M_idx)
            kernel = W1[K_idx,M_idx,:,:].reshape(l**2,1).squeeze(1).detach()
            M_idx+=1
            for k in range(n):
                if(k+len(kernel) < D):
                    block_matrix[k, k:k + len(kernel)] = kernel
            large_matrix[i:i + n, j:j + D] = block_matrix 
        K_idx+=1
    return large_matrix


import torch
import torch.nn.functional as F

def calculate_psnr(original, reconstructed):
    # Calculate the Mean Squared Error (MSE)
    mse = F.mse_loss(original, reconstructed)
    
    # Calculate the maximum pixel value (assuming 8-bit images)
    max_pixel_value = 255.0
    
    # Calculate the PSNR
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    
    return psnr.item()



class FP(nn.Module):
    def __init__(self, pretrained_weights = None):
        super(FP, self).__init__()
        # Define a 2D convolutional layer and initialize its weights with the pretrained weights
        self.conv_layer = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=1, padding=1, bias=False).double()
        if pretrained_weights is not None:  
            self.conv_layer.weight = nn.Parameter(pretrained_weights)
        
    def forward(self, x):
        conv_result = self.conv_layer(x)
        return conv_result


# Hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001
Scaling  = 8
# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((Scaling, Scaling)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)


# Initialize ResNet model
resnet = ResNet()

loading = False
if(loading):
    # Load the trained weights
    resnet.load_state_dict(torch.load('resnet.pth'))



# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=learning_rate)



#### Visualizing the images

images, labels = next(iter(train_loader))

# Plot the images
# plt.figure(figsize=(10, 10))
# for i in range(3):
#     plt.subplot(8, 8, i + 1)
#     plt.imshow(images[i].permute(1, 2, 0))  # Transpose to (H, W, C) format for Matplotlib
#     plt.axis('off')
#     plt.title(f"Label: {labels[i]}")
# plt.show()



# Test the model
resnet.eval()  # Set the model to evaluation mode
correct = 0
total = 0

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

with torch.no_grad():
    for images, labels in test_loader:
        res_block_in,res_block_out,out1,out2,outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")
### Doing Forward Pass on one image
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1,shuffle=False)

with torch.no_grad():
    for images, labels in test_loader:
        res_block_in,res_block_out,out1,out2,outputs = resnet(images)
        break
        
res_block1_weights = resnet.res_block1.state_dict()
W1= res_block1_weights['conv1.weight'] 
W2 = res_block1_weights['conv2.weight']



def solve_optimization_problem(W1_weights = None, W2_weights = None,res_block_in = None,res_block_out = None):
    
    input_tensor = res_block_in.double()
    y = res_block_out.double()
    if W1_weights is not None and W2_weights is not None:
        W1 = FP(pretrained_weights=W1_weights.double())
        W2 = FP(pretrained_weights=W2_weights.double())

    else:
        W1 = FP(pretrained_weights=None)
        W2 = FP(pretrained_weights=None)
    # Get the shape of the intermediate convolutional result
    p_shape = F.relu(W1(input_tensor)).shape

    # Define the optimization variables
    var_in = {"x": input_tensor.shape, "p": p_shape, "n": p_shape}

    def user_fn(X_struct):
        x = X_struct.x
        p = X_struct.p
        n = X_struct.n

        # Define the objective function
        f = torch.norm( y - x - W2(p))**2

        # Inequality constraints (element-wise)
        ci = pygransoStruct()
        ci.c1 = -p
        ci.c2 = -n

        # Equality constraint
        ce = pygransoStruct()
        ce.c1 = W1(x) - p + n
        ce.c2 = torch.matmul(n.view(n.size(0), -1), p.view(p.size(0), -1).T).squeeze()
        #ce.c3 = y - x - W2(p)

        return [f, ci, ce]

    # Combine the user-defined functions
    comb_fn = lambda X_struct: user_fn(X_struct)

    # Define options and initial guess
    opts = pygransoStruct()
    opts.maxit = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opts.torch_device = device
    x0 = torch.randn(*input_tensor.shape, dtype=torch.double)
    p0 = torch.zeros(*p_shape, dtype=torch.double)
    n0 = torch.zeros(*p_shape, dtype=torch.double)
    opts.x0 = torch.cat((x0.view(-1, 1), p0.view(-1, 1), n0.view(-1, 1)), dim=0)
    opts.print_frequency = 10

    # Solve the optimization problem
    soln = pygranso(var_spec=var_in, combined_fn=comb_fn, user_opts=opts)

    # Get the optimal values
    optimal_x = soln.final.x[:x0.numel()].view(*input_tensor.shape)
    optimal_p = soln.final.x[x0.numel():x0.numel()+p0.numel()].view(*p_shape)
    optimal_n = soln.final.x[x0.numel()+p0.numel():].view(*p_shape)

    return optimal_x, optimal_p, optimal_n


optimal_x, optimal_p, optimal_n = solve_optimization_problem(W1, W2,res_block_in,res_block_out)

print("Optimal x:", optimal_x)
print("Optimal p:", optimal_p)
print("Optimal n:", optimal_n)


torch.norm(optimal_x - res_block_in)


import matplotlib.pyplot as plt

# Assuming image_tensor is your multi-channel image tensor
image_tensor = optimal_x

# Visualize individual channels as grayscale images
num_channels = image_tensor.shape[1]
fig, axes = plt.subplots(1, num_channels, figsize=(12, 4))

for i in range(num_channels):
    channel_image = image_tensor[:,i, :, :]
    print(channel_image.squeeze(0).shape)
    axes[i].imshow(channel_image.squeeze(0), cmap='plasma')
    axes[i].set_title(f'Reconstructed_Ch {i + 1}')

plt.show()


import matplotlib.pyplot as plt

# Assuming image_tensor is your multi-channel image tensor
image_tensor = res_block_in

# Visualize individual channels as grayscale images
num_channels = image_tensor.shape[1]
fig, axes = plt.subplots(1, num_channels, figsize=(12, 4))

for i in range(num_channels):
    channel_image = image_tensor[:,i, :, :]
    print(channel_image.squeeze(0).shape)
    axes[i].imshow(channel_image.squeeze(0), cmap='plasma')
    axes[i].set_title(f'Orignal_Ch {i + 1}')

plt.show
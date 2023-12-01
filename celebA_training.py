import os
import time

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter
from torchvision import datasets
from torchvision import transforms
from sklearn.model_selection import train_test_split
import time
import random
import matplotlib.pyplot as plt
from PIL import Image
from IR_152_ReLU import IR152
from IR_152_PReLU import IR152_PReLU

torch.manual_seed(0)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


def test(model, criterion=None, dataloader=None, device='cpu'):
    tf = time.time()
    model.to(device)
    model.eval()
    loss, cnt, ACC, correct_top5 = 0.0, 0, 0, 0
    with torch.no_grad():
        for i, (img, iden) in enumerate(dataloader):
            img, iden = img.to(device), iden.to(device)
            img, iden = img.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)
            _, out_prob = model(img)
            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()

            _, top5 = torch.topk(out_prob, 5, dim=1)
            for ind, top5pred in enumerate(top5):
                if iden[ind] in top5pred:
                    correct_top5 += 1

            cnt += bs

    return ACC * 100.0 / cnt, correct_top5 * 100.0 / cnt

# Define the train_reg function
def train_reg(model, criterion, optimizer, trainloader, testloader, n_epochs, device='cpu'):
    best_ACC = (0.0, 0.0)
    model.to(device)
    for epoch in range(n_epochs):
        tf = time.time()
        ACC, cnt, loss_tot = 0, 0, 0.0
        model.train()
		
        for i, (img, iden) in enumerate(trainloader):
            img, iden = img, iden
            img, iden = img.to(device), iden.to(device)
            #import pdb;pdb.set_trace()
            bs = img.size(0)
            
            feat,out_prob = model(img)
           # import pdb;pdb.set_trace()
            if(i == 2):
                pass
                #import pdb;pdb.set_trace()
            print("Index:",i)
            #print("Labels:",iden)
            cross_loss = criterion(out_prob, iden)
            loss = cross_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_iden = torch.argmax(out_prob, dim=1)
            ACC += torch.sum(iden == out_iden).item()
            loss_tot += loss.item() * bs
            cnt += bs

        train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
        test_acc = test(model, criterion, testloader,device = device)  # Implement your test function accordingly

        interval = time.time() - tf
        if test_acc[0] > best_ACC[0]:
            best_ACC = test_acc
            best_model = deepcopy(model)
            # Save the best model's state to a checkpoint file
            checkpoint = {
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_accuracy': best_ACC[0],
        }
        torch.save(checkpoint, 'best_model_checkpoint_Prelu.pth')

        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval, train_loss, train_acc, test_acc[0]))

    print("Best Acc:{:.2f}".format(best_ACC[0]))
    return best_model, best_ACC
    

class CelebADataset(Dataset):
    def __init__(self, root_dir, image_names, label_names, transform=None,num_labels_to_keep=1000):
        self.root_dir = root_dir
        self.image_names = image_names
        self.label_names = label_names
        self.transform = transform
        self.num_labels_to_keep = num_labels_to_keep
        
        # Filter the labels to keep only the top N most frequent labels
        self.label_counts = Counter(label_names)
        self.top_labels = [label for label, count in self.label_counts.most_common(num_labels_to_keep)]
        
        # Filter image and label names based on the top labels
        filtered_image_names = []
        filtered_label_names = []
        for image_name, label_name in zip(image_names, label_names):
            if label_name in self.top_labels:
                filtered_image_names.append(image_name)
                filtered_label_names.append(label_name)

        self.image_names = filtered_image_names
        self.label_names = filtered_label_names

        self.label_to_idx = {}  # Mapping of label names to unique integers
        self.idx_to_label = {}  # Mapping of unique integers to label names

        # Create a mapping of label names to unique integers
        self._create_label_mapping()

    def _create_label_mapping(self):
        unique_labels = set(self.label_names)
        for idx, label in enumerate(unique_labels):
            self.label_to_idx[label] = idx
            self.idx_to_label[idx] = label

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = Image.open(img_name)
        label = self.label_names[idx]

        # Map label string to a unique integer
        label = self.label_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, label


### Settings ###

##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Architecture
NUM_FEATURES = 128*128
NUM_CLASSES = 2
BATCH_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use CUDA device if available

#DEVICE = 'cpu' # default GPU device
GRAYSCALE = False



##### Use the code for reconititi 


# Initialize empty lists to store label names and image names
label_names = []
image_names = []

# Specify the path to your file
file_path = 'identity_CelebA.txt'  # Replace with the actual file path
image_folder = 'img_align_celeba'



#specific_image_name = '2880'

# Read the file line by line
with open(file_path, 'r') as file:
    lines = file.readlines()

# Process each line in the file
for line in lines:
    # Split the line by space (assuming that the label name and image name are separated by space)
    parts = line.strip().split()

    # Make sure there are at least two parts (label and image name)
    if len(parts) >= 2:
        label_name = parts[1]
        image_name = parts[0]
        
        #print(image_name)
        #import pdb;pdb.set_trace()
        #image_path = os.path.join(image_folder, image_name)
        #image = Image.open(image_path)
        
        # Display the image using matplotlib
        
        
        # if label_name == specific_image_name:
        #     plt.imshow(image)
        #     plt.title(label_name)
        #     plt.axis('off')
        #     plt.show()
            

        # Append the label name and image name to their respective lists
        label_names.append(label_name)
        image_names.append(image_name)

# Now, label_names and image_names contain the extracted data

crop_size = 108
crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
offset_height = (218 - crop_size) // 2
offset_width = (178 - crop_size) // 2
proc = []
proc.append(transforms.ToTensor())
proc.append(transforms.Lambda(crop))
proc.append(transforms.ToPILImage())
proc.append(transforms.Resize((112, 112)))
proc.append(transforms.ToTensor())
proc.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

# Create a composed transform using your custom steps
custom_transform = transforms.Compose(proc)
celeba_dataset = CelebADataset(image_folder, image_names, label_names, custom_transform,num_labels_to_keep=1000)



train_loader = DataLoader(celeba_dataset, batch_size=1, shuffle=False)

#import pdb;pdb.set_trace()


# # Iterate over the first batch (assuming batch size is 1)
# for batch_images, batch_labels in train_loader:
#     image = batch_images[0].permute(1, 2, 0).cpu().numpy()  # Convert to NumPy and change channel order
#     label = batch_labels[0].item()  # Get the label as a Python integer
    
#     # Display the image
#     plt.imshow(image)
#     plt.title(f"Label: {label}")
#     plt.axis('off')
#     plt.show()
#     break

# # Find the maximum label value
max_label = max(celeba_dataset.label_to_idx.values())

print("Maximum Label Value:", max_label)


batch_size = 64  # Adjust this value according to your needs
# Define the desired split ratio (e.g., 80% train, 20% test)
split_ratio = 0.8

# Get the total number of samples in the dataset
total_samples = len(celeba_dataset)

# Calculate the number of samples for the training and testing sets
num_train_samples = int(total_samples * split_ratio)
num_test_samples = total_samples - num_train_samples

# Use random sampling to split the dataset into train and test sets
random.seed(RANDOM_SEED)  # Set a random seed for reproducibility
indices = list(range(total_samples))
#random.shuffle(indices)

# Split the indices into train and test indices
train_indices = indices[:num_train_samples]
test_indices = indices[num_train_samples:]

# Create separate datasets for training and testing
train_dataset = torch.utils.data.Subset(celeba_dataset, train_indices)
test_dataset = torch.utils.data.Subset(celeba_dataset, test_indices)

# Create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#import pdb;pdb.set_trace()


model = IR152_PReLU(num_classes=max_label+1).to(DEVICE) # IR152_PReLU IR152
optimizer = torch.optim.SGD(params=model.parameters(),
 							    lr=1e-2, 
             					momentum= 0.9, 
             					weight_decay=1e-4)

criterion = nn.CrossEntropyLoss()

# # Call the train_reg function to start training
best_model, best_acc = train_reg(model, criterion, optimizer, train_loader, test_loader, n_epochs=40, device=DEVICE)







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
#from IR_152_ReLU import IR152
#from IR_152_PReLU import IR152_PReLU


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
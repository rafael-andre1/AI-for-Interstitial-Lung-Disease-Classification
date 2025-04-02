import pandas as pd
import numpy as np
import random
import os

# -----------  ResNet

# Core library
import torch

# Essentials for development
import torch.nn as nn
import torchvision.models as models

# Data resize (ResNet uses (224,224) inputs)
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

# Allows for paralell batch processing
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Define paths
annotations_file = r"D:\Rafa\A1Uni\2semestre\Estágio\np_ROI_data"  # CSV with image filenames & labels
img_dir = "fibrosis_annotations.csv"  # Folder with np files

# Define transformations (if needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (for grayscale)
])

# Create dataset instance
dataset = CustomImageDataset(annotations_file, img_dir, transform=transform)

print("done")
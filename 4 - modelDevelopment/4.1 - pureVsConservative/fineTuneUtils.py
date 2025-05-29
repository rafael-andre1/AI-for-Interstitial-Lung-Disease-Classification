from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import random
import math
import copy
import os


# Core library
import torch

# Essentials for development
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# Data resize (ResNet uses (224,224) inputs)
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

# Allows for paralell batch processing
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import decode_image

# --------------------------------------------------------- #

def savePickles(custom_loss, custom_val_loss):
    with open(r'pickleJar\custom_loss.pkl', 'wb') as f:
        pickle.dump(custom_loss, f)

    with open(r'pickleJar\custom_val_loss.pkl', 'wb') as f:
        pickle.dump(custom_val_loss, f)

# --------------------------------------------------------- #

def saveRun(pure_vs_conservative, base_path):
    for i, run in enumerate(pure_vs_conservative):
        pair_pure, pair_cons, train_loss, val_loss = run
        folder_path = os.path.join(base_path, f"iteration_{i}")
        os.makedirs(folder_path, exist_ok=True)

        model_pure, epoch_pure = pair_pure
        model_cons, epoch_cons = pair_cons

        # Save model states
        torch.save({
            'model_state_dict': model_pure.state_dict(),
            'epoch': epoch_pure,
            'train_loss': train_loss,
            'val_loss': val_loss
        }, os.path.join(folder_path, 'pure_model.pt'))

        torch.save({
            'model_state_dict': model_cons.state_dict(),
            'epoch': epoch_cons,
            'train_loss': train_loss,
            'val_loss': val_loss
        }, os.path.join(folder_path, 'cons_model.pt'))

        # Save metadata
        metadata = {
            'epoch_pure': epoch_pure,
            'epoch_cons': epoch_cons,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        with open(os.path.join(folder_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

# --------------------------------------------------------- #

def getROC(model, val_dataset): 

    all_labels, all_scores = [], []
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Translate logits to Class 1 probabilities using softmax
            # (pulls column 1 values and assigns probabilities)
            probs = F.softmax(outputs, dim=1)[:, 1]  

            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Compute distance to (0,1) for each point on the ROC curve
    distances = np.sqrt((fpr)**2 + (1 - tpr)**2)

    # Gets closest point to the perfect discriminator (0,1)
    best_idx = np.argmin(distances)
    best_threshold = thresholds[best_idx]

    # ---------- Display and Results ---------- 

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f'Best Threshold = {best_threshold:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    print("Area Under Curve:", roc_auc)
    print("Best Threshold (closest to (0,1)):", best_threshold)

    return best_threshold, roc_auc

# --------------------------------------------------------- #
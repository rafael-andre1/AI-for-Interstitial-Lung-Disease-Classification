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
import copy
import math
import os
import re

# -----------  ResNet -----------

# Core library
import torch

# Essentials for development
import torch.nn as nn
from torchsummary import summary
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# Data resize (ResNet uses (224,224) inputs)
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

# Allows for paralell batch processing
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import decode_image

# ---------- Distances ----------

from scipy.spatial.distance import mahalanobis
from sklearn.covariance import LedoitWolf
import torch.nn.functional as F
from numpy.linalg import inv
import math

# ------------ t-SNE ------------

from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE




# ------------------------- Main Class ------------------------- #


class FibrosisDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, albumentations=None, gauss=False):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.albumentations = albumentations
        self.gauss = gauss

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # idx represents index
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        slice_id = self.img_labels.iloc[idx, 0]
        patient_id = getPatientID(slice_id)

        # Load the .npy file
        image = np.load(img_path)
        
        #image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]

        # Adds randomly selected gauss noise or blur
        if self.gauss:
            # Gaussian Noise
            gauss_noise = image + np.random.normal(loc=0, scale=random.choice(range(10,40)), size=image.shape)
            # Gaussian Blur
            gauss_blur = gaussian_filter(image, sigma=(random.choice(range(10,16))/10)) 
            # Random choice
            image = random.choice((gauss_noise,gauss_blur))

        # Guarantee compatibility
        if self.gauss or self.albumentations: image = image.astype(np.float32)

        # Already has resize prior to augments
        if self.albumentations:
            stacked_aug = self.albumentations(image=image)
            image = stacked_aug['image']

        # Applies necessary ResNet input transformations
        if self.transform:
            image = self.transform(image)

        return image, label, patient_id



# ------------------------- Main Training Loop ------------------------- #


def trainResNet(train_dataset, val_dataset, num_epochs=90, batch_size=32, lr=5e-7, patience=5, improve_min=0.001):
    # New ResNet instance
    resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Considers only absolute lowest val_loss value
    best_model_pure = copy.deepcopy(resnet18)
    best_val_pure, epoch_pure = 10, 0

    # Use only 2 output neurons
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 2)  

    # If graphics card is available, uses it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18.to(device)

    print("*"+("-"*29)+"*")
    print("|{:^29}|".format(f"Using {device}"))
    print("*"+("-"*29)+"*")

    # Generate data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss function 
    criterion = nn.CrossEntropyLoss()  
    # ADAM
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=lr)

    # add learning_rate_scheduler e step learning rate

    # Initializing early stop counter (based on validation loss)
    best_loss = 10
    early_stop_ctr = 0

    loss_array = []
    val_loss_array = []


    for epoch in range(num_epochs):
        resnet18.train()
        running_loss = 0.0
        running_val_loss = 0.0


        # --------------- Weight updates ---------------
        
        # Iterates over each image in dataset, updates weights
        for images, labels, patient_id in tqdm((train_loader), desc = "Training..."):
            images, labels = images.to(device), labels.to(device)
            
            # Applies ADAM 
            optimizer.zero_grad()

            # Generates output
            outputs = resnet18(images)

            # Computes loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Computes average training loss
        avg_loss_train = running_loss/len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train loss: {avg_loss_train:.6f}")
        loss_array.append(avg_loss_train)


        # ---------------- Validation Loss -----------------

        # Iterates over each image in dataset, computes validation loss
        for images, labels, patient_id in (val_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Applies ADAM 
            optimizer.zero_grad()

            # Generates output
            outputs = resnet18(images)

            # Computes loss
            val_loss = criterion(outputs, labels)
            
            running_val_loss += val_loss.item()

        # Computes average validation loss
        avg_loss_val = running_val_loss/len(val_loader)
        print(f"Validation loss: {avg_loss_val:.6f}")
        val_loss_array.append(avg_loss_val)


        # ------------------ Best Model -------------------

        if epoch >= 35:
            if avg_loss_val < best_val_pure:
                best_val_pure = avg_loss_val
                best_model_pure = copy.deepcopy(resnet18)
                epoch_pure = epoch
        
        
        # ---------------- Early Stopping -----------------

        # Only after a high number of epochs (around 60)
        if epoch > 60:
            # Check for improvement (1% standard)
            if abs(best_loss - avg_loss_val) > (best_loss * improve_min):
                best_loss = avg_loss_val
                early_stop_ctr = 0
            else: early_stop_ctr += 1

            # Checks early stopping condition
            if early_stop_ctr >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!!!")
                break
    
    return best_model_pure, epoch_pure, loss_array, val_loss_array    




# ------------------------- Individual Slice Evaluation ------------------------- #


def evalResNet(resnet18, test_dataset, threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Counters for each class
    correct, total = 0, 0
    correct_class_0, total_class_0 = 0, 0
    correct_class_1, total_class_1 = 0, 0

    # Needed for F1 score and confusion matrix
    y_true = []
    y_pred = []

    # for feature based
    #lat=resnet.define_hook(fc)

    with torch.no_grad():
        for images, labels, patient_id in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Transform outputs to probabilities
            outputs = resnet18(images)

            # Translate logits to Class 1 probabilities using softmax
            probs = F.softmax(outputs, dim=1)[:, 1] 

            # Translate prob vs threshold to predictions
            # True if prob >= threshold else False
            # .long() transforms True/False to 1/0
            predicted = (probs >= threshold).long()
            
            # Update total and correct for general accuracy
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            # Class-specific accuracy
            for label, pred in zip(labels, predicted):
                if label == 0:
                    total_class_0 += 1
                    if pred == label:
                        correct_class_0 += 1
                elif label == 1:
                    total_class_1 += 1
                    if pred == label:
                        correct_class_1 += 1

            # Add data to lists
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())



    # Weights == inverse proportions
    weight_0, weight_1 = 0.133, 0.867

    print("Total examples:", total)

    # -------------------------------     Perfomance Metrics     -------------------------------

    # Accuracy
    accuracy_class_0 = 100 * (correct_class_0 / total_class_0) if total_class_0 > 0 else 0
    accuracy_class_1 = 100 * (correct_class_1 / total_class_1) if total_class_1 > 0 else 0
    accuracy = 100 * (correct / total)
    accuracy2 = 100 * ((correct_class_0 + correct_class_1) / total)
    weighted_accuracy = ((accuracy_class_0*weight_0) + (accuracy_class_1*weight_1))

    # F1 scores
    f1_macro = f1_score(y_true, y_pred, average='macro') # Assigns same importance to each class
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_class_0 = f1_score(y_true, y_pred, pos_label=0)
    f1_class_1 = f1_score(y_true, y_pred, pos_label=1)

    # Confusion Matrix
    conf_mat = confusion_matrix(y_true, y_pred)


    # -------------------------------     Print Results     -------------------------------

    # Accuracy
    print("\n --------------------- \n")
    print(f"Accuracy for Class 0: {accuracy_class_0:.2f}%  ({correct_class_0} in {total_class_0})")
    print(f"Accuracy for Class 1: {accuracy_class_1:.2f}%  ({correct_class_1} in {total_class_1})")
    print(f"Test Accuracy: {accuracy:.2f}%")
    if f"{accuracy:.2f}" != f"{accuracy2:.2f}": print("ERROR CALCULATING ACCURACIES")
    print(f"Weighted Accuracy: {weighted_accuracy:.2f}%")


    # F1 scores
    print("\n --------------------- \n")
    print(f"F1 Score (Macro): {f1_macro:.3f}")
    print(f"F1 Score (Weighted): {f1_weighted:.3f}")
    print(f"F1 Score Class 0: {f1_class_0:.3f}")
    print(f"F1 Score Class 1: {f1_class_1:.3f}")

    # Confusion matrix
    print("\n --------------------- \n")
    print("\nConfusion Matrix: \n", conf_mat)




# ------------------------- Patient-Wise Classification (Probabilities) ------------------------- #

def evalPatientProbResNet(resnet18, test_dataset, threshold, aggregate_criteria="mean", n=1, ratio= 0.5, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Counters for each class
    correct, total = 0, 0
    correct_class_0, total_class_0 = 0, 0
    correct_class_1, total_class_1 = 0, 0

    # Needed for F1 score and confusion matrix
    y_true = []
    y_pred = []

    # Needed for patient-wise classification
    patient_prob = {}
    patient_class = {}

    # Used to store values after aggregation
    patient_aggregate = {}


    # 1. This loop populates dictionaries with values pulled from the model
    with torch.no_grad():
        for images, labels, patient_id in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Transform outputs to probabilities
            outputs = resnet18(images)

            # Translate logits to Class 1 probabilities using softmax
            probs = F.softmax(outputs, dim=1)[:, 1] 

            # Associate probabilities to dictionary for each patient
            for pid, prob, label in zip(patient_id, probs.tolist(), labels.tolist()):
                if pid not in patient_prob:
                    patient_prob[pid] = []  # Initializes key:value pair if it doesn't exist
                    patient_class[pid] = 0  # Same principle

                patient_prob[pid].append(prob)  # Adds probability to list
                
                # Label aggregation condition
                if label == 1: patient_class[pid] = 1 
                       
    
    # 2. After pulling every slice value for every patient, apply aggregate_criteria
    # and evaluate the aggregated values using criteria-specific conditions
    for id, prob_list in patient_prob.items():
        if aggregate_criteria == "mean":
            prob = np.mean(prob_list)
            patient_aggregate[id] = prob

            # If larger -> True -> int -> 1
            # If smaller -> False -> int -> 0
            predicted = int(prob >= threshold)

        # Absolute and Relative slice ammount thresholds (number of slices vs ratio)
        elif aggregate_criteria == "ratio" or aggregate_criteria == "n_is_enough":
            
            # Obtain fibrosis ratio
            ctr0,ctr1 = 0,0
            for prob in prob_list:
                slice_class = int(prob >= threshold)
                if slice_class == 1: ctr1 +=1
                else: ctr0 +=1

            # If ratio of slices == 1 is >= threshold_ratio 
            # OR if number of slices >= threshold_n 
            # THEN patient_classification == 1
            # 0 otherwise
            predicted = int((ctr1/(ctr1+ctr0)) >= ratio) if aggregate_criteria == "ratio" else int(ctr1>=n)

        label = patient_class[id]

        # Counters for metrics   
        total += 1
        if patient_class[id] == 0:
            total_class_0 += 1
            if predicted == patient_class[id]:
                correct_class_0 += 1
                correct += 1
        elif patient_class[id] == 1:
            total_class_1 += 1
            if predicted == patient_class[id]:
                correct_class_1 += 1
                correct += 1

        # Populate y_true and y_pred lists for F1 Score
        # print(id, patient_class[id])
        y_true.append(patient_class[id])
        y_pred.append(predicted)
        

    
    # Weights change here -> Class == 1 is the majority class
    weight_0, weight_1 = 14/24, 10/24

    print("Total examples:", total)

    # -------------------------------     Perfomance Metrics     -------------------------------

    # Accuracy
    accuracy_class_0 = 100 * (correct_class_0 / total_class_0) if total_class_0 > 0 else 0
    if correct_class_0 == total_class_0: accuracy_class_0 = 100

    accuracy_class_1 = 100 * (correct_class_1 / total_class_1) if total_class_1 > 0 else 0
    if correct_class_1 == total_class_1: accuracy_class_1 = 100

    accuracy = 100 * (correct / total)
    accuracy2 = 100 * ((correct_class_0 + correct_class_1) / total)
    weighted_accuracy = ((accuracy_class_0*weight_0) + (accuracy_class_1*weight_1))

    print("Labels:    ",y_true)
    print("Predicted: ", y_pred)
    # print(type(y_true), type(y_pred))


    # F1 scores
    f1_macro = f1_score(y_true, y_pred, average='macro') # Assigns same importance to each class
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_class_0 = f1_score(y_true, y_pred, pos_label=0)
    f1_class_1 = f1_score(y_true, y_pred, pos_label=1)

    # Confusion Matrix
    conf_mat = confusion_matrix(y_true, y_pred)


    # -------------------------------     Print Results     -------------------------------

    if verbose:

        # Accuracy
        print("\n --------------------- \n")
        print(f"Accuracy for Class 0: {accuracy_class_0:.2f}%  ({correct_class_0} in {total_class_0})")
        print(f"Accuracy for Class 1: {accuracy_class_1:.2f}%  ({correct_class_1} in {total_class_1})")
        print(f"Test Accuracy: {accuracy:.2f}%")
        if f"{accuracy:.2f}" != f"{accuracy2:.2f}": print("ERROR CALCULATING ACCURACIES")
        print(f"Weighted Accuracy: {weighted_accuracy:.2f}%")

        # F1 scores
        print("\n --------------------- \n")
        print(f"F1 Score (Macro): {f1_macro:.3f}")
        print(f"F1 Score (Weighted): {f1_weighted:.3f}")
        print(f"F1 Score Class 0: {f1_class_0:.3f}")
        print(f"F1 Score Class 1: {f1_class_1:.3f}")

        # Confusion matrix
        print("\n --------------------- \n")
        print("\nConfusion Matrix: \n", conf_mat)    

# ------------------------------------------------ #


def getROCAggregateOLD(model, dataset, threshold, aggregate_criteria="mean", n=1, show_plot=True): 

    all_labels, all_scores = [], []
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Needed for patient-wise classification
    patient_prob = {}
    patient_class = {}

    model.eval()
    with torch.no_grad():
        for images, labels, patient_id in loader:
            images, labels = images.to(device), labels.to(device)

            # Transform outputs to probabilities
            outputs = model(images)

            # Translate logits to probabilities using softmax
            # and then chooses only values for Class 1 (0 or 1)
            probs = F.softmax(outputs, dim=1)[:, 1] 
        
            """ Previous idea:

            # If there isn't a base threshold to start from,
            # we need to classify based on highest likelyhood
            # This code chooses the Class with highest probability,
            # automatically classifying/transforming logits to 0 and 1.
            # They are simply named "probs" for convenience, but be mindful 
            # THEY AREN'T PROBABILITIES!!!
            else: probs = torch.argmax(outputs, dim=1)
            """


            # Associate probabilities/values to dictionary for each patient
            for pid, prob, label in zip(patient_id, probs.tolist(), labels.tolist()):
                if pid not in patient_prob:
                    patient_prob[pid] = []  # Initializes key:value pair if it doesn't exist
                    patient_class[pid] = 0  # Sets classification as 0 until otherwise

                patient_prob[pid].append(prob)  # Adds probability to list
                
                # Label aggregation condition
                if label == 1: patient_class[pid] = 1 # Updates Classification
    
    # 2. After pulling every slice value for every patient, apply aggregate_criteria
    # and evaluate the aggregated values using criteria-specific conditions
    for id, prob_list in patient_prob.items():
        if aggregate_criteria == "mean":
            final_prob = np.mean(prob_list)
   
        elif aggregate_criteria == "majority_vote":
            ctr0, ctr1 = 0, 0
            for prob in prob_list:
                predicted = int(prob >= threshold)
                if predicted == 1: ctr1 += 1
                elif predicted == 0: ctr0 += 1
                else: print("Error: Invalid prediction for patient", id)
            
            # From a medical standpoint, I decided 50/50 calls for more
            # tests, as a prevention for false negatives in these cases
            if (ctr1 >= ctr0): predicted = 1
            else: predicted = 0

            # 3. After obtaining the aggregated classification, apply mean of slices with
            # Class == Prediction, creating final_prob, which represents the patient-wise probability
            final_prob_maj_vote = []
            for prob in prob_list:
                slice_class = int(prob >= threshold)
                if slice_class == predicted: final_prob_maj_vote.append(prob)

            final_prob = np.mean(final_prob_maj_vote)

        elif aggregate_criteria == "n_is_enough":
            n_ctr = 0

            # Absolute slice number
            if isinstance(n, int):
                for prob in prob_list:
                    predicted = int(prob >= threshold)
                    if predicted == 1: n_ctr += 1
                    elif predicted != 0: print("Error: Invalid prediction for patient", id)

                    # Early stopping
                    if n_ctr == n: break

                # Assigning correct values
                if n_ctr >= n: predicted = 1
                else: predicted = 0

            # Relative slice ammount
            elif isinstance(n, float):
                size = len(prob_list)
                for prob in prob_list:
                    predicted = int(prob >= threshold)
                    if predicted == 1: n_ctr += 1
                    elif predicted != 0: print("Error: Invalid prediction for patient", id)

                    # Early stopping
                    if n_ctr >= n * size: break

                # Assigning correct values
                if n_ctr >= n * size: predicted = 1
                else: predicted = 0

                #if predicted == 1: print(" +++++ ", id, " | ", n_ctr, "out of", size, "slices!")
                #else: print(" ----- ", id, " | ", n_ctr, "out of", size, "slices!")
            
            else: print("Error: Invalid n value for patient", id)

            # Running step 3. for N is enough
            final_prob_n_enough = []
            for prob in prob_list:
                slice_class = int(prob >= threshold)
                if slice_class == predicted: final_prob_n_enough.append(prob)

            final_prob = np.mean(final_prob_n_enough)

        label = patient_class[id]

        #  y_true, y_pred -> all_scores, all_labels
        all_labels.append(label)
        all_scores.append(final_prob)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Compute distance to (0,1) for each point on the ROC curve
    distances = np.sqrt((fpr)**2 + (1 - tpr)**2)

    # Gets closest point to the perfect discriminator (0,1)
    best_idx = np.argmin(distances)
    best_threshold = thresholds[best_idx]

    # ---------- Display and Results ---------- 

    if show_plot:
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f'Best Threshold for method {aggregate_criteria} = {best_threshold:.2f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

        print("Area Under Curve:", roc_auc)
        print("Best Threshold (closest to (0,1)):", best_threshold)

    return best_threshold, roc_auc


# ------------------------------------------------ #

def getROCAggregate(model, dataset, threshold, aggregate_criteria="mean", show_plot=True): 

    all_labels, all_scores = [], []
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Needed for patient-wise classification
    patient_prob = {}
    patient_class = {}

    model.eval()
    with torch.no_grad():
        for images, labels, patient_id in loader:
            images, labels = images.to(device), labels.to(device)

            # Transform outputs to probabilities
            outputs = model(images)

            # Translate logits to probabilities using softmax
            # and then chooses only values for Class 1 (0 or 1)
            probs = F.softmax(outputs, dim=1)[:, 1] 
        
            # Associate probabilities/values to dictionary for each patient
            for pid, prob, label in zip(patient_id, probs.tolist(), labels.tolist()):
                if pid not in patient_prob:
                    patient_prob[pid] = []  # Initializes key:value pair if it doesn't exist
                    patient_class[pid] = 0  # Sets classification as 0 until otherwise

                patient_prob[pid].append(prob)  # Adds probability to list
                
                # Label aggregation condition
                if label == 1: patient_class[pid] = 1 # Updates Classification
    
    # 2. After pulling every slice value for every patient, apply aggregate_criteria
    # and evaluate the aggregated values using criteria-specific conditions
    for id, prob_list in patient_prob.items():
        if aggregate_criteria == "mean":
            final_prob = np.mean(prob_list)
        
        # Absolute and Relative slice ammount thresholds (number of slices vs ratio)
        elif aggregate_criteria == "ratio" or aggregate_criteria == "n_is_enough":
            
            # Obtain fibrosis ratio
            ctr0,ctr1 = 0,0
            for prob in prob_list:
                slice_class = int(prob >= threshold)
                if slice_class == 1: ctr1 +=1
                else: ctr0 +=1

            final_prob = (ctr1/(ctr1+ctr0)) if aggregate_criteria == "ratio" else ctr1

        label = patient_class[id]

        #  y_true, y_pred -> all_scores, all_labels
        all_labels.append(label)
        all_scores.append(final_prob)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Compute distance to (0,1) for each point on the ROC curve
    distances = np.sqrt((fpr)**2 + (1 - tpr)**2)

    # Gets closest point to the perfect discriminator (0,1)
    best_idx = np.argmin(distances)
    best_threshold = thresholds[best_idx]

    # ---------- Display and Results ---------- 

    if show_plot:
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f'Best Threshold for method {aggregate_criteria} = {best_threshold:.2f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

        print("Area Under Curve:", roc_auc)
        print("Best Threshold (closest to (0,1)):", best_threshold)

    return best_threshold, roc_auc


# ------------------------- Feature-Wise Classification & Clustering ------------------------- #

class HookFC:
    def __init__(self, model, layer='avgpool'):
        self.model = model
        self.intercepts = []
        self.hook = None

        # Register and save hook
        for name, module in self.model.named_modules():
            if layer in name: self.hook = module.register_forward_hook(self.getLayerOutput)

    def getLayerOutput(self, module, input, output):
        # Clear previous outputs on each forward pass → prevents accumulation
        self.intercepts = [output.detach().cpu()]

    def __call__(self, input_tensor):
        self.model.zero_grad()
        _ = self.model(input_tensor)
        return self.intercepts[0]  # Returns only the latest output

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

# ------------------------------------------------ #

def getFeatures(resnet18, train_dataset, val_dataset, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Used for initial t-SNE
    tsne_features, tsne_labels = [], []

    # Evaluation task data
    eval_features, eval_labels = [], []

    #resnet18.fc = torch.nn.Identity()
    hook = HookFC(resnet18)

    # This loop populates dictionaries with features pulled from the train_dataset
    # using interceptions made by the model hook
    with torch.no_grad():
        for images, labels, patient_id in tqdm(train_loader, desc="Populating training features..."):
            images, labels = images.to(device), labels.to(device)

            #outputs = resnet18(images)
            outputs = hook(images)

            # Associate features to labels
            for ftr, label in zip(outputs, labels.tolist()):
                tsne_features.append(ftr.cpu())  # Adds feature to list
                tsne_labels.append(label)

    
    # Same concept, but for validation_dataset
    with torch.no_grad():
        for images, labels, patient_id in tqdm(val_loader, desc="Populating validation features..."):
            images, labels = images.to(device), labels.to(device)

            outputs = hook(images)

            # Associate features and labels
            for ftr, label in zip(outputs, labels.tolist()):
                tsne_features.append(ftr.cpu())  # Adds feature to list
                tsne_labels.append(label)

    # Same concept, but for test_dataset
    with torch.no_grad():
        for images, labels, patient_id in tqdm(test_loader, desc="Populating test features..."):
            images, labels = images.to(device), labels.to(device)

            outputs = hook(images)

            # Associate features and labels
            for ftr, label in zip(outputs, labels.tolist()):
                eval_features.append(ftr.cpu())  # Adds feature to list
                eval_labels.append(label)
    
    return tsne_features, tsne_labels, eval_features, eval_labels

# ------------------------------------------------ #

def plotTSNE(train_val_features, train_val_labels, test_features=None, apply_pca=False):
    X_train_val = torch.stack(train_val_features).cpu().numpy()
    y_train_val = np.array(train_val_labels)

    if test_features:
        # Associate to label == 2 

        X_test = torch.stack([f.view(-1) for f in test_features], dim=0).cpu().numpy()
        y_test = np.full(len(X_test), 2) 
         
        # Combine for joint t-SNE
        X = np.concatenate((X_train_val, X_test), axis=0)
        y = np.concatenate((y_train_val, y_test), axis=0)
    
    else: X, y = X_train_val, y_train_val
        
    if apply_pca:
        pca = PCA(n_components=50)
        X = pca.fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    # Assign colors
    colors = np.array([
        'green' if label == 0 else
        'red' if label == 1 else
        'blue' for label in y
    ])

    plt.figure(figsize=(10, 8))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, s=10)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Class 0', markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Class 1', markerfacecolor='red', markersize=8)
    ]
    
    if test_features: legend_elements.append(Line2D([0], [0], marker='o', 
                                            color='w', label='Test', 
                                            markerfacecolor='blue', markersize=8))

    plt.legend(handles=legend_elements, title="Classes")
    plt.title("t-SNE of ResNet Features (with PCA)" if apply_pca else "t-SNE of ResNet Features")
    plt.show()

# ------------------------------------------------ #

def euclideanDistance(a, b):
    # Torch features normal distance for tensors, makes this quite easy
    return torch.norm(a - b)

# ------------------------------------------------ #

def cosineDistance(a, b):
    # Torch functionalities also make this pretty straightforward
    cosine_similarity = F.cosine_similarity(a, b, dim=0)
    return 1 - cosine_similarity

# ------------------------------------------------ #

def customMahalanobisDistance(x, X):     
    # Transform slice (x) and cluster (X == matrix of all slices) to np
    x, X = x.numpy(), torch.stack(X).numpy()

    # Calculate centroid of X
    centroid = np.mean(X, axis=0)

    if len(X) >= 512: 
        # Calculate covariance of X
        # It's important to set rowvar to False, otherwise
        # column == obs (slice)  | row == feature, which is the
        # inverse of what we need
        covariance_matrix = (np.cov(X, rowvar=False))
        inv_cov = np.linalg.inv(covariance_matrix)
    else: 
        # This implements LedoitWolf, so there's no need for
        # rowvar == False
        lw = LedoitWolf().fit(X)
        inv_cov = np.linalg.inv(lw.covariance_)
        

    # Compute values
    delta = (x-centroid)
    squared_dist = ((delta.T) @ inv_cov) @ (delta)
    #print(squared_dist)
    return math.sqrt(squared_dist)

# ------------------------------------------------ #

def featuresKNN_CPU(train_val_features, labels, test_features, test_labels, criteria="euclidean",k=1):
    if k==0: return f"K CANNOT BE ZERO. SHUTTING DOWN..."
    if criteria not in ["euclidean", "cosine", "mahalanobis"]: 
        return f"INVALID DISTANCE CRITERIA. SHUTTING DOWN..."

    # Counters for each class
    correct, total = 0, 0
    correct_class_0, total_class_0 = 0, 0
    correct_class_1, total_class_1 = 0, 0

    # Needed for F1 score and confusion matrix
    y_true = test_labels
    y_pred = []

    # Needed for K-NN
    distances = []

    # Needed for Mahalanobis
    cluster0, cluster1 = [], []
    for i in range(len(train_val_features)):
        if labels[i] == 0: cluster0.append(train_val_features[i])
        else: cluster1.append(train_val_features[i])

    print(f"Length of class1: {len(cluster1)}")


    # Avoids loop and if-statement repetitiveness
    dist_func = {
        "euclidean": euclideanDistance,
        "cosine": cosineDistance,
        "mahalanobis": customMahalanobisDistance
    }

    description = f"Computing {criteria} distances"
    if criteria in ["euclidean", "cosine"]: description += f" for k=={k}... "
    else: description += f"... "
    
    for test_slice in tqdm(test_features, desc = description):
        # 1. Compute distances
        if criteria in ["euclidean", "cosine"]:
            for i in range(len(train_val_features)):
                train_slice, train_slice_label = train_val_features[i], labels[i]

                # Creates (distance, label) tuple for ranking and majority voting classification
                dist_pair = (dist_func[criteria](test_slice, train_slice), train_slice_label)
                distances.append(dist_pair)

            # Sorts distances in ascending order, making the 
            # first elements the closest to the slice 
            sorted_distances = sorted(distances, key=lambda x: x[0])

            # Creates ranking
            ranking = [sorted_distances[i][0] for i in range(k)]

            # 2.1. Classify slices based on k value

            # Gets classification based on majority voting:
            # number of ones in an array with only 1s and 0s == sum of every element,
            # if the sum represents at least half the elements, then it's fibrosis 
            # (rather assume it's fibrosis when not sure), 0 otherwise
            y_pred.append(1 if (np.sum(ranking) >= (len(ranking) / 2)) else 0)
            total +=1

        
        else:
            # Compute distances to each cluster centroid
            try:
                dist0 = dist_func[criteria](test_slice, cluster0)
                dist1 = dist_func[criteria](test_slice, cluster1)
            except Exception as e:    
                print(e)
            

            # 2.2. Classify slices based on centroid proximity
            y_pred.append(1 if dist1 <= dist0 else 0)
            total +=1


        
    # 3. Evaluate classifications
    # Class-specific accuracy
    for label, pred in zip(y_true, y_pred):
        if label == 0:
            total_class_0 += 1
            if pred == label:
                correct_class_0 += 1
                correct += 1
        elif label == 1:
            total_class_1 += 1
            if pred == label:
                correct_class_1 += 1
                correct += 1


    # Weights == inverse proportions
    weight_0, weight_1 = 0.133, 0.867

    print("Total examples:", total)

    # -------------------------------     Perfomance Metrics     -------------------------------

    # Accuracy
    accuracy_class_0 = 100 * (correct_class_0 / total_class_0) if total_class_0 > 0 else 0
    accuracy_class_1 = 100 * (correct_class_1 / total_class_1) if total_class_1 > 0 else 0
    accuracy = 100 * (correct / total)
    accuracy2 = 100 * ((correct_class_0 + correct_class_1) / total)
    weighted_accuracy = ((accuracy_class_0*weight_0) + (accuracy_class_1*weight_1))

    # F1 scores
    f1_macro = f1_score(y_true, y_pred, average='macro') # Assigns same importance to each class
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_class_0 = f1_score(y_true, y_pred, pos_label=0)
    f1_class_1 = f1_score(y_true, y_pred, pos_label=1)

    # Confusion Matrix
    conf_mat = confusion_matrix(y_true, y_pred)


    # -------------------------------     Print Results     -------------------------------

    # Accuracy
    print("\n --------------------- \n")
    print(f"Accuracy for Class 0: {accuracy_class_0:.2f}%  ({correct_class_0} in {total_class_0})")
    print(f"Accuracy for Class 1: {accuracy_class_1:.2f}%  ({correct_class_1} in {total_class_1})")
    print(f"Test Accuracy: {accuracy:.2f}%")
    if f"{accuracy:.2f}" != f"{accuracy2:.2f}": print("ERROR CALCULATING ACCURACIES")
    print(f"Weighted Accuracy: {weighted_accuracy:.2f}%")


    # F1 scores
    print("\n --------------------- \n")
    print(f"F1 Score (Macro): {f1_macro:.3f}")
    print(f"F1 Score (Weighted): {f1_weighted:.3f}")
    print(f"F1 Score Class 0: {f1_class_0:.3f}")
    print(f"F1 Score Class 1: {f1_class_1:.3f}")

    # Confusion matrix
    print("\n --------------------- \n")
    print("\nConfusion Matrix: \n", conf_mat)

# ------------------------------------------------ #

def euclideanDistance_batch(test_batch, train_batch):
    # test_batch: shape → [num_test, feat_dim]
    # train_batch: shape → [num_train, feat_dim]
    # Computes pairwise distances: output shape → [num_test, num_train]
    return torch.cdist(test_batch, train_batch, p=2)

# ------------------------------------------------ #

def cosineDistance_batch(test_batch, train_batch):
    # Normalize first
    test_norm = F.normalize(test_batch, p=2, dim=1)
    train_norm = F.normalize(train_batch, p=2, dim=1)
    # Cosine similarity: shape → [num_test, num_train]
    sim = torch.matmul(test_norm, train_norm.T)
    return 1 - sim

# ------------------------------------------------ #

def featuresKNN_GPU(train_val_features, labels, test_features, test_labels, criteria="euclidean",k=1, verbose=1):
    if k==0: return f"K CANNOT BE ZERO. SHUTTING DOWN..."
    if criteria not in ["euclidean", "cosine", "mahalanobis"]: 
        return f"INVALID DISTANCE CRITERIA. SHUTTING DOWN..."
    
    # Moves data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_val_features = torch.stack(train_val_features).to(device)
    labels = torch.tensor(labels).to(device)
    test_features = torch.stack(test_features).to(device)
    test_labels = torch.tensor(test_labels).to(device)

    # Counters for each class
    correct, total = 0, 0
    correct_class_0, total_class_0 = 0, 0
    correct_class_1, total_class_1 = 0, 0

    # Needed for F1 score and confusion matrix
    y_true = test_labels
    y_pred = []

    # Needed for K-NN
    distances = []

    # Needed for Mahalanobis
    cluster0, cluster1 = [], []
    for i in range(len(train_val_features)):
        if labels[i] == 0: cluster0.append(train_val_features[i])
        else: cluster1.append(train_val_features[i])

    #print(f"Length of class1: {len(cluster1)}")

    dist_func = {
        "euclidean": euclideanDistance_batch,
        "cosine": cosineDistance_batch,
    }

    description = f"Computing {criteria} distances"
    if criteria in ["euclidean", "cosine"]: description += f" for k=={k}... "
    else: description += f"... "
    
    if criteria in ["euclidean", "cosine"]:
        # Distances between test and train: shape → [num_test, num_train]
        distances = dist_func[criteria](test_features, train_val_features)  

        # For each array of elements dist(test_slice, train_val_slice), 
        # pulls indexes of the smallest values. 
        topk_indices = torch.topk(distances, k=k, largest=False, dim=1).indices  # shape → [num_test, k]

        # Since labels and train_val_features are index-synchronized,
        # the indexes can be used in label[index] to pull the classes of the closest examples
        neighbor_labels = labels[topk_indices]  # shape → [num_test, k]

        # Same concept as before, but uses dim=1 to sum only
        # along the rows (each array of labels!)
        vote_sums = neighbor_labels.sum(dim=1)

        # Finally, applies majority voting, using .long()
        # to convert False and True to 0s and 1s, respectively
        predictions = (vote_sums >= (k / 2)).long()  # shape → [num_test]
        
        y_pred = predictions.tolist()
        total = len(test_features)

    # Class-specific accuracy
    for label, pred in zip(y_true, y_pred):
        if label == 0:
            total_class_0 += 1
            if pred == label:
                correct_class_0 += 1
                correct += 1
        elif label == 1:
            total_class_1 += 1
            if pred == label:
                correct_class_1 += 1
                correct += 1

    # Moving data to cpu
    y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)
    y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else np.array(y_pred)

    # Weights == inverse proportions
    weight_0, weight_1 = 0.133, 0.867

    #print("Total examples:", total)

    # -------------------------------     Perfomance Metrics     -------------------------------

    # Accuracy
    accuracy_class_0 = 100 * (correct_class_0 / total_class_0) if total_class_0 > 0 else 0
    accuracy_class_1 = 100 * (correct_class_1 / total_class_1) if total_class_1 > 0 else 0
    accuracy = 100 * (correct / total)
    accuracy2 = 100 * ((correct_class_0 + correct_class_1) / total)
    weighted_accuracy = ((accuracy_class_0*weight_0) + (accuracy_class_1*weight_1))

    # F1 scores
    f1_macro = f1_score(y_true, y_pred, average='macro') # Assigns same importance to each class
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_class_0 = f1_score(y_true, y_pred, pos_label=0)
    f1_class_1 = f1_score(y_true, y_pred, pos_label=1)

    # Confusion Matrix
    conf_mat = confusion_matrix(y_true, y_pred)

    if verbose==0: return k, weighted_accuracy/100, f1_weighted


    # -------------------------------     Print Results     -------------------------------

    # Accuracy
    print("\n --------------------- \n")
    print(f"Accuracy for Class 0: {accuracy_class_0:.2f}%  ({correct_class_0} in {total_class_0})")
    print(f"Accuracy for Class 1: {accuracy_class_1:.2f}%  ({correct_class_1} in {total_class_1})")
    print(f"Test Accuracy: {accuracy:.2f}%")
    if f"{accuracy:.2f}" != f"{accuracy2:.2f}": print("ERROR CALCULATING ACCURACIES")
    print(f"Weighted Accuracy: {weighted_accuracy:.2f}%")


    # F1 scores
    print("\n --------------------- \n")
    print(f"F1 Score (Macro): {f1_macro:.3f}")
    print(f"F1 Score (Weighted): {f1_weighted:.3f}")
    print(f"F1 Score Class 0: {f1_class_0:.3f}")
    print(f"F1 Score Class 1: {f1_class_1:.3f}")

    # Confusion matrix
    print("\n --------------------- \n")
    print("\nConfusion Matrix: \n", conf_mat)

# ------------------------------------------------ #

def distanceCriteriaEvolution(k_list, acc_list, f1_list):
    combined_score = (np.array(acc_list) + np.array(f1_list)) / 2
    best_k_index = np.argmax(combined_score)
    best_k = k_list[best_k_index]

    plt.figure(figsize=(10, 5))
    plt.plot(k_list, acc_list, label='Weighted Accuracy', color='blue')
    plt.plot(k_list, f1_list, label='Weighted F1 Score', color='green')
    plt.plot(k_list, combined_score, label='Combined Score', linestyle='--', color='orange')
    plt.axvline(best_k, color='red', linestyle='--', label=f'Best k = {best_k}')

    plt.xlabel('k')
    plt.ylabel('Normalized Score')
    plt.title('KNN Performance Evolution with k')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------- General Utility Functions ------------------------- #


def checkShape(train_dataset, df_fibrosis):

    # Random choice of Slice
    dataset_size=len(train_dataset)
    idx = random.choice(range(dataset_size))

    # Identifies name of file corresponding to the first line of the train dataset
    slice_id = train_dataset.img_labels.iloc[idx, 0]  
    print(f"SliceID at index {idx}: {slice_id}")

    # Pulls first index of value of pandas series element with corresponding id -> 
    # -> gets np array and displays original shape
    print("\nOriginal np.array shape:")
    print((df_fibrosis["SliceData"][df_fibrosis["SliceID"] == slice_id[:-4]]).values[0].shape)

    print("----------------------")

    # Pulls "image" value from the (image, label) pair in the train dataset -> 
    # -> gets torch tensor and displays new shape
    print("\nTransformed tensor shape:")
    print(train_dataset[idx][0].shape)


# ---------------------------------------------------- #

def tensorVSnp(dataset, df_fibrosis, rgb=True):

    # Random choice of Slice
    idx = random.choice(range(len(dataset)))

    # Identifies name of file corresponding to random line of the train dataset
    slice_id = dataset.img_labels.iloc[idx, 0]  
    print(f"SliceID at index {idx}: {slice_id}")

    # Pulls first index of value of pandas series element with corresponding id -> 
    # -> gets np array 
    np_array = (df_fibrosis["SliceData"][df_fibrosis["SliceID"] == slice_id[:-4]]).values[0]

    # Pulls "image" value from the (image, label) pair in the train dataset -> 
    # -> gets torch tensor 
    tensor_rgb = (dataset[idx][0])

    # Normalize tensor (needed for plotting)
    full_tensor_rgb = (tensor_rgb - tensor_rgb.min()) / (tensor_rgb.max() - tensor_rgb.min())

    # Convert PyTorch tensor to NumPy (swaps dimension orders for compatibility with plot)
    full_tensor_rgb_np = full_tensor_rgb.permute(1, 2, 0).cpu().numpy()
    tensor_rgb_np = tensor_rgb.permute(1, 2, 0).cpu().numpy()

    print(tensor_rgb_np.shape)

    # Plot images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Numpy Image
    axes[0].imshow(np_array, cmap='gray')  
    axes[0].set_title("Original np.array")

    # Tensor Image (full RGB)
    axes[1].imshow(full_tensor_rgb_np)  
    axes[1].set_title("Transformed Tensor")

    # ----------- RGB

    if rgb:

        # Separate plot for RGB channels 
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

        # Display the Red channel
        axes2[0].imshow(tensor_rgb_np[:, :, 0], cmap="gray")  # R channel
        axes2[0].set_title("Red Channel")

        # Display the Green channel
        axes2[1].imshow(tensor_rgb_np[:, :, 1], cmap="gray")  # G channel
        axes2[1].set_title("Green Channel")

        # Display the Blue channel
        axes2[2].imshow(tensor_rgb_np[:, :, 2], cmap="gray")  # B channel
        axes2[2].set_title("Blue Channel")

    plt.show()


# ---------------------------------------------------- #


def getROC(model, val_dataset): 

    all_labels, all_scores = [], []
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    with torch.no_grad():
        for images, labels, patientID in val_loader:
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

# ---------------------------------------------------- #


def plotLoss(train_loss, val_loss, best_epoch):
    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='x')

    # Add purple cross for best epoch on validation loss
    best_val_loss = val_loss[best_epoch - 1]  # -1 because epochs start from 1
    plt.plot(best_epoch, best_val_loss, 'X', color='purple', markersize=10, label='Best Epoch')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------- #


def getPatientID(s):
    # HRCT_Pilot__PatientID
    match = re.search(r'HRCT_Pilot__(\d+)__', s)
    if match: return match.group(1)
        
    # PatientID__sliceSpecific
    match = re.match(r'(\d+)__', s)
    if match:
        return match.group(1)
    return None  

# ---------------------------------------------------- #

def getMaxSliceNumber(dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient_prob = {}
    for images, labels, patient_id in loader:
        images, labels = images.to(device), labels.to(device)

        for pid, slice, label in zip(patient_id, images.tolist(), labels.tolist()):
                # Initializes key:value pair if it doesn't exist
                if pid not in patient_prob: patient_prob[pid] = 1  
                patient_prob[pid] += 1

    return max(patient_prob.values())


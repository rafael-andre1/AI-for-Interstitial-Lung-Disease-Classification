# Intro

Due to the nature of this internship project, learning from previous similar projects could be extremely helpful.

Furthermore, this represents a collection of problem solving concepts and tutorials. 

Throghout the development of my work, I used and cited all of these, even if just for informative reasons.

# Model Development

### GitHub Project Repositories with source code

**[FibrosisNet](https://github.com/darwinai/FibrosisNet)**

 - This open-source project introduces a deep convolutional neural network designed for predicting pulmonary fibrosis progression from chest CT images
 - **`Ideas`**: Use agains my own CNN, if the performance is overwhelmingly superior consider adopting architectural concepts 

### Fine-Tuning and Optimization

 - ImageNet
 - Adam
 - Optuna
 - Early-Stopping

### Ensembles 



### Transfer Learning 

 For feature extraction, we can use transfer learning due to the reduced dataset size, in case we want to use different classifiers (I want to try boosting methods).


### 3d Spacial Dependencies

### Classification vs Segmentation


### Architectural Inspiration


 - **`Comparison_of_CNNs_for_Lung_Biopsy_Images_Classification`** (4 cnns, with different ds sizes as well)
 - **`Lung_Pattern_Classification_for_Interstitial_Lung_Diseases_Using_a_Deep_Convolutional_Neural_Network`** (new evaluation metric, need to learn about it before using. Uses LeakyReLU and different hyperparameter combinations)
 - **``**

# Dataset Issues

#### Data Normalization

 - HU normalization

#### Image Preprocessing

**[Lung Fibrosis DICOM Image Preprocessing](https://www.kaggle.com/code/digvijayyadav/lung-fibrosis-dicom-image-preprocessing)**

 - Big project (very well documented), developed for a competition, will allow me to learn how to correctly input dicom files for feature extractor training
 - **`Ideas`**: slice plotting, scan gifs, segmentation using Watershed

**[LIDC-IDRI competition playlist](https://www.youtube.com/playlist?list=PLQVvvaa0QuDd5meH8cStO9cMi98tPT12_)**

 - **`Ideas`**: kickstarter for initial data processing pipeline, as well as architectural concepts
  

#### Small dataset → Data augmentation 

**[Medical image data augmentation: techniques, comparisons and interpretations](https://link.springer.com/article/10.1007/s10462-023-10453-z#Sec17)**

 - Article related to data augmentation options, uses the LIDC-IDRI (and others, pg27 of the pdf) dataset to apply 11 different methods, comparing their performance after classification
 - Uses ResNet to prove effectiveness of said augmentations, allows me to justify choice of augmentation method
 - Includes number of images that result from a single image's augmentation, very useful for preliminary dataset size analysis
 - **`Ideas`**: method ideas for my dataset, maybe comparing performance difference regarding LIDC-IDRI obtained results (only relative, as in if the best methods are the same for both datasets)

This pdf citation might be useful:

"Optimization has been provided by Adam, and activation has been provided by ReLU. The initial learning rate has been chosen as 0.0003, and cross-entropy loss has been computed to update the weights in the training phase. The number of epochs and mini-batch size have been set as 6 and 10, respectively."


**After analysis of the augmentation options:**
###### *"The ideal augmentation methods should provide high classification accuracy, sensitivity, specificity, F1-score, and MCC while also significantly increasing the number of training samples. These choices should balance strong classification performance and dataset growth, which enhancing model robustness and generalization."*

1. **9th Augmentation Method (Translation and Shearing)**
   - **Performance:** Highest accuracy (85.74%), high sensitivity (86.16%), excellent specificity (97.65%), strong F1-score (84.35%), and highest MCC (82.81%).
   - **Dataset Expansion:** Increases dataset by **10x**.

2. **10th Augmentation Method (Translation, Shearing, and Rotation)**
   - **Performance:** Second highest accuracy (83.82%), high sensitivity (84.83%), excellent specificity (96.29%), strong F1-score (82.24%), and high MCC (81.24%).
   - **Dataset Expansion:** Increases dataset by **10x**.

3. **4th Augmentation Method (Gaussian Noise Addition)**
   - **Performance:** High accuracy (84.12%), high sensitivity (85.10%), excellent specificity (96.11%), strong F1-score (83.15%), and high MCC (81.62%).
   - **Dataset Expansion:** Increases dataset by **3x**.

4. **7th Augmentation Method (Gaussian Noise Addition and Rotation)**
   - **Performance:** Good accuracy (75.00%), high sensitivity (83.38%), decent specificity (76.67%), strong F1-score (76.92%), and moderate MCC (54.71%).
   - **Dataset Expansion:** Increases dataset by **30x**.

5. **6th Augmentation Method (Salt and Pepper Noise Addition and Shearing)**
   - **Performance:** Moderate accuracy (62.38%) but excellent specificity (93.76%), making it useful for reducing false positives.
   - **Dataset Expansion:** Increases dataset by **30x**.

#### Elaborating on choices (AI for format and summarize)

1. **Translation and Shearing (9th method)**
   - **What it does:** Moves (translates) the image in different directions and skews (shears) it.  
   - **Why it’s good:** Simulates real-world variations in CT scans due to patient positioning and scanner angle changes. It also had the **best classification performance**.  

2. **Translation, Shearing, and Rotation (10th method)**  
   - **What it does:** Combines translation, shearing, and rotation to modify orientation.  
   - **Why it’s good:** Introduces diverse spatial transformations, making the model robust to patient movement and scanner variability. Achieved **second-best classification performance**.  

3. **Gaussian Noise Addition (4th method)**  
   - **What it does:** Adds small random noise that mimics scanner artifacts.  
   - **Why it’s good:** Helps the model generalize by reducing overfitting to noise-free images. It provided **high classification accuracy**.  

4. **Gaussian Noise Addition and Rotation (7th method)**  
   - **What it does:** Introduces noise (as in the 4th method) and rotates images.  
   - **Why it’s good:** Enhances robustness to both scanner artifacts and slight rotations due to patient positioning. Also **expands the dataset significantly (30x)**.  

5. **Salt and Pepper Noise Addition and Shearing (6th method)**  
   - **What it does:** Adds random black-and-white noise (salt & pepper) and applies shearing.  
   - **Why it’s good:** Simulates imaging artifacts and distortions seen in medical scans. **Largest dataset expansion (30x), helping model learn better**.  



#### Imbalanced Classes


# Results

### Lack of Standardized Metrics for Fibrosis 

 Standard classification metrics (accuracy, AUC) may not fully capture fibrosis progression. Segmentation-based metrics like Dice score, Jaccard index, and Mean Surface Distance (MSD) might be better.

 Reading data augmentatio article might help with that.

### Explainability


### Ethical Concerns

 - Wrong use, unnecessary (false positive) or delayed/ignored (false negative) biopsies, segmentation (where and how much) and classification (has, doesn't) for decision making, regarding both accuracy and benefits for treatment (sometimes seeing that it's there with more accuracy is better than saying where it is with less accuracy)
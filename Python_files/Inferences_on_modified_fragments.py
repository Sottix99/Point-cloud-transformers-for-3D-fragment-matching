#!/usr/bin/env python
# coding: utf-8

# ## Import packages and functions

# In[1]:


import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import random
import pickle
import datetime
import os
import torch.nn.functional as F
from scipy.stats import special_ortho_group
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import wandb
import itertools
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
from torch.nn.functional import softmax


# In[2]:


from Transformations import apply_translation
from PCT_definition import sample_and_group, farthest_point_sample, index_points, square_distance, Local_op, SA_Layer, StackedAttention
from Model_7_features import use_GPU, PairModel1, Branch


# Set the seeds:

# In[ ]:


seed=999
os.environ["PL_GLOBAL_SEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# ## First Case: Modification of random points with their mean

# import the proper functions:

# In[ ]:


from Mod_pointclouds import apply_obscure


# Load the data:

# In[ ]:


with open('C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\test_subset_random.pkl', 'rb') as file:
    test_randomized = pickle.load(file)


# Activate GPU:

# In[ ]:


device=use_GPU()


# Define the Test dataloader:

# In[ ]:


test_loader_basic = DataLoader(test_randomized, batch_size=16) 


# Inference cycle:

# In[ ]:


percent_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9]
results = []

# loop through percent_values
for percent in percent_values:

    # load pre-trained model each time
    model = PairModel1().to(device)
    model.double()
    W_stored = torch.load(r'C:\\Users\\Alessandro\\Desktop\\Weights\\Main_epoch116.pt')
    model.load_state_dict(W_stored)

    # set up loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0001)
    num_epochs = 1
    best_val_accuracy = 0.0 

    # set checkpoint directory
    checkpoint_dir = r'C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\Check_points'

    model.eval()  
    progress_bar_val = tqdm(test_loader_basic, desc=f'Validation', leave=False)    
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0
    val_contrast = 0.0
    y_true_val = []
    y_pred_val = []
    y_scores_val = [] 

    with torch.no_grad():
        for val_batch in progress_bar_val:
            val_frags_a, val_frags_b, val_labels = val_batch
             
            # translations in the origin 
            val_frags_a = apply_translation(val_frags_a)
            val_frags_b = apply_translation(val_frags_b)
 
            # obscure random points
            val_frags_b = apply_obscure(val_frags_b, percent)
            val_frags_a = apply_obscure(val_frags_a, percent)

            val_frags_a = val_frags_a.double().to(device)
            val_frags_b = val_frags_b.double().to(device)

            val_labels = val_labels.to(device)
                
            val_outputs = model(val_frags_a, val_frags_b)
            
            val_outputs_probs = softmax(val_outputs, dim=1)
            val_loss += criterion(val_outputs, val_labels).item()
          
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total_samples += val_labels.size(0)
            val_correct_predictions += (val_predicted == val_labels).sum().item()
            
            y_true_val.extend(val_labels.cpu().numpy())
            y_pred_val.extend(val_predicted.cpu().numpy())
            y_scores_val.extend(val_outputs_probs.cpu().numpy()[:, 1])  
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_val, y_scores_val)

    # AUC
    roc_auc = roc_auc_score(y_true_val, y_scores_val)

    # F1-score, Accuracy, Val_loss
    f1_val = f1_score(y_true_val, y_pred_val, average='weighted') 
    val_accuracy = val_correct_predictions / val_total_samples
    val_loss /= len(test_loader_basic)

    # store the results
    results.append({
        'percent': percent,
        'val_accuracy': val_accuracy,
        'f1_val': f1_val,
        'roc_auc': roc_auc,
        'val_loss': val_loss,
        'conf_matrix': confusion_matrix(y_true_val, y_pred_val)
    })
    
    # find the indices of the cases predicted well, predicted wrong, the cases predicted as zero and the cases as one
    correctly_predicted_indices = np.where(np.array(y_true_val) == np.array(y_pred_val))[0]
    incorrectly_predicted_indices = np.where(np.array(y_true_val) != np.array(y_pred_val))[0]
    preds_0_indices = np.where(np.array(y_pred_val) == 0)[0]
    preds_1_indices = np.where(np.array(y_pred_val) == 1)[0]

    # create the directory if it doesn't exist
    directory = 'first_case'
    os.makedirs(directory, exist_ok=True)
    

    # save all the indices associated with the various predictions:
    with open(f'{directory}\\correctly_predicted_indices'+'_'+str(percent)+'.pkl', 'wb') as file:
     pickle.dump(correctly_predicted_indices, file)

    with open(f'{directory}\\incorrectly_predicted_indices'+'_'+str(percent)+'.pkl', 'wb') as file:
     pickle.dump(incorrectly_predicted_indices, file)

    with open(f'{directory}\\preds_0_indices'+'_'+str(percent)+'.pkl', 'wb') as file:
     pickle.dump(preds_0_indices, file)

    with open(f'{directory}\\preds_1_indices'+'_'+str(percent)+'.pkl', 'wb') as file:
     pickle.dump(preds_1_indices, file)

# store all the metrics     
with open(f'{directory}\\results_tot.pkl', 'wb') as file:
    pickle.dump(results, file)

# print
for result in results:
    print(f"Percent: {result['percent']}")
    print("Validation Accuracy:", result['val_accuracy'])
    print("Validation F1 Score:", result['f1_val'])
    print("Validation AUC:", result['roc_auc'])
    print("Validation Loss", result['val_loss'])
    print("Confusion Matrix:")
    print(result['conf_matrix'])
    print("\n")


# ## Second Case: Modification of selected points with their mean

# import the proper functions:

# In[ ]:


from Mod_pointclouds import apply_obscure_with_mean_selected_points


# Load the data:

# In[ ]:


with open('C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\test_subset_random.pkl', 'rb') as file:
    test_randomized = pickle.load(file)


# Activate GPU:

# In[ ]:


device=use_GPU()


# Define the Test dataloader:

# In[ ]:


test_loader_basic = DataLoader(test_randomized, batch_size=16) 


# Inference cycle:

# In[ ]:


axes_values = [0, 1, 2]
fraction_values = [3, 2.5, 2, 1.5, 1.2]
results = []

# loop through axes and fractions
for ax in tqdm(axes_values, desc='Axes'):
    for fraction in tqdm(fraction_values, desc='Fractions', leave=False):
        
        # load pre-trained model each time
        model = PairModel1().to(device)
        model.double()
        W_stored = torch.load(r'C:\\Users\\Alessandro\\Desktop\\Weights\\Main_epoch116.pt')
        model.load_state_dict(W_stored)
        
        # set up loss and optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0001)
        num_epochs = 1
        best_val_accuracy = 0.0
        
        # set checkpoint directory
        checkpoint_dir = r'C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\Check_points'

        model.eval()
        progress_bar_val = tqdm(test_loader_basic, desc=f'Validation', leave=False)
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        val_contrast = 0.0
        y_true_val = []
        y_pred_val = []
        y_scores_val = []

        with torch.no_grad():
            for val_batch in progress_bar_val:
                val_frags_a, val_frags_b, val_labels = val_batch
                
                # translations in the origin 
                val_frags_a = apply_translation(val_frags_a)
                val_frags_b = apply_translation(val_frags_b)

                # obscure selected points with their mean
                val_frags_b = apply_obscure_with_mean_selected_points(val_frags_b, ax, frac=fraction)
                val_frags_a = apply_obscure_with_mean_selected_points(val_frags_a, ax, frac=fraction)

                val_frags_a = val_frags_a.double().to(device)
                val_frags_b = val_frags_b.double().to(device)

                val_labels = val_labels.to(device)

                val_outputs = model(val_frags_a, val_frags_b)

                val_outputs_probs = softmax(val_outputs, dim=1)
                val_loss+= criterion(val_outputs, val_labels).item()

                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total_samples += val_labels.size(0)
                val_correct_predictions += (val_predicted == val_labels).sum().item()

                y_true_val.extend(val_labels.cpu().numpy())
                y_pred_val.extend(val_predicted.cpu().numpy())
                y_scores_val.extend(val_outputs_probs.cpu().numpy()[:, 1])

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_val, y_scores_val)

        # AUC
        roc_auc = roc_auc_score(y_true_val, y_scores_val)

        # F1-score, Accuracy, Val_loss
        f1_val = f1_score(y_true_val, y_pred_val, average='weighted')
        val_accuracy = val_correct_predictions / val_total_samples
        val_loss /= len(test_loader_basic)

        # store the results 
        results.append({
            'ax': ax,
            'fraction': fraction,
            'val_accuracy': val_accuracy,
            'f1_val': f1_val,
            'roc_auc': roc_auc,
            'val_loss': val_loss,
            'conf_matrix': confusion_matrix(y_true_val, y_pred_val)
        })
        
        # find the indices of the cases predicted well, predicted wrong, the cases predicted as zero and the cases as one
        correctly_predicted_indices = np.where(np.array(y_true_val) == np.array(y_pred_val))[0]
        incorrectly_predicted_indices = np.where(np.array(y_true_val) != np.array(y_pred_val))[0]
        preds_0_indices = np.where(np.array(y_pred_val) == 0)[0]
        preds_1_indices = np.where(np.array(y_pred_val) == 1)[0]
        
        # create the directory if it doesn't exist
        directory = 'second_case_mean'
        os.makedirs(directory, exist_ok=True)

        # save all the indices associated with the various predictions:
        with open(f'{directory}\\correctly_predicted_indices{ax}_{fraction}.pkl', 'wb') as file:
            pickle.dump(correctly_predicted_indices, file)

        with open(f'{directory}\\incorrectly_predicted_indices{ax}_{fraction}.pkl', 'wb') as file:
            pickle.dump(incorrectly_predicted_indices, file)

        with open(f'{directory}\\preds_0_indices{ax}_{fraction}.pkl', 'wb') as file:
            pickle.dump(preds_0_indices, file)

        with open(f'{directory}\\preds_1_indices{ax}_{fraction}.pkl', 'wb') as file:
            pickle.dump(preds_1_indices, file)


# store all the metrics   
with open(f'{directory}\\results_tot.pkl', 'wb') as file:
    pickle.dump(results, file)

# print
for result in results:
    print("Axis:", result['ax'])
    print("Fraction:", result['fraction'])
    print("Validation Accuracy:", result['val_accuracy'])
    print("Validation F1 Score:", result['f1_val'])
    print("Validation AUC:", result['roc_auc'])
    print("Validation Loss", result['val_loss'])
    print("Confusion Matrix:")
    print(result['conf_matrix'])
    print("\n")


# ## Third Case: Modification of selected points with coordinates from existing points

# import the proper functions:

# In[ ]:


from Mod_pointclouds import apply_obscure_selected_points


# Load the data:

# In[ ]:


with open('C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\test_subset_random.pkl', 'rb') as file:
    test_randomized = pickle.load(file)


# Activate GPU:

# In[ ]:


device=use_GPU()


# Define the Test dataloader:

# In[ ]:


test_loader_basic = DataLoader(test_randomized, batch_size=16) 


# Inference cycle:

# In[ ]:


axes_values = [0, 1, 2]
fraction_values = [3, 2.5, 2, 1.5, 1.2]
results = []

# loop through axes and fractions
for ax in tqdm(axes_values, desc='Axes'):
    for fraction in tqdm(fraction_values, desc='Fractions', leave=False):
        
        # load pre-trained model each time
        model = PairModel1().to(device)
        model.double()
        W_stored = torch.load(r'C:\\Users\\Alessandro\\Desktop\\Weights\\Main_epoch116.pt')
        model.load_state_dict(W_stored)
        
        # set up loss and optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0001)
        num_epochs = 1
        best_val_accuracy = 0.0
        
        # set checkpoint directory
        checkpoint_dir = r'C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\Check_points'

        model.eval()
        progress_bar_val = tqdm(test_loader_basic, desc=f'Validation', leave=False)
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        val_contrast = 0.0
        y_true_val = []
        y_pred_val = []
        y_scores_val = []

        with torch.no_grad():
            for val_batch in progress_bar_val:
                val_frags_a, val_frags_b, val_labels = val_batch
                
                # translations in the origin 
                val_frags_a = apply_translation(val_frags_a)
                val_frags_b = apply_translation(val_frags_b)

                # obscure selected points with coordinates from existing points
                val_frags_b = apply_obscure_selected_points(val_frags_b, ax, frac=fraction)
                val_frags_a = apply_obscure_selected_points(val_frags_a, ax, frac=fraction)

                val_frags_a = val_frags_a.double().to(device)
                val_frags_b = val_frags_b.double().to(device)

                val_labels = val_labels.to(device)

                val_outputs = model(val_frags_a, val_frags_b)

                val_outputs_probs = softmax(val_outputs, dim=1)
                val_loss+= criterion(val_outputs, val_labels).item()

                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total_samples += val_labels.size(0)
                val_correct_predictions += (val_predicted == val_labels).sum().item()

                y_true_val.extend(val_labels.cpu().numpy())
                y_pred_val.extend(val_predicted.cpu().numpy())
                y_scores_val.extend(val_outputs_probs.cpu().numpy()[:, 1])

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_val, y_scores_val)

        # AUC
        roc_auc = roc_auc_score(y_true_val, y_scores_val)

        # F1-score, Accuracy, Val_loss
        f1_val = f1_score(y_true_val, y_pred_val, average='weighted')
        val_accuracy = val_correct_predictions / val_total_samples
        val_loss /= len(test_loader_basic)

        # store the results 
        results.append({
            'ax': ax,
            'fraction': fraction,
            'val_accuracy': val_accuracy,
            'f1_val': f1_val,
            'roc_auc': roc_auc,
            'val_loss': val_loss,
            'conf_matrix': confusion_matrix(y_true_val, y_pred_val)
        })
        
        # find the indices of the cases predicted well, predicted wrong, the cases predicted as zero and the cases as one
        correctly_predicted_indices = np.where(np.array(y_true_val) == np.array(y_pred_val))[0]
        incorrectly_predicted_indices = np.where(np.array(y_true_val) != np.array(y_pred_val))[0]
        preds_0_indices = np.where(np.array(y_pred_val) == 0)[0]
        preds_1_indices = np.where(np.array(y_pred_val) == 1)[0]
        
        # create the directory if it doesn't exist
        directory = 'third_case_existing_points'
        os.makedirs(directory, exist_ok=True)

        # save all the indices associated with the various predictions:
        with open(f'{directory}\\correctly_predicted_indices{ax}_{fraction}.pkl', 'wb') as file:
            pickle.dump(correctly_predicted_indices, file)

        with open(f'{directory}\\incorrectly_predicted_indices{ax}_{fraction}.pkl', 'wb') as file:
            pickle.dump(incorrectly_predicted_indices, file)

        with open(f'{directory}\\preds_0_indices{ax}_{fraction}.pkl', 'wb') as file:
            pickle.dump(preds_0_indices, file)

        with open(f'{directory}\\preds_1_indices{ax}_{fraction}.pkl', 'wb') as file:
            pickle.dump(preds_1_indices, file)


# store all the metrics   
with open(f'{directory}\\results_tot.pkl', 'wb') as file:
    pickle.dump(results, file)

# print
for result in results:
    print("Axis:", result['ax'])
    print("Fraction:", result['fraction'])
    print("Validation Accuracy:", result['val_accuracy'])
    print("Validation F1 Score:", result['f1_val'])
    print("Validation AUC:", result['roc_auc'])
    print("Validation Loss", result['val_loss'])
    print("Confusion Matrix:")
    print(result['conf_matrix'])
    print("\n")


# ## Fourth case: Modification of selected points by adding noise (generating new points)

# import the proper functions:

# In[ ]:


from Mod_pointclouds import apply_generate_selected_points


# Load the data:

# In[ ]:


with open('C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\test_subset_random.pkl', 'rb') as file:
    test_randomized = pickle.load(file)


# Activate GPU:

# In[ ]:


device=use_GPU()


# Define the Test dataloader:

# In[ ]:


test_loader_basic = DataLoader(test_randomized, batch_size=16) 


# Inference cycle:

# In[ ]:


axes_values = [0, 1, 2]
fraction_values = [3, 2.5, 2, 1.5, 1.2]
results = []

# loop through axes and fractions
for ax in tqdm(axes_values, desc='Axes'):
    for fraction in tqdm(fraction_values, desc='Fractions', leave=False):
        
        # load pre-trained model each time
        model = PairModel1().to(device)
        model.double()
        W_stored = torch.load(r'C:\\Users\\Alessandro\\Desktop\\Weights\\Main_epoch116.pt')
        model.load_state_dict(W_stored)
        
        # set up loss and optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0001)
        num_epochs = 1
        best_val_accuracy = 0.0
        
        # set checkpoint directory
        checkpoint_dir = r'C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\Check_points'

        model.eval()
        progress_bar_val = tqdm(test_loader_basic, desc=f'Validation', leave=False)
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        val_contrast = 0.0
        y_true_val = []
        y_pred_val = []
        y_scores_val = []

        with torch.no_grad():
            for val_batch in progress_bar_val:
                val_frags_a, val_frags_b, val_labels = val_batch
                
                # translations in the origin 
                val_frags_a = apply_translation(val_frags_a)
                val_frags_b = apply_translation(val_frags_b)

                # modify selected points by adding noise to coordinates
                val_frags_b = apply_generate_selected_points(val_frags_b, ax, frac=fraction)
                val_frags_a = apply_generate_selected_points(val_frags_a, ax, frac=fraction)

                val_frags_a = val_frags_a.double().to(device)
                val_frags_b = val_frags_b.double().to(device)

                val_labels = val_labels.to(device)

                val_outputs = model(val_frags_a, val_frags_b)

                val_outputs_probs = softmax(val_outputs, dim=1)
                val_loss+= criterion(val_outputs, val_labels).item()

                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total_samples += val_labels.size(0)
                val_correct_predictions += (val_predicted == val_labels).sum().item()

                y_true_val.extend(val_labels.cpu().numpy())
                y_pred_val.extend(val_predicted.cpu().numpy())
                y_scores_val.extend(val_outputs_probs.cpu().numpy()[:, 1])

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_val, y_scores_val)

        # AUC
        roc_auc = roc_auc_score(y_true_val, y_scores_val)

        # F1-score, Accuracy, Val_loss
        f1_val = f1_score(y_true_val, y_pred_val, average='weighted')
        val_accuracy = val_correct_predictions / val_total_samples
        val_loss /= len(test_loader_basic)

        # store the results 
        results.append({
            'ax': ax,
            'fraction': fraction,
            'val_accuracy': val_accuracy,
            'f1_val': f1_val,
            'roc_auc': roc_auc,
            'val_loss': val_loss,
            'conf_matrix': confusion_matrix(y_true_val, y_pred_val)
        })
        
        # find the indices of the cases predicted well, predicted wrong, the cases predicted as zero and the cases as one
        correctly_predicted_indices = np.where(np.array(y_true_val) == np.array(y_pred_val))[0]
        incorrectly_predicted_indices = np.where(np.array(y_true_val) != np.array(y_pred_val))[0]
        preds_0_indices = np.where(np.array(y_pred_val) == 0)[0]
        preds_1_indices = np.where(np.array(y_pred_val) == 1)[0]
        
        # create the directory if it doesn't exist
        directory = 'fourth_case'
        os.makedirs(directory, exist_ok=True)

        # save all the indices associated with the various predictions:
        with open(f'{directory}\\correctly_predicted_indices{ax}_{fraction}.pkl', 'wb') as file:
            pickle.dump(correctly_predicted_indices, file)

        with open(f'{directory}\\incorrectly_predicted_indices{ax}_{fraction}.pkl', 'wb') as file:
            pickle.dump(incorrectly_predicted_indices, file)

        with open(f'{directory}\\preds_0_indices{ax}_{fraction}.pkl', 'wb') as file:
            pickle.dump(preds_0_indices, file)

        with open(f'{directory}\\preds_1_indices{ax}_{fraction}.pkl', 'wb') as file:
            pickle.dump(preds_1_indices, file)


# store all the metrics   
with open(f'{directory}\\results_tot.pkl', 'wb') as file:
    pickle.dump(results, file)

# print
for result in results:
    print("Axis:", result['ax'])
    print("Fraction:", result['fraction'])
    print("Validation Accuracy:", result['val_accuracy'])
    print("Validation F1 Score:", result['f1_val'])
    print("Validation AUC:", result['roc_auc'])
    print("Validation Loss", result['val_loss'])
    print("Confusion Matrix:")
    print(result['conf_matrix'])
    print("\n")


#!/usr/bin/env python
# coding: utf-8

# ## Import packages and functions

# In[ ]:


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


# In[ ]:


from From_clusters_to_couples import CreateCouples
from Transformations import apply_translation, apply_randomrotations
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


# The code is the same as Main.ipynb, the only difference is that apply_randomrotations() is not applied in the train loop

# ## Data

# Load the data:

# In[ ]:


train = torch.load("C:\\Users\\Alessandro\\Desktop\\Tesi\\pair_dataset\\dataset_1024_AB\\train_pair_dataset_REG.pt")

with open('C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\val_subset_random.pkl', 'rb') as file:
    val_randomized = pickle.load(file)

with open('C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\test_subset_random.pkl', 'rb') as file:
    test_randomized = pickle.load(file)


# Unroll the pairs from the clusters:

# In[ ]:


train_couples = CreateCouples(train)


# Identify positive and negative couples:

# In[ ]:


train_couples_0 = [item for item in train_couples if item[2] == 0]
train_couples_1 = [item for item in train_couples if item[2] == 1]


# Define the validation dataloader:

# In[ ]:


val_loader_basic = DataLoader(val_randomized, batch_size=16) 


# ## Train

# Activate GPU:

# In[ ]:


device=use_GPU()


# Define the model:

# In[ ]:


model = PairModel1().to(device)
model.double()


# Only when using pre-trained weights:

# In[ ]:


W_stored = torch.load(r'C:\\Users\\Alessandro\\Desktop\\Weights\\Main_not_aug_epoch98.pt')
model.load_state_dict(W_stored)


# Set hyperparameters and prepare Weights & Biases:

# In[ ]:


wandb.init(
      project="train", 
      notes = "No Data Augmentation",
      config={
      "learning_rate": 0.00005,
      "architecture": "Model2_mod",
      "epochs": 20,
      "weight_decay": 0.0001,
      "W_crossentropy":1,
      "couples_per_epoch": 10000,
      "seed": seed # the seed defined at the beginning
      })
      
config = wandb.config
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
num_epochs = config.epochs
best_val_accuracy = 0.0 


# Sets the path where the model weights will be stored.
# make sure to enter the right destination folder, otherwise the training cycle will stop by not finding the folder
checkpoint_dir = r'C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\Check_points'


# Training cycle:

# In[ ]:


checkpoint_interval = 1
epoch_number = 0  


for epoch in range(num_epochs):
    model.train() 

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    y_true = []
    y_pred = []
    
    random.shuffle(train_couples_0)
    random.shuffle(train_couples_1)

    # 5000 positive pairs and 5000 negative pairs are used
    balanced_train_list = []
    couples_per_epoch = config.couples_per_epoch
    
    sampled_couples_0 = random.sample(train_couples_0, couples_per_epoch // 2)
    sampled_couples_1 = random.sample(train_couples_1, couples_per_epoch // 2)

    balanced_train_list.extend(sampled_couples_0)
    balanced_train_list.extend(sampled_couples_1)
        
    random.shuffle(balanced_train_list)
    
    train_loader = DataLoader(balanced_train_list, batch_size=16,shuffle=True) 
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

    ###########
    ## Train ##
    ###########


    for batch_data in progress_bar:

        optimizer.zero_grad() 
        frags_a, frags_b, labels = batch_data
        
        # No random rotations applied

        # Translations in the origin
        frags_a = apply_translation(frags_a)
        frags_b = apply_translation(frags_b)

        frags_a = frags_a.double().to(device)
        frags_b = frags_b.double().to(device)
        labels = labels.to(device)

        # Model output
        outputs = model(frags_a, frags_b)
        loss = criterion(outputs, labels)

        # Optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        progress_bar.set_postfix({'Loss': loss.item(), 'Accuracy': correct_predictions / total_samples})
        
    # Metrics computation
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = correct_predictions / total_samples
    train_loss = total_loss/len(train_loader)

    metrics_train = {"train_loss": train_loss, 
                       "accuracy": accuracy,
                       "f1":f1}
    
    # Upload results to wandb
    wandb.log(metrics_train) 



    ###############
    ## Inference ##
    ###############

    model.eval()  
    
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
        for val_batch in val_loader_basic:
            val_frags_a, val_frags_b, val_labels = val_batch
            
            # Translations in the origin
            val_frags_a = apply_translation(val_frags_a)
            val_frags_b = apply_translation(val_frags_b)

            val_frags_a = val_frags_a.double().to(device)
            val_frags_b = val_frags_b.double().to(device)
            val_labels = val_labels.to(device)

            # Model output
            val_outputs = model(val_frags_a, val_frags_b)

            val_loss += criterion(val_outputs, val_labels).item()
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total_samples += val_labels.size(0)
            val_correct_predictions += (val_predicted == val_labels).sum().item()
            y_true_val.extend(val_labels.cpu().numpy())
            y_pred_val.extend(val_predicted.cpu().numpy())

    # Metrics computation
    f1_val = f1_score(y_true_val, y_pred_val, average='weighted')        
    val_accuracy = val_correct_predictions / val_total_samples
    val_loss /= len(val_loader_basic)
    val_metrics = {"val_loss": val_loss, 
                       "val_accuracy": val_accuracy,
                       "f1_val": f1_val}
    
    # Upload results to wandb
    wandb.log(val_metrics)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {accuracy:.4f}, F1_train:{f1:.4f} ',
    f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, F1_val : {f1_val:.4f}')


    # Store the results    
    current_time = datetime.datetime.now()
    checkpoint_name = f"{current_time.strftime('%m%d_%H%M%S')}_{epoch + 1}.pt"    
    # make sure to enter the right destination folder, otherwise the training cycle will stop by not finding the folder
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name) 
    torch.save(model.state_dict(), checkpoint_path)
    wandb.run.log_artifact(checkpoint_path,name=str(epoch+1))


# ## Inference time:

# Pre-trained weight loading:

# In[ ]:


W_stored = torch.load(r'C:\\Users\\Alessandro\\Desktop\\Weights\\Main_not_aug_epoch98.pt')
model.load_state_dict(W_stored)


# Define the Test dataloader:

# In[ ]:


test_loader_basic = DataLoader(test_randomized, batch_size=16) 


# Set hyperparameters:

# In[ ]:


criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0001)
num_epochs = 1
best_val_accuracy = 0.0 
# Sets the path where the model weights will be stored.
checkpoint_dir = r'C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\Check_points'


# Inference cycle:

# In[ ]:


model.eval()  
progress_bar_val = tqdm(test_loader_basic, desc=f'Validation', leave=False)    
val_loss = 0.0
val_correct_predictions = 0
val_total_samples = 0

y_true_val = []
y_pred_val = []
y_scores_val = []

with torch.no_grad():
    for val_batch in progress_bar_val:
        val_frags_a, val_frags_b, val_labels = val_batch
        
        # Translations in the origin
        val_frags_a = apply_translation(val_frags_a)
        val_frags_b = apply_translation(val_frags_b)
        
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

# Print
print("Test Accuracy:", val_accuracy)
print("Test F1 Score:", f1_val)
print("Test AUC:", roc_auc)
print("Test Loss", val_loss)

conf_matrix = confusion_matrix(y_true_val, y_pred_val)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)


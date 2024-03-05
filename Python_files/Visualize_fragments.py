#!/usr/bin/env python
# coding: utf-8

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
from torch.nn.functional import softmax
import open3d  as o3d


# In[ ]:


from Visualization_functions import *
from Mod_pointclouds import *


# Set the seeds:

# In[ ]:


seed=999
os.environ["PL_GLOBAL_SEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# In this notebook are presented examples of how visualization functions can be applied in the data. Clearly, only a few example cases have been reported

# Load the data:

# In[ ]:


with open('C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\test_subset_random.pkl', 'rb') as file:
    test_randomized = pickle.load(file)


# ### Single fragments

# Print a single fragment:

# In[ ]:


visualize_single_pointcloud(test_randomized,0,1,0) # Fragment 1 from Couple 0 without random rotation applied.


# In[ ]:


visualize_single_pointcloud(test_randomized,0,1,1) # Fragment 1 from Couple 0 with a random rotation applied.


# It is possible to directly introduce a point cloud as input to the visualize_single_pointcloud_from_input() function:

# In[ ]:


a= test_randomized[0][1]


# In[ ]:


visualize_single_pointcloud_from_input(a,0) # Fragment 1 from Couple 0 without random rotation applied.


# Therefore, in case modifications are going to be made to the input tensor, it is possible to obtain the views of the modified fragment:

# In[ ]:


a= test_randomized[0][1]
a = torch.from_numpy(a)
a= obscure(a,0.9)


# In[ ]:


visualize_single_pointcloud_from_input(a,0) # Fragment 1 from Couple 0 in which obscure() is applied with p=0.9 and no random rotation.


# ### Couples

# Print a Couple:

# In[ ]:


visualize_couple_pointcloud(test_randomized, 0,spacing=1) # Couple 0 where spacing = 1 is applied


# Print a Couple in which obscure() is applied:

# In[ ]:


with open('C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\test_subset_random.pkl', 'rb') as file:
    test_randomized = pickle.load(file)


# In[ ]:


visualize_couple_pointcloud_obscure(test_randomized, 0, p=0.9, spacing=1) # Couple 0 where obscure() is applied with p=0.9


# Print a Couple in which obscure_with_mean_selected_points() is applied:

# In[ ]:


with open('C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\test_subset_random.pkl', 'rb') as file:
    test_randomized = pickle.load(file)


# In[ ]:


visualize_couple_pointcloud_obscure_zone_mean(test_randomized, 0, axes=0, f= 3,spacing=1) # Couple 0 where obscure_with_mean_selected_points() is applied with axes=0 and frac =3


# Print a Couple in which obscure_selected_points() is applied:

# In[ ]:


with open('C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\test_subset_random.pkl', 'rb') as file:
    test_randomized = pickle.load(file)


# In[ ]:


visualize_couple_pointcloud_obscure_zone(test_randomized, 0, axes=0, f= 3,spacing=1) # Couple 0 where obscure_points_selected() is applied with axes=0 and frac =3


# Print a Couple in which generate_selected_points() is applied:

# In[ ]:


with open('C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\test_subset_random.pkl', 'rb') as file:
    test_randomized = pickle.load(file)


# In[ ]:


visualize_couple_pointcloud_obscure_zone_generate(test_randomized, 0, axes=0, f= 3,spacing=1) # Couple 0 where generate_selected_points() is applied with axes=0 and frac =3


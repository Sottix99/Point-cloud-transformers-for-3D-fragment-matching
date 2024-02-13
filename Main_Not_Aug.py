#!/usr/bin/env python
# coding: utf-8

# # Import packages e define functions

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
#from pytorch_metric_learning import miners, losses
import wandb
import itertools
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
from torch.nn.functional import softmax


# wandb api key: 96753ce682b21ba1903b8bf57d7786ba04926b15

# Let's fix the seed:

# In[2]:


seed=999
os.environ["PL_GLOBAL_SEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# In[3]:


def CreateCouples(pt):
    
    """ This function, modifies the shape of the tensor to fit the model 

    """
    clusters= pt.shape[0]
    couples = []
    # for each subcluster
    for i in tqdm(range(0,clusters)):
        
        # discover the number of fragments
        n_frags = pt[i][0].shape[0]

        # exract the adj matrix
        matr= pt[i][0]

        # exract the cluster of fragments
        data = pt[i][1]
        
        for j in range(0,n_frags -1):
            
            init = j+1
            for k in range(init,n_frags): 

             couples.append([data[j], data[k], matr[j][k]])
    return couples 

def CreateCouples_cluster(pt):
    """ This function, modifies the shape of the tensor to fit the model """
    clusters = pt.shape[0]
    cluster_list = []

    for i in tqdm(range(clusters)):
        n_frags = pt[i][0].shape[0]
        matr = pt[i][0]
        data = pt[i][1]
        cluster_couples = []

        for j in range(n_frags - 1):
            init = j + 1
            for k in range(init, n_frags):
                cluster_couples.append([data[j], data[k], matr[j][k]])

        cluster_list.append(cluster_couples)

    return cluster_list
def center_in_origin(frag):
    min_vals, _ = torch.min(frag[:, 0:3], axis=0)
    max_vals, _ = torch.max(frag[:, 0:3], axis=0)
    frag[:, 0:3] = (frag[:, 0:3] - min_vals) / (max_vals - min_vals)
    
    return frag
    
def normalize(batch):
    out=[]
    for element in batch:
        out.append(center_in_origin(element))
    out_tensor = torch.stack(out)    
    return out_tensor  



def use_GPU():
    """ This function activates the gpu 
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(0), "is available and being used")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead") 
    return device  



def translate_to_origin(frag):
    """ This function translate each fragment in the origin
    """
    frag[:,:3] -= torch.mean(frag[:,:3]) 
    return frag

def apply_translation(batch):
    """ This function apply translate_to_origin() to each fragment in the batch
    """
    out=[]
    for element in batch:
        out.append(translate_to_origin(element))
    out_tensor = torch.stack(out)    
    return out_tensor


def random_rotation(frag):

    randrot = (torch.rand(3)*360).tolist()
    r = R.from_euler('zyx', randrot, degrees=True)
    frag[:,:3] = torch.from_numpy(r.apply(frag[:,:3]))
    frag[:,3:6] = torch.from_numpy(r.apply(frag[:,3:6]))
    return frag

def apply_randomrotations(batch):
    """ This function apply random_rotation() to each fragment in the batch
    """
    out=[]
    for element in batch:
        out.append(random_rotation(element))
    out_tensor = torch.stack(out)    
    return out_tensor








class ContrastiveLoss(torch.nn.Module):
    def __init__(self, m=2.0):
        super(ContrastiveLoss, self).__init__()
        self.m = m

    def forward(self, y1, y2, d):
        euc_dist = torch.nn.functional.pairwise_distance(y1, y2)

        if d.dim() == 0:  # Se d è uno scalare
            if d == 0:
                return torch.mean(torch.pow(euc_dist, 2))  # Distanza quadratica
            else:  # d == 1
                delta = self.m - euc_dist
                delta = torch.clamp(delta, min=0.0, max=None)
                return torch.mean(torch.pow(delta, 2))
        else:  # Se d è un tensore di valori 0 e 1
            is_same = d == 0
            is_diff = d == 1

            loss_same = torch.pow(euc_dist[is_same], 2).mean() if torch.any(is_same) else torch.tensor(0.0).to(euc_dist.device)
            loss_diff = torch.pow(torch.clamp(self.m - euc_dist[is_diff], min=0.0), 2).mean() if torch.any(is_diff) else torch.tensor(0.0).to(euc_dist.device)

            return (loss_same + loss_diff) / (1.0 + torch.any(is_same).float() + torch.any(is_diff).float())
        
def divide_macro_element(val_couples_c, macro_index):
    macro = val_couples_c[macro_index]
    #random.shuffle(macro)
    nuova_lista=[]
    primi = []
    secondi = []
    terzi = []
    
    for tripletta in macro:
        primi.append(tripletta[0])
        secondi.append(tripletta[1])
        terzi.append(tripletta[2])
    
    primi_divisi = [[x] for x in primi]
    secondi_divisi = [[x] for x in secondi]
    terzi_divisi = [[x] for x in terzi]
    
    nuova_lista.append([primi_divisi, secondi_divisi, terzi_divisi])
    
    primi = []
    secondi = []
    terzi = []

    for macro in nuova_lista:
        primi.append(macro[0])
        secondi.append(macro[1])
        terzi.append(macro[2])

    primi = list(itertools.chain.from_iterable(primi))
    secondi = list(itertools.chain.from_iterable(secondi))
    terzi = list(itertools.chain.from_iterable(terzi))
    val_divisione_nuova = []
    for i in range(len(primi)):
        val_divisione_nuova.append([primi[i], secondi[i], terzi[i]])
    
    return val_divisione_nuova        


# In[4]:


# https://github.com/qq456cvb/Point-Transformers

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 
    
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]

    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)

    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    

class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


# In[5]:


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        
        d_points = 7 # we have 7 features for each point
        self.conv1 = nn.Conv1d(d_points, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = StackedAttention()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        
    def forward(self, x):
        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x= x.double()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)
        
        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0] # Returns the maximum value of all elements in the input tensor. (2 elementes for each vector)
        x = x.view(batch_size, -1) # Returns a new tensor with the same data as the self tensor but of a different shape.
        
        return x
    
    
class PairModel1(nn.Module):
    def __init__(self):
        super().__init__()
        
        output_channels = 2 # it's a binary classification

        self.branch1 = Branch()
        self.branch2 = Branch()
        self.dp1 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
            
        # classificator
        self.linear1 = nn.Linear(2048, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp3 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)
        
    def forward(self, batch_1, batch_2):
        #for param in self.branch1.parameters():
           #param.requires_grad = False
        #for param in self.branch2.parameters():
           #param.requires_grad = False
        x_1 = self.branch1(batch_1)
        x_2 = self.branch2(batch_2)
        #print(x_1.shape)
        #print(x_2.shape)
        x_mult = x_1 * x_2 # let's sum the output of the two branches 
        x_sum = x_1 + x_2
        x = torch.cat((x_mult, x_sum), dim=1) 
        #x = self.dp1(x)

        # classificator
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.dp2(x)
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.dp3(x)
        x = self.linear3(x)
        #x = F.softmax(x, dim=1)
        return x


# # Preparation

# Data loading:

# In[6]:


train = torch.load("C:\\Users\\Alessandro\\Desktop\\Tesi\\pair_dataset\\dataset_1024_AB\\train_pair_dataset_REG.pt")
val = torch.load("C:\\Users\\Alessandro\\Desktop\\Tesi\\pair_dataset\\dataset_1024_AB\\val_pair_dataset_REG.pt")
test = torch.load("C:\\Users\\Alessandro\\Desktop\\Tesi\\pair_dataset\\dataset_1024_AB\\test_pair_dataset_REG.pt")


# In[7]:


# Change the shape of the data
train_couples = CreateCouples(train)
val_couples = CreateCouples(val)
test_couples = CreateCouples(test)


# In[8]:


print("Train",len(train_couples))
print("Val",len(val_couples))
print("Test",len(test_couples))


# In[9]:


train_couples_0 = [item for item in train_couples if item[2] == 0]
train_couples_1 = [item for item in train_couples if item[2] == 1]


# In[10]:


print("Positive Coupels",len(train_couples_1))
print("Negative Coupels",len(train_couples_0))


# In[11]:


# if i want to use the random validation subset 
with open('val_subset_random.pkl', 'rb') as file:
    val_randomized = pickle.load(file)
val_loader_basic_randomized = DataLoader(val_randomized, batch_size=16)    


# In[23]:


#val_1 = [[item[0], item[1],item[2]] for item in val_couples if item[2] == 1]
#val_0 = [[item[0], item[1],item[2]] for item in val_couples if item[2] == 0]
#val_0_s = val_0[0:3000]
#val_1_s = val_1[0:3000]

#val_list = val_1_s + val_0_s
#random.shuffle(val_list) 
#val_loader_basic = DataLoader(val_list, batch_size=16)    


# # Train

# In[12]:


device=use_GPU()


# In[13]:


model = PairModel1().to(device)
model.double()


# In[16]:


wandb.init(
      project="train_lungo", 
      notes = "Not using Data Augumentation",
      config={
      "learning_rate": 0.00005,
      "architecture": "Model2_mod",
      "epochs": 100,
      "weight_decay": 0.0001,
      "W_crossentropy":1,
      "W_contrastive":0,
      "couples_per_epoch": 10000,
      "seed": seed # the seed defined at the beginning
      })
      
config = wandb.config

criterion = nn.CrossEntropyLoss().to(device)
contrast_criterion = ContrastiveLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
num_epochs = config.epochs
best_val_accuracy = 0.0 


# Sets the path where the model weights will be stored.
checkpoint_dir = r'C:\\Users\\Alessandro\\Desktop\\Tesi\\PairModel\\Check_points'


# In[17]:


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
        
        
        #frags_a = apply_randomrotations(frags_a)
        #frags_b = apply_randomrotations(frags_b)
        
        frags_a = apply_translation(frags_a)
        frags_b = apply_translation(frags_b)

        frags_a = frags_a.double().to(device)
        frags_b = frags_b.double().to(device)
        labels = labels.to(device)
        
        outputs = model(frags_a, frags_b)
        loss_ = criterion(outputs, labels)
        
        contrast_loss = contrast_criterion(frags_a, frags_b, labels)
        loss = loss_ + config.W_contrastive*contrast_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        progress_bar.set_postfix({'Loss': loss.item(), 'Accuracy': correct_predictions / total_samples})
        

    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = correct_predictions / total_samples
    train_loss = total_loss/len(train_loader)

    metrics_train = {"train_loss": train_loss, 
                       "accuracy": accuracy,
                       "f1":f1}
    wandb.log(metrics_train) 



    ###############
    ## Inference ##
    ###############

    model.eval()  
    
    val_loss_ = 0.0
    val_correct_predictions = 0
    val_total_samples = 0
    val_contrast = 0.0
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
        for val_batch in val_loader_basic_randomized:
            val_frags_a, val_frags_b, val_labels = val_batch
            

            val_frags_a = apply_translation(val_frags_a)
            val_frags_b = apply_translation(val_frags_b)

            val_frags_a = val_frags_a.double().to(device)
            val_frags_b = val_frags_b.double().to(device)

            val_labels = val_labels.to(device)
            
            val_outputs = model(val_frags_a, val_frags_b)
            val_loss_ += criterion(val_outputs, val_labels).item()
            val_contrast += contrast_criterion(val_frags_a, val_frags_b, val_labels).item()
            val_loss = val_loss_ +config.W_contrastive*val_contrast
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total_samples += val_labels.size(0)
            val_correct_predictions += (val_predicted == val_labels).sum().item()
            y_true_val.extend(val_labels.cpu().numpy())
            y_pred_val.extend(val_predicted.cpu().numpy())

    f1_val = f1_score(y_true_val, y_pred_val, average='weighted')        
    val_accuracy = val_correct_predictions / val_total_samples
    val_loss /= len(val_loader_basic_randomized)
    val_metrics = {"val_loss": val_loss, 
                       "val_accuracy": val_accuracy,
                       "f1_val": f1_val}
    wandb.log(val_metrics)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {accuracy:.4f}, F1_train:{f1:.4f} ',
    f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, F1_val : {f1_val:.4f}')


    # Store the results    
    current_time = datetime.datetime.now()
    checkpoint_name = f"{current_time.strftime('%m%d_%H%M%S')}_{epoch + 1}.pt"    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)
    
    # log the parameters 
    wandb.run.log_artifact(checkpoint_path,name=str(epoch+1))


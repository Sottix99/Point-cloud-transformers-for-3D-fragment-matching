import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from PCT_definition import sample_and_group, farthest_point_sample, index_points, square_distance, Local_op, SA_Layer, StackedAttention


def use_GPU():
    """ This function activates the GPU if available; otherwise, it defaults to using the CPU.

    Output:
    - device (torch.device): The device (GPU or CPU) being used. 
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(0), "is available and being used")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead") 
    return device  




class Branch(nn.Module):
    """ This is the class that defines the branch used in the model of this work, which is responsible for extracting the global features associated with each point cloud.
    This class, represents the original PCT Encoder presented in the paper "PCT: Point Cloud Transformer" by Menghao et al. and implemented in: https://github.com/qq456cvb/Point-Transformers.
    The changes made only affect the size in order to be used for the data used in this study.
    This branch, refers to the case where the model accepts data having 3 features for each point of the input pointclouds.



    Architecture:
    - Input: a tensor ('batch_1' or 'batch_2') of point clouds having 3 features (x,y,z) for each point
    - Two Convolution layers with Batch Normalization and ReLU activations
    - Local operations using 'gather_local_0' and 'gather_local_1'
    - StackedAttention layer ('pt_last')
    - Sequential block containing a convolutional layer, Batch Normalization and LeakyReLU activation ('conv_fuse')

    """
    def __init__(self):
        super().__init__()
        
        d_points = 3 # 3 features for each point
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
    """ This class defines the model for couples classification using two instances of the 'Branch' class. 
    The model input are the two tensors 'batch_1' and 'batch_2' containing the first and second elements of the various pairs, respectively.

    Architecture:
    - Input: the two tensors 'batch_1' and 'batch_2' of point clouds having 3 features (x,y,z) for each point.
    - Two instances of the 'Branch' class where the PCT encoder is located and the global characteristics of each individual point cloud are produced.
    - Aggregation operation that collects as input the two tensors containing the global features of the individual point clouds forming the pairs and returns as output the global features associated with each pair.
    - Dropout layers and ReLU.
    - A combination of linear layers, batch normalizations, and dropouts to classify global features.

    """
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

        # extraction of global features associated with each individual fragment:
        x_1 = self.branch1(batch_1)
        x_2 = self.branch2(batch_2)

        # aggregation:
        x_mult = x_1 * x_2 #  sum the output of the two branches
        x_sum = x_1 + x_2 #  multiply the output of the two branches
        x = torch.cat((x_mult, x_sum), dim=1) # concatenate the sum and multiplication performed to obtain the global features associated with the pair


        # classificator:
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.dp2(x)
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.dp3(x)
        x = self.linear3(x)

        return x
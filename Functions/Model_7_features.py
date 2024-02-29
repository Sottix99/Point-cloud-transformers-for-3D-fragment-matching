import torch
import torch.nn as nn
from PCT_definition import sample_and_group, farthest_point_sample, index_points, square_distance, Local_op, SA_Layer, StackedAttention


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


class ContrastiveLoss(torch.nn.Module):
    """ Contrastive loss (Was no longer used for this work)
    """
    def __init__(self, m=2.0):
        super(ContrastiveLoss, self).__init__()
        self.m = m

    def forward(self, y1, y2, d):
        euc_dist = torch.nn.functional.pairwise_distance(y1, y2)

        if d.dim() == 0:  # if d is a scalar
            if d == 0:
                return torch.mean(torch.pow(euc_dist, 2))  # square distance
            else:  # d == 1
                delta = self.m - euc_dist
                delta = torch.clamp(delta, min=0.0, max=None)
                return torch.mean(torch.pow(delta, 2))
        else:  # if d is a tensor of the type 0 e 1
            is_same = d == 0
            is_diff = d == 1

            loss_same = torch.pow(euc_dist[is_same], 2).mean() if torch.any(is_same) else torch.tensor(0.0).to(euc_dist.device)
            loss_diff = torch.pow(torch.clamp(self.m - euc_dist[is_diff], min=0.0), 2).mean() if torch.any(is_diff) else torch.tensor(0.0).to(euc_dist.device)

            return (loss_same + loss_diff) / (1.0 + torch.any(is_same).float() + torch.any(is_diff).float())


class Branch(nn.Module):
    """ The branch when features = 7
    """
    def __init__(self):
        super().__init__()
        
        d_points = 7 # 7 features for each point
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
    """ The main mdoel
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

        x_1 = self.branch1(batch_1)
        x_2 = self.branch2(batch_2)

        x_mult = x_1 * x_2 # let's sum the output of the two branches 
        x_sum = x_1 + x_2
        x = torch.cat((x_mult, x_sum), dim=1) 


        # classificator
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.dp2(x)
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.dp3(x)
        x = self.linear3(x)

        return x
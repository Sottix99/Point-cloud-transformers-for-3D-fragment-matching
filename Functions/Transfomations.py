

import torch
from scipy.spatial.transform import Rotation as R

def normalize(frag):
    """ This function normalizes each fragment, bringing them to have the same size
    """
    min_vals, _ = torch.min(frag[:, 0:3], axis=0)
    max_vals, _ = torch.max(frag[:, 0:3], axis=0)
    frag[:, 0:3] = (frag[:, 0:3] - min_vals) / (max_vals - min_vals)
    
    return frag
    
def apply_normalize(batch):
    """ This function apply normalize() to each fragment in the batch
    """
    out=[]
    for element in batch:
        out.append(normalize(element))
    out_tensor = torch.stack(out)    
    return out_tensor  


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
    """ This function randomly rotates each fragment by Euler angles
    """
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
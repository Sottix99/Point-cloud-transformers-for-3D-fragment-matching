

import torch
from scipy.spatial.transform import Rotation as R

def normalize(frag):
    """ Normalize each fragment, bringing them to have the same size.

    Args:
    - frag (torch.Tensor): Input fragment tensor.

    Returns:
    - frag (torch.Tensor): Normalized fragment tensor.
    """
    min_vals, _ = torch.min(frag[:, 0:3], axis=0)
    max_vals, _ = torch.max(frag[:, 0:3], axis=0)
    frag[:, 0:3] = (frag[:, 0:3] - min_vals) / (max_vals - min_vals)
    
    return frag
    
def apply_normalize(batch):
    """ Apply normalize() to each fragment in the batch.

    Args:
    - batch (torch.Tensor): Input batch of fragments.

    Returns:
    - out_tensor (torch.Tensor): Normalized batch of fragments.
    """
    out=[]
    for element in batch:
        out.append(normalize(element))
    out_tensor = torch.stack(out)    
    return out_tensor  


def translate_to_origin(frag):
    """ Translate each fragment to the origin.

    Args:
    - frag (torch.Tensor): Input fragment tensor.

    Returns:
    - frag (torch.Tensor): Translated fragment tensor.
    """
    frag[:,:3] -= torch.mean(frag[:,:3]) 
    return frag

def apply_translation(batch):
    """ Apply translate_to_origin() to each fragment in the batch.

    Args:
    - batch (torch.Tensor): Input batch of fragments.

    Returns:
    - out_tensor (torch.Tensor): Translated batch of fragments.
    """
    out=[]
    for element in batch:
        out.append(translate_to_origin(element))
    out_tensor = torch.stack(out)    
    return out_tensor


def random_rotation(frag):
    """ Randomly rotate each fragment by Euler angles (works only if the number of features is at least 6).

    Args:
    - frag (torch.Tensor): Input fragment tensor.

    Returns:
    - frag (torch.Tensor): Rotated fragment tensor.

    """
    randrot = (torch.rand(3)*360).tolist()
    r = R.from_euler('zyx', randrot, degrees=True)
    frag[:,:3] = torch.from_numpy(r.apply(frag[:,:3]))
    frag[:,3:6] = torch.from_numpy(r.apply(frag[:,3:6]))
    return frag




def apply_randomrotations(batch):
    """ Apply random_rotation() to each fragment in the batch (works only if the number of features is at least 6).

    Args:
    - batch (torch.Tensor): Input batch of fragments.

    Returns:
    - out_tensor (torch.Tensor): Batch of fragments with random rotations.
    """
    out=[]
    for element in batch:
        out.append(random_rotation(element))
    out_tensor = torch.stack(out)    
    return out_tensor



# Special functions, when only the first 3 features are used in the data:

def reduce_to_3_features(frag):
    """ Reduce fragment to only the first 3 features.

    Args:
    - frag (torch.Tensor): Input fragment tensor.

    Returns:
    - frag (torch.Tensor): Fragment tensor with only the first 3 features.
    """
    frag = frag[:,:3]
    return frag

def apply_reduce_to_3_features(batch):
    """ Apply reduce_to_3_features() to each fragment in the batch.

    Args:
    - batch (torch.Tensor): Input batch of fragments.

    Returns:
    - out_tensor (torch.Tensor): Batch of fragments with only the first 3 features.
    """
    out=[]
    for element in batch:
        out.append(reduce_to_3_features(element))
    out_tensor = torch.stack(out)    
    return out_tensor


def random_rotation_3_features(frag):
    """ Randomly rotate each fragment by Euler angles (used only in case the number of features is equal to 3).

    Args:
    - frag (torch.Tensor): Input fragment tensor having only the first 3 features.

    Returns:
    - frag (torch.Tensor): Rotated fragment tensor having only the first 3 features.

    """
    randrot = (torch.rand(3)*360).tolist()
    r = R.from_euler('zyx', randrot, degrees=True)
    frag[:,:3] = torch.from_numpy(r.apply(frag[:,:3]))
    return frag

def apply_randomrotations_3_features(batch):
    """ Apply random rotation to each fragment in the batch (used only in case the number of features is equal to 3).

    Args:
    - batch (torch.Tensor): Input batch of fragments, that have only the first 3 features.

    Returns:
    - out_tensor (torch.Tensor): Batch of fragments, that have only the first 3 features, with random rotations.
    """
    out=[]
    for element in batch:
        out.append(random_rotation_3_features(element))
    out_tensor = torch.stack(out)    
    return out_tensor


# Special functions, when only the first 6 features are used in the data:

def reduce_to_6_features(frag):
    """ Reduce fragment to only the first 6 features.

    Args:
    - frag (torch.Tensor): Input fragment tensor.

    Returns:
    - frag (torch.Tensor): Fragment tensor with only the first 6 features.
    """
    frag = frag[:,:6]
    return frag

def apply_reduce_to_6_features(batch):
    """ Apply reduce_to_6_features() to each fragment in the batch.

    Args:
    - batch (torch.Tensor): Input batch of fragments.

    Returns:
    - out_tensor (torch.Tensor): Batch of fragments with only the first 6 features.
    """
    out=[]
    for element in batch:
        out.append(reduce_to_6_features(element))
    out_tensor = torch.stack(out)    
    return out_tensor


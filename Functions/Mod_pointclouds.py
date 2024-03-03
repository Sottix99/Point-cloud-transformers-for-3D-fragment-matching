import torch
import random
import numpy as np

def obscure(tensor, percent):
    """
    Obscure a specified percentage of rows in a pointcloud by replacing them with the mean values of their respective columns.

    Parameters:
    - tensor (torch.Tensor): Input 2D tensor to be obscured.
    - percent (float): Percentage of rows to obscure, specified as a value between 0 and 1.

    Returns:
    - torch.Tensor: Obscured tensor with specified rows replaced by column means.
    """
    # calculate the number of points to obscure based on the specified percentage
    num_points_to_obscure = int(percent * tensor.shape[0])

    # randomly select indices of rows to obscure
    indices_to_obscure = torch.randperm(tensor.shape[0])[:num_points_to_obscure]
    
    # calculate the mean of each column
    column_means = tensor.mean(dim=0)
    
    # replace the chosen points with the mean of each column
    tensor[indices_to_obscure, :] = column_means
    
    return tensor


def apply_obscure(batch,percent):
    """
    Apply obscure() to each element in a batch of tensors.

    Parameters:
    - batch (list of torch.Tensor): List of tensors to be modified.
    - percent (float): Percentage of rows to obscure in each tensor, specified as a value between 0 and 1.

    Returns:
    - torch.Tensor: Stack of obscured tensors obtained by applying obscure() to each element in the batch.
    """
    out=[]
    # apply obscure() to each element in the batch
    for element in batch:
        out.append(obscure(element,percent))

    # stack the obscured tensors
    out_tensor = torch.stack(out)    
    return out_tensor




def get_indices_to_obscure(tensor, theta, min_value, max_value):
    """
    Get indices of points in a tensor based on a specified condition for a given coordinate.

    Parameters:
    - tensor (torch.Tensor): Input tensor containing data points.
    - theta (int): Index of the coordinate for which the condition is applied. Valid values: 0 (x), 1 (y) or 2 (z)
    - min_value (float): Minimum value for the specified coordinate.
    - max_value (float): Maximum value for the specified coordinate.

    Returns:
    - torch.Tensor: Indices of points satisfying the specified condition.
    """
    # get indices based on the condition on the specified coordinate
    indices_to_obscure = torch.nonzero((tensor[:, theta] >= min_value) & (tensor[:, theta] <= max_value)).squeeze(dim=1)
    
    return indices_to_obscure


def obscure_zone(tensor, indices_to_obscure=None, axes=None, max_value=None):
    """
    Obscure specific points in a tensor by replacing them with column means and assigning a specific value to selected axes.
    To avoid the formation of flying dots in the coordinate used as the reference axis, the maximum value is assigned, for all points to be modified by the mean, to that coordinate 

    Parameters:
    - tensor (torch.Tensor): Input tensor to be modified.
    - indices_to_obscure (torch.Tensor): Indices of points to be modified.
    - axes (int): Index of the coordinate for which the condition is applied. Valid values: 0 (x), 1 (y) or 2 (z)
    - max_value (float): Value to be assigned to the specified axes.

    Returns:
    - torch.Tensor: Tensor with obscured points on specified axes.
    """
    # calculate the mean of each column
    column_means = tensor.mean(dim=0)

    # replace obscured points with column means
    tensor[indices_to_obscure, :] = column_means

    # assign the specified value only to the selected axes
    tensor[indices_to_obscure, axes] = max_value

    return tensor


def obscure_with_mean_selected_points(a,axes,frac):
    """
    This function is used to be able to automatize and dynamically apply the editing procedure to all the various point clouds. 
    Each point cloud has a very different shape and size. Here the maximum and minimum are first found for the chosen axis to be based on for editing. 
    The values of the two extremes of the point range that fall within that condition are then obtained and, through the get_indices_to_obscure() function, the relevant indices are obtained. 
    Finally, obscure_zone() is applied to the input tensor, resulting in the output of the edited point cloud.


    Parameters:
    - a (torch.Tensor): Input tensor to be modified.
    - axes (int): Index of the coordinate for which the condition is applied. Valid values: 0 (x), 1 (y) or 2 (z).
    - frac (float): Fraction used to calculate the interval for selecting points to be modified.

    Returns:
    - torch.Tensor: Tensor with modified points on specified axes.
    """
    coords = a[:, :3]

    # finds the maximum, minimum and width
    max_per_col, _ = torch.max(coords, dim=0)
    min_per_col, _ = torch.min(coords, dim=0)
    width = max_per_col.numpy()[axes] - min_per_col.numpy()[axes]
    
    # get indices_to_obscure
    indices_to_obscure = get_indices_to_obscure(a,axes, min_per_col.numpy()[axes] -1, min_per_col.numpy()[axes] + abs(width)/frac)

    # get the modified tensor
    tensor = obscure_zone(a,  indices_to_obscure=indices_to_obscure,  axes=axes, max_value=min_per_col.numpy()[axes] + abs(width) / frac)

    return tensor    


def apply_obscure_with_mean_selected_points(batch,axes,frac):
    """
    Apply obscure_with_mean_selected_points() to each element in a batch of tensors.

    Parameters:
    - batch (list of torch.Tensor): List of input tensors to be modified.
    - axes (int): Index of the coordinate for which the condition is applied. Valid values: 0 (x), 1 (y) or 2 (z).
    - frac (float): Fraction used to calculate the interval for selecting points to be modified.

    Returns:
    - torch.Tensor: Stack of modified tensors obtained by applying obscure_with_mean_selected_points() to each element in the batch.
    """
    out=[]
    # apply obscure_with_mean_selected_points() to each element in the batch
    for element in batch:
        out.append(obscure_with_mean_selected_points(element,axes=axes,frac=frac))

    # stack the obscured tensors
    out_tensor = torch.stack(out)    
    return out_tensor




def obscure_points(a, indices_to_obscure):
    """
    It is a variant of the obscure_zone() function. unlike this one, the points are not replaced with the mean, but the coordinates of random points that already exist but are outside the range are assigned.
    
    Parameters:
    - a (torch.Tensor): Input tensor to be modified.
    - indices_to_obscure (torch.Tensor): Indices of points to be modified.

    Returns:
    - torch.Tensor: Tensor with obscured points on specified axes.
    """
    # find points that do not fall within the obscured zone
    non_obscured_points_indices = [index for index in range(len(a)) if index not in indices_to_obscure]

    # randomly samples
    if non_obscured_points_indices:
        replacement_index = random.choice(non_obscured_points_indices)
        replacement_point = a[replacement_index].clone()  # Make a copy to avoid modifying the original point

        # replace the selected points with the replacement point
        for index in indices_to_obscure:
            a[index] = replacement_point

    return a


def obscure_selected_points(a, axes, frac):
    """
    It is a variant of the obscure_with_mean_selected_points() function. unlike this one, the points are not replaced with the mean, but the coordinates of random points that already exist but are outside the range are assigned.


    Parameters:
    - a (torch.Tensor): Input tensor to be modified.
    - axes (int): Index of the coordinate for which the condition is applied. Valid values: 0 (x), 1 (y) or 2 (z).
    - frac (float): Fraction used to calculate the interval for selecting points to be modified.

    Returns:
    - torch.Tensor: Tensor with modified points on specified axes.
    """
    coords = a[:, :3]

    # finds the maximum, minimum and width
    max_per_col, _ = torch.max(coords, dim=0)
    min_per_col, _ = torch.min(coords, dim=0)
    width = max_per_col.numpy()[axes] - min_per_col.numpy()[axes]

    # get indices_to_obscure
    indices_to_obscure = get_indices_to_obscure(a, axes, min_per_col.numpy()[axes] - 1, min_per_col.numpy()[axes] + abs(width) / frac)
    
    # get the modified tensor
    tensor = obscure_points(a, indices_to_obscure=indices_to_obscure)

    return tensor


def apply_obscure_selected_points(batch,axes,frac):
    """
    Apply obscure_selected_points() to each element in a batch of tensors.

    Parameters:
    - batch (list of torch.Tensor): List of input tensors to be modified.
    - axes (int): Index of the coordinate for which the condition is applied. Valid values: 0 (x), 1 (y) or 2 (z).
    - frac (float): Fraction used to calculate the interval for selecting points to be modified.

    Returns:
    - torch.Tensor: Stack of modified tensors obtained by applying obscure_with_mean_selected_points() to each element in the batch.
    """
    out=[]
    # apply obscure_selected_points() to each element in the batch
    for element in batch:
        out.append(obscure_selected_points(element,axes=axes,frac=frac))

    # stack the obscured tensors
    out_tensor = torch.stack(out)    
    return out_tensor




def obscure_and_generate_zone(tensor, indices_to_obscure, column_index):
    """
    Obscuring specific indices in a tensor by introducing noise into the coordinates.
    It is as if new points are being generated that take the place of old ones.
    The three columns related to normals and the one associated with the triangle area are excluded from the changes. 
    As for the column related to the chosen axis (0 or 1 or 2) a minor random noise is applied through the average of the column multiplied by a random weight between 0 and 1.

    Parameters:
    - tensor (torch.Tensor): Input tensor.
    - indices_to_obscure (torch.Tensor): Indices of points to be modified.
    - column_index (int): Index of the column to be modified. Valid values: 0 (x), 1 (y) or 2 (z).

    Returns:
    - torch.Tensor: Modified tensor with obscured points and a generated zone.
    """
    # create a copy of the original tensor
    modified_tensor = tensor.clone()

    # calculate the minimum and maximum values in the original tensor
    tensor_min = torch.min(modified_tensor)
    tensor_max = torch.max(modified_tensor)

    # create a random tensor normalized between tensor_min and tensor_max
    random_tensor = torch.rand(tensor.shape, dtype=modified_tensor.dtype) * (tensor_max - tensor_min) + tensor_min
    
    # generate random weights for each row
    for i in range(len(indices_to_obscure)):
        weight_generator = torch.rand(1, dtype=modified_tensor.dtype)
        random_binary = np.random.randint(2,)

        # apply modifications based on random weights and binary choice
        if random_binary ==1:
         modified_tensor[indices_to_obscure[i]] += random_tensor[indices_to_obscure[i]] * weight_generator * np.random.uniform(0.1, 0.4)
        else: 
           modified_tensor[indices_to_obscure[i]] -= random_tensor[indices_to_obscure[i]] * weight_generator * np.random.uniform(0.1, 0.4)

    # copy the column associated with 'column_index' from the original tensor
    original_column_1 = tensor[:, column_index].clone()

    # copy additional columns from the original tensor (nx, ny, nz, A)
    original_column_3 = tensor[:, 3].clone()
    original_column_4 = tensor[:, 4].clone()
    original_column_5 = tensor[:, 5].clone()
    original_column_6 = tensor[:, 6].clone()

    #swaps the values with the original ones for the columns: nx, ny, nz, A and for the one related to the chosen axes
    modified_tensor[:, column_index] = original_column_1
    modified_tensor[:, 3] = original_column_3
    modified_tensor[:, 4] = original_column_4
    modified_tensor[:, 5] = original_column_5
    modified_tensor[:, 6] = original_column_6
    

    # generate random weights for each row in specific columns
    weight_generator_columns = torch.rand(len(modified_tensor), 2, dtype=modified_tensor.dtype) 
    
    for i in range(len(indices_to_obscure)):
        random_binary = np.random.randint(2,)

        # apply modifications based on random weights and binary choice
        if random_binary == 1: 
            modified_tensor[indices_to_obscure[i], column_index] += weight_generator_columns[i, 0] * torch.mean(tensor[:, column_index])
        else: 
            modified_tensor[indices_to_obscure[i], column_index] -= weight_generator_columns[i, 0] * torch.mean(tensor[:, column_index])

         

    return modified_tensor


def generate_selected_points(a, axes, frac):
 """
    Generates selected points in a tensor by modifying a specified zone.

    Parameters:
    - a (torch.Tensor): Input tensor.
    - axes (list of int): Axis indices used to modify the zone.
    - frac (float): Fraction used to calculate the interval for selecting points for modification.

    Returns:
    - torch.Tensor: Tensor with modified selected points.
 """

 coords = a[:, :3]

 # finds the maximum, minimum and width
 max_per_col, _ = torch.max(coords, dim=0)
 min_per_col, _ = torch.min(coords, dim=0)
 width = max_per_col.numpy()[axes] - min_per_col.numpy()[axes]
 
 # get indices_to_obscure
 indices_to_obscure = get_indices_to_obscure(a,axes, min_per_col.numpy()[axes] -1, min_per_col.numpy()[axes] + abs(width)/frac)

 # get the modified tensor
 tensor = obscure_and_generate_zone(a,  indices_to_obscure,axes)
 return tensor 


def apply_generate_selected_points(batch,axes,frac):
    """
    Apply generate_selected_points() to each element in a batch of tensors.

    Parameters:
    - batch (list of torch.Tensor): List of input tensors.
    - axes (list of int): Axis indices used to modify the zone.
    - frac (float): Fraction used to calculate the interval for selecting points for modification.

    Returns:
    - torch.Tensor: Stack of modified tensors obtained by generate_selected_points() to each element in the batch.
    """
    out=[]
    # apply generate_selected_points() to each element in the batch
    for element in batch:
        out.append(generate_selected_points(element,axes=axes,frac=frac))

    # stack the modified tensors    
    out_tensor = torch.stack(out)    
    return out_tensor
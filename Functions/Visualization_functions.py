import open3d as o3d
import torch
import numpy as np
from Transformations import *
from Mod_pointclouds import obscure, obscure_with_mean_selected_points, obscure_selected_points, generate_selected_points

# Single fragments:

def visualize_single_pointcloud(list_of_pointcloud, couple, element, rotate, size = 8):
    """
    Visualize a single point cloud using Open3D. The application of translate_to_origin() to fragments is applied by default.

    Args:
    - list_of_pointcloud (list): List of point clouds, where each point cloud is represented as a tensor.
    - couple (int): Index of the point cloud couple in the list.
    - element (int): Index of the specific point cloud within the couple.
    - rotate (int): Flag indicating whether to apply random rotation to the point cloud (1 for rotation, 0 for no rotation).
    - size (int, optional): Size of the points in the visualization. Default is 8.

    Returns:
    None

    """
    # retrieve the specified point cloud tensor
    tensor = list_of_pointcloud[couple][element]

    # apply rotation if specified
    if rotate ==1:
     tensor = random_rotation(tensor)
    else:
        tensor = tensor

    # extract XYZ coordinates from the tensor     
    xyz_coordinates = tensor[:, :3]
    
    # create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz_coordinates)

    # create a visualization window and add the point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Couple' + str(couple) +', Fragment' + str(element))
    vis.add_geometry(point_cloud)

    # set point size and display
    opt = vis.get_render_option()
    opt.point_size = size
    vis.run()
    vis.destroy_window()


def visualize_single_pointcloud_from_input(tensor, rotate, size = 8):
    """
    Visualize a single point cloud using Open3D. Unlike visualize_single_pointcloud(), the point cloud to be represented is directly input to the function, without having to be retrieved from the list of couple.
    

    Args:
    - tensor (numpy.ndarray or torch.Tensor): Input point cloud.
    - rotate (int): Flag indicating whether to apply random rotation to the point cloud (1 for rotation, 0 for no rotation).
    - size (int, optional): Size of the points in the visualization. Default is 8.

    Returns:
    None

    """

    # apply rotation if specified
    if rotate ==1:
     tensor = random_rotation(tensor)
    else:
        tensor = tensor

    # extract XYZ coordinates from the tensor     
    xyz_coordinates = tensor[:, :3]
    
    # create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz_coordinates)

    # create a visualization window and add the point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)

    # set point size and display
    opt = vis.get_render_option()
    opt.point_size = size
    vis.run()
    vis.destroy_window()




# Couples:
    
def visualize_couple_pointcloud(list_of_pointcloud, index, indeces_corrects=None, spacing=0, rotate=0, size=8):
    """
    Visualizes a pair of point clouds using Open3D. The application of translate_to_origin() to fragments is applied by default.
    This function also allows the original point clouds to be represented by applying some functions on them, such as spacing between them or random rotations to better visualize the pair

    Args:
    - list_of_pointcloud (list): List of point clouds, where each point cloud is represented as a tensor.
    - index (int): Index of the point cloud couple in the list.
    - indeces_corrects (numpy.ndarray, optional): Contains the index values associated with the pairs predicted correctly by the model. 
      Is used to report both the true and predicted label in the title of the display window Default is None.

    - spacing (float, optional): Distance to translate the second point cloud along the X-axis. Can be used in cases where the two fragments are too close together. Default is 0.
    - rotate (int, optional): Flag indicating whether to apply random rotation (1) or not (0) to the point clouds. Default is 0.
    - size (int, optional): Point size for visualization. Default is 8.

    Returns:
    None
    """
    # extract point clouds
    a = list_of_pointcloud[index][0]
    b = list_of_pointcloud[index][1]
    
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    
    # translations in the origin
    a = translate_to_origin(a)
    b = translate_to_origin(b)

    # apply rotation if specified
    if rotate ==1:
     a = random_rotation(a)
     b = random_rotation(b)
    else:
        a=a
        b=b

    # extract the first three columns for XYZ coordinates
    a_xyz_coordinates = a[:, :3]
    b_xyz_coordinates = b[:, :3]

    # create PointCloud objects for both arrays
    point_cloud_a = o3d.geometry.PointCloud()
    point_cloud_a.points = o3d.utility.Vector3dVector(a_xyz_coordinates)

    point_cloud_b = o3d.geometry.PointCloud()
    point_cloud_b.points = o3d.utility.Vector3dVector(b_xyz_coordinates)

    # translate the second point cloud along the X-axis
    b_xyz_coordinates[:, 2] += spacing
    point_cloud_b.points = o3d.utility.Vector3dVector(b_xyz_coordinates)

    # create a visualizer and add both point clouds
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Couples')
    vis.add_geometry(point_cloud_a)
    vis.add_geometry(point_cloud_b)
    
    # print the title based on correctness and ground truth label
    if indeces_corrects:
        y_true = list_of_pointcloud[index][2]
        intro_string = "Couple" + str(index)
        print(f"index: {index}, color: {y_true}")

        if (sum(indeces_corrects == index) == 1) and (list_of_pointcloud[index][2] == 1):
            vis.create_window(window_name=str(intro_string) + ' Y_true=1, Y_preds=1')

        elif (sum(indeces_corrects == index) == 0) and (list_of_pointcloud[index][2] == 1):
            vis.create_window(window_name=str(intro_string) + ' Y_true=1, Y_preds=0')

        elif (sum(indeces_corrects == index) == 1) and (list_of_pointcloud[index][2] == 0):
            vis.create_window(window_name=str(intro_string) + ' Y_true=0, Y_preds=0')

        elif (sum(indeces_corrects == index) == 0) and (list_of_pointcloud[index][2] == 0):
            vis.create_window(window_name=str(intro_string) + ' Y_true=0, Y_preds=1')

        else:
            pass

    # print the title based on ground truth label    
    else:
        intro_string = "Couple" + str(index)
        vis.create_window(window_name= intro_string + 'Y_True = ' + str(list_of_pointcloud[index][2])) 

    # set point size and display 
    opt = vis.get_render_option()
    opt.point_size = size
    vis.run()
    vis.destroy_window()    


def visualize_couple_pointcloud_obscure(list_of_pointcloud, index, p, indeces_corrects=None, spacing=0, rotate=0, size=8):
    """
    Visualizes a pair of point clouds using Open3D. The obscure() function with the parameter p is applied to the fragment pair before the display.
    The application of translate_to_origin() to fragments is applied by default.
    This function also allows the original point clouds to be represented by applying some functions on them, such as spacing between them or random rotations to better visualize the pair

    Args:
    - list_of_pointcloud (list): List of point clouds, where each point cloud is represented as a tensor.
    - index (int): Index of the point cloud couple in the list.
    - p (float): Percentage of rows to obscure in each tensor, specified as a value between 0 and 1.
    - indeces_corrects (numpy.ndarray, optional): Contains the index values associated with the pairs predicted correctly by the model. 
      Is used to report both the true and predicted label in the title of the display window Default is None.

    - spacing (float, optional): Distance to translate the second point cloud along the X-axis. Can be used in cases where the two fragments are too close together. Default is 0.
    - rotate (int, optional): Flag indicating whether to apply random rotation (1) or not (0) to the point clouds. Default is 0.
    - size (int, optional): Point size for visualization. Default is 8.

    Returns:
    None
    """
    # extract point clouds
    a = list_of_pointcloud[index][0]
    b = list_of_pointcloud[index][1]
    
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)

    # translations in the origin
    a = translate_to_origin(a)
    b = translate_to_origin(b)
    
    # obscure random points
    a= obscure(a,p)
    b= obscure(b,p)

    # apply rotation if specified
    if rotate ==1:
     a = random_rotation(a)
     b = random_rotation(b)
    else:
        a=a
        b=b

    # extract the first three columns for XYZ coordinates
    a_xyz_coordinates = a[:, :3]
    b_xyz_coordinates = b[:, :3]

    # create PointCloud objects for both arrays
    point_cloud_a = o3d.geometry.PointCloud()
    point_cloud_a.points = o3d.utility.Vector3dVector(a_xyz_coordinates)

    point_cloud_b = o3d.geometry.PointCloud()
    point_cloud_b.points = o3d.utility.Vector3dVector(b_xyz_coordinates)

    # translate the second point cloud along the X-axis
    b_xyz_coordinates[:, 2] += spacing
    point_cloud_b.points = o3d.utility.Vector3dVector(b_xyz_coordinates)

    # create a visualizer and add both point clouds
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Couples')
    vis.add_geometry(point_cloud_a)
    vis.add_geometry(point_cloud_b)
    
    # print the title based on correctness and ground truth label
    if indeces_corrects:
        y_true = list_of_pointcloud[index][2]
        intro_string = "Couple" + str(index)
        print(f"index: {index}, color: {y_true}")

        if (sum(indeces_corrects == index) == 1) and (list_of_pointcloud[index][2] == 1):
            vis.create_window(window_name=str(intro_string) + ' Y_true=1, Y_preds=1')

        elif (sum(indeces_corrects == index) == 0) and (list_of_pointcloud[index][2] == 1):
            vis.create_window(window_name=str(intro_string) + ' Y_true=1, Y_preds=0')

        elif (sum(indeces_corrects == index) == 1) and (list_of_pointcloud[index][2] == 0):
            vis.create_window(window_name=str(intro_string) + ' Y_true=0, Y_preds=0')

        elif (sum(indeces_corrects == index) == 0) and (list_of_pointcloud[index][2] == 0):
            vis.create_window(window_name=str(intro_string) + ' Y_true=0, Y_preds=1')

        else:
            pass

    # print the title based on ground truth label    
    else:
        intro_string = "Couple" + str(index)
        vis.create_window(window_name= intro_string + 'Y_True = ' + str(list_of_pointcloud[index][2])) 

    # set point size and display 
    opt = vis.get_render_option()
    opt.point_size = size
    vis.run()
    vis.destroy_window()    


def visualize_couple_pointcloud_obscure_zone_mean(list_of_pointcloud, index, axes, f, indeces_corrects=None, spacing=0, rotate=0, size=8):
    """
    Visualizes a pair of point clouds using Open3D. The obscure_with_mean_selected_points() function with the parameters axes and f is applied to the fragment pair before the display.
    The application of translate_to_origin() to fragments is applied by default.
    This function also allows the original point clouds to be represented by applying some functions on them, such as spacing between them or random rotations to better visualize the pair

    Args:
    - list_of_pointcloud (list): List of point clouds, where each point cloud is represented as a tensor.
    - index (int): Index of the point cloud couple in the list.
    - axes (int): Index of the coordinate for which the condition is applied. Valid values: 0 (x), 1 (y) or 2 (z).
    - f (float): Fraction used to calculate the interval for selecting points to be modified.
    - indeces_corrects (numpy.ndarray, optional): Contains the index values associated with the pairs predicted correctly by the model. 
      Is used to report both the true and predicted label in the title of the display window Default is None.

    - spacing (float, optional): Distance to translate the second point cloud along the X-axis. Can be used in cases where the two fragments are too close together. Default is 0.
    - rotate (int, optional): Flag indicating whether to apply random rotation (1) or not (0) to the point clouds. Default is 0.
    - size (int, optional): Point size for visualization. Default is 8.

    Returns:
    None
    """

    # extract point clouds
    a = list_of_pointcloud[index][0]
    b = list_of_pointcloud[index][1]
    
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    
    # translations in the origin
    a = translate_to_origin(a)
    b = translate_to_origin(b)
    
    # obscure selected points with their mean
    a= obscure_with_mean_selected_points(a,axes,f)
    b= obscure_with_mean_selected_points(b,axes,f)

    # apply rotation if specified
    if rotate ==1:
     a = random_rotation(a)
     b = random_rotation(b)
    else:
        a=a
        b=b

    # extract the first three columns for XYZ coordinates
    a_xyz_coordinates = a[:, :3]
    b_xyz_coordinates = b[:, :3]

    # create PointCloud objects for both arrays
    point_cloud_a = o3d.geometry.PointCloud()
    point_cloud_a.points = o3d.utility.Vector3dVector(a_xyz_coordinates)

    point_cloud_b = o3d.geometry.PointCloud()
    point_cloud_b.points = o3d.utility.Vector3dVector(b_xyz_coordinates)

    # translate the second point cloud along the X-axis
    b_xyz_coordinates[:, 2] += spacing
    point_cloud_b.points = o3d.utility.Vector3dVector(b_xyz_coordinates)

    # create a visualizer and add both point clouds
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Couples')
    vis.add_geometry(point_cloud_a)
    vis.add_geometry(point_cloud_b)
    
    # print the title based on correctness and ground truth label
    if indeces_corrects:
        y_true = list_of_pointcloud[index][2]
        intro_string = "Couple" + str(index)
        print(f"index: {index}, color: {y_true}")

        if (sum(indeces_corrects == index) == 1) and (list_of_pointcloud[index][2] == 1):
            vis.create_window(window_name=str(intro_string) + ' Y_true=1, Y_preds=1')

        elif (sum(indeces_corrects == index) == 0) and (list_of_pointcloud[index][2] == 1):
            vis.create_window(window_name=str(intro_string) + ' Y_true=1, Y_preds=0')

        elif (sum(indeces_corrects == index) == 1) and (list_of_pointcloud[index][2] == 0):
            vis.create_window(window_name=str(intro_string) + ' Y_true=0, Y_preds=0')

        elif (sum(indeces_corrects == index) == 0) and (list_of_pointcloud[index][2] == 0):
            vis.create_window(window_name=str(intro_string) + ' Y_true=0, Y_preds=1')

        else:
            pass

    # print the title based on ground truth label    
    else:
        intro_string = "Couple" + str(index)
        vis.create_window(window_name= intro_string + 'Y_True = ' + str(list_of_pointcloud[index][2])) 

    # set point size and display 
    opt = vis.get_render_option()
    opt.point_size = size
    vis.run()
    vis.destroy_window()    


def visualize_couple_pointcloud_obscure_zone(list_of_pointcloud, index, axes, f, indeces_corrects=None, spacing=0, rotate=0, size=8):
    """
    Visualizes a pair of point clouds using Open3D. The obscure_selected_points() function with the parameters axes and f is applied to the fragment pair before the display.
    The application of translate_to_origin() to fragments is applied by default.
    This function also allows the original point clouds to be represented by applying some functions on them, such as spacing between them or random rotations to better visualize the pair

    Args:
    - list_of_pointcloud (list): List of point clouds, where each point cloud is represented as a tensor.
    - index (int): Index of the point cloud couple in the list.
    - axes (int): Index of the coordinate for which the condition is applied. Valid values: 0 (x), 1 (y) or 2 (z).
    - f (float): Fraction used to calculate the interval for selecting points to be modified.
    - indeces_corrects (numpy.ndarray, optional): Contains the index values associated with the pairs predicted correctly by the model. 
      Is used to report both the true and predicted label in the title of the display window Default is None.

    - spacing (float, optional): Distance to translate the second point cloud along the X-axis. Can be used in cases where the two fragments are too close together. Default is 0.
    - rotate (int, optional): Flag indicating whether to apply random rotation (1) or not (0) to the point clouds. Default is 0.
    - size (int, optional): Point size for visualization. Default is 8.

    Returns:
    None
    """

    # extract point clouds
    a = list_of_pointcloud[index][0]
    b = list_of_pointcloud[index][1]
    
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    
    # translations in the origin
    a = translate_to_origin(a)
    b = translate_to_origin(b)
    
    # obscure selected points with coordinates from existing points
    a= obscure_selected_points(a,axes,f)
    b= obscure_selected_points(b,axes,f)

    # apply rotation if specified
    if rotate ==1:
     a = random_rotation(a)
     b = random_rotation(b)
    else:
        a=a
        b=b

    # extract the first three columns for XYZ coordinates
    a_xyz_coordinates = a[:, :3]
    b_xyz_coordinates = b[:, :3]

    # create PointCloud objects for both arrays
    point_cloud_a = o3d.geometry.PointCloud()
    point_cloud_a.points = o3d.utility.Vector3dVector(a_xyz_coordinates)

    point_cloud_b = o3d.geometry.PointCloud()
    point_cloud_b.points = o3d.utility.Vector3dVector(b_xyz_coordinates)

    # translate the second point cloud along the X-axis
    b_xyz_coordinates[:, 2] += spacing
    point_cloud_b.points = o3d.utility.Vector3dVector(b_xyz_coordinates)

    # create a visualizer and add both point clouds
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Couples')
    vis.add_geometry(point_cloud_a)
    vis.add_geometry(point_cloud_b)
    
    # print the title based on correctness and ground truth label
    if indeces_corrects:
        y_true = list_of_pointcloud[index][2]
        intro_string = "Couple" + str(index)
        print(f"index: {index}, color: {y_true}")

        if (sum(indeces_corrects == index) == 1) and (list_of_pointcloud[index][2] == 1):
            vis.create_window(window_name=str(intro_string) + ' Y_true=1, Y_preds=1')

        elif (sum(indeces_corrects == index) == 0) and (list_of_pointcloud[index][2] == 1):
            vis.create_window(window_name=str(intro_string) + ' Y_true=1, Y_preds=0')

        elif (sum(indeces_corrects == index) == 1) and (list_of_pointcloud[index][2] == 0):
            vis.create_window(window_name=str(intro_string) + ' Y_true=0, Y_preds=0')

        elif (sum(indeces_corrects == index) == 0) and (list_of_pointcloud[index][2] == 0):
            vis.create_window(window_name=str(intro_string) + ' Y_true=0, Y_preds=1')

        else:
            pass

    # print the title based on ground truth label    
    else:
        intro_string = "Couple" + str(index)
        vis.create_window(window_name= intro_string + 'Y_True = ' + str(list_of_pointcloud[index][2])) 

    # set point size and display 
    opt = vis.get_render_option()
    opt.point_size = size
    vis.run()
    vis.destroy_window()


def visualize_couple_pointcloud_obscure_zone_generate(list_of_pointcloud, index, axes, f, indeces_corrects=None, spacing=0, rotate=0, size=8):
    """
    Visualizes a pair of point clouds using Open3D. The generate_selected_points() function with the parameters axes and f is applied to the fragment pair before the display.
    The application of translate_to_origin() to fragments is applied by default.
    This function also allows the original point clouds to be represented by applying some functions on them, such as spacing between them or random rotations to better visualize the pair

    Args:
    - list_of_pointcloud (list): List of point clouds, where each point cloud is represented as a tensor.
    - index (int): Index of the point cloud couple in the list.
    - axes (int): Index of the coordinate for which the condition is applied. Valid values: 0 (x), 1 (y) or 2 (z).
    - f (float): Fraction used to calculate the interval for selecting points to be modified.
    - indeces_corrects (numpy.ndarray, optional): Contains the index values associated with the pairs predicted correctly by the model. 
      Is used to report both the true and predicted label in the title of the display window Default is None.

    - spacing (float, optional): Distance to translate the second point cloud along the X-axis. Can be used in cases where the two fragments are too close together. Default is 0.
    - rotate (int, optional): Flag indicating whether to apply random rotation (1) or not (0) to the point clouds. Default is 0.
    - size (int, optional): Point size for visualization. Default is 8.

    Returns:
    None
    """

    # extract point clouds
    a = list_of_pointcloud[index][0]
    b = list_of_pointcloud[index][1]
    
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    
    # translations in the origin
    a = translate_to_origin(a)
    b = translate_to_origin(b)
    
    # modify selected points by adding noise to coordinates
    a= generate_selected_points(a,axes,f)
    b= generate_selected_points(b,axes,f)

    # apply rotation if specified
    if rotate ==1:
     a = random_rotation(a)
     b = random_rotation(b)
    else:
        a=a
        b=b

    # extract the first three columns for XYZ coordinates
    a_xyz_coordinates = a[:, :3]
    b_xyz_coordinates = b[:, :3]

    # create PointCloud objects for both arrays
    point_cloud_a = o3d.geometry.PointCloud()
    point_cloud_a.points = o3d.utility.Vector3dVector(a_xyz_coordinates)

    point_cloud_b = o3d.geometry.PointCloud()
    point_cloud_b.points = o3d.utility.Vector3dVector(b_xyz_coordinates)

    # translate the second point cloud along the X-axis
    b_xyz_coordinates[:, 2] += spacing
    point_cloud_b.points = o3d.utility.Vector3dVector(b_xyz_coordinates)

    # create a visualizer and add both point clouds
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Couples')
    vis.add_geometry(point_cloud_a)
    vis.add_geometry(point_cloud_b)
    
    # print the title based on correctness and ground truth label
    if indeces_corrects:
        y_true = list_of_pointcloud[index][2]
        intro_string = "Couple" + str(index)
        print(f"index: {index}, color: {y_true}")

        if (sum(indeces_corrects == index) == 1) and (list_of_pointcloud[index][2] == 1):
            vis.create_window(window_name=str(intro_string) + ' Y_true=1, Y_preds=1')

        elif (sum(indeces_corrects == index) == 0) and (list_of_pointcloud[index][2] == 1):
            vis.create_window(window_name=str(intro_string) + ' Y_true=1, Y_preds=0')

        elif (sum(indeces_corrects == index) == 1) and (list_of_pointcloud[index][2] == 0):
            vis.create_window(window_name=str(intro_string) + ' Y_true=0, Y_preds=0')

        elif (sum(indeces_corrects == index) == 0) and (list_of_pointcloud[index][2] == 0):
            vis.create_window(window_name=str(intro_string) + ' Y_true=0, Y_preds=1')

        else:
            pass

    # print the title based on ground truth label    
    else:
        intro_string = "Couple" + str(index)
        vis.create_window(window_name= intro_string + 'Y_True = ' + str(list_of_pointcloud[index][2])) 

    # set point size and display 
    opt = vis.get_render_option()
    opt.point_size = size
    vis.run()
    vis.destroy_window()    
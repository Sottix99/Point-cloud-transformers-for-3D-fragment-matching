

import itertools

def CreateCouples(pt):
    
    """ This function modifies the shape of the tensor to fit the model by unrolling clusters and forming pairs.
    
        
    Input:
    - pt (tensor): The input tensor representing clusters of fragments. 
      The input tensor is assumed to have a shape of (num_clusters, 2), where pt[i][0] is the adjacency matrix and pt[i][1] is the cluster of fragments for the i-th subcluster.
    

    Output:
    - couples (list): A list of pairs formed from the input clusters, including corresponding adjacency matrix values. 
      The output 'couples' list contains triples [fragment1, fragment2, adjacency_value].

    """
    clusters= pt.shape[0]
    couples = []
    # for each subcluster
    for i in range(0,clusters):
        
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

    

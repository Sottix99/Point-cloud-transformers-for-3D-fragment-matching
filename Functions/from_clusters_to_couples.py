

import itertools

def CreateCouples(pt):
    
    """ This function, modifies the shape of the tensor to fit the model 
        It unrolls the clusters by forming the pairs

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

    

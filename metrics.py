import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist

"""
Shape of the point clouds:
pc1.shape = (NumberOfPoints1, Dimension)
pc2.shape = (NumberOfPoints2, Dimension)
"""
    
def gromov_wasserstein(pc1:np.ndarray, pc2:np.ndarray)->float: 
    def dist_ecc_fast(ecc, u):
        return(np.mean(ecc <= u))
    
    d1 = distance_matrix(pc1, pc1)
    d2 = distance_matrix(pc2, pc2)
    out = 0
    
    ecc1 = d1.mean(0)
    ecc2 = d2.mean(0)
    
    unique_ecc = np.unique(np.concatenate((ecc1,ecc2)))
    for i in range(unique_ecc.shape[0] - 1):
        u = unique_ecc[i]
        out += (unique_ecc[i+1] - unique_ecc[i]) * np.abs(dist_ecc_fast(ecc1, u) - dist_ecc_fast(ecc2, u))
        
    return(0.5*out) 
    
def chamfer_distance(pc1:np.ndarray, pc2:np.ndarray)->float:
    dist = cdist(pc1, pc2)
    ch_dist = (np.min(dist, axis=1).mean() + np.min(dist, axis=0).mean()) / 2
    return ch_dist
    
def average_ratio(pc1:np.ndarray, pc2:np.ndarray, Dist_list:list)->float:
    d = cdist(pc1, pc2)
    avr = 0
    for i,dist in enumerate(Dist_list):
        avr += (i+1) * ((d.min(1) <= dist).sum()/pc1.shape[0] + (d.min(0) <= dist).sum()/pc2.shape[0])
        
    return avr / (len(Dist_list)**2 + len(Dist_list))
    
def ratio(pc1:np.ndarray, pc2:np.ndarray, thr_list = np.array([0.1, 0.5, 1, 2, 4]))->list:
    d = np.min(cdist(pc1, pc2), axis=1)
    r = (d <= thr_list[:, None]).mean(1)
    return thr_list, r


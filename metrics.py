import numpy as np
from scipy.spatial import distance_matrix
"""
Shape of the point clouds:
pc1.shape = (NumberOfPoints1, Dimension)
pc2.shape = (NumberOfPoints2, Dimension)
"""
    
def gromov_wasserstein(pc1, pc2): 
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
    
def chamfer_distance(pc1, pc2):
    d = np.square(distance_matrix(pc1, pc2))

    return d.min(1).mean() + d.min(0).mean()
    
def average_ratio(pc1, pc2, Dist_list):
    n = len(Dist_list)
    d = np.square(distance_matrix(pc1, pc2))
    avr = 0
    for i in range(n):
        dist = Dist_list[i]
        avr += (i+1) * ((d.min(1) <= dist).sum()/pc1.shape[0] + (d.min(0) <= dist).sum()/pc2.shape[0])
        
    return avr / (n**2 + n)
    
def ratio(pc1, pc2, thr_list = [0.1, 0.5, 1, 2, 4]):
    d = np.square(distance_matrix(pc1, pc2))
    r = []
    for thr in thr_list:
        r.append((d.min(1) <= thr).sum()/pc1.shape[0])

    return thr_list, r


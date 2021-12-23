from itertools import combinations
import numpy as np
from numpy.linalg import norm as vector_norm
from scipy.spatial import ConvexHull

# a naive implementation
def diameter(x: np.array):
    
    hull = ConvexHull(x,qhull_options='QJ') # rescale option to avoid QHull errors   
    vertices = x[hull.vertices,:]
    diameter = 0
    for v,w in [*combinations(vertices,2)]:
        l = vector_norm(v-w)
        diameter = l if l>diameter else diameter
        
    return diameter
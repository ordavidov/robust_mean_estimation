import argparse
from typing import Union
import itertools as it
import numpy as np
from numpy.random import uniform
from scipy.stats import dirichlet
from scipy.stats import multivariate_normal
from scipy.stats import random_correlation

def _yield_standard_simplex(dim: int) -> Union[np.array,list]:
    
    while True:
    
        yield dirichlet(np.ones(dim)).rvs().flatten()                                                              

def sample_standard_simplex(dim: int, N: int=1, upper_bound: float=1) -> Union[np.array,list]:
    
    upper_bound = min(1,upper_bound)
    assert upper_bound>=1/dim, 'upper bound must be >= 1/dim'
    
    return np.concatenate(list(it.islice(filter(lambda w: np.all(w<=upper_bound),_yield_standard_simplex(dim)),N)))

def _point_sampler(num_centers: int, weights: list, **kwargs):
    
    while True:
        
        i = np.random.choice(np.arange(num_centers),replace=True,p=weights)
    
        yield np.atleast_2d(multivariate_normal(**{k: v[i] for k,v in kwargs.items()}).rvs())

def _data_sampler(N: int, weights: list, num_centers, **kwargs):

    return np.concatenate(list(it.islice(_point_sampler(num_centers,weights,**kwargs),N)))

def mixed_gaussian_dataset_generator(lower_bnds: list, upper_bnds: list, num_centers: int, N: int) -> np.array:

    bounding_box = np.array([lower_bnds,upper_bnds])
    dim = bounding_box.shape[-1]    
    domain_scale = np.power((bounding_box[1,:]-bounding_box[0,:]).prod(),1/dim)
    
    centers = uniform(low=lower_bnds, high=upper_bnds, size=(num_centers,dim))
    eigenvals = np.vstack([sample_standard_simplex(dim)*dim for _ in range(num_centers)])
    corrs = [random_correlation.rvs(e) for e in eigenvals]
    sigmas = [np.random.rand(dim)*domain_scale for _ in range(num_centers)]
    covs = [np.diag(sigma) @ corr @ np.diag(sigma) for corr, sigma in zip(corrs,sigmas)]
    weights = sample_standard_simplex(num_centers)
    
    return _data_sampler(N, weights, num_centers, mean=centers, cov=covs), centers, weights, covs     

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dimension", type=int, default=3, help="Spatial dimension for samples.")
    args = parser.parse_args()

    d = args.dimension
    w = sample_standard_simplex(d)

    print('Standard simplex sample w={}\nWeights add up to: {:.2f}'.format(w,np.sum(w)))

    num_centers = 1
    lower_bnds = np.zeros(d)
    upper_bnds = np.ones(d)
    N = 1

    _, centers, _, covs = mixed_gaussian_dataset_generator(lower_bnds,upper_bnds,num_centers,N)

    print('Generated a multivariate normal distribution at random:\nMean: {}\nCovariance: {}'.format(centers[0],covs[0]))

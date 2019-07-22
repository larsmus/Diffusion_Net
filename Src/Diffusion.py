'''
Created on Sep 1, 2016

@author: urishaham
'''
import numpy as np
from sklearn.neighbors import NearestNeighbors
import itertools
#from scipy.sparse.linalg import svds


def diffusion_map(X, k=16, dim=3, sigma=0, ref_neighbor=7, t=1, kernel="gaussian"):
    K_mat, K_aff = ComputeLBAffinity(X, k=k, sig=sigma, ref_neighbor=ref_neighbor, kernel=kernel)
    P = makeRowStoch(K_mat) 
    E1, v1, eig = Diffusion(P, nEigenVals=dim+1)
    embed_train = np.matmul(E1, np.diag(v1)**t)
    eig["affinity"] = K_aff
    return embed_train, eig


def Diffusion(P, nEigenVals = 12): # Laplace - Beltrami
    Vals, Vecs = np.linalg.eig(P)
    # sort eigenvalues
    I = Vals.argsort()[::-1]
    Vals = Vals[I]
    Vecs = Vecs[:,I]
    Vecs = Vecs / Vecs[0,0]
    eig = {"vals": Vals, "vecs": Vecs, "mat": P}
    Vecs = Vecs[:, 1:nEigenVals]
    Vals = Vals[1:nEigenVals]
    return Vecs, Vals, eig


def makeRowStoch(K):
    d = 1. / np.sum(K, axis = 0)
    D_inv = np.diag(d)
    return np.dot(D_inv,K)


def ComputeLBAffinity(X, k, sig, ref_neighbor, kernel):
    Idx, Dx = Knnsearch(X, X, k, metric=kernel)
    K_aff, W_aff = ComputeKernel(Idx, Dx, sig=sig, ref_neighbor=ref_neighbor, kernel=kernel)
    d = 1 / np.sum(K_aff, axis=0)
    D_inv = np.diag(d)
    K = np.dot(np.dot(D_inv,K_aff), D_inv)
    return K, K_aff


def Knnsearch(X, Y, k, metric):
    if metric == "laplacian":
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', p=1).fit(X)
    elif metric == "gaussian":
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', p=2).fit(X)
    else:
        print("Unknown metric, use gaussian instead")
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', p=2).fit(X)
    Dx, Idx = nbrs.kneighbors(Y)
    return (Idx,Dx)


def ComputeKernel(Idx,Dx,sig, ref_neighbor, kernel, epsilon=1):
    n, m = Dx.shape
    if sig == 0:
        ep = np.median(Dx,axis=1)
        ep = epsilon*np.tile(ep[:,None],(1,Idx.shape[1]))
    elif sig == "adaptive":
        n, m = Dx.shape
        ep = np.zeros((n,m))
        for i, j in itertools.product(range(n), range(m)):
            ep[i, j] = np.sqrt(Dx[i, ref_neighbor]*Dx[j, ref_neighbor])
    else:
        ep = sig
        
    if kernel == "laplacian":
        temp = np.exp(-np.power(np.divide(Dx,ep),2))
    else:
        temp = np.exp(-np.abs(np.divide(Dx,ep)))
        #temp[np.where(temp<1.0e-3)] = 0
    
    W = np.zeros(shape=(n,n))
    for i in range(n):
        W[i,Idx[i,:]] = temp[i,:]
        
    A = (np.transpose(W) + W)/2  
    return (A,W)
    

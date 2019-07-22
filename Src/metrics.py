import numpy as np
import Diffusion as df
from scipy.linalg import block_diag


def get_metrics(y, met = None, embedding=None, x_train=None, eig=None, sigma=0, k=None, ref_neighbor=None, kernel="gaussian"):
    if met is None:
        met = {}
        met["geometric"] = geometric(embedding, y)
        met["spectral"] = spectral(x_train, y, k, ref_neighbor, kernel)
        met["probabilistic"] = probabilistic(eig, y)
        return met
    else:
        met["geometric"].append(geometric(embedding, y))
        met["spectral"].append(spectral(x_train, y, sigma=sigma, k=9))
        met["probabilistic"].append(probabilistic(eig, y))
    
    return met


def geometric(embedding, y):
    classes = np.unique(y)
    global_center_mass = np.mean(embedding, axis=0)
    class_center_mass = [np.mean(embedding[y==i], axis=0) for i in classes]
    class_mean_dist = [np.mean(np.linalg.norm(embedding[y==i] - class_center_mass[i], ord=2, axis=1)**2) for i in classes]
    all_mean_dist = np.mean(np.linalg.norm(embedding - global_center_mass, axis=1)**2)
    return all_mean_dist / np.sum(class_mean_dist)


def spectral(x_train, y, sigma, k=None, ref_neighbor=None, kernel="gaussian"):
    if k is None: k = x_train.shape[0]
    idx, dx = df.Knnsearch(x_train, x_train, k, metric="gaussian")
    _ , K = df.ComputeKernel(idx, dx, sig=sigma, ref_neighbor=ref_neighbor, kernel=kernel)
    K_bar = block_diag(*[K[np.ix_(np.where(y==i)[0],np.where(y==i)[0])] for i in np.unique(y)])
    d = 1 / np.sum(K_bar, axis=0)
    D = np.diag(d)
    A_bar = D @ K_bar @ D
    A_tilde = D @ K @ D
    vals, _ = np.linalg.eig(A_tilde)
    vals.sort()
    vals = vals[::-1]
    num_classes = len(np.unique(y))
    return (vals[num_classes] - vals[num_classes + 1]).real


def probabilistic(eig, y):
    transition = eig["mat"]
    np.fill_diagonal(transition, 0)
    clusters = [transition[np.ix_(np.where(y==i)[0],np.where(y==i)[0])] for i in np.unique(y)]
    gcut = [np.sum(cluster) for cluster in clusters]
    return (1. / len(y)) * np.sum(gcut)

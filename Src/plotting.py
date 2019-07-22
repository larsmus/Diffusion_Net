import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from scipy.spatial.distance import pdist
from mpl_toolkits.mplot3d import Axes3D
init_notebook_mode(connected=True)


def plot_embedding_space(embed, y_train, dimension=2):
    if dimension == 2:
        fig, (a1) = plt.subplots(1, 1)
        a1.scatter(embed[:, 0], embed[:, 1], c=y_train, cmap='jet', label=y_train)
        plt.title('diffusion embedding of train')
        # a1.set_aspect('equal')
        plt.show()
    elif dimension == 3:
        fig, (a1) = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        a1.scatter(embed[:, 0], embed[:, 1], embed[:, 2], c=y_train, cmap='gist_ncar', label=y_train)
        plt.title('diffusion embedding of train')
    else:
        raise NotImplementedError


def plot_3d_interactive(embed, y_train):
    data = []
    for number in range(10):
        embed_subset = embed[y_train == number]
        trace = go.Scatter3d(x=embed_subset[:, 0], y=embed_subset[:, 1], z=embed_subset[:, 2], mode="markers",
                             marker=dict(size=12, color=number, colorscale="Rainbow", opacity=0.5),
                             name=str(number))
        data.append(trace)
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=True)
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def plot_eigenvalues(eig, ax=None, label=None, show=True):
    if ax is None:
        fig, ax = plt.subplots()
    vals = eig["vals"]
    vals.sort()
    ax.plot(np.linspace(0, 1, len(vals)), vals, label=label)
    if show:
        plt.show()


def plot_distance_ratio_sum(eig, ax=None, label=None, show=True):
    if ax is None:
        fig, ax = plt.subplots()

    vals = eig["vals"]
    vecs = eig["vecs"]
    I = vals.argsort()[::-1]
    vals = vals[I]
    vecs = vecs[:, I]
    all_coords = np.matmul(vecs, np.diag(vals))
    diff_coordinates = [np.sum(pdist(all_coords[:, 1:i])) for i in range(all_coords.shape[1])]
    distances_ratio = diff_coordinates / diff_coordinates[-1]

    ax.plot(np.linspace(0, 1, len(vals)), distances_ratio, label=label)
    if show:
        plt.show()


def plot_distance_ratio_min(eig, ax=None, label=None, show=True):
    if ax is None:
        fig, ax = plt.subplots()

    vals = eig["vals"]
    vecs = eig["vecs"]
    I = vals.argsort()[::-1]
    vals = vals[I]
    vecs = vecs[:, I]
    all_coords = np.matmul(vecs, np.diag(vals))
    diff_coordinates = [pdist(all_coords[:, 1:i]) for i in range(all_coords.shape[1])]
    distances_ratio = diff_coordinates / diff_coordinates[-1]
    distances_min = [np.min(ratio) for ratio in distances_ratio]

    ax.plot(np.linspace(0, 1, len(vals)), distances_min, label=label)
    if show:
        plt.show()

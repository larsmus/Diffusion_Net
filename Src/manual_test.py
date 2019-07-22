import numpy as np
import metrics as metrics
import Diffusion as df
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotting as plotting
from sklearn.utils import shuffle
import random
random.seed(123)

sigma_m = 2
sigma_v = 1.5
N = 100
num_classes = 2
centers = [np.random.normal(np.zeros(6), sigma_m**2, size=6) for i in range(num_classes)]
classes = np.array([np.array([np.random.normal(center, sigma_v**2) for i in range(N)]) for center in centers])
x_train = np.concatenate((classes[0], classes[1]))
y_train = np.concatenate((np.ones(N, dtype="int"), np.zeros(N, dtype="int")))
x_train, y_train = shuffle(x_train, y_train, random_state=0)


sigmas = np.linspace(0.1, 100, 100)
met = {"geometric": [], "spectral": [], "probabilistic": []}
for sigma in sigmas:
    embed, _, eig = df.diffusion_map(x_train, k=10, dim=2, sigma=np.sqrt(sigma))
    met["geometric"].append(metrics.geometric(embed, y_train))
    met["spectral"].append(metrics.spectral(x_train, y_train, sigma=sigma))
    met["probabilistic"].append(metrics.probabilistic(eig, y_train))

fig, ax = plt.subplots()
ax.plot(sigmas, met["geometric"], label="Geometric")
ax.plot(sigmas, met["spectral"], label="Spectral")
ax.plot(sigmas, met["probabilistic"], label="Probabilistic")
ax.legend()
plt.show()

classes = np.unique(y_train)
global_center_mass = np.mean(embed, axis=0)
class_center_mass = [np.mean(embed[y_train==i], axis=0) for i in classes]
print(global_center_mass)
print(class_center_mass)
plotting.plot_embedding_space(embed, y_train)
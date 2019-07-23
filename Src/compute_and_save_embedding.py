import numpy as np
from Src.utils import load_and_prepare_mnist_data
from Src.Diffusion import diffusion_map
import time
import json
import os

parameters = {"k": 9, "n_train": 5000, "n_test": 500, "sigma": 0}

(x_train, y_train), _ = load_and_prepare_mnist_data(n_train=parameters["n_train"], n_test=parameters["n_test"])
time = str(int(time.time()))
os.makedirs(f"../Data/experiment_{time}")

embed_train, eig_train = diffusion_map(x_train, k=parameters["k"], dim=784, sigma=parameters["sigma"])
np.save(f"../Data/experiment_{time}/embed_train", embed_train)
np.save(f"../Data/experiment_{time}/eig", eig_train, allow_pickle=True)
np.save(f"../Data/experiment_{time}/x_train", x_train)
np.save(f"../Data/experiment_{time}/y_train", y_train)
np.save(parameters, f"../Data/experiment_{time}/parameters.json")

print(f"Done with {str(time)}")



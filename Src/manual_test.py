import numpy as np
from Src.utils import load_and_prepare_mnist_data
from Src.Diffusion import diffusion_map
import time
import os


(x_train, y_train),(x_test, y_test) = load_and_prepare_mnist_data(n_train=5000, n_test=500)
time = str(int(time.time()))

os.makedirs(f"../Data/experiment_{time}")
embed_train, eig_train = diffusion_map(x_train, k=9, dim=784, sigma=0)
np.save(f"../Data/embed_train_{time}", embed_train)
np.save(f"../Data/eig_{time}", eig_train, allow_pickle=True)
np.save(f"../Data/x_train{time}", x_train)
np.save(f"../Data/y_train{time}", y_train)

print(f"Done with {str(time)}")







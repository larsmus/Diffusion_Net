import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical



def get_embedding_mse(embed_1, embed_2):
    embed_rotated = get_rotated_embedding(embed_1, embed_2)
    return np.mean(np.linalg.norm(embed_1 - embed_rotated, ord=2, axis=1)**2)


def get_rotated_embedding(embed_1, embed_2):
    # return rotated version of embed_2
    matrix_shape = embed_1.shape[1]
    S = np.zeros((matrix_shape, matrix_shape))
    for i in range(matrix_shape):
        for j in range(matrix_shape):
            S[i, j] = np.dot(embed_1[:, i], embed_2[:, j])
    u, _, v = np.linalg.svd(S)
    R = v.T @ u.T
    return (R @ embed_2.T).T


def subsample_data(x_train, y_train, x_test, y_test, num_train, num_test=0):
    train_idx = np.random.choice(x_train.shape[0], num_train, replace=False)
    test_idx = np.random.choice(x_test.shape[0], num_test, replace=False)
    x_train = x_train[train_idx,:,:]
    y_train = y_train[train_idx]
    x_test = x_test[test_idx,:,:]
    y_test = y_test[test_idx]
    return (x_train, y_train), (x_test, y_test)


def preprocess_data(x_train, x_test):
    x_train = x_train.astype("float")/255.0
    x_test = x_test.astype("float")/255.0
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
    return x_train, x_test


def load_and_prepare_mnist_data(n_train, n_test):
    (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()
    (x_train, y_train),(x_test, y_test) = subsample_data(x_train_full, y_train_full, x_test_full, y_test_full,
                                                           n_train, n_test)
    x_train, x_test = preprocess_data(x_train, x_test)
    y_train = to_categorical(y_train, len(np.unique(y_train)))
    y_test = to_categorical(y_test, len(np.unique(y_test)))
    return (x_train, y_train),(x_test, y_test)



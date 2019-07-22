import numpy as np


def get_embedding_mse(embed_1, embed_2):
    matrix_shape = embed_1.shape[1]
    S = np.zeros((matrix_shape, matrix_shape))
    for i in range(matrix_shape):
        for j in range(matrix_shape):
            S[i, j] = np.dot(embed_1[:, i], embed_2[:, j])
    u, _, v = np.linalg.svd(S)
    R = v.T @ u.T
    return np.mean(np.linalg.norm(embed_1 - (R @ embed_2.T).T, ord=2, axis=1)**2)


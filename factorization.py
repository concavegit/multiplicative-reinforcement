import numpy as np
from scipy import optimize


def factorization(n):
    """
    input a number to get a matrix describing factors of 2, 3, 5, 7, whether the number is 1, and the number itself.
    """
    arr = []
    for i in range(n):
        onenumber = []
        onenumber.append(1 if i % 2 == 0 else 0)
        onenumber.append(1 if i % 3 == 0 else 0)
        onenumber.append(1 if i % 5 == 0 else 0)
        onenumber.append(1 if i % 7 == 0 else 0)
        onenumber.append(1 if i == 1 else 0)
        onenumber.append(i)
        arr.append(onenumber)
    a = np.array(arr)
    return a


def factorization2(n):
    factors = np.array([2, 3, 5, 7])
    table = np.tile(np.arange(n), (factors.size, 1)).T
    table = table % factors == 0
    representation = np.hstack([table, np.arange(n)[:, np.newaxis]])
    return representation


def learn_constrained_features_helper(x, *args):
    target, representation, nfeatures, = args
    matrix_size = x.size // 2
    matrix_rows = representation.shape[1]
    matrix_cols = nfeatures
    matrix1 = x[:matrix_size].reshape((matrix_rows, matrix_cols))
    matrix2 = x[matrix_size:].reshape((matrix_cols, matrix_rows))
    result = representation.dot(matrix1).dot(matrix2).dot(representation.T)
    mask = ~np.isnan(target)
    norm_squared = ((result[mask] - target[mask])**2).sum()
    return norm_squared


def learn_constrained_features(table, representation, numfeatures):
    matrix_rows = representation.shape[1]
    matrix_size = matrix_rows * numfeatures
    results = optimize.minimize(learn_constrained_features_helper,
                                np.zeros(matrix_size),
                                table, representation, numfeatures)

    return (results.x[:matrix_size].reshape(matrix_rows, numfeatures),
            results.x[matrix_size:].reshape(numfeatures, matrix_rows))


print(factorization(21))

#!/usr/bin/env python
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

import model


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
    initial_guess = np.ones(matrix_size * 2)
    results = optimize.minimize(
        learn_constrained_features_helper, initial_guess, (table, representation, numfeatures))

    return (results.x[:matrix_size].reshape(matrix_rows, numfeatures),
            results.x[matrix_size:].reshape(numfeatures, matrix_rows),)


def main():
    """Fill in a multiplication table and visualize it. """
    parser = argparse.ArgumentParser("Get predicted multiplication times.")
    parser.add_argument('filename', nargs='?', default='data.csv', type=str)
    args = parser.parse_args()

    training_data = pd.read_csv(args.filename)
    training_table = model.to_table(training_data)
    representation = factorization2(21)
    matrix1, matrix2 = learn_constrained_features(
        training_table, representation, 6)
    reconstructed = representation.dot(
        matrix1).dot(matrix2).dot(representation.T)
    print(matrix1)
    print(matrix2)
    print(training_table)
    fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 20), sharex=True, sharey=True)
    fig2, (axes3, axes4) = plt.subplots(1, 2, figsize=(20, 20))
    model.plot_table(training_table, axes1)
    model.plot_table(reconstructed, axes2)
    model.plot_table(matrix1, axes3)
    model.plot_table(matrix2, axes4)
    # fig.savefig('reconstructed.png')
    plt.show()


if __name__ == '__main__':
    main()

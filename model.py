#!/usr/bin/env python
"""
usage: Get predicted multiplication times. [-h] [filename]

positional arguments:
  filename

optional arguments:
  -h, --help  show this help message and exit
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import NMF


def train_model(table):
    """Train a model to guess how long it would take to fill in the
    missing multiplication table entries.
    """
    nmf = NMF(3, init='random', solver='mu', max_iter=1000)
    nmf.fit(table)
    return nmf


def to_table(data):
    """Converts a table of multiplied numbers and times to a
    multiplication table where all the cells are how long it took to
    solve the problem.

    Args:
        data: A pandas dataframe where the first column is num1, next
            is num2, and the last is time.
    """
    num_rows = data.num1.max() + 1
    num_cols = data.num2.max() + 1
    matrix = np.empty((num_rows, num_cols))
    matrix[:] = np.nan
    times_by_problem = {}
    for _, row in data.iterrows():
        times_by_problem.setdefault(
            (row.num1, row.num2), []).append(row.elapsed)
    mean_times_by_problem = np.array(
        [[k[0], k[1], np.mean(v)] for k, v in times_by_problem.items()])
    matrix[mean_times_by_problem[:, 0].astype(
        int), mean_times_by_problem[:, 1].astype(int)] =\
        mean_times_by_problem[:, 2]
    return matrix


def reconstruct_table(table):
    """Given an incomplete multiplication table, fill in the predicted
    time taken to solve each problem.

    Args:
        table: The multiplication table to fill.

    Returns:
        The completed multiplication table.
     """
    trained_model = train_model(table)
    trained_model_h = trained_model.components_
    trained_model_w = trained_model.transform(table)
    reconstructed = trained_model_w.dot(trained_model_h)
    return reconstructed


def plot_table(table, axes):
    """Given a table, plot the time taken to solve each problem.

    Args:
        table: The tabulated results to plot.
        axis: The axis to plot on.

    Returns:
        An axes with the information plotted on it.
    """
    plt.xticks(np.arange(table.shape[1] + 2) - 1)
    plt.yticks(np.arange(table.shape[0] + 2) - 1)
    axes.imshow(table)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if not np.isnan(table[i, j]):
                axes.text(j, i, f'{table[i, j]:.2g}',
                          ha='center', va='center', color='w')
    return axes


def main():
    """Fill in a multiplication table and visualize it. """
    parser = argparse.ArgumentParser("Get predicted multiplication times.")
    parser.add_argument('filename', nargs='?', default='data.csv', type=str)
    args = parser.parse_args()

    training_data = pd.read_csv(args.filename)
    training_table = to_table(training_data)
    reconstructed = reconstruct_table(training_table)
    fig, (axes1, axes2) = plt.subplots(
        1, 2, figsize=(20, 20), sharex=True, sharey=True)
    plot_table(training_table, axes1)
    plot_table(reconstructed, axes2)
    fig.savefig('reconstructed.png')


if __name__ == '__main__':
    main()

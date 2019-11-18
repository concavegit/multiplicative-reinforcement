#!/usr/bin/env python
import pandas as pd
import argparse
import numpy as np
from sklearn.decomposition import NMF


def train_model(table):
    """Train a model to guess how long it would take to fill in the
    missing multiplication table entries.
    """
    nmf = NMF(3)
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
    matrix[:] = 0
    matrix[data.num1, data.num2] = data.time
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='?', default='data.csv', type=str)
    args = parser.parse_args()

    training_data = pd.read_csv(args.filename)
    training_table = to_table(training_data)
    trained_model = train_model(training_table)
    trained_model_h = trained_model.components_
    trained_model_w = trained_model.transform(training_table)
    reconstructed = trained_model_w.dot(trained_model_h)
    print(trained_model.components_)
    print(training_data)
    print(reconstructed)
    np.savetxt("table.csv",training_table,delimiter=",")


if __name__ == '__main__':
    main()

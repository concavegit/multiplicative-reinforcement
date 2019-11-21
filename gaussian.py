import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

import model

def gaussian(training_data):
    columns = np.zeros((21,2))
    rows = np.zeros((21,2))
    for i in range(len(training_data)):
        for j in range(len(training_data[i])):
            if (not np.isnan(training_data[i][j])):
                columns[j,0] += training_data[i][j]
                columns[j,1] += 1
                rows[i,0] += training_data[i][j]
                rows[i,1] += 1
    for i in columns:
        i[0] = i[0] / i[1]
    for i in rows:
        i[0] = i[0] / i[1]
    print(columns)
    print(rows)
    results = np.zeros((21,21))
    for i in range(len(training_data)):
        for j in range(len(training_data)):
            if(np.isnan(training_data[i][j])):
                results[i][j] = (columns[i,0] + rows[j,0]) / 2
            else:
                results[i][j] = (columns[i,0] + rows[j,0] + training_data[i][j] * 5)/7
    return results

def main():
    """Fill in a multiplication table and visualize it. """
    parser = argparse.ArgumentParser("Get predicted multiplication times.")
    parser.add_argument('filename', nargs='?', default='data.csv', type=str)
    args = parser.parse_args()

    training_data = pd.read_csv(args.filename)
    training_table = model.to_table(training_data)
    reconstructed = gaussian(training_table)
    #print(matrix1)
    #print(matrix2)
    print(training_table)
    fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 20), sharex=True, sharey=True)
    fig2, (axes3, axes4) = plt.subplots(1, 2, figsize=(20, 20))
    model.plot_table(training_table, axes1)
    model.plot_table(reconstructed, axes2)
    #model.plot_table(matrix1, axes3)
    #model.plot_table(matrix2, axes4)
    # fig.savefig('reconstructed.png')
    plt.show()


if __name__ == '__main__':
    main()

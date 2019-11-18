#!/usr/bin/env python
"""
usage: play.py [-h] [-n HIST] [-p NUM_PROBS] [filename]

Play the multiplication game.

positional arguments:
  filename      The file to interpret problem difficulty with.

optional arguments:
  -h, --help    show this help message and exit
  -n HIST       The maximum amount of recent problems to infer difficulty
                from.
  -p NUM_PROBS  The number of problems to ask.
"""

import argparse
import pandas as pd
import numpy as np

import trainer
import model


def draw_problem(table):
    """Takes a multiplication table and returns a multiplication
    problem at a probability increasing with difficulty.

    Args:
        table: A multiplication table.
            The first index is the first number being multiplied, the
            second index is the second number being multiplied, and
            the value is the time taken to solve it.
    Returns:
        The first number to multiply and the second number to
        multiply.
        The probability of drawing a problem increases with the
        problem difficulty.
    """
    probabilities = table.ravel() / table.sum()
    problem_index = np.random.choice(np.arange(table.size), p=probabilities)
    return problem_index // table.shape[1], problem_index % table.shape[1]


def play(filename, num_probs=40, hist=40):
    """Play the game using an existing dataset of timed problems.

    The results of the game are appended to the dataset file.

    Args:
        filename (str): The file to read the multiplication problems and
            their completion times.

        num_probs (int): The amount of problems to ask.

        hist (int): The amount of recent problems to use in computing
            problem difficulty.
    """
    training_data = pd.read_csv(filename)[-hist:]
    training_table = model.to_table(training_data)
    trained_model = model.train_model(training_table)
    trained_model_h = trained_model.components_
    trained_model_w = trained_model.transform(training_table)
    reconstructed = trained_model_w.dot(trained_model_h)
    for _ in range(num_probs):
        problem = draw_problem(reconstructed)
        trainer.log_problem(problem, filename)


def main():
    """Play the game."""
    parser = argparse.ArgumentParser(
        description="Play the multiplication game.")
    parser.add_argument('filename', nargs='?', default='data.csv', type=str,
                        help="The file to interpret problem difficulty" +
                        " with.")
    parser.add_argument('-n', dest='hist', default=40, type=int,
                        help="The maximum amount of recent problems to"
                        " infer difficulty from.")
    parser.add_argument('-p', dest='num_probs', default=20, type=int,
                        help="The number of problems to ask.")
    args = parser.parse_args()
    play(args.filename, args.num_probs, args.hist)


if __name__ == '__main__':
    main()
# def select_problem(timed_multiplication_table):

#!/usr/bin/env python
"""Command line utility to collect multiplication test results.

usage: trainer.py [-h] [-f FILENAME] N

Gather info on multiplication time. This script prompts, times, and records
multiplication problems. If the answer is incorrect, the problem is asked
again until the correct answer is reached.

positional arguments:
  N            Upperbound of the numbers to be multiplied.

optional arguments:
  -h, --help   show this help message and exit
  -f FILENAME  The file to populate with results.
"""
import time
import os
import argparse
import random
import collections


def prompt_problem(upperbound):
    """Displays a multiplication problem and waits until the correct
    answer is entered.

    It presents a problem with two numbers from 0 to the upperbound.

    Args:
        upperbound (int): The upperbound of numbers to multiply.
    Returns:
        num1 (int): The first number in the multiplication problem.
        num2 (int): The second number in the multiplication problem.
        elapsed (float): The time taken to solve the probleem.
    """
    num1 = random.randint(0, upperbound)
    num2 = random.randint(0, upperbound)

    prompt = f'{num1} x {num2}\n>>> '
    start = time.time()
    answer = int(input(prompt))
    while answer != num1 * num2:
        answer = int(input(prompt))
    elapsed = time.time() - start
    results = collections.namedtuple('Result', ['num1', 'num2', 'elapsed'])
    results.num1 = num1
    results.num2 = num2
    results.elapsed = elapsed
    return results


def main():
    """Takes the command line arguments and populates a csv with the
    multiplication test results.
    """
    parser = argparse.ArgumentParser(
        description='Gather info on multiplication time.'
        ' This script prompts, times, and records multiplication problems.'
        ' If the answer is incorrect, the problem is asked again until the'
        ' correct answer is reached.')
    parser.add_argument('upperbound', metavar='N',
                        help='Upperbound of the numbers to be multiplied.',
                        default=20, type=int, nargs='?')
    parser.add_argument('-f', dest='filename', default='data.csv',
                        help='The file to populate with results.')
    args = parser.parse_args()

    with open(args.filename, 'a+') as filename:
        if os.stat(args.filename).st_size == 0:
            filename.write('num1,num2,time\n')
        while True:
            result = prompt_problem(args.upperbound)
            filename.write(f'{result.num1},{result.num2},{result.elapsed}\n')


if __name__ == '__main__':
    main()

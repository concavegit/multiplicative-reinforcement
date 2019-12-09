import tensorflow as tf
import numpy as np
import model
import pandas as pd

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def factorization(n):
    factors = np.array([2, 3, 5, 7])
    table = np.tile(np.arange(n), (factors.size, 1)).T
    table = table % factors == 0
    representation = np.hstack([table, np.arange(n)[:, np.newaxis]])
    return representation


@tf.function
def train(opt, guess, representation, nfeatures, target):
    matrix_rows = tf.shape(representation)[1]
    matrix_cols = nfeatures
    matrix_size = matrix_rows * matrix_cols
    loss_history = []
    mask = ~tf.math.is_nan(target)

    with tf.GradientTape() as tape:
        tape.watch(guess)
        matrix1 = tf.reshape(guess[:matrix_size], (matrix_rows, matrix_cols))
        matrix2 = tf.reshape(guess[matrix_size:], (matrix_cols, matrix_rows))
        first_half = tf.matmul(representation, matrix1)
        second_half = tf.matmul(matrix2, tf.transpose(representation))
        reconstructed = tf.matmul(first_half, second_half)
        diff_squared = (reconstructed[mask] - target[mask])**2
        loss = tf.reduce_sum(diff_squared)
        loss_history.append(loss)

    gradients = tape.gradient(loss, guess)
    opt.apply_gradients(zip([gradients], [guess]))


training_data = pd.read_csv('nathan.csv')
training_table = tf.constant(model.to_table(training_data), dtype=tf.float64)

opt = tf.keras.optimizers.Adam()
representation = tf.constant(factorization(21), dtype=tf.float64)
nfeatures = tf.constant(3, dtype=tf.int32)
guess = tf.Variable(tf.random.normal(
    (tf.shape(representation)[1] * nfeatures * 2,), dtype=tf.float64))
for _ in range(1000000):
    train(opt, guess, representation, nfeatures, training_table)
print(guess)

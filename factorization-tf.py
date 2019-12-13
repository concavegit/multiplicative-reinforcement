import tensorflow as tf
import io
import datetime
import numpy as np
import model
import pandas as pd
import matplotlib.pyplot as plt

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
    mask = ~tf.math.is_nan(target)

    with tf.GradientTape() as tape:
        tape.watch(guess)
        positive_guess = tf.exp(guess)
        matrix1 = tf.reshape(
            positive_guess[:matrix_size], (matrix_rows, matrix_cols))
        matrix2 = tf.reshape(
            positive_guess[matrix_size:], (matrix_cols, matrix_rows))
        first_half = tf.matmul(representation, matrix1)
        second_half = tf.matmul(matrix2, tf.transpose(representation))
        reconstructed = tf.matmul(first_half, second_half)
        diff_squared = (reconstructed[mask] - target[mask])**2
        loss = tf.reduce_sum(diff_squared)

    gradients = tape.gradient(loss, guess)
    opt.apply_gradients(zip([gradients], [guess]))
    return gradients, loss, matrix1, matrix2, reconstructed


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


training_data = pd.read_csv('nathan.csv')
training_table = model.to_table(training_data).astype(np.float64)

opt = tf.keras.optimizers.Adam()
representation = factorization(21).astype(np.float64)
nfeatures = 3
guess = tf.Variable(tf.random.normal(
    (tf.shape(representation)[1] * nfeatures * 2,), dtype=tf.float64))


current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
training_log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(training_log_dir)

for epoch in range(50000):
    gradients, loss, matrix1, matrix2, reconstructed = train(
        opt, guess, representation, nfeatures, training_table)
    with summary_writer.as_default():
        tf.summary.scalar('gradient', tf.norm(gradients), step=epoch)
        tf.summary.scalar('loss', loss, step=epoch)
    if epoch % 1000 == 0:
        table_figure, table_axes = plt.subplots(figsize=(10, 10))
        model.plot_table(reconstructed, table_axes)
        image = plot_to_image(table_figure)

        matrix_figure, matrix_axes = plt.subplots(1, 2, figsize=(8, 8))
        model.plot_table(matrix1, matrix_axes[0], False)
        model.plot_table(matrix2, matrix_axes[1], False)
        matrix_image = plot_to_image(matrix_figure)

        with summary_writer.as_default():
            tf.summary.image('Reconstructed', image, step=epoch)
            tf.summary.image('Matrices', matrix_image, step=epoch)


plt.imshow(reconstructed)
plt.show()
print(guess)
print(reconstructed)
print(reconstructed)

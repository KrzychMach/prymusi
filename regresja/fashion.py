import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def decision(observations, weights, bias):
    lin_results = tf.tensordot(observations, weights, 1) + bias
    log_results = tf.sigmoid(lin_results)
    # bin_result = log_results > 0.5
    # return bin_result
    return log_results


def binary_decision(observation, weights, bias):
    lin_results = tf.tensordot(observation, weights, 1)
    log_results = tf.sigmoid(lin_results)
    if log_results > 0.5:
        return 1
    else:
        return 0


def filter_dataset(train_images, train_labels, test_images, test_labels):
    """
    Zostawia tylko rekordy o labelu 0 lub 1 (t-shirty lub spodnie)
    """
    i = 0
    while i < len(train_labels):
        if train_labels[i] > 1:
            train_labels = np.delete(train_labels, i, 0)
            train_images = np.delete(train_images, i, 0)
        else:
            i += 1
    i = 0
    while i < len(test_labels):
        if test_labels[i] > 1:
            test_labels = np.delete(test_labels, i, 0)
            test_images = np.delete(test_images, i, 0)
        else:
            i += 1
    return train_images, train_labels, test_images, test_labels


def main():
    # przygotowanie pod animacje
    frames = []
    fig = plt.figure()

    # Wczytanie + obróbka
    dataset = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
    train_images, train_labels, test_images, test_labels =\
        filter_dataset(train_images, train_labels, test_images, test_labels)

    # skalowanie
    # (mianownik 1000 wybrany arbitralnie, bo wartości z zakresu 0-1 były za duże dla sigmoida, a to działa OK)
    train_images = (train_images - (255 / 2)) / 1000
    test_images = (test_images - (255 / 2)) / 1000

    # zamiana na tf.Variable
    train_images, train_labels, test_images, test_labels =\
        tf.Variable(train_images), tf.Variable(train_labels), tf.Variable(test_images), tf.Variable(test_labels)

    # nadanie odpowiedniego kształtu/typu
    train_labels, test_labels =\
        tf.cast(train_labels, 'float64'), tf.cast(test_labels, 'float64')
    train_images, test_images =\
        tf.reshape(train_images, [-1, 28 * 28]), tf.reshape(test_images, [-1, 28 * 28])

    # przygotowanie datasetów
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(len(train_dataset))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.shuffle(len(test_dataset))

    # przygotowanie parametrów
    weights = tf.Variable(tf.ones(len(train_images[0]), dtype='float64'))
    bias = tf.Variable(0.0, dtype='float64')

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.07, momentum=0.6)

    # trening
    batch_size = 10
    for observations, labels in train_dataset.batch(batch_size):
        frames.append([plt.imshow(weights.numpy().reshape([28, 28]), cmap='coolwarm')])
        with tf.GradientTape() as tape:
            results = decision(observations, weights, bias)
            losses = - (labels * tf.math.log(results) + (1 - labels) * tf.math.log(1 - results))
            loss = tf.reduce_sum(losses) / batch_size
        gradients = tape.gradient(loss, [weights, bias])
        optimizer.apply_gradients(zip(gradients, [weights, bias]))

    # test
    score = 0
    total = 0
    for observation, label in test_dataset.batch(1):
        result = binary_decision(observation, weights, bias)
        if result == label.numpy():
            score += 1
        total += 1

    print(f'Score: {score}/{total}, {(score/total) * 100}%')

    # animacja
    my_animation = animation.ArtistAnimation(fig, frames)
    # my_animation.save('./fashion_mnist_weights.gif', writer=animation.PillowWriter(fps=10))


main()

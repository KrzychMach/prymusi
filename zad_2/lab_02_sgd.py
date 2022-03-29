import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def funny_function(x):
    return (x - 5.0) ** 2


def wavy_function(x):
    return 60.0 * tf.sin(x)


def combined_function(x):
    return 0.5 * funny_function(x) + 0.5 * wavy_function(x)


def main():
    current_x = tf.Variable(-47.0)

    xs = np.linspace(-50.0, 50.0, 1000)
    ys = combined_function(xs)

    fig, ax = plt.subplots()

    optimiser = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.9)

    def step(i):
        nonlocal current_x
        with tf.GradientTape() as tape:
            current_y = combined_function(current_x)

        gradients = tape.gradient(current_y, [current_x])

        ax.clear()
        ax.plot(xs, ys)
        ax.scatter([current_x.numpy()], [current_y.numpy()])

        optimiser.apply_gradients(zip(gradients, [current_x]))

    animation = mpl_animation.FuncAnimation(fig, step, interval=50)
    plt.show()


if __name__ == '__main__':
    main()

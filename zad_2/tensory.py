import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def funny_function(x, y):
    return x ** 2 + y ** 2


def wavy_function(x, y):
    return 10 * y * tf.sin(x)


def combined_function(x, y):
    return 0.5 * funny_function(x, y) + 0.5 * wavy_function(x, y)


def main():
    current_args = [tf.Variable(-47.0), tf.Variable(20.0)]

    xs = np.linspace(-50.0, 50.0, 1000)
    ys = np.linspace(-50.0, 50.0, 1000)
    xs, ys = np.meshgrid(xs, ys)
    zs = combined_function(xs, ys)

    fig, ax = plt.subplots()

    optimiser = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.8)

    def step(i):
        nonlocal current_args
        with tf.GradientTape() as tape:
            if random.randint(0, 1) == 0:
                current_z = funny_function(current_args[0], current_args[1])
            else:
                current_z = wavy_function(current_args[0], current_args[1])

        gradients = tape.gradient(current_z, current_args)

        ax.clear()
        contour = ax.contour(xs, ys, zs, 30, cmap='coolwarm')
        ax.clabel(contour, inline=True)
        ax.scatter([current_args[0].numpy()], [current_args[1].numpy()], color='black')

        optimiser.apply_gradients(zip(gradients, current_args))

    anim_list.append(mpl_animation.FuncAnimation(fig, step, interval=20))  # bez tego jakiś bug powoduje, że animacja nie działa
    plt.show()


if __name__ == '__main__':
    anim_list = []
    main()

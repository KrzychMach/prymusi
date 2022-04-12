import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def foo(x, y):
    return x * y * y


def visualise_surface(w_x, w_y, w_z, b):
    c = b / w_z
    a = w_x / w_z
    b = w_y / w_z

    x = np.linspace(-50, 50, 10)
    y = np.linspace(-50, 50, 10)
    x, y = np.meshgrid(x, y)
    z = a * x + b * y + c

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, z)

    plt.show()


def visualise_scatter():
    x = np.linspace(-50, 50, 15)
    y = np.linspace(-50, 50, 15)
    x, y = np.meshgrid(x, y)
    z = foo(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)
    plt.show()


visualise_scatter()
visualise_surface(3, -5, 1, 7)

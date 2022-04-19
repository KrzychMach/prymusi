import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sklearn.preprocessing as skl_preprocessing


def decide(observations, weights, bias):
    linear_results = tf.tensordot(observations, weights, 1) + bias
    logistic_results = tf.sigmoid(linear_results)
    return logistic_results


def change(prev_weights, weights, prev_bias, bias):
    diff_weights = (weights - prev_weights).numpy()
    diff_weights = sum(i ** 2 for i in diff_weights)
    diff_bias = tf.math.abs(bias - prev_bias).numpy() ** 2
    return (diff_weights + diff_bias) ** 0.5


def main():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.linspace(0, 100, 10)
    y = np.linspace(0, 100, 10)
    x, y = np.meshgrid(x, y)

    df = pd.read_csv('./hehe.csv')
    df['Label'] = pd.Categorical(df['Decision']).codes
    scaler = skl_preprocessing.MinMaxScaler()
    df['Age'] = scaler.fit_transform(np.array(df['Age']).reshape(-1, 1))
    df['Height'] = scaler.fit_transform(np.array(df['Height']).reshape(-1, 1))
    df['Performance'] = scaler.fit_transform(np.array(df['Performance']).reshape(-1, 1))

    observations_df = df[['Age', 'Height', 'Performance', 'Label']]
    observations_df = observations_df.astype({'Label': 'float64'})
    results_df = observations_df.pop('Label')
    dataset = tf.data.Dataset.from_tensor_slices((observations_df.values, results_df.values))
    dataset = dataset.shuffle(len(dataset))

    weights = tf.Variable(tf.ones(3, dtype='float64'))
    prev_weights = tf.identity(weights)
    bias = tf.Variable(0.0, dtype='float64')
    prev_bias = tf.identity(bias)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.3)

    batch_size = 5
    diff = 10

    def frame_generator():
        nonlocal diff
        i = 0
        while diff > 0.05:
            yield i
            i += 1

    def step(i):
        nonlocal diff, prev_weights, prev_bias
        for observations, labels in dataset.batch(batch_size):
            with tf.GradientTape() as tape:
                results = decide(observations, weights, bias)
                losses = -(labels * tf.math.log(results) + (1 - labels) * tf.math.log(1 - results))
                loss = tf.reduce_sum(losses) / batch_size
            gradients = tape.gradient(loss, [weights, bias])
            optimizer.apply_gradients(zip(gradients, [weights, bias]))

        diff = change(prev_weights, weights, prev_bias, bias)
        prev_weights = tf.identity(weights)
        prev_bias = tf.identity(bias)

        a = weights.numpy()[0] / weights.numpy()[2]
        b = weights.numpy()[1] / weights.numpy()[2]
        c = bias.numpy() / weights.numpy()[2]
        z = a * x + b * y + c
        ax.clear()
        ax.plot_surface(x, y, z)

    anim = animation.FuncAnimation(fig, step, interval=20, frames=frame_generator())
    anim.save('./hehe.gif', writer=animation.PillowWriter(fps=5))
    print('done')


main()

import numpy as np
import tensorflow as tf
import pandas as pd
import sklearn.preprocessing as skl_preprocessing


def decide(observations, weights, bias):
    linear_results = tf.tensordot(observations, weights, 1) + bias
    logistic_results = tf.sigmoid(linear_results)
    # binary_results = logistic_results > 0.5
    return logistic_results


def main():
    df = pd.read_csv('./weight-height-short.csv')
    df['Label'] = pd.Categorical(df['Gender']).codes
    scaler = skl_preprocessing.MinMaxScaler()
    df['Height'] = scaler.fit_transform(np.array(df['Height']).reshape(-1, 1))
    df['Weight'] = scaler.fit_transform(np.array(df['Weight']).reshape(-1, 1))

    observations_df = df[['Height', 'Weight', 'Label']]
    observations_df = observations_df.astype({'Label': 'float64'})
    results_df = observations_df.pop('Label')
    dataset = tf.data.Dataset.from_tensor_slices((observations_df.values, results_df.values))
    dataset = dataset.shuffle(len(dataset))

    weights = tf.Variable(tf.ones(2, dtype='float64'))
    bias = tf.Variable(0.0, dtype='float64')
    print(weights.value(), bias.value())

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.3)

    batch_size = 5
    for observations, labels in dataset.batch(batch_size):
        with tf.GradientTape() as tape:
            results = decide(observations, weights, bias)
            losses = -(labels * tf.math.log(results) + (1 - labels) * tf.math.log(1 - results))
            loss = tf.reduce_sum(losses) / batch_size
        gradients = tape.gradient(loss, [weights, bias])
        optimizer.apply_gradients(zip(gradients, [weights, bias]))
        print(weights.value(), bias.value())


main()

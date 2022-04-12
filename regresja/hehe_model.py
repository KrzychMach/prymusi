import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as skl_preprocessing


def decide(observations, weights, bias):
    linear_results = tf.tensordot(observations, weights, 1) + bias
    logistic_results = tf.sigmoid(linear_results)
    return logistic_results


def main():
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
    bias = tf.Variable(0.0, dtype='float64')

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.3)

    batch_size = 5
    for observations, labels in dataset.batch(batch_size):
        with tf.GradientTape() as tape:
            results = decide(observations, weights, bias)
            losses = -(labels * tf.math.log(results) + (1 - labels) * tf.math.log(1 - results))
            loss = tf.reduce_sum(losses) / batch_size
        gradients = tape.gradient(loss, [weights, bias])
        optimizer.apply_gradients(zip(gradients, [weights, bias]))

    df = pd.read_csv('./hehe_check.csv')
    df['Label'] = pd.Categorical(df['Decision']).codes
    scaler = skl_preprocessing.MinMaxScaler()
    df['Age'] = scaler.fit_transform(np.array(df['Age']).reshape(-1, 1))
    df['Height'] = scaler.fit_transform(np.array(df['Height']).reshape(-1, 1))
    df['Performance'] = scaler.fit_transform(np.array(df['Performance']).reshape(-1, 1))

    observations_df = df[['Age', 'Height', 'Performance', 'Label']]
    observations_df = observations_df.astype({'Label': 'float64'})
    results_df = observations_df.pop('Label')
    dataset = tf.data.Dataset.from_tensor_slices((observations_df.values, results_df.values))

    score = 0
    total = 0
    for observation, label in dataset.batch(1):
        results = decide(observation, weights, bias)
        decision = results > 0.5
        if decision == label.numpy():
            score += 1
        total += 1

    print(f"Score: {score}/{total}, {score/total * 100}%")



main()

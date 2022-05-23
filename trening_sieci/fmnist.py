import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt


def rozpoznanie():
    dataset = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # train_images, train_labels, test_images, test_labels = \
    #     tf.Variable(train_images), tf.Variable(train_labels), tf.Variable(test_images), tf.Variable(test_labels)
    # train_labels, test_labels = \
    #     tf.cast(train_labels, 'float64'), tf.cast(test_labels, 'float64')

    # train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    # train_dataset = train_dataset.shuffle(len(train_dataset))
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    # test_dataset = test_dataset.shuffle(len(test_dataset))

    # inputs = tf.keras.layers.Input(shape=(28, 28), name='input_layer')
    # flat = tf.keras.layers.Flatten(input_shape=(28, 28))
    # dense = tf.keras.layers.Dense(128, activation='relu', name='dense_layer')
    # final = tf.keras.layers.Dense(10, name='logit_layer')
    # outputs = final(dense(flat(inputs)))
    # model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='podstawy')
    #
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    #
    # model.fit(train_images, train_labels, epochs=20)
    # model.evaluate(test_images, test_labels, verbose=2)
    # model.summary()
    # model.save("./fmnist_model.keras")

    model = tf.keras.models.load_model("./fmnist_model.keras")
    model.summary()
    model.evaluate(test_images, test_labels, verbose=2)


if __name__ == "__main__":
    rozpoznanie()

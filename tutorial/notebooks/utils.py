#!/usr/bin/python
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

(mnist_imgs, mnist_lbls), (mnist_test_imgs, mnist_test_lbls) = tf.keras.datasets.mnist.load_data()
mnist_imgs = mnist_imgs.reshape(-1, 28, 28, 1) / 255.0
mnist_test_imgs = mnist_test_imgs.reshape(-1, 28, 28, 1) / 255.0

def mnist_convnet():
    convnet = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, 3, activation="relu", padding="same"),
        Conv2D(64, 3, activation="relu", padding="same"),
        MaxPooling2D(),
        Conv2D(64, 3, activation="relu", padding="same"),
        Conv2D(64, 2, activation="relu", padding="same"),
        MaxPooling2D(),
        Flatten(),
        Dense(1024, activation="relu"),
        Dense(10, activation="softmax")
    ])

    convnet.compile(
        optimizer="Adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return(convnet)


def build_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Build a simple conv net for image classification
    :return: the compiled model
    """
    # input is 28 x 28 x 1
    inputs = Input(shape=input_shape, name='inputs')
    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='dense_1')(inputs)
    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='dense_2')(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model

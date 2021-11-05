# Day_03_02_MultiLayer.py
import tensorflow.keras as keras
import numpy as np


def mnist_softmax():
    # 퀴즈
    # mnist 데이터의 shape을 출력하세요
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # print(x_train.shape, x_test.shape)    # (60000, 28, 28) (10000, 28, 28)
    # print(y_train.shape, y_test.shape)    # (60000,) (10000,)

    # print(np.min(x_train), np.max(x_train))   # 0 255

    # 퀴즈
    # mnist 데이터셋에 대해 동작하는 모델을 구축하세요
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    x_train = x_train / 255
    x_test = x_test / 255

    model = keras.Sequential()
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=10, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))


def mnist_multi_layers():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    x_train = x_train / 255
    x_test = x_test / 255

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[784]))
    # (?, 512) = (?, 784) @ (784, 512) + 512
    model.add(keras.layers.Dense(512, activation='relu'))
    # (?, 128) = (?, 512) @ (512, 128) + 128
    model.add(keras.layers.Dense(128, activation='relu'))
    # (?, 10) = (?, 128) @ (128, 10) + 10
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=10, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))


# mnist_softmax()
mnist_multi_layers()

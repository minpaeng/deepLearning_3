# Day_03_01_SoftmaxRegression.py
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection


def softmax_regression():
    x = [[1, 2],            # C
         [2, 1],
         [4, 5],            # B
         [5, 4],
         [8, 9],            # A
         [9, 8]]
    y = [[0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=10, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    p = model.predict(x)
    print(p)

    # 퀴즈
    # 예측 결과의 합계가 1이 되는지 증명하세요
    # for row in p:
    #     print(np.sum(row))
    print(np.sum(p, axis=1))    # 0(수직) 1(수평)
    print('-' * 30)

    # 퀴즈
    # argmax 함수를 사용해서 정확도를 구하세요
    p_arg = np.argmax(p, axis=1)
    y_arg = np.argmax(y, axis=1)
    print(p_arg)
    print(y_arg)

    print('acc :', np.mean(p_arg == y_arg))


# 퀴즈
# iris_onehot.csv 파일을 읽어서
# 70%로 학습하고 30%에 대해 정확도를 구하세요
def softmax_regression_iris():
    iris = pd.read_csv('data/iris_onehot.csv', index_col=0)

    values = iris.values
    np.random.shuffle(values)

    x = values[:, :-3]
    y = values[:, -3:]
    # print(x.shape, y.shape)       # (150, 4) (150, 3)

    train_size = int(len(x) * 0.7)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=10, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))


def softmax_regression_iris_dense():
    iris = pd.read_csv('data/iris.csv', index_col=0)

    x = iris.values[:, :-1]
    y = iris.values[:, -1:]
    # print(x.shape, y.shape)       # (150, 4) (150, 1)
    # print(x.dtype, y.dtype)       # object object

    x = np.float32(x)

    enc = preprocessing.LabelBinarizer()
    y = enc.fit_transform(y)

    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=10, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))


def softmax_regression_iris_sparse():
    iris = pd.read_csv('data/iris.csv', index_col=0)

    x = iris.values[:, :-1]
    y = iris.values[:, -1:]
    # print(x.shape, y.shape)       # (150, 4) (150, 1)
    # print(x.dtype, y.dtype)       # object object

    x = np.float32(x)

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(y)

    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=10, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))

    # 퀴즈
    # 정확도를 직접 계산하세요
    p = model.predict(x_test)

    p_arg = np.argmax(p, axis=1)
    print('acc :', np.mean(p_arg == y_test))


# softmax_regression()
# softmax_regression_iris()

# softmax_regression_iris_dense()
softmax_regression_iris_sparse()

# 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 -> 7




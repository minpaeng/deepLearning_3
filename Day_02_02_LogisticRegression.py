# Day_02_02_LogisticRegression.py
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from sklearn import preprocessing


def logistic_regression():
    x = [[1, 2],            # 탈락
         [2, 1],
         [4, 5],            # 통과
         [5, 4],
         [8, 9],
         [9, 8]]
    y = [[0],
         [0],
         [1],
         [1],
         [1],
         [1]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=10, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    # 퀴즈
    # predict 함수를 사용해서 정확도를 직접 계산하세요
    p = model.predict(x)
    print(p)

    p_bools = (p > 0.5)
    print(p_bools)

    p_ints = np.int32(p_bools)
    print(p_ints)

    equals = (p_ints == y)
    print(equals)

    print('acc :', np.mean(equals))


# 퀴즈
# 피마 인디언 당뇨병 데이터를 가져와서
# 70%로 학습하고 30%에 대해 정확도를 구하세요 (목표 75%)
def logistic_regression_pima():
    pima = pd.read_csv('data/pima-indians-diabetes.csv',
                       skiprows=9, header=None)
    # print(pima)

    x = pima.values[:, :-1]
    y = pima.values[:, -1:]
    # print(x.shape, y.shape)       # (768, 8) (768, 1)

    x = preprocessing.scale(x)
    # x = preprocessing.minmax_scale(x)        # 값을 0~1사이의 값으로 바꿔줌

    train_size = int(len(x) * 0.7)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=10, verbose=2,
              validation_data=(x_test, y_test))
    # print(model.evaluate(x_test, y_test, verbose=0))


logistic_regression()
# logistic_regression_pima()

# http://192.168.0.48/

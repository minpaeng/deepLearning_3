# Day_02_01_MultipleRegression.py
import tensorflow.keras as keras
import numpy as np


# 퀴즈
# 아래 데이터에 대해 모델을 구축하고 결과를 보여주세요
def multiple_regression():
    x = [[1, 0],
         [0, 2],
         [3, 0],
         [0, 4],
         [5, 0]]
    y = [[1],
         [2],
         [3],
         [4],
         [5]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.mse)

    model.fit(x, y, epochs=100)
    print(model.predict(x))


# 퀴즈
# 보스턴 집값 데이터에 대해
# 80%의 데이터로 학습하고 20%의 데이터에 대해 평균 오차를 구하세요
def multiple_regression_boston():
    # 퀴즈
    # 보스턴 집값 데이터에 포함된 학습과 검사 데이터의 shape을 알려주세요
    boston = keras.datasets.boston_housing.load_data()
    # print(type(boston), len(boston))          # <class 'tuple'> 2

    train, test = boston
    # print(type(train), len(train))            # <class 'tuple'> 2

    x_train, y_train = train
    x_test, y_test = test
    # print(x_train.shape, x_test.shape)        # (404, 13) (102, 13)
    # print(y_train.shape, y_test.shape)        # (404,) (102,)

    # print(y_train[:5])                        # [15.2 42.3 50.  21.1 17.7]

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    # print(y_train.shape, y_test.shape)        # (404, 1) (102, 1)

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.SGD(0.000001),
                  loss=keras.losses.mse,
                  metrics=['mae'])

    model.fit(x_train, y_train, epochs=10, verbose=2)

    p = model.predict(x_test)
    p = p.reshape(-1)
    e = p - y_test.reshape(-1)

    print('mae :', np.mean(np.absolute(e)))


# multiple_regression()
multiple_regression_boston()

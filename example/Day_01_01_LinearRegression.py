# Day_01_01_LinearRegression.py
import numpy as np
import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt

# ctrl + shift + f10
# alt + 1
# alt + 4
# ctrl + /


# 퀴즈
# 아래 데이터에 대해 동작하는 케라스 모델을 구축하세요
def linear_regression():
    x = [1, 2, 3]
    y = [1, 2, 3]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.mse)  # mean square error

    model.fit(x, y, epochs=100)
    print(model.predict(x))

    p = model.predict(x)
    p = p.reshape(-1)
    e = p - y
    print(e)

    print('mae :', np.mean(np.absolute(e)))
    print('mse :', np.mean(e ** 2))


# 퀴즈
# 속도가 30과 50일 때의 제동 거리를 구하세요

# 퀴즈
# 구축한 모델을 시각화하세요 (matplotlib)
def linear_regression_cars():
    cars = pd.read_csv('data/cars.csv', index_col=0)
    # print(cars.values)

    x = cars.values[:, 0]
    y = cars.values[:, 1]
    # print(x.shape, y.shape)       # (50,) (50,)

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.SGD(0.001),
                  loss=keras.losses.mse)

    model.fit(x, y, epochs=100)
    p = model.predict([0, 30, 50])
    p = p.reshape(-1)
    p0, p1, p2 = p
    print(p0, p1, p2)

    plt.plot(x, y, 'ro')
    plt.plot([0, 30], [0, p1], 'g')
    plt.plot([0, 30], [p0, p1], 'b')
    plt.show()


linear_regression()
# linear_regression_cars()

# 192.168.0.48

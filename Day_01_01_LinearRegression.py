import numpy as np
import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt

# ctrl + shift + f10(디버깅 없이 실행)
# alt + 1 : 왼쪽 프로젝트 탭 닫기
# alt + 4 : 하단 실행 탭 닫기
#  ctrl + / : 전체 주석, 한 줄씩 주석 가능
print('hello')


# 퀴즈
# 아래 데이터에 대해 동작하는 케라스 모델을 구축하세요
def linear_regression():
    x = [1, 2, 3]
    y = [1, 2, 3]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.mse)  # mean square error
    model.fit(x, y, epochs=300)

    p = model.predict(x)
    p = p.reshape(-1)
    e = p - y  # 오차값

    print(e)
    print('mae: ', np.mean(np.absolute(e)))
    print('mae: ', np.mean(e**2))


# 퀴즈
# 속도가 30과 50일 때의 제동 거리를 구하세요

# 구축한 모델을 시각화하세요
def linear_regression_cars():
    cars = pd.read_csv('data/cars.csv', index_col=0)
    # print(cars.values)

    x = cars.values[:, 0]
    y = cars.values[:, 1]
    # print(x.shape, y.shape)  # (50,) (50,)

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.SGD(10e-5), loss=keras.losses.mse)  # mean square error
    model.fit(x, y, epochs=50)

    p = model.predict([0, 30, 50])
    p = p.reshape(-1)
    p0, p1, p2 = p
    print('speed 30: ', p1)
    print('speed 50: ', p2)

    plt.plot(x, y, 'ro')
    plt.plot([0, 30], [0, p1], 'g')
    plt.plot([0, 30], [p0, p1], 'b')
    plt.show()


# linear_regression()
linear_regression_cars()


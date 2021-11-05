# Day_08_01_google.py
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import numpy as np
import tensorflow.keras as keras
from sklearn import preprocessing, model_selection


# 퀴즈
# GOOG.csv 파일을 읽고 70%로 학습하고 30%에 대해 결과를 예측하세요 (Close 예측)

def get_xy():
    stock = pd.read_csv('../data/GOOG.csv', index_col=0)
    # print(stock)      # [252 rows x 6 columns]

    values = [stock['Open'], stock['High'], stock['Low'], stock['Volume'], stock['Close']]
    values = np.transpose(values)
    # print(values.shape)           # (252, 5)

    # ----------------------------- #

    scaler = preprocessing.MinMaxScaler()
    values = scaler.fit_transform(values)
    # values = values[::-1]                     # 앞쪽이 옛날 데이터

    grams = nltk.ngrams(values, 7+1)
    grams = np.float32(list(grams))
    # print(grams.shape)            # (725, 8, 5)

    x = np.float32([g[:-1] for g in grams])
    y = np.float32([g[-1, -1:] for g in grams])
    # print(x.shape, y.shape)       # (725, 7, 5) (725, 1)

    return x, y, scaler.data_min_[-1], scaler.data_max_[-1]


# 퀴즈
# 앞에서 만든 데이터에 대해 모델을 구축하세요
def model_stock():
    x, y, data_min, data_max = get_xy()

    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))
    model.add(keras.layers.SimpleRNN(32, return_sequences=False))
    model.add(keras.layers.Dense(1))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.mse,
                  metrics='mae')

    model.fit(x_train, y_train, epochs=100, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))

    # 퀴즈
    # 정답과 예측 결과를 시각화하세요
    p = model.predict(x_test)

    plt.subplot(1, 2, 1)
    plt.plot(y_test, 'r', label='target')
    plt.plot(p, 'g', label='prediction')
    plt.legend()

    # 퀴즈
    # 예측 결과를 원래 값으로 복구하세요
    p = data_min + (data_max - data_min) * p
    y_test = data_min + (data_max - data_min) * y_test

    plt.subplot(1, 2, 2)
    plt.plot(y_test, 'r')
    plt.plot(p, 'g')
    plt.ylim(1000, 3000)

    plt.show()


model_stock()


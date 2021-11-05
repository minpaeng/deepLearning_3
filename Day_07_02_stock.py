import pandas as pd
import tensorflow.keras as keras
import numpy as np
import nltk
from sklearn import preprocessing, model_selection


# 퀴즈 1
# stock_daily.csv 파일로부터 x, y를 반환하는 함수를 만드세요
# batch_size, seq_length, n_features = 32, 7, 5

# 퀴즈 2
# 80%의 데이터로 학습하고 20%의 데이터에 대해 결과를 예측하세요
def get_xy():
    stock = pd.read_csv('data/stock_daily.csv', skiprows=2, header=None)
    # print(stock)

    values = preprocessing.minmax_scale(stock.values)
    values = values[::-1]    # 최신날짜 -> 오래된날짜 순으로 뒤집기(rnn에 최신데이터 먼저 넣어줘야함)

    grams = list(nltk.ngrams(values, 7 + 1))
    grams = np.float32(grams)
    # print(grams.shape)

    x = np.float32([g[:-1] for g in grams])
    y = np.float32([g[-1, -1:] for g in grams])
    # print(x.shape, y.shape)  # (725, 7, 5) (725, 1)

    print(x.shape, y.shape)
    return x, y


# 퀴즈
# 앞에서 만든 데이터에 대해 모델을 구축하세요
def modef_stock():
    x, y = get_xy()

    data = model_selection.train_test_split(x, y, train_size=0.8)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x_train.shape[1:]))
    model.add(keras.layers.SimpleRNN(32, return_sequences=False))
    model.add(keras.layers.Dense(1))
    model.summary()

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.mse,
                  metrics='mae')

    model.fit(x_train, y_train, epochs=100, verbose=2, batch_size=32)
    model.evaluate(x_test, y_test, verbose=2)
    p = model.predict(x_test)


modef_stock()

# applekoong@naver.com

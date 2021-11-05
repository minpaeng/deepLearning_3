import tensorflow.keras as keras
import numpy as np
from sklearn import preprocessing


# RNN: Recurrent Neural Network
# 시계열 데이터. 순서가 있는 데이터에 사용

# tensor
# tenso -> ensor

# 퀴즈
# tensor를 원핫 벡터로 변환해서 x, y로 분할하고
# 해당 데이터로 동작하는 모델을 만드세요
def char_rnn_1_dense():
    x = [[1, 0, 0, 0, 0, 0],  # tenso
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0]]
    y = [[0, 1, 0, 0, 0, 0],  # ensor
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(6, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))


# 퀴즈
# 앞의 코드를 sparse 버전으로 수정하세요
def char_rnn_1_sparse():
    x = [[1, 0, 0, 0, 0, 0],  # tenso
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0]]
    y = [[1],  # ensor
         [2],
         [3],
         [4],
         [5]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(6, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))


# char_rnn_1_dense()
char_rnn_1_sparse()

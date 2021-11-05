# Day_05_02_char_rnn_2.py
import tensorflow.keras as keras
import numpy as np


# 퀴즈
# 아래처럼 정렬하세요
# tensor -> enorst
def char_rnn_2_sorted():
    x = [[0, 0, 0, 0, 0, 1],  # tenso
         [1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0]]
    y = [0, 1, 4, 2, 3]       # ensor

    model = keras.Sequential()
    model.add(keras.layers.Dense(6, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))


def char_rnn_2_simple_rnn():
    x = [[0, 0, 0, 0, 0, 1],  # tenso
         [1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0]]
    y = [0, 1, 4, 2, 3]       # ensor

    x = np.float32([x])
    y = np.float32([y])

    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(2, return_sequences=True))
    model.add(keras.layers.Dense(6, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=10, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    # 퀴즈
    # predict 함수를 사용해서 직접 정확도를 구하세요
    p = model.predict(x)
    print(p.shape)
    print(y.shape)

    p_arg = np.argmax(p[0], axis=1)
    y_arg = y[0]
    print(p_arg)
    print(y_arg)

    print('acc :', np.mean(p_arg == y_arg))


# char_rnn_2_sorted()
char_rnn_2_simple_rnn()


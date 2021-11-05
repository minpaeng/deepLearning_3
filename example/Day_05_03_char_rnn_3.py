# Day_05_03_char_rnn_3.py
import tensorflow.keras as keras
import numpy as np
from sklearn import preprocessing


# 퀴즈
# 단어를 x, y로 변환하는 함수를 만드세요
def make_xy(word):
    lb = preprocessing.LabelBinarizer()
    onehot = lb.fit_transform(list(word))

    x = onehot[:-1]
    y = onehot[1:]

    y = np.argmax(y, axis=1)
    # return x[np.newaxis], y[np.newaxis]       # int32라서 실패
    return np.float32([x]), np.float32([y])


def char_rnn_3(word):
    x, y = make_xy(word)
    # print(x.shape, y.shape)   # (1, 5, 6) (1, 5)

    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(2, return_sequences=True))
    model.add(keras.layers.Dense(x.shape[-1], activation='softmax'))

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


char_rnn_3('tensor')
# char_rnn_3('rainbow eyes')

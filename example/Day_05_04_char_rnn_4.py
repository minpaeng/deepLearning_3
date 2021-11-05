# Day_05_04_char_rnn_4.py
import tensorflow.keras as keras
import numpy as np
from sklearn import preprocessing


# 퀴즈
# 여러 개의 단어를 x, y로 변환하는 함수를 만드세요
def make_xy(words):
    long_text = ''.join(words)
    # print(long_text)      # yellowcoffeetensor
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(long_text))

    x, y = [], []
    for w in words:
        onehot = lb.transform(list(w))
        xx = onehot[:-1]
        yy = onehot[1:]

        yy = np.argmax(yy, axis=1)

        x.append(xx)
        y.append(yy)

    return np.float32(x), np.float32(y)


def char_rnn_4(words):
    x, y = make_xy(words)
    # print(x.shape, y.shape)   # (3, 5, 11) (3, 5)

    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(32, return_sequences=True))
    model.add(keras.layers.Dense(x.shape[-1], activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    # 퀴즈
    # predict 함수를 사용해서 직접 정확도를 구하세요
    p = model.predict(x)
    print(p.shape)
    print(y.shape)

    for i in range(len(x)):
        p_arg = np.argmax(p[i], axis=1)
        y_arg = y[i]
        print(p_arg)
        print(y_arg)

        print('acc :', np.mean(p_arg == y_arg))


char_rnn_4(['yellow', 'coffee', 'tensor'])

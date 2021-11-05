import tensorflow.keras as keras
import numpy as np
from sklearn import preprocessing


# 퀴즈
# 단어를 x, y로 변환하는 함수를 만드세요
def make_xy(word):
    enc = preprocessing.LabelBinarizer()
    x = enc.fit_transform(list(word))
    y = np.argmax(x, axis=1)
    # print(x, y)
    x = x[:-1]
    y = y[1:]
    # print(x, y)

    x = np.float32([x])
    y = np.float32([y])

    return x, y


def char_rnn_3(word):
    x, y = make_xy(word)

    model = keras.Sequential()

    model.add(keras.layers.SimpleRNN(12, return_sequences=True))
    model.add(keras.layers.Dense(x.shape[-1], activation='softmax'))    # x.shape[-1]을 통해 여러 단어에 대해 사용 가능

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    # 퀴즈
    # predict 함수를 사용해서 직접 정확도를 구하세요
    p = model.predict(x)
    print(p.shape, y.shape)  # (1, 5, 6) (1, 5)

    p_arg = np.argmax(p[0], axis=1)
    y_arg = y[0]
    print(p_arg)
    print(y_arg)

    print('acc :', np.mean(p_arg == y_arg))


char_rnn_3('apple')

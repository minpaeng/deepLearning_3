# Day_05_05_char_rnn_5.py
import tensorflow.keras as keras
import numpy as np
from sklearn import preprocessing


# 퀴즈
# 길이가 다른 단어들의 목록에 동작하도록 수정하세요
def make_xy(words):
    long_text = ''.join(words)
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(long_text))

    max_len = max([len(w) for w in words])

    x, y = [], []
    for w in words:
        if len(w) < max_len:
            w += '*' * (max_len - len(w))

        onehot = lb.transform(list(w))
        xx = onehot[:-1]
        yy = onehot[1:]

        yy = np.argmax(yy, axis=1)

        x.append(xx)
        y.append(yy)

    return np.float32(x), np.float32(y), lb.classes_


def char_rnn_5(words):
    x, y, vocab = make_xy(words)

    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(32, return_sequences=True))
    model.add(keras.layers.Dense(x.shape[-1], activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    p = model.predict(x)
    p_arg = np.argmax(p, axis=2)

    # 퀴즈
    # 패딩에 대해 예측한 결과를 버리세요
    # lengths = [len(w) for w in words]

    # for i, w in zip(vocab[p_arg], words):
    #     # print(w, len(w))
    #     valid = len(w) - 1
    #     print(''.join(i[:valid]))
    #
    # for i in range(len(words)):
    #     pred = p_arg[i]
    #     w = words[i]
    #     converted = vocab[pred]
    #     w_len = len(w) - 1
    #
    #     print(converted[:w_len])

    print(''.join(vocab[p_arg[0]]), end='')

    for i in range(1, len(p_arg)):
        print(vocab[p_arg[i, -1]], end='')


if __name__ == '__main__':
    char_rnn_5(['yellow', 'sky', 'blood_game'])     # 패딩 : *

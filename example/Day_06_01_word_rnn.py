# Day_06_01_word_rnn.py
# Day_05_04_char_rnn_4.py
import tensorflow.keras as keras
import numpy as np
from sklearn import preprocessing


# 퀴즈
# 여러 개의 단어를 x, y로 변환하는 함수를 만드세요
def make_xy(sentences):
    tokens = [word for sent in sentences for word in sent.split()]
    # print(tokens)

    # tokens = []
    # for sent in sentences:
    #     for word in sent.split():
    #         # print(word)
    #         tokens.append(word)

    lb = preprocessing.LabelBinarizer()
    lb.fit(tokens)

    x, y = [], []
    for sent in sentences:
        onehot = lb.transform(sent.split())
        xx = onehot[:-1]
        yy = onehot[1:]

        yy = np.argmax(yy, axis=1)

        x.append(xx)
        y.append(yy)

    return np.float32(x), np.float32(y), lb.classes_


def char_rnn_4(words):
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

    for i in vocab[p_arg]:
        print(' '.join(i))


# 퀴즈
# char_rnn_4 모델을 word 버전으로 수정하세요
sentences = ['jeonju is the most beautiful korea',
             'bibimbap is the most famous food',
             'tomorrow i am going to market']

char_rnn_4(sentences)

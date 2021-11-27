import tensorflow.keras as keras
import numpy as np
from sklearn import preprocessing


def make_xy(sentences):
    tokens = [word for sent in sentences for word in sent.split()]
    # 위 코드와 같은 for문
    # long_text = []
    # for word in words:
    #     for w in word.split():
    #         print(w)
    #         long_text.append(w)

    lb = preprocessing.LabelBinarizer()
    lb.fit(list(tokens))

    x, y = [], []
    for sent in sentences:
        onehot = lb.transform(list(sent.split()))
        xx = onehot[:-1]
        yy = onehot[1:]

        yy = np.argmax(yy, axis=1)

        x.append(xx)
        y.append(yy)

    return np.float32(x), np.float32(y), lb.classes_    # classes_: 유니크한 토큰


def char_rnn_4(sentences):
    x, y, vocab = make_xy(sentences)
    # print(x.shape, y.shape)     # (3, 5, 11) (3, 5)
    # print(vocab)

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

    # 위 컴프리핸션?이랑 같은코드
    for i in vocab[p_arg]:
        print(' '.join(i))


# 퀴즈
# char_rnn_4 모델을 word 버전으로 수정하세요
sentences = ['jeonju is the most beautiful korea',
             'bibimbap is the most famous food',
             'tomorrow i am going to market']

char_rnn_4(sentences)

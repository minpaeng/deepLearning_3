import tensorflow.keras as keras
import numpy as np
from sklearn import preprocessing


def make_xy(words):
    long_text = ''.join(words)
    # print(long_text)        # yellowcoffeetensor

    lb = preprocessing.LabelBinarizer()
    lb.fit(list(long_text))     # 나중에 원핫벡터로 변환하기 위해 당장 변환은 하지 않고 단어 리스트에 대한 학습만 함

    x, y = [], []
    for w in words:
        onehot = lb.transform(list(w))      # transform으로 문자를 원핫벡터로 변환
        # print(onehot)
        xx = onehot[:-1]
        yy = onehot[1:]

        yy = np.argmax(yy, axis=1)

        x.append(xx)
        y.append(yy)

    return np.float32(x), np.float32(y), lb.classes_    # classes_: 유니크한 토큰


def char_rnn_4(words):
    x, y, vocab = make_xy(words)
    # print(x.shape, y.shape)     # (3, 5, 11) (3, 5) (3, 5, 11)에서  5는 sequence length(셀을 몇번 전개할 것인지)
    # print(vocab)        # ['c' 'e' 'f' 'l' 'n' 'o' 'r' 's' 't' 'w' 'y']

    model = keras.Sequential()

    model.add(keras.layers.SimpleRNN(32, return_sequences=True))
    model.add(keras.layers.Dense(x.shape[-1], activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    p = model.predict(x)
    # print(p.shape, y.shape)       # (3, 5, 11) (3, 5)

    # for i in range(len(x)):
    #     p_arg = np.argmax(p[i], axis=1)
    #     y_arg = y[i]
    #     print(p_arg)      # [[[1 3 3 5 9] [5 2 2 1 1] [1 4 7 5 6]]
    #     print(y_arg)
    #
    #     print('acc :', np.mean(p_arg == y_arg))

    p_arg = np.argmax(p, axis=2)
    print(p_arg)
    print('acc :', np.mean(p_arg == y, axis=1))

    # 퀴즈
    # vocab 을 사용해서 예측 결과를 디코딩하세요

    # for i in range(len(p_arg)):
    #     print([vocab[j] for j in p_arg[i]])   # 컴프리핸션.......? 출력의 결과로 리스트가 나옴

    for pred, yy in zip(p_arg, y):
        print('y : ', ''.join([vocab[j] for j in np.int32(yy)]))
        print('p : ', ''.join([vocab[j] for j in pred]))
        print()

    # 인덱스 배열
    print(vocab[p_arg])
    print([''.join(i) for i in vocab[p_arg]])

    # 위 컴프리핸션?이랑 같은코드
    for i in vocab[p_arg]:
        print(''.join(i))

    # print(predict_word)


char_rnn_4(['yellow', 'coffee', 'tensor'])

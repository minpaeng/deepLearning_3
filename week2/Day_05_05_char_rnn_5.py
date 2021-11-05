import tensorflow.keras as keras
import numpy as np
from sklearn import preprocessing


# 퀴즈
# 길이가 다른 단어들의 목록에 동작하도록 수정하세요
def make_xy(words):
    long_text = ''.join(words)
    # print(long_text)        # yellowcoffeetensor

    lb = preprocessing.LabelBinarizer()
    lb.fit(list(long_text))  # 나중에 원핫벡터로 변환하기 위해 당장 변환은 하지 않고 단어 리스트에 대한 학습만 함

    max_len = max([len(w) for w in words])

    x, y = [], []
    for w in words:
        if len(w) < max_len:
            w += '*' * (max_len - len(w))
            # print(w)

        onehot = lb.transform(list(w))  # transform으로 문자를 원핫벡터로 변환
        # print(onehot)
        xx = onehot[:-1]
        yy = onehot[1:]

        yy = np.argmax(yy, axis=1)

        x.append(xx)
        y.append(yy)

    return np.float32(x), np.float32(y), lb.classes_  # classes_: 유니크한 토큰


def char_rnn_5(words):
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

    p_arg = np.argmax(p, axis=2)

    # print([''.join(i) for i in vocab[p_arg]])

    # 퀴즈
    # 패디에 대해 예측한 결과를 버리세요
    max_len = max([len(w) for w in words])
    for i, w in zip(vocab[p_arg], words):
        # print(w, len(w))
        valid = len(w) - 1
        print(''.join(i[:valid]))
        # word[words[i]+1:] = ''


if __name__ == '__main__':
    char_rnn_5(['yellow', 'sky', 'blood_game'])  # 패딩: *

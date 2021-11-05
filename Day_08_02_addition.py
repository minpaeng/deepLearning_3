# Day_08_02_addition.py
import random
import tensorflow.keras as keras
from sklearn import model_selection

# 235+17=252
# x : 235+17
# y : 252


# 퀴즈
# 자릿수에 맞는 숫자를 만드는 함수를 구현하세요
# digits: 자릿수
import numpy as np


def make_number(digits):
    # return random.randrange(10 ** digits)

    d = random.randrange(digits) + 1
    return random.randrange(10 ** d)


def make_data(size, digits, reverse):
    questions, expected, seen = [], [], set()

    while len(questions) < size:
        a = make_number(digits)
        b = make_number(digits)

        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)

        q = '{}+{}'.format(a, b)
        q += '#' * (digits * 2 + 1 - len(q))    # 86+7###

        t = str(a + b)
        t += '#' * (digits + 1 - len(t))        # 93##

        if reverse:
            t = t[::-1]                         # ##39

        questions.append(q)
        expected.append(t)

    return questions, expected


def make_onehot(texts, chr2idx):
    batch_size, seq_length, n_features = len(texts), len(texts[0]), len(chr2idx)
    v = np.zeros([batch_size, seq_length, n_features])

    for i, t in enumerate(texts):
        for j, c in enumerate(t):
            k = chr2idx[c]
            v[i, j, k] = 1
    return v


questions, expected = make_data(size=50000, digits=3, reverse=True)

vocab = '#+0123456789'

chr2idx = {c: i for i, c in enumerate(vocab)}
idx2chr = {i: c for i, c in enumerate(vocab)}
# print(chr2idx)        # {'#': 0, '+': 1, '0': 2, ...}
# print(idx2chr)        # {0: '#', 1: '+', 2: '0', ...}

# print(questions[:3])  # ['936+0##', '723+26#', '7+91###']
# print(expected[:3])   # ['936#', '749#', '98##']

x = make_onehot(questions, chr2idx)     # (50000, 7, 12)
y = make_onehot(expected, chr2idx)      # (50000, 4, 12)

# print(x[0, 0])    # [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
# print(x[0, -1])   # [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

# 퀴즈
# 앞에서 만든 x, y에 대해 80%로 학습하고 20%에 대해 정확도를 구하세요
data = model_selection.train_test_split(x, y, train_size=0.8)
x_train, x_test, y_train, y_test = data

model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))
# bad
# model.add(keras.layers.SimpleRNN(128, return_sequences=True))
# model.add(keras.layers.Reshape([4, -1]))
# excellent
model.add(keras.layers.LSTM(128, return_sequences=False))
model.add(keras.layers.RepeatVector(y.shape[1]))                    # 3+1
model.add(keras.layers.LSTM(128, return_sequences=True))
model.add(keras.layers.Dense(y.shape[-1], activation='softmax'))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.categorical_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=30, verbose=2,
          validation_data=(x_test, y_test))
# print(model.evaluate(x_test, y_test, verbose=0))

# 퀴즈
# 테스트 데이터셋에서 랜덤한 데이터를 뽑아서
# 한번에 하나씩 문제, 정답, 예측 결과를 출력하세요
# 출력 결과는 문자열로 디코딩합니다.
for _ in range(10):
    idx = random.randrange(len(x_test))

    q = x_test[idx][np.newaxis]     # [np.newaxis, :, :]와 동일. [:,np.newaxis] 등으로 활용
    a = y_test[idx][np.newaxis]
    p = model.predict(q)
    # print(q.shape, a.shape, p.shape)      # 1, 7, 12) (1, 4, 12) (1, 4, 12)

    q_arg = np.argmax(q[0], axis=1)
    a_arg = np.argmax(a[0], axis=1)
    p_arg = np.argmax(p[0], axis=1)
    # print(q_arg)              # [ 5  4  2  1  3  9 10]
    # print(p_arg)              # [ 5 11 11  0]

    # q_dec = ''.join([idx2chr[n] for n in q_arg])
    # q_dec = q_dec.replace('#', '')
    q_dec = ''.join([idx2chr[n] for n in q_arg]).replace('#', '')
    a_dec = ''.join([idx2chr[n] for n in a_arg]).replace('#', '')
    p_dec = ''.join([idx2chr[n] for n in p_arg]).replace('#', '')
    print('문제 :', q_dec)
    print('정답 :', a_dec)
    print('예측 :', p_dec)
    print()









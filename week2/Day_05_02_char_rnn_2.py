import tensorflow.keras as keras
import numpy as np
from sklearn import preprocessing

# https://m.blog.naver.com/ekbae98/221265719881

# 퀴즈
# 아래처럼 정렬하세요
# tensor-> enorst: enorst를 vocabulary라고 함
def char_rnn_2_sorted():
    x = [[0, 0, 0, 0, 0, 1],  # tenso
         [1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0]]
    y = [0, 1, 4, 2, 3]

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
         [0, 0, 0, 1, 0, 0]]
    y = [0, 1, 4, 2, 3]

    x = np.float32([x])
    y = np.float32([y])

    model = keras.Sequential()
    # return_sequences: 기본 false로 예측한 값들 중 마지막 하나만 사용
    # rnn의 입력데이터는 3차원(cnn은 4차원)
    model.add(keras.layers.SimpleRNN(2, return_sequences=True))
    model.add(keras.layers.Dense(6, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    # 퀴즈
    # predict 함수를 사용해서 직접 정확도를 구하세요
    p = model.predict(x)
    print(p.shape, y.shape)     # (1, 5, 6) (1, 5)
    print(p)
    # p_arg = np.argmax(p[0], axis=1)    # 이거 왜 6개가나오죠..?
    # print(p_arg)
    p_arg = np.argmax(p[0], axis=1)
    print(p_arg)
    y_arg = y[0]
    print(p_arg)
    print(y_arg)

    print('acc :', np.mean(p_arg == y_arg))


# char_rnn_2_sorted()
char_rnn_2_simple_rnn()

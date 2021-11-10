import nltk
from sklearn import preprocessing
import numpy as np
import tensorflow.keras as keras


# 퀴즈
# 니체 파일을 x, y 데이터로 반환하는 함수를 만드세요
def make_data(seq_length):
    # 1. 파일 읽기
    f = open('../data/nietzsche.txt', "r", encoding='utf-8')
    nietzsche = f.read()  # f.read(10000): 10000글자만 읽어옴
    nietzsche = nietzsche.lower()
    f.close()
    # print(len(nietzsche))            # 600893
    nietzsche = nietzsche[:100000]

    # 2. 숫자로 변환
    bin = preprocessing.LabelBinarizer()
    onehot = bin.fit_transform(list(nietzsche))
    # print(onehot.shape)              # (10000, 46)
    # print(bin.classes_)

    # 2. x, y로 변환
    grams = list(nltk.ngrams(onehot, seq_length + 1))
    grams = np.float32(grams)
    # print(grams.shape)               # (9940, 61, 46)

    # # 퀴즈
    # # grams를 x, y로 분할하세요
    # x = np.int32([w[:-1] for w in grams])
    # y = np.argmax([w[-1] for w in grams], axis=1)
    # # print(x.shape, y.shape)            # (9940, 60, 46) (9940,)

    # 위 코드와 같음: 컨프리핸션?은 넘파이를 사용하기 때문
    x = grams[:, :-1]
    y = np.argmax(grams[:, -1], axis=1)
    # print(x.shape, y.shape)                 # (9940, 60, 46) (9940,)

    return x, y, bin.classes_


# 퀴즈
# 모델을 구축해서 결과를 예측하세요 (디코딩 포함)
def make_model(vocab_size):
    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(128))   # LSTM, GRU 등 바꿔가며 써보기
    model.add(keras.layers.Dense(vocab_size, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    return model


def predict_basic(model, x, vocab):
    p = model.predict(x)
    # print(p.shape)                      # (9940, 46)
    p_arg = np.argmax(p, axis=1)
    # print(p_arg.shape)                  # (9940,)

    print(vocab[p_arg])
    print(''.join(vocab[p_arg]))


# 퀴즈
# start_token부터 시작해서 한 글자씩 100글자를 만들어서 결과를 출력하세요
def predict_by_argmax(model, tokens, vocab):
    for i in range(100):
        # print(tokens.shape)             # (60, 46)
        # rnn은 3차원 데이터를 predict에 인자로 넘겨줘야함
        p = model.predict(tokens[np.newaxis])       # tokens: (1, 60, 46)
        # print(p.shape)                              # (1, 46)
        p = p[0]
        p_arg = np.argmax(p)

        print(vocab[p_arg], end='')

        # 1 + 59 + 1 글자 -> 가장 오래된 숫자 하나를 삭제하고 예측한 값을 뒤에 추가
        tokens[:-1] = tokens[1:]
        tokens[-1] = p


def weighted_pick(p):
    t = np.cumsum(p)

    n = np.random.rand(1)[0]
    return np.searchsorted(t, n)


# 퀴즈
# 가중치 합계를 이용해서 예측 결과를 디코딩하세요
def predict_by_weighted(model, tokens, vocab):
    for i in range(100):
        p = model.predict(tokens[np.newaxis])
        p = p[0]
        p_arg = weighted_pick(p)
        print(vocab[p_arg], end='')

        # 1 + 59 + 1 글자 -> 가장 오래된 숫자 하나를 삭제하고 예측한 값을 뒤에 추가
        tokens[:-1] = tokens[1:]
        tokens[-1] = p
    print()
    print('-'*30)

def temperature_pick(z, t):
    z = np.log(z) / t
    z = np.exp(z)                   # np.e ** z
    s = np.sum(z)
    return weighted_pick(z / s)


# 퀴즈
# 온도에 따라 다른 결과를 만들어 내는 함수를 구현하고, 예측 결과를 디코딩하세요
def predict_by_temperature(model, tokens, vocab, temperature):
    for i in range(100):
        p = model.predict(tokens[np.newaxis])
        p = p[0]
        p_arg = temperature_pick(p, temperature)
        print(vocab[p_arg], end='')

        # 1 + 59 + 1 글자 -> 가장 오래된 숫자 하나를 삭제하고 예측한 값을 뒤에 추가
        tokens[:-1] = tokens[1:]
        tokens[-1] = p


seq_length = 60
x, y, vocab = make_data(seq_length)

model = make_model(len(vocab))
model.fit(x, y, epochs=10, verbose=2)

# predict_basic(model, x, vocab)

pos = np.random.randint(0, len(x) - seq_length, 1)
pos = pos[0]

tokens = x[pos]
print(tokens.shape)                 # (60, 46)

# predict_by_argmax(model, tokens, vocab)
predict_by_weighted(model, tokens, vocab)
predict_by_temperature(model, tokens, vocab, 0.2)
predict_by_temperature(model, tokens, vocab, 0.5)
predict_by_temperature(model, tokens, vocab, 0.8)
predict_by_temperature(model, tokens, vocab, 2.0)
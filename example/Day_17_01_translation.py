# Day_17_01_translation.py
import tensorflow.keras as keras
import numpy as np


# 퀴즈
# 토큰 사전과 토큰 인덱스(chr2idx, 딕셔너리)를 반환하는 함수를 만드세요
def make_vocab(data):
    eng = sorted(set(''.join([e for e, _ in data])))
    kor = sorted(set(''.join([k for _, k in data])))
    # print(eng)
    # print(kor)

    vocab = eng + kor + list('SEP')     # Start, End, Pad
    # print(vocab)      # ['a', 'd', 'e', 'f', ...]

    chr2idx = {v: i for i, v in enumerate(vocab)}
    # print(chr2idx)    # {'a': 0, 'd': 1, 'e': 2, ...}

    return vocab, chr2idx


def make_xy(data, chr2idx):
    onehots = np.eye(len(chr2idx), dtype=np.int32)

    enc_x, dec_x, dec_y = [], [], []
    for e, k in data:
        enc_in = [chr2idx[c] for c in e]
        dec_in = [chr2idx[c] for c in 'S'+k]
        target = [chr2idx[c] for c in k+'E']
        # print(enc_in, dec_in, target)     # [3, 8, 8, 1] [25, 22, 19] [22, 19, 26]

        # 퀴즈
        # enc_in을 원핫 벡터로 변환해서 추가하세요
        # enc_x.append([onehots[i] for i in enc_in])
        enc_x.append(onehots[enc_in])
        dec_x.append(onehots[dec_in])
        dec_y.append(target)                # sparse로 처리하기 때문에 변환하지 않는다
        # print(enc_x[-1])

    return np.int32(enc_x), np.int32(dec_x), np.int32(dec_y)


data = [('food', '음식'), ('pink', '분홍'),
        ('wind', '바람'), ('desk', '책상'),
        ('head', '머리'), ('hero', '영웅')]

vocab, chr2idx = make_vocab(data)
enc_x, dec_x, dec_y = make_xy(data, chr2idx)
# print(enc_x.shape, dec_x.shape, dec_y.shape)  # (6, 4, 28) (6, 3, 28) (6, 3)

# 인코더
enc_input = keras.layers.Input(enc_x.shape[1:])
_, enc_state = keras.layers.SimpleRNN(128, return_state=True)(enc_input)

# 디코더
dec_input = keras.layers.Input(dec_x.shape[1:])
dec_output = keras.layers.SimpleRNN(128, return_sequences=True)(dec_input, initial_state=enc_state)
dec_output = keras.layers.Dense(len(vocab), activation='softmax')(dec_output)

model = keras.Model([enc_input, dec_input], dec_output)
model.summary()

# 퀴즈
# 학습하고 결과를 예측하세요
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit([enc_x, dec_x], dec_y, epochs=100, verbose=2)

# 퀴즈
# 학습한 데이터 중에서 2개를 골라서 번역하세요
# 'hero', 'pink'
data = [('pink', 'PP'), ('hero', 'PP')]

enc_x, dec_x, _ = make_xy(data, chr2idx)
p = model.predict([enc_x, dec_x])
# print(p.shape)        # (2, 3, 28)

# 퀴즈
# 예측한 결과를 디코딩하세요
p_arg = np.argmax(p, axis=2)
print(p_arg)
print([''.join([vocab[v] for v in t[:-1]]) for t in p_arg])



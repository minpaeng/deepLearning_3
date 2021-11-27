# Day_18_01_translation_word.py
import tensorflow.keras as keras
import numpy as np


# 퀴즈
# 토큰 사전과 토큰 인덱스(chr2idx, 딕셔너리)를 반환하는 함수를 만드세요
def make_vocab(data):
    eng = sorted(set([t for e, _ in data for t in e.split()]))
    kor = sorted(set([t for _, k in data for t in k.split()]))
    # print(eng)
    # print(kor)

    vocab = eng + kor + ['_SOS_', '_EOS_', '_PAD_']
    # print(vocab)      # ['about', 'am', 'apple', ...]

    chr2idx = {v: i for i, v in enumerate(vocab)}
    # print(chr2idx)    # {'about': 0, 'am': 1, 'apple': 2, ...}

    return vocab, chr2idx


def make_xy(data, chr2idx):
    onehots = np.eye(len(chr2idx), dtype=np.int32)

    enc_x, dec_x, dec_y = [], [], []
    for e, k in data:
        e = e.split()
        k = k.split()

        enc_in = [chr2idx[c] for c in e]
        dec_in = [chr2idx[c] for c in ['_SOS_']+k]
        target = [chr2idx[c] for c in k+['_EOS_']]
        print(enc_in, dec_in, target)     # [17, 10, 18, 6] [35, 22, 23, 24] [22, 23, 24, 36]

        enc_x.append(onehots[enc_in])
        dec_x.append(onehots[dec_in])
        dec_y.append(target)                # sparse로 처리하기 때문에 변환하지 않는다
        # print(enc_x[-1])

    return np.int32(enc_x), np.int32(dec_x), np.int32(dec_y)


# 퀴즈
# 영어 문장을 한글 문장으로 번역하는 모델을 구축하세요
# 데이터 갯수는 6개.
data = [('what is your goal', '너의 목표는 뭐니'),
        ('house is very expensive', '집이 너무 비싸'),
        ('how about your day', '오늘 하루 어땠어'),
        ('i went to school', '오늘 학교 갔어'),
        ('i like apple pie', '애플 파이를 좋아해'),
        ('i am your father', '내가 너의 앱이다'),
        ]

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

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit([enc_x, dec_x], dec_y, epochs=100, verbose=2)

# 퀴즈
# 학습한 데이터 중에서 2개를 골라서 번역하세요
data = [('how about your day', '오늘 하루 어땠어'),
        ('i went to school', '오늘 학교 갔어')]

enc_x, dec_x, _ = make_xy(data, chr2idx)
p = model.predict([enc_x, dec_x])
# print(p.shape)        # (2, 4, 38)

p_arg = np.argmax(p, axis=2)
print(p_arg)
print([' '.join([vocab[v] for v in t[:-1]]) for t in p_arg])

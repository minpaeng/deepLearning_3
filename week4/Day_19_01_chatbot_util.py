# Day_19_01_chatbot_util.py
import keras.layers
import numpy as np
import tensorflow.keras as keras
# 참고 사이트:https://github.com/golbin/TensorFlow-Tutorials

_PAD_, _SOS_, _EOS_, _UNK_ = 0, 1, 2, 3  # '_PAD_', '_SOS_', '_EOS_', '_UNK_'


def train_and_save_model(enc_x, dec_x, dec_y, vocab):
    # 인코더
    enc_input = keras.layers.Input(enc_x.shape[1:])
    _, enc_state = keras.layers.SimpleRNN(128, return_state=True)(enc_input)
    # _ : enc_output(출력에 대한 값),  enc_state : state

    # 디코더
    dec_input = keras.layers.Input(dec_x.shape[1:])
    dec_output = keras.layers.SimpleRNN(128, return_sequences=True)(dec_input, initial_state=enc_state)
    dec_output = keras.layers.Dense(len(vocab), activation='softmax')(dec_output)

    model = keras.Model([enc_input, dec_input], dec_output)  # 입력이 두개, 출력이 한개
    model.summary()

    # 모델 학습, 결과 예측
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit([enc_x, dec_x], dec_y, epochs=100, verbose=2)

    model.save('model/chat.h5')


def read_vocab():
    f = open('chat_data/vocab.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    x = [line.strip() for line in lines]
    # vocab = f.read().split()
    f.close()
    # print(x[:3])  # ['_PAD_', '_SOS_', '_EOS_']

    return x


def read_vector():
    f = open('chat_data/vectors.txt', 'r', encoding='utf-8')
    vector = [[int(v) for v in row.split(',')] for row in f.read().split()]
    f.close()
    # print(vector[:3])  # [['105'], ['105'], ['114', '128', '85', '79']]

    return vector


def add_pad(sequence, max_len):
    if len(sequence) > max_len:
        return sequence[:max_len]  # 우리 모델의 글자 수보다 더 크게 할 경우 잘라준다

    return sequence + (max_len - len(sequence)) * [_PAD_]


def make_xy(questions, answers, vocab):
    q_len = max([len(q) for q in questions])
    a_len = max([len(a) for a in answers]) + 1
    # print(q_len, a_len)  # 9 10

    onehots = np.eye(len(vocab), dtype=np.int32)

    enc_x, dec_x, dec_y = [], [], []
    for q, a in zip(questions, answers):
        # print(q)
        enc_in = add_pad(q, q_len)
        dec_in = add_pad([_SOS_]+a, a_len)
        target = add_pad(a+[_EOS_], a_len)
        # print(enc_in, dec_in, target)

        enc_x.append(onehots[enc_in])
        dec_x.append(onehots[dec_in])
        dec_y.append(target)
        # print(enc_x[-1])

    return np.int32(enc_x), np.int32(dec_x), np.int32(dec_y)


def make_dataset():
    vocab = read_vocab()
    vector = read_vector()

    # 나중에 이 모델을 이용해서 심심이 같은 챗봇도 만들어볼 수 있다! (오,,)
    # 퀴즈 1
    # 질문과 대답에서 최대 길이를 구하세요 ( 먼저 질문과 대답으로 분리)
    questions = [q for q in vector[::2]]  # 슬라이싱을 이용해서 2칸씩 건너뜀
    answers = [a for a in vector[1::2]]

    # print(add_pad([1, 2, 3], 10))  # [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
    # print(add_pad([1, 2, 3], 3))  # [1, 2, 3]

    enc_x, dec_x, dec_y = make_xy(questions, answers, vocab)
    # print(enc_x.shape, dec_x.shape, dec_y.shape)  # (52, 9, 164) (52, 10, 164) (52, 10)

    return enc_x, dec_x, dec_y, vocab


def load_model(enc_x, dec_x, dec_y, vocab):
    model = keras.models.load_model('model/chat.h5')
    p = model.predict([enc_x, dec_x])

    x_arg = np.argmax(enc_x, axis=2)
    p_arg = np.argmax(p, axis=2)
    for xx, yy, pp in zip(x_arg, dec_y, p_arg):
        decode_prediction(xx, vocab, '질문')
        decode_prediction(yy, vocab, '대답')
        decode_prediction(pp, vocab, '예측')
        print()

    # print(p_arg)
    # print([''.join([vocab[v] for v in t[:-1]]) for t in p_arg])


def decode_prediction(p, vocab, title):
    p = list(p)
    p = [i for i in p if i != _PAD_]  # 질문 : 안녕 _PAD_ _PAD_ _ ... 해결 -> 만약 패드라면 무시하세요
    pos = p.index(_EOS_) if _EOS_ in p else len(p)
    p = p[:pos]
    print(title, ':', ' '.join([vocab[i] for i in p]))  # 안녕 _EOS_ _PAD_ _PAD_ _PAD_ _PAD_... 해결 -> 위의 코드들결


if __name__ == '__main__':
    enc_x, dec_x, dec_y, vocab = make_dataset()
    # train_and_save_model(enc_x, dec_x, dec_y, vocab)
    load_model(enc_x, dec_x, dec_y, vocab)











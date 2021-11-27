# Day_19_01_chatbot_util.py
import tensorflow.keras as keras
import numpy as np

_PAD_, _SOS_, _EOS_, _UNK_ = 0, 1, 2, 3


def train_and_save_model(enc_x, dec_x, dec_y, vocab):
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

    model.save('model/chat.h5')


def read_vocab():
    f = open('chat_data/vocab.txt', 'r', encoding='utf-8')
    vocab = f.read().split()
    f.close()

    return vocab


def read_vector():
    f = open('chat_data/vectors.txt', 'r', encoding='utf-8')
    vector = [[int(v) for v in row.split(',')] for row in f.read().split()]
    f.close()

    return vector


def add_pad(sequence, max_len):
    if len(sequence) > max_len:
        return sequence[:max_len]

    return sequence + (max_len - len(sequence)) * [_PAD_]


def make_xy(questions, answers, vocab):
    q_len = max([len(q) for q in questions])
    a_len = max([len(a) for a in answers]) + 1

    onehots = np.eye(len(vocab), dtype=np.int32)

    enc_x, dec_x, dec_y = [], [], []
    for q, a in zip(questions, answers):
        enc_in = add_pad(q, q_len)
        dec_in = add_pad([_SOS_]+a, a_len)
        target = add_pad(a+[_EOS_], a_len)

        enc_x.append(onehots[enc_in])
        dec_x.append(onehots[dec_in])
        dec_y.append(target)                # sparse로 처리하기 때문에 변환하지 않는다

    # (52, 9, 164) (52, 10, 164) (52, 10)
    return np.int32(enc_x), np.int32(dec_x), np.int32(dec_y)


def make_dataset():
    vocab = read_vocab()
    vector = read_vector()
    # print(vocab[:5])    # ['_PAD_', '_SOS_', '_EOS_', '_UNK_', '가']
    # print(vector[:3])   # [['105'], ['105'], ['114', '128', '85', '79']]

    # 퀴즈
    # 질문과 대답에서 최대 길이를 구하세요 (먼저 질문과 대답으로 분리)
    questions = [q for q in vector[::2]]
    answers = [a for a in vector[1::2]]

    enc_x, dec_x, dec_y = make_xy(questions, answers, vocab)
    # print(enc_x.shape, dec_x.shape, dec_y.shape)

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


def decode_prediction(p, vocab, title):
    p = list(p)
    p = [i for i in p if i != _PAD_]
    pos = p.index(_EOS_) if _EOS_ in p else len(p)
    p = p[:pos]
    print(title, ':', ' '.join([vocab[i] for i in p]))


if __name__ == '__main__':
    enc_x, dec_x, dec_y, vocab = make_dataset()
    # train_and_save_model(enc_x, dec_x, dec_y, vocab)
    load_model(enc_x, dec_x, dec_y, vocab)

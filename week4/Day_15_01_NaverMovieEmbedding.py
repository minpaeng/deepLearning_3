import tensorflow.keras as keras
import re
import matplotlib.pyplot as plt
import numpy as np


# 원핫 사용하지 않고 임베딩 버전으로 변경
def get_xy(file_path):
    f = open(file_path, "r", encoding='utf-8')

    # skip header
    f.readline()  # 첫번째 줄을 읽어버림

    x, y = [], []
    for line in f:
        # print(line.strip().split('\t'))
        _, doc, label = line.strip().split('\t')  # _ : 사용하지 않겠다는 의미
        x.append(clean_str(doc).split())
        y.append(int(label))
    f.close()
    # small = int(len(x) * 0.1)  # 크기를 줄이기위해 씀. 주석처리하고 x, y 리턴해도 됨
    # return x[:small], np.int32(y[:small])
    return x, np.int32(y)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def save_model():
    x_train, y_train = get_xy('../data/ratings_train.txt')
    x_test, y_test = get_xy('../data/ratings_test.txt')

    vocab_size = 2000
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)

    max_len = 35
    x_train_pad = keras.preprocessing.sequence.pad_sequences(x_train_seq, maxlen=max_len)
    x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=max_len)

    print(x_train_pad.shape, x_test_pad.shape)      # (150000, 35) (50000, 35)

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x_train_pad.shape[1:]))        # [max_len, vocab_size]
    model.add(keras.layers.Embedding(vocab_size, 100))      # 2차원 입력을 3차원으로 변환함
    model.add(keras.layers.LSTM(50))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    model.fit(x_train_pad, y_train, epochs=1, batch_size=32, verbose=1,
              validation_data=(x_test_pad, y_test))
    model.save('../model/embedding.h5')


def load_model():
    model = keras.models.load_model('../model/embedding.h5')

    x_train, y_train = get_xy('../data/ratings_train.txt')
    vocab_size, max_len = 2000, 35
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    # x_test_seq = tokenizer.texts_to_sequences(x_test)
    # x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=max_len)
    # onehots = np.eye(vocab_size, dtype=np.int32)
    # x_test_onehot = onehots[x_test_pad]
    # print(model.evaluate(x_test_onehot, y_test, verbose=0))

    # 퀴즈
    # 자신이 작성한 리뷰에 대해 긍정/부정 결과를 알려주세요
    review = '엄청 재밌었어요 추천합니다'
    review = [clean_str(review).split()]
    print(review)
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=2000)
    x_train, y_train = get_xy('../data/ratings_train.txt')
    tokenizer.fit_on_texts(x_train)

    review_seq = tokenizer.texts_to_sequences(review)
    max_len = 35
    review_pad = keras.preprocessing.sequence.pad_sequences(review_seq, maxlen=max_len)

    y = model.predict(review_pad)
    y = y[0]
    print('예측 결과: ', end='')
    if y > 0.6:
        print('긍정문입니다.', y)
    else:
        print('부정문입니다.', y)


save_model()
# load_model()

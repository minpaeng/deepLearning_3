# Day_16_01_popcorn.py
import pandas as pd
import re
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os


def clean_str(string):
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


def get_train():
    df_train = pd.read_csv('popcorn/labeledTrainData.tsv',
                           delimiter='\t', index_col=0)
    # print(df_train)

    x_train = [clean_str(r).split() for r in df_train['review']]
    y_train = df_train['sentiment']
    # print(y_train.dtype)      # int64
    # print(x_train[:3])

    return x_train, np.int32(y_train)


def get_test():
    df = pd.read_csv('popcorn/testData.tsv', delimiter='\t', index_col=0)
    # print(df)

    x = [clean_str(r).split() for r in df['review']]
    y = df.index.values
    # print(y.dtype)      # object
    # print(x[:3])

    return x, y


def save_model_rnn():
    x_train, y_train = get_train()
    # print(y_train.shape, user_ids.shape)      # (25000,) (25000,)
    # print(type(y_train))                      # <class 'numpy.ndarray'>

    # heights = sorted([len(t) for t in x_train])
    # plt.plot(heights)
    # plt.show()

    vocab_size, max_len = 2000, 350
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_train_pad = keras.preprocessing.sequence.pad_sequences(x_train_seq, maxlen=max_len)

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x_train_pad.shape[1:]))
    model.add(keras.layers.Embedding(vocab_size, 100))
    model.add(keras.layers.LSTM(50))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    checkpoint = keras.callbacks.ModelCheckpoint('model/popcorn_{epoch:02d}_{val_loss:.2f}.h5',
                                                 save_best_only=True)

    model.fit(x_train_pad, y_train, epochs=100, batch_size=64, verbose=2,
              validation_split=0.2, callbacks=[checkpoint])


def save_model_cnn():
    x_train, y_train = get_train()

    vocab_size, max_len = 2000, 350
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_train_pad = keras.preprocessing.sequence.pad_sequences(x_train_seq, maxlen=max_len)

    # ---------------------------------------------- #

    inputs = keras.layers.Input(x_train_pad.shape[1:])
    embeds = keras.layers.Embedding(vocab_size, 100)(inputs)

    output3 = keras.layers.Conv1D(128, 3, activation='relu', name='conv3')(embeds)
    output3 = keras.layers.GlobalAvgPool1D(name='gap3')(output3)

    output4 = keras.layers.Conv1D(128, 4, activation='relu', name='conv4')(embeds)
    output4 = keras.layers.GlobalAvgPool1D(name='gap4')(output4)

    output5 = keras.layers.Conv1D(128, 5, activation='relu', name='conv5')(embeds)
    output5 = keras.layers.GlobalAvgPool1D(name='gap5')(output5)

    concat = keras.layers.concatenate([output3, output4, output5])

    output = keras.layers.Dense(256, activation='relu')(concat)
    output = keras.layers.Dense(1, activation='sigmoid')(output)

    model = keras.Model(inputs, output)

    # ---------------------------------------------- #
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    checkpoint = keras.callbacks.ModelCheckpoint('model/popcorn_cnn_{epoch:02d}_{val_loss:.2f}.h5',
                                                 save_best_only=True)

    model.fit(x_train_pad, y_train, epochs=100, batch_size=64, verbose=2,
              validation_split=0.2, callbacks=[checkpoint])


def make_submission(user_ids, predictions, filename):
    f = open(os.path.join('model', filename), 'w', encoding='utf-8')

    f.write('"id","sentiment"\n')
    for uid, p in zip(user_ids, predictions):
        # print(uid, int(p > 0.5))
        f.write('"{}",{}\n'.format(uid, int(p > 0.5)))

    f.close()


def load_model(model_name, submission_name):
    x_train, y_train = get_train()
    x_test, user_ids = get_test()

    vocab_size, max_len = 2000, 350
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    x_test_seq = tokenizer.texts_to_sequences(x_test)
    x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=max_len)

    model = keras.models.load_model(model_name)
    p = model.predict(x_test_pad)
    # print(p.shape)        # (25000, 1)

    p = p.reshape(-1)       # (25000,)
    make_submission(user_ids, p, submission_name)


def load_ensemble(model_name1, model_name2, submission_name):
    x_train, y_train = get_train()
    x_test, user_ids = get_test()

    vocab_size, max_len = 2000, 350
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    x_test_seq = tokenizer.texts_to_sequences(x_test)
    x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=max_len)

    model1 = keras.models.load_model(model_name1)
    p1 = model1.predict(x_test_pad)
    p1 = p1.reshape(-1)

    model2 = keras.models.load_model(model_name2)
    p2 = model2.predict(x_test_pad)
    p2 = p2.reshape(-1)

    make_submission(user_ids, (p1 + p2) / 2, submission_name)


# save_model_rnn()
# save_model_cnn()

# load_model('model/popcorn_03_0.31.h5', 'popcorn_rnn.csv')
# load_model('model/popcorn_cnn_03_0.29.h5', 'popcorn_cnn.csv')

load_ensemble('model/popcorn_03_0.31.h5', 'model/popcorn_cnn_03_0.29.h5', 'popcorn_ensemble.csv')

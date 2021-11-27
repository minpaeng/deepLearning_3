import tensorflow.keras as keras
import pandas as pd
import re


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


def save_model(v_size, m_len):
    train = pd.read_csv("../data/labeledTrainData.tsv",
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("../data/testData.tsv",
                       header=0, delimiter="\t", quoting=3)

    # print(train.shape)      # (25000, 3)
    # print(train.columns.values)     # ['id' 'sentiment' 'review']
    # print(train["review"][0])

    x_train, y_train = train['review'].values, train['sentiment'].values
    x_test = train['review'].values
    print(x_train[0])
    print(y_train[:5])  # [1 1 0 0 1]
    print(x_train.dtype, y_train.dtype)  # object int64

    vocab_size, max_len = v_size, m_len
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)

    x_train_pad = keras.preprocessing.sequence.pad_sequences(x_train_seq, maxlen=max_len)
    x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=max_len)

    print(x_train_pad.shape, x_test_pad.shape)

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x_train_pad.shape[1:]))  # [max_len, vocab_size]
    model.add(keras.layers.Embedding(vocab_size, 100))  # 2차원 입력을 3차원으로 변환함
    model.add(keras.layers.LSTM(50))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    checkpoint = keras.callbacks.ModelCheckpoint('../model/popcorn_{epoch:02d}-{val_loss:.2f}.h5',
                                                 save_best_only=True)
    model.fit(x_train_pad, y_train, epochs=20, batch_size=32, verbose=1,
              validation_split=0.2, callbacks=[checkpoint])
    # print(model.evaluate(x_train, y_train, verbose=0))


save_model(10000, 1000)


def load_model(model_path, v_size, m_len, out_path):
    model = keras.models.load_model(model_path)

    train = pd.read_csv("../data/labeledTrainData.tsv",
                        header=0, delimiter="\t", quoting=3)
    x_train = train['review'].values
    vocab_size, max_len = v_size, m_len
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    test = pd.read_csv("../data/testData.tsv",
                       header=0, delimiter="\t", quoting=3)

    x_test = test['review'].values
    x_test_seq = tokenizer.texts_to_sequences(x_test)

    x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=max_len)

    y = model.predict(x_test_pad)
    print(y.shape)  # (25000, 1)

    print(y)
    print(test["id"].values)
    test_sentiment = []
    for res in y:
        if res >= 0.5:
            test_sentiment.append("1")
        else:
            test_sentiment.append("0")

    f = open(out_path, "w")
    f.write("\"id\", \"sentiment\"" + '\n')
    test_id = test['id'].values

    for i in range(len(test_id)):
        f.write(test_id[i] + ',' + test_sentiment[i] + '\n')

    f.close()


# load_model()

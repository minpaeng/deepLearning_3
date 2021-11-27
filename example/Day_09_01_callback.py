# Day_09_01_callback.py
import random
import tensorflow.keras as keras
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt


def make_number(digits):
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


questions, expected = make_data(size=5000, digits=3, reverse=True)

vocab = '#+0123456789'

chr2idx = {c: i for i, c in enumerate(vocab)}
idx2chr = {i: c for i, c in enumerate(vocab)}

x = make_onehot(questions, chr2idx)     # (50000, 7, 12)
y = make_onehot(expected, chr2idx)      # (50000, 4, 12)

data = model_selection.train_test_split(x, y, train_size=0.8)
x_train, x_test, y_train, y_test = data

model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))
model.add(keras.layers.LSTM(128, return_sequences=False))
model.add(keras.layers.RepeatVector(y.shape[1]))
model.add(keras.layers.LSTM(128, return_sequences=True))
model.add(keras.layers.Dense(y.shape[-1], activation='softmax'))
# model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics='acc')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
plateau = keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1)
checkpoint = keras.callbacks.ModelCheckpoint(
    'model/addition_{epoch:02d}_{val_loss:.2f}.h5',
    save_best_only=True)

history = model.fit(x_train, y_train, epochs=20, verbose=2, validation_data=(x_test, y_test),
                    callbacks=[checkpoint])

# 모델 읽어오기
# model = keras.models.load_model('model/addition_17_0.90.h5')
# print(model.evaluate(x_test, y_test, verbose=0))

# print(history.history)
# print(history.history.keys())     # dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

# 퀴즈
# loss와 acc 그래프를 하나의 피겨에 두 개의 플랏으로 그려주세요
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], 'r', label='train')
# plt.plot(history.history['val_loss'], 'g', label='valid')
# plt.legend()
# plt.title('loss')
#
# plt.subplot(1, 2, 2)
# plt.plot(history.history['acc'], 'r', label='train')
# plt.plot(history.history['val_acc'], 'g', label='valid')
# plt.legend()
# plt.title('accuracy')
# plt.ylim(0, 1)
#
# plt.show()







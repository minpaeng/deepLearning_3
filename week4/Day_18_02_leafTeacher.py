# Day_18_02_leaf.py
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np
import tensorflow.keras as keras
import os
import re


# 퀴즈 1
# 캐글에서 나뭇잎 경진대회의 모델을 만들어서 결과를 제출하세요
def get_train():
    leaf = pd.read_csv('leaf-classification/train.csv', index_col=0)
    # print(leaf)  # [990 rows x 193 columns]

    x = leaf.values[:, 1:]

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(leaf.species)

    return np.float32(x), y, enc.classes_


def get_test():
    leaf = pd.read_csv('leaf-classification/test.csv', index_col=0)
    # print(leaf)

    return np.float32(leaf.values), np.int32(leaf.index.values)


def make_submission(user_ids, predictions, filename):
    f = open(os.path.join('model', filename), 'w', encoding='utf-8')
    f.write('id')

    for c in classes:
        f.write(',{}'.format(c))
    f.write('\n')

    for uid, p in zip(user_ids, predictions):
        f.write('{}'.format(uid))
        for num in p:
            f.write(',{}'.format(num))
        f.write('\n')
    f.close()


x_train, y_train, classes = get_train()
# print(x_train.shape, y_train.shape)  # (990, 192) (990,)
# print(y_train[:5])  # [ 3 49 65 94 84]
print(classes)

x_test, leaf_ids = get_test()
# print(x_test.shape, leaf_ids.shape)  # (594, 192) (594,)
# print(leaf_ids[:5])  # [ 4  7  9 12 13]


model = keras.Sequential()
model.add(keras.layers.Dense(len(classes), activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=100, verbose=2, validation_split=0.2)

# 퀴즈
# 정확도를 직접 계산하세요
p = model.predict(x_test)
print(p.shape)

make_submission(leaf_ids, p, "leaf_sample")




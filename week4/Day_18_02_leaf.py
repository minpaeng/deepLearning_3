# Day_18_02_leaf.py
""" 앞으로 해야할 것
1. 수업 시간에 풀었던 퀴즈 다시 풀기 (풀고, 풀고, 또 풀고~~) **제일 중요
2. 모두를 위한 딥러닝 시즌1
3. UCI 머신러닝 popular data set 풀어보기
4. 텐서플로우 자격증 취득

- 이력서에 어떤거 풀었는지 kaggle의 어떤 문제 풀었는지 말하기
- 취업사이트 먼저 찾아보기 ( 일반 딥러닝을 할 수 있는 곳 )
- 최소 6개월-1년 다니고 이직
- 급여보다 딥러닝을 할 수 있는 곳인지 찾아보기
"""
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np
import tensorflow.keras as keras
import os
import re


# 퀴즈 1
# 캐글에서 나뭇잎 경진대회의 모델을 만들어서 결과를 제출하세요
def get_train():
    train = pd.read_csv('leaf-classification/train.csv', index_col=0)
    # print(train)  # [990 rows x 193 columns]

    x_train = train.values[:, 1:]
    y_train = train.values[:, 0]
    # print(x_train.dtype, y_train.dtype)   # object object

    x_train = np.float32(x_train)
    # print(x_train.dtype, y_train.dtype)   # float32 object

    return x_train, y_train


def get_test():
    test = pd.read_csv('leaf-classification/test.csv')
    # print(test)  # [594 rows x 193 columns]

    x_test = test.values[:, 1:]
    id_test = test.values[:, :1]
    # print(x_test.dtype, id_test.dtype)   # float64 float64

    return x_test, id_test


def save_model():
    x_train, y_train = get_train()
    # x_test, id_test = get_test()
    # print(x_train.shape, y_train.shape)  # (990, 192) (990,)
    # print(x_test.shape, id_test.shape)  # (594, 192) (594, 1)

    enc = preprocessing.LabelEncoder()  # y의 문자를 숫자로 바꿈
    y_train = enc.fit_transform(y_train)
    print(y_train)

    # print(x_train.dtype, y_train.dtype)  # float32 int32

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x_train.shape[1:]))
    model.add(keras.layers.Embedding(100, 100))
    model.add(keras.layers.LSTM(50))
    model.add(keras.layers.Dense(len(enc.classes_), activation='softmax'))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=100, verbose=2)
    exit()
    checkpoint = keras.callbacks.ModelCheckpoint('model/popcornTeacher_{epoch:02d}_{val_loss:.2f}.h5',
                                                 save_best_only=True)  # save_best_only 성능이 좋아질때마다 저장

    model.fit(x_train_pad, y_train, epochs=100, batch_size=64, verbose=2,
              validation_split=0.2, callbacks=[checkpoint])


save_model()





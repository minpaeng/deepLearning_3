# Day_04_01_abalone.py
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

# 퀴즈
# abalone 데이터를 읽어서 80%로 학습하고 20%에 대해 결과를 예측하세요
# 1단계 : 파일 읽기
abalone = pd.read_csv('data/abalone.data', header=None)
# print(abalone)

# 2단계 : x, y 데이터 분리
x = abalone.values[:, 1:]
y = abalone.values[:, 0]
# print(x.shape, y.shape)       # (4177, 8) (4177,)
# print(y[:5])                  # ['M' 'M' 'F' 'M' 'I']
# print(x.dtype)                # object

x = np.float32(x)
# print(x.dtype)                # float32

enc = preprocessing.LabelEncoder()
y = enc.fit_transform(y)
# print(y[:5])                  # [2 2 0 2 1]

# 3단계
x = preprocessing.scale(x)
# x = preprocessing.minmax_scale(x)

data = model_selection.train_test_split(x, y, train_size=0.8)
x_train, x_test, y_train, y_test = data

# 4단계
model = keras.Sequential()
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(12, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=100, verbose=2,
          validation_data=(x_test, y_test))
# print(model.evaluate(x_test, y_test, verbose=0))

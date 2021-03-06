import tensorflow.keras as keras
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection


# 퀴즈
# abalone 데이터를 읽어서 80%로 학습하고 20%에 대해 결과를 예측하세요
# https://archive.ics.uci.edu/ml/index.php

# 1단계: 파일 읽기
abalone = pd.read_csv('../data/abalone.data', header=None)
# print(data)

# 2단계: x, y 데이터 분리
x = abalone.values[:, 1:]
y = abalone.values[:, 0]
# print(x.shape, y.shape)     # (4177, 8) (4177,)
# print(y[:5])                # ['M' 'M' 'F' 'M' 'I']
# print(x.dtype, y.dtype)     # object object

x = np.float32(x)
enc = preprocessing.LabelEncoder()    # 문자열을 0, 1의 숫자형태로 바꿈: 1차원 데이터만 전달 가능
y = enc.fit_transform(y)
# print(y[:5])      # [2 2 0 2 1]
# print(x.dtype, y.dtype)     # float32 int32

# 3단계: train, test 로 분리
x = preprocessing.scale(x)      # 단위가 모두 다르면 학습률이 떨어지므로 스케일링을 해줌
data = model_selection.train_test_split(x, y, train_size=0.8)
x_train, x_test, y_train, y_test = data
print(x_train.shape, x_test.shape)      # (3341, 8) (836, 8)
print(y_train.shape, y_test.shape)      # (3341,) (836,)

# 4단계: 모델 구축
model = keras.Sequential()
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

# sparse_categorical_crossentropy: 숫자를 원핫 벡터로 바꿔줌
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=100, verbose=2,
          validation_data=(x_test, y_test))
print(model.evaluate(x_test, y_test, verbose=0))

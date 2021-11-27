# Day_09_02_cars.py
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection


# 퀴즈
# car.data 파일을 읽어서
# 60%로 학습하고 20%로 검증하고
# 최종적으로 나머지 20%에 대해 결과를 예측하는 모델을 구축하세요
def model_car_sparse():
    car = pd.read_csv('data/car.data', header=None,
                      names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

    enc = preprocessing.LabelEncoder()
    buying = enc.fit_transform(car['buying'])
    maint = enc.fit_transform(car['maint'])
    doors = enc.fit_transform(car['doors'])
    persons = enc.fit_transform(car['persons'])
    lug_boot = enc.fit_transform(car['lug_boot'])
    safety = enc.fit_transform(car['safety'])

    x = np.transpose([buying, maint, doors, persons, lug_boot, safety])
    y = enc.fit_transform(car['class'])
    # print(x.shape, y.shape)               # (1728, 6) (1728,)

    data = model_selection.train_test_split(x, y, train_size=0.8)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.Dense(4, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=100, verbose=2, validation_split=0.75)
    print(model.evaluate(x_test, y_test, verbose=0))


def model_car_dense():
    car = pd.read_csv('data/car.data', header=None,
                      names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

    enc = preprocessing.LabelBinarizer()
    buying = enc.fit_transform(car['buying'])
    maint = enc.fit_transform(car['maint'])
    doors = enc.fit_transform(car['doors'])
    persons = enc.fit_transform(car['persons'])
    lug_boot = enc.fit_transform(car['lug_boot'])
    safety = enc.fit_transform(car['safety'])

    x = np.concatenate([buying, maint, doors, persons, lug_boot, safety], axis=1)

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(car['class'])
    # print(x.shape, y.shape)               # (1728, 21) (1728,)

    data = model_selection.train_test_split(x, y, train_size=0.8)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.Dense(4, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=100, verbose=2, validation_split=0.75)
    print(model.evaluate(x_test, y_test, verbose=0))


# model_car_sparse()
model_car_dense()

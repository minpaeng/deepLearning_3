# Day_12_01_VGG16.py
import tensorflow.keras as keras


# 퀴즈
# VGG16 모델을 구축하세요
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=[224, 224, 3]))

model.add(keras.layers.Conv2D(64, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.Conv2D(64, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

model.add(keras.layers.Conv2D(128, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.Conv2D(128, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

model.add(keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dense(1000, activation='softmax'))
model.summary()

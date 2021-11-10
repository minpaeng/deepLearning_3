import tensorflow.keras as keras


# 수용영역(Receptive Field): feature 추출 영역(3x3, 5,5)
# FC layer가 파라미터의 대부분을 갖고있음

# 퀴즈
# VGG16 모델을 구축하세요
# 파라미터 약 1억 3천 8백만개
def make_VGG():

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
    model.add(keras.layers.MaxPool2D([3, 3], [2, 2], 'same'))
    model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([3, 3], [2, 2], 'same'))
    model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([3, 3], [2, 2], 'same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(1000, activation='softmax'))
    model.summary()


make_VGG()




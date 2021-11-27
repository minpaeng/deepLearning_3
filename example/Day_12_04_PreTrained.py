# Day_12_04_PreTrained.py
import tensorflow.keras as keras

# flowers5 -- train -- buttecup
#                   +- coltsfoot
#                   +- daffodil
#          +- test  -- buttecup
#                   +- coltsfoot
#                   +- daffodil

gen_train = keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                         zoom_range=0.5)
flow_train = gen_train.flow_from_directory('flowers3/train',
                                           target_size=[224, 224],
                                           class_mode='sparse')   # "categorical", "binary", "sparse"
# x, y = flow_train.next()
# print(x.shape, y.shape)   # (32, 224, 224, 3) (32,)
# print(y)                  # [2. 1. 0. 2. 2. 0. ...]

gen_test = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
flow_test = gen_test.flow_from_directory('flowers3/test',
                                         target_size=[224, 224],
                                         class_mode='sparse')

conv_base = keras.applications.VGG16(include_top=False,
                                     input_shape=[224, 224, 3])
conv_base.trainable = False

model = keras.Sequential()
# model.add(keras.layers.InputLayer(input_shape=[224, 224, 3]))
model.add(conv_base)

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit(flow_train, epochs=10, batch_size=16, verbose=2,
          validation_data=flow_test)


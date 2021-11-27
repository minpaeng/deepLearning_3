import tensorflow.keras as keras

# flowers5 -- train -- buttecup
#                   +- coltsfoot
#                   +- daffodil
#          -- train -- buttecup
#                   +- coltsfoot
#                   +- daffodil

# 로컬폴더에서  train, test 폴더를 읽어서 데이터셋을 만듦
gen_train = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)  # zoom_range 등의 다양한 파라미터 이용하기
flow_train = gen_train.flow_from_directory('../flowers3/train',
                                           target_size=(224, 224),
                                           class_mode='sparse')       # "categorical", "binary", "sparse"
gen_test = keras.preprocessing.image.ImageDataGenerator()
flow_test = gen_test.flow_from_directory('../flowers3/test',
                                           target_size=(224, 224),
                                           class_mode='sparse')       # "categorical", "binary", "sparse"


conv_base = keras.applications.VGG16(include_top=False, input_shape=[224, 224, 3])
conv_base.trainable = False     # True로 두는 것을 파인튜닝이라고 함. 자신없으면 false로 두어 내부를 건드리지 않아야 함

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

# https://word2vec.kr/search/?query=%ED%95%9C%EA%B5%AD-%EC%84%9C%EC%9A%B8%2B%EB%8F%84%EC%BF%84 word2vec 관련 자료

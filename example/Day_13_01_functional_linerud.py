# Day_13_01_functional_linerud.py
from sklearn import datasets, preprocessing
import tensorflow.keras as keras


# 퀴즈
# linnerud 데이터셋에 대해 동작하는 딥러닝 모델을 구축하세요
# linnerud = datasets.load_linnerud()
# print(linnerud.keys())
# print(linnerud['feature_names'])      # ['Chins', 'Situps', 'Jumps']
# print(linnerud['target_names'])       # ['Weight', 'Waist', 'Pulse']

x, y = datasets.load_linnerud(return_X_y=True)
# print(x.shape, y.shape)               # (20, 3) (20, 3)
# print(y[:3])              # [[191.  36.  50.] [189.  37.  52.] [193.  38.  58.]]

x = preprocessing.scale(x)
y = preprocessing.scale(y)
# x = preprocessing.minmax_scale(x)

y1 = y[:, :1]
y2 = y[:, 1:2]
y3 = y[:, 2:]

inputs = keras.layers.Input(shape=[3])
output = keras.layers.Dense(6, activation='relu')(inputs)

output1 = keras.layers.Dense(6, activation='relu')(output)
output1 = keras.layers.Dense(1, name='weights')(output1)

output2 = keras.layers.Dense(6, activation='relu')(output)
output2 = keras.layers.Dense(1, name='waist')(output2)

output3 = keras.layers.Dense(6, activation='relu')(output)
output3 = keras.layers.Dense(1, name='pulse')(output3)

model = keras.Model(inputs, [output1, output2, output3])
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.mse)

# model.fit(x, y, epochs=100, verbose=2)
model.fit(x, [y1, y2, y3], epochs=100, verbose=2)

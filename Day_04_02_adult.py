import tensorflow.keras as keras
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection


# 퀴즈
# adult.data 파일로 학습하고 adult.test 파일에 대해 결과를 예측하세요
def get_data_encoder(file_path):
    # 1단계: 파일 읽기
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
             'marital-status', 'occupation', 'relationship', 'race',
             'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
             'native-country', 'income']
    adult = pd.read_csv(file_path,
                        names=names,
                        sep=', ',
                        engine='python')  # sep가 2글자 이상이기 때문에 기본적으로 사용하는 c 엔진에서 파이썬 엔진으로 바꿔줘야 함
    # print(adult)

    # print(adult.values[:5, 0])        # [39 50 38 53 28]
    # print(adult.values[:5, 1])        # ['State-gov' 'Self-emp-not-inc' 'Private' 'Private' 'Private']

    # print(adult['age'].values)      # .values를 붙이면 인덱스 없이 데이터만 넘파이로 가져올 수 있음

    # adult.info()    # 컬럼 명, 타입 등을 확인 가능

    # 2단계
    # 6행 32561열 : 전치(transpose)되어있음
    x = [adult['age'].values, adult['fnlwgt'].values,
         adult['education-num'].values, adult['capital-gain'].values,
         adult['capital-loss'].values, adult['hours-per-week'].values]
    x = np.int32(x)
    # print(x[:, :5])

    x = np.transpose(x)
    # print(x.shape, x.dtype)     # (32561, 6) int32
    # print(x[:5, :])

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(adult['income'].values)
    # print(y[:10])

    # 퀴즈
    # 문자열 데이터를 x 데이터에 추가하세요
    workclass = enc.fit_transform(adult['workclass'].values)
    education = enc.fit_transform(adult['education'].values)
    marital = enc.fit_transform(adult['marital-status'].values)
    occupation = enc.fit_transform(adult['occupation'].values)
    relationship = enc.fit_transform(adult['relationship'].values)
    race = enc.fit_transform(adult['race'].values)
    sex = enc.fit_transform(adult['sex'].values)
    native = enc.fit_transform(adult['native-country'].values)

    added = [workclass, education, marital, occupation, relationship, race, sex, native]
    # added = np.array(added)
    # print(added.shape)
    added = np.transpose(added)
    print(added.shape, added.dtype)     # (16281, 8) int32

    x = np.concatenate([x, added], axis=1)

    x = preprocessing.minmax_scale(x)  # 단위가 다르면 스케일링 꼭 고려! minmax_scale, scale 중 잘나오는거 사용

    return x, y


# 똑같은 데이터를 원핫 벡터로 바꾸면 좀 더 성능이 좋아짐
def get_data_binarizer(file_path):
    # 1단계: 파일 읽기
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
             'marital-status', 'occupation', 'relationship', 'race',
             'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
             'native-country', 'income']
    adult = pd.read_csv(file_path,
                        names=names,
                        sep=', ',
                        engine='python')  # sep가 2글자 이상이기 때문에 기본적으로 사용하는 c 엔진에서 파이썬 엔진으로 바꿔줘야 함
    # print(adult)

    # print(adult.values[:5, 0])        # [39 50 38 53 28]
    # print(adult.values[:5, 1])        # ['State-gov' 'Self-emp-not-inc' 'Private' 'Private' 'Private']

    # print(adult['age'].values)      # .values를 붙이면 인덱스 없이 데이터만 넘파이로 가져올 수 있음

    # adult.info()    # 컬럼 명, 타입 등을 확인 가능

    # 2단계
    # 6행 32561열 : 전치(transpose)되어있음
    x = [adult['age'].values, adult['fnlwgt'].values,
         adult['education-num'].values, adult['capital-gain'].values,
         adult['capital-loss'].values, adult['hours-per-week'].values]
    x = np.int32(x)
    # print(x[:, :5])

    x = np.transpose(x)
    # print(x.shape, x.dtype)     # (32561, 6) int32
    # print(x[:5, :])

    enc = preprocessing.LabelBinarizer()
    y = enc.fit_transform(adult['income'].values)
    # print(y[:10])

    # 퀴즈
    # 문자열 데이터를 x 데이터에 추가하세요
    bin = preprocessing.LabelBinarizer()
    workclass = bin.fit_transform(adult['workclass'].values)
    education = bin.fit_transform(adult['education'].values)
    marital = bin.fit_transform(adult['marital-status'].values)
    occupation = bin.fit_transform(adult['occupation'].values)
    relationship = bin.fit_transform(adult['relationship'].values)
    race = bin.fit_transform(adult['race'].values)
    sex = bin.fit_transform(adult['sex'].values)
    # train과 test 클래스 갯수가 달라서  skip
    # print(native.shape, sex.shape)      # (32561, 42) (32561, 1)

    x = np.concatenate([x, workclass, education, marital, occupation, relationship, race, sex], axis=1)
    x = preprocessing.minmax_scale(x)  # 단위가 다르면 스케일링 꼭 고려! minmax_scale, scale 중 잘나오는거 사용

    return x, y


# x_train, y_train = get_data_encoder('data/adult.data')
# x_test, y_test = get_data_encoder('data/adult.test')
# print(x_train.shape, y_train.shape)   # (32561, 14) (32561,)
# print(x_test.shape, y_test.shape)     # (16281, 14) (16281,)

x_train, y_train = get_data_binarizer('data/adult.data')
x_test, y_test = get_data_binarizer('data/adult.test')
# print(x_train.shape, y_train.shape)     # (32561, 65) (32561, 1)
# print(x_test.shape, y_test.shape)       # (16281, 65) (16281, 1)

# 퀴즈
# 앞에서 읽어온 데이터에 대해 모델을 구축하세요
# 4단계
model = keras.Sequential()
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(12, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.binary_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=100, verbose=2,
          batch_size=100,
          validation_data=(x_test, y_test))
# print(model.evaluate(x_test, y_test, verbose=0))
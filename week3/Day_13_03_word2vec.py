# Day_13_03_word2vec.py
import numpy as np
import tensorflow.keras as keras


# 관련 사이트
# http://www.rex-ai.info/docs/AI_study_NLP3

# 텍스트 생성 -> 토큰으로 분할(2차원) -> 불용어 제거 -> 단어장 생성 -> 벡터로 변환
# -> skip-gram/cbow 데이터 생성 -> 딥러닝 데이터(원핫)로 변환 -> 모델 구축
def make_vocab_and_vector():
    corpus = ['king is a strong man',
              'queen is a wise woman',
              'boy is a young man',
              'girl is a young woman',
              'prince is a young king',
              'princess is a young queen',
              'man is strong',
              'woman is pretty',
              'prince is a boy will be king',
              'princess is a girl will be queen']

    # 퀴즈
    # 문자열 코퍼스를 토큰으로 구분되어 있는 2차원 코퍼스로 변환하세요
    # 'king is a strong man' -> ['king', 'is', 'a', 'strong', 'man']
    word_tokens = [line.split() for line in corpus]
    # print(word_tokens)    # [['king', 'is', 'a', 'strong', 'man'], ...]

    # 퀴즈
    # 코퍼스로부터 불용어를 제거하세요
    stop_words = ['is', 'a', 'will', 'be']
    word_tokens = [[w for w in tokens if w not in stop_words] for tokens in word_tokens]
    # word_tokens = [w for tokens in word_tokens for w in tokens if w not in stop_words]
    # print(word_tokens)    # [['king', 'strong', 'man'], ...]

    # 퀴즈
    # 단어장을 만드세요
    vocab = {w for tokens in word_tokens for w in tokens}
    vocab = sorted(vocab)
    # print(vocab)
    # ['boy', 'girl', 'king', 'man', 'pretty', 'prince',
    # 'princess', 'queen', 'strong', 'wise', 'woman', 'young']

    # 퀴즈
    # word_tokens를 vocab을 사용해서 숫자로 변환하세요
    # print(word_tokens[0])     # ['king', 'strong', 'man'] -> [2, 8, 3]
    word_vectors = [[vocab.index(w) for w in tokens] for tokens in word_tokens]
    # print(word_vectors)       # [[2, 8, 3], [7, 9, 10], ...]

    return word_vectors, vocab


def extract(tokens, center, window_size):
    first = max(center - window_size, 0)
    last = min(center + window_size + 1, len(tokens))

    return [tokens[i] for i in range(first, last) if i != center]


def make_xy(word_vectors, vocab):
    skip_gram = False
    xx, yy = [], []
    for tokens in word_vectors:
        # print(tokens)
        for center in range(len(tokens)):
            # print(tokens[center], extract(tokens, center, 1))
            surrounds = extract(tokens, center, 1)
            if skip_gram:
                for target in surrounds:
                    xx.append(tokens[center])
                    yy.append(target)
            else:
                xx.append(surrounds)
                yy.append(tokens[center])

    # print(xx)
    # print(yy)

    # 퀴즈
    # x를 만드세요
    # xx를 활용해서 x를 원핫 벡터로 변환하세요
    x = np.zeros([len(xx), len(vocab)])
    # for i in range(len(xx)):
    #     x[i, xx[i]] = 1

    for i, p in enumerate(xx):
        if skip_gram:
            x[i, p] = 1
        else:
            # print(p)
            onehots = [[int(t == k) for k in range(len(vocab))] for t in p]
            x[i] = np.mean(onehots, axis=0)

    # print(x[:5])

    return x, np.int32(yy)


word_vectors, vocab = make_vocab_and_vector()
x, y = make_xy(word_vectors, vocab)
# print(x.shape, y.shape)           # (28, 12) (28,)

# 퀴즈
# 앞에서 만든 데이터로 모델을 구축하세요
model = keras.Sequential()
model.add(keras.layers.Dense(len(vocab), activation='softmax'))

model.compile(optimizer=keras.optimizers.SGD(0.01),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit(x, y, epochs=10, verbose=2)



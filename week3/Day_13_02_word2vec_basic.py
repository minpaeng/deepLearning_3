# Day_13_02_word2vec_basic.py


# 퀴즈
# 주변 단어 인덱스만 추출하는 함수를 만드세요
def extract(token_count, center, window_size):
    first = max(center - window_size, 0)
    last = min(center + window_size + 1, token_count)

    return [i for i in range(first, last) if i != center]


def show_word2vec(tokens, skipgram):
    for center in range(len(tokens)):
        # print(center, extract(len(tokens), center, 2))
        surrounds = extract(len(tokens), center, 2)

        # 퀴즈
        # skip-gram 방식으로 결과를 출력하세요
        # for t in surrounds:
        #     print(center, t)
        # print([(center, t) for t in surrounds])
        if skipgram:
            print(*[(center, t) for t in surrounds])  # unpacking
            # print(*[(tokens[center], tokens[t]) for t in surrounds])
        else:
            print(extract(len(tokens), center, 2), center)
            # 퀴즈
            # cbow 방식도 skip-gram 처럼 토큰을 직접 출력하세요
            # print([tokens[t] for t in extract(len(tokens), center, 2)], tokens[center])
    print('-' * 30)


tokens = 'The quick brown fox jumps over the lazy dog'.split()
print(tokens)

show_word2vec(tokens, skipgram=True)
show_word2vec(tokens, skipgram=False)








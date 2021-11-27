# 퀴즈
# 토큰 사전과 토큰 인덱스(chr2idx, 딕셔너리)을 반환하는 함수를 만드세요
def make_vocab(data):
    idx, dic = [], []
    for e, k in data:
        print(e, k)
        idx.append(e)
        dic.append(k)
    return idx, dic


data = [('food', '음식'), ('pink', '분홍'),
        ('wind', '바람'), ('desk', '책상'),
        ('head', '머리'), ('hero', '영웅')]

make_vocab(data)


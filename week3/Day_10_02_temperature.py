import numpy as np


# 0~1사이의 값
# 값이 0 일 때 0.5 반환(이진분류)
def sigmoid(z):
    return 1 / (1 + np.e ** -z)


def softmax_1(z):
    s = np.sum(z)
    return z / s


def softmax_2(z):
    z = np.exp(z)                   # np.e ** z
    s = np.sum(z)
    return z / s


def temperature(z, t):
    z = np.log(z) / t
    z = np.exp(z)                   # np.e ** z
    s = np.sum(z)
    return z / s


def weighted_pick(p):
    t = np.cumsum(p)                # 누적합계를 구함
    # print(t)                      # [0.3 0.5 0.9 1. ]

    # 정렬된 배열에서 0.6이 어디에 들어가야 정렬된 상태를 유지하는지 인덱스를 반환
    print(np.searchsorted(t, 0.6))

    n = np.random.rand(1)[0]        # 0~1 사이의 균등분포
    print(np.searchsorted(t, n), n)


# a = [2.0, 1.0, 0.1]
# a = np.float32(a)
#
# print(np.e)                       # 2.718281828459045
#
# print(sigmoid(a))                 # [0.880797  0.7310586 0.5249792]
# print(softmax_1(a))               # [0.64516133 0.32258067 0.03225807]
# print(softmax_2(a))               # [0.6590011  0.24243295 0.09856589]
#
# print(temperature(a, 0.1))        # [9.9902439e-01 9.7560976e-04 9.7561038e-14]
# print(temperature(a, 0.5))        # [0.79840314 0.19960079 0.00199601]
# print(temperature(a, 0.8))        # [0.69247675 0.29115063 0.0163726 ]

weighted_pick([0.3, 0.2, 0.4, 0.1])







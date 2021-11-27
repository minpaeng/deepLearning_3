# Day_05_06_char_rnn_6.py
import nltk
import numpy as np
import Day_05_05_char_rnn_5
import itertools


def show_ngrams():
    tokens = 'i work at intel'.split()
    print(list(nltk.bigrams(tokens)))
    print(list(nltk.trigrams(tokens)))
    print(list(nltk.ngrams(tokens, 2)))

    w = 'tensor'
    print(list(nltk.ngrams(w, 2)))


long_sentence = ("if you want to build a ship, " 
                 "don't drum up people to collect wood and don't assign them tasks and work, "
                 "but rather teach them to long for the endless immensity of the sea.")

# 퀴즈
# 시퀀스 길이 20개로 문장을 나누세요
# 20개 안에는 x가 19개, y가 19개 들어있습니다 (ngrams 사용)
# don't drum up people to collect wood and don't assign them tasks and work
# don't drum up people
#   x: don't drum up peopl
#   y: on't drum up people
# on't drum up people
# n't drum up people t
seq_length = 20
# x = [long_sentence[i:i+seq_length] for i in range(len(long_sentence) - seq_length)]

words = nltk.ngrams(long_sentence, seq_length)

words = [''.join(w) for w in words]
Day_05_05_char_rnn_5.char_rnn_5(words)

# x와 y로 분할하는 코드
# 1번
# words = list(words)
# x = [w[:-1] for w in words]
# y = [w[1:] for w in words]
# print(np.array(x).shape, np.array(y).shape)   # (152, 19) (152, 19)

# 2번
# x = nltk.ngrams(long_sentence[:-1], seq_length-1)
# y = nltk.ngrams(long_sentence[1:], seq_length-1)
# print(np.array(x).shape, np.array(y).shape)                 # () ()
# print(np.array(list(x)).shape, np.array(list(y)).shape)     # (152, 19) (152, 19)

# Day_14_01_tokenizer.py
import tensorflow.keras as keras


long_sentence = ("if you want to build a ship, " 
                 "don't drum up people to collect wood and don't assign them tasks and work, "
                 "but rather teach them to long for the endless immensity of the sea.")

tokenizer = keras.preprocessing.text.Tokenizer(num_words=10)
tokenizer.fit_on_texts(long_sentence.split())

print(tokenizer.index_word)
# {1: 'to', 2: "don't", 3: 'and', 4: 'them', 5: 'the', 6: 'if', 7: 'you', 8: 'want', 9: 'build',
# 10: 'a', 11: 'ship', 12: 'drum', 13: 'up', 14: 'people', 15: 'collect', 16: 'wood',
# 17: 'assign', 18: 'tasks', 19: 'work', 20: 'but', 21: 'rather', 22: 'teach', 23: 'long',
# 24: 'for', 25: 'endless', 26: 'immensity', 27: 'of', 28: 'sea'}

print(tokenizer.texts_to_sequences(['build', 'a', 'ship']))         # [[9], [], []]
print(tokenizer.texts_to_sequences([['build'], ['a'], ['ship']]))   # [[9], [], []]
print(tokenizer.texts_to_sequences([['build', 'a', 'ship']]))       # [[9]]

# 퀴즈
# 아래 시퀀스를 문자열로 변환하세요
seq = [[5, 1, 13, 2, 3], [4, 5, 6], [9, 10, 11]]
print(tokenizer.sequences_to_texts(seq))
# ["the to don't and", 'them the if', 'build']

print(keras.preprocessing.sequence.pad_sequences(seq), end='\n\n')
print(keras.preprocessing.sequence.pad_sequences(seq, padding='post'), end='\n\n')

print(keras.preprocessing.sequence.pad_sequences(seq, maxlen=4), end='\n\n')
print(keras.preprocessing.sequence.pad_sequences(seq, maxlen=4, truncating='post'), end='\n\n')

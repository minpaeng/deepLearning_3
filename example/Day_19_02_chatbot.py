# Day_19_02_chatbot.py
import Day_19_01_chatbot_util
import tensorflow.keras as keras
import numpy as np


enc_x, dec_x, dec_y, vocab = Day_19_01_chatbot_util.make_dataset()
q_len = len(enc_x[0])
a_len = len(dec_x[0])

onehots = np.eye(len(vocab), dtype=np.int32)

dec_in = Day_19_01_chatbot_util.add_pad([Day_19_01_chatbot_util._SOS_], a_len)
dec_in = onehots[dec_in]
dec_in = np.int32([dec_in])

model = keras.models.load_model('model/chat.h5')

while True:
    q = input('왕자 : ')

    if '끝' in q:
        break

    tokens = q.split()
    seq = [vocab.index(t) if t in vocab else Day_19_01_chatbot_util._UNK_ for t in tokens]

    enc_in = Day_19_01_chatbot_util.add_pad(seq, q_len)
    enc_in = onehots[enc_in]
    enc_in = np.int32([enc_in])

    p = model.predict([enc_in, dec_in])
    p_arg = np.argmax(p, axis=2)

    Day_19_01_chatbot_util.decode_prediction(p_arg[0], vocab, '여우')
    print()


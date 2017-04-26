from encoder import Encoder
from nets import generateData
from loader import trained_trio
import numpy as np

encoder = Encoder()

sess, trio = trained_trio('./modelcheckpoints/modelchkpt.chk')


def neighbors(in_string):
    # returns a list of strings that are nearest-neighbors of in_string

    s = encoder.encode(in_string)
    neighbor_list = []
    for i in range(len(s)):
        l = list(s)
        l[i] = str((int(l[i]) + 1) % 2)
        neighbor_enc = ''.join(l)
        neighbor_string = encoder.decode(neighbor_enc)
        neighbor_list += [neighbor_string]
    return neighbor_list


def l1_distances(sess, trio, s, key):
    # s is a string
    # currently works with four-letter words (heh) because we haven't
    # implemented a padding procedure

    # returns a list of integers, one for each neighbor. the integer is
    # the number of bits where the encoding of the ciphertext of the
    # neighbor differs from the encoding of the ciphertext of s.

    c0 = trio.encryptPlaintext(sess, s, key)
    distances = []
    for neighbor in neighbors(s):
        c = trio.encryptPlaintext(sess, neighbor, key)
        distance = sum(np.abs(c - c0))
        distances += [distance]

    return distances


key = generateData(20,1) #  NB: a new key with each run
D = l1_distances(sess, trio, 'test',key)

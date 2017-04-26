from encoder import Encoder
from nets import generateData
from loader import trained_trio

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


def distances(s):
    # s is a string
    # currently works with four-letter words (heh) because we haven't
    # implemented a padding procedure
    

    # returns a list of integers, one for each neighbor. the integer is
    # the number of bits where the encoding of the ciphertext of the
    # neighbor differs from the encoding of the ciphertext of s.

    k = generateData(len(s)*5, 1)

    c0 = encoder.encode(trio.encryptPlaintext(sess, s, k))

    distances = []
    for neighbor in neighbors(s):
        ciphertext = trio.encryptPlaintext(sess, neighbor, k)
        c = encoder.encode(ciphertext)
        distance = sum([b != b0 for b, b0 in zip(c, c0)])
        # snazzy; found online

        distances += [distance]

        # # to test, uncomment this and distances('test') below
        # # NOTE: I get distance 0 in ciphertext space for some neighbors.
        # # could fix if trio.decryptBob gave floats. is it worth it?
        # print(c)
        # print(c0)
        # print('supposed distance = ',distance)
        # print('neighbor is ',neighbor)
        # print()

    return distances

# distances('test')

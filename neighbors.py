from encoder import Encoder
from nets import generateData
from loader import trained_trio
import numpy as np

encoder = Encoder()

sess, trio = trained_trio('./modelcheckpoints/modelchkpt.chk')


def neighbors(in_bin):
    # returns a list of strings that are nearest-neighbors of in_bin
    # where in_bin is a binary string
    s = in_bin
    neighbor_list = []
    for i in range(len(s)):
        l = list(s)
        l[i] = str((int(l[i]) + 1) % 2)
        neighbor_enc = ''.join(l)
        neighbor_list += [neighbor_enc]
    return neighbor_list


def l1_distances(sess, trio, s, key):
    # compute l1 distances for a single plaintext
    nearest_ns = [np.array(list(n), dtype='float32')
                  for n in neighbors(encoder.encode(s))]
    nearest_ns = np.concatenate([[n] for n in nearest_ns], axis=0)
    keys = np.concatenate([key for i in range(20)], axis=0)
    encrypted_ns = trio.encryptBatch(sess, nearest_ns, keys)

    c0 = trio.encryptPlaintext(sess, s, key)
    distances = []
    for c in encrypted_ns:
        distance = sum(np.abs(c - c0))
        distances += [distance]

    return distances
# tester = l1_distances(sess, trio, 'test', generateData(20,1))


B = [bin(i)[2:].zfill(20) for i in range(2**20-128, 2**20)]
B = [bin(i)[2:].zfill(20) for i in range(2**5)]


def batch_neighbors(B):
    # B is a list of binary strings
    all_ns = np.array([])
    L = []
    for b in B:
        #print(''.join([str(int(x)) for x in b]))
        #print(str(''.join(b)))
        L += [np.array(list(n), dtype='float32') for n in neighbors(''.join([str(int(x)) for x in b]))]
    all_ns = np.concatenate([[l] for l in L], axis=0)
    return all_ns


def batch_min_l1_distances(sess, trio, B, key):
    # B is a list of binary strings
    # key is an array, e.g., generateData(20,1)

    # generate and encrypt neighbors
    nearest_ns = batch_neighbors(B)
    keys = np.concatenate([key for i in range(len(nearest_ns))], axis=0)
    encrypted_ns = trio.encryptBatch(sess, nearest_ns, keys)

    # encrypt the strings in B
    C = [np.array(list(b), dtype='float32') for b in B]
    C = np.concatenate([[c] for c in C], axis=0)
    keys = np.concatenate([key for i in range(len(C))], axis=0)
    encrypted_bs = trio.encryptBatch(sess, C, keys)

    min_distances = []
    for c0 in encrypted_bs:
        distances = []
        for c in encrypted_ns[:20]:
            distance = sum(np.abs(c - c0))
            distances += [distance]
        min_distances += [min(distances)]
        #This originally was bs, but I think it should be ns if we're adjusting the nearest-neighbor vector
        encrypted_ns = encrypted_ns[20:]

    return min_distances

def batch_extreme_l1_distances(sess, trio, B, key, log_loc):
    # B is a list of binary strings
    # key is an array, e.g., generateData(20,1)

    # generate and encrypt neighbors
    nearest_ns = batch_neighbors(B) 
    keys = np.concatenate([key for i in range(len(nearest_ns))], axis=0)
    encrypted_ns = trio.encryptBatch(sess, nearest_ns, keys)
    print("length of encrypted_ns is",len(encrypted_ns))

    # encrypt the strings in B
    C = [np.array(list(b), dtype='float32') for b in B]
    C = np.concatenate([[c] for c in C], axis=0)
    keys = np.concatenate([key for i in range(len(C))], axis=0)
    encrypted_bs = trio.encryptBatch(sess, C, keys)

    min_distances = []
    max_distances = []
    for c0 in encrypted_bs:
        distances = []
        for c in encrypted_ns[:20]:
            distance = sum(np.abs(c - c0))
            distances += [distance]
        min_distances += [min(distances)]
        max_distances += [max(distances)]
        encrypted_ns = encrypted_ns[20:]
        print("length of encrypted_ns is now",len(encrypted_ns))

    with open(log_loc, 'a') as log_file:
        for pair in list(zip(min_distances, max_distances)):
            log_file.write(str(pair) + '\n')



def get_all_distances(key):
    B = [bin(i)[2:].zfill(20) for i in range(2**10)]
    counter = 0
    while B != []:
        batch_extreme_l1_distances(B[:2**10], key,
                                   'kill-these-extrema')
        B = B[2**10:]
        counter += 1
        print('finished batch ',counter)


# the key I used to make the big file is
# [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1]

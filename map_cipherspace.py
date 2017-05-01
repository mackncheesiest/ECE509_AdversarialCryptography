from encoder import Encoder
from nets import generateData
from loader import trained_trio
from neighbors import batch_extreme_l1_distances
import numpy as np


# I just really wanted to use the word "cipherspace"
# Load the trained model
sess, trio = trained_trio('./modelcheckpoints/modelchkpt.chk')
myEnc = Encoder()

# Define a output filenames
# This file holds the ciphertext encodings for all 2^20 plaintext inputs
ciphertexts_filename = './cipherspace_values.txt'
# This file holds the L1 nearest-neighbor distances for all 2^20 plaintext inputs
nn_filename = './nn_values.txt'

# Use a fixed key for now
k = generateData(plaintext_len=20, batch_size=1)
keys = k * np.ones([2**12, 20])

# Each element is like 4 bytes, so...only like 80 MB of memory? Not terrible
"""
myCiphertexts = np.zeros([2**20, 20])


print("calculating all possible ciphertexts with a random key...")
# 256 total batches of 4096 each gives 256 * 4096 = 1048576 = 2^20 combinations
for i in range(2**8):
    #Batch size of 4096
    tempArr = np.zeros([2**12, 20])
    for j in range(2**12):
        tempArr[j, :] = [np.float32(x) for x in format(j + 4096*i, 'b').zfill(20)]
    
    #Encrypt this batch and put it in its array locations
    myCiphertexts[range(i*4096, (i+1)*4096), :] = trio.encryptBatch(sess, tempArr, keys)
    #print("computed ciphertexts for plaintext values: " + str(4096*i) + " to " + str(4096*(i+1)))
    if i % 2**4 == 0:
        print(str(4096*i/(2**20) * 100) + "% done...")

print("writing ciphertexts to file...")    
np.savetxt(ciphertexts_filename, myCiphertexts)
print("finished writing ciphertexts to file...")
"""

# There are 2**20 different plaintexts, each of them has 20 nearest neighbors, 
# of which we are collecting the min and max as a tuple (one element)
myNNs = np.zeros([2**20, 2])

print("calculating nearest neighbors...")
# Next, calculate all the nearest neighbors
# Use a fixed random key
k = generateData(plaintext_len=20, batch_size=1)
for i in range(2**8):
    # Batch size of 4096
    tempArr = np.zeros([2**12, 20])
    for j in range(2**12):
        tempArr[j, :] = [np.float32(x) for x in format(j + 4096*i, 'b').zfill(20)]
    
    # Get the array of extreme nearest-neighbor distances for this batch
    myNNs[(4096*i):(4096*(i+1)), :] = batch_extreme_l1_distances(sess, trio, tempArr, k)
    if i % 2**4 == 0:
        print(str(4096*i/(2**20) * 100) + "% done...")

# And write this big numpy array to a file
print("writing nearest neighbors to file...")
np.savetxt(nn_filename, myNNs)    
print("finished writing nearest neighbors to file...")
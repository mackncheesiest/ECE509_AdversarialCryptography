import tensorflow as tf
from nets import *
import matplotlib.pyplot as plt
import numpy as np

def main():
    in_length = 64  # mesage of length 32, so expect output of length 32
    conv_params = [
        [[4, 1, 2], 1], 
        [[2, 2, 4], 2], 
        [[1, 4, 4], 1], 
        [[1, 4, 1], 1]
    ]
    
    sess = tf.Session()
    
    myTrio = Trio(in_length, conv_params)
    
    [bobAvgErrVect, eveAvgErrVect] = myTrio.train(sess, iterations=50000)
    
    plt.plot(bobAvgErrVect)
    plt.title('Average Bit Error Across 1000 iterations/epoch for Bob')
    plt.show()
    plt.plot(eveAvgErrVect)
    plt.title('Average Bit Error Across 1000 iterations/epoch for Eve')
    plt.show()
    
    
    #http://stackoverflow.com/questions/7396849/convert-binary-to-ascii-and-vice-versa
    mathBin = bin(int.from_bytes('door'.encode(), 'big'))[2:]
    mathBin = mathBin.zfill(in_length//2)
    mathBin = [float(c) for c in mathBin]
    #Pick a key
    key = generateData(plaintext_len=in_length//2)
    key = [float(i) for i in key]
    
    encryptedData = myTrio.encryptPlaintext(sess, mathBin, key)
    decryptedData = myTrio.decryptBob(sess, encryptedData[0], key)
    
    asciiEncryptedData = [int(round(abs(c))) for c in encryptedData[0]]
    asciiEncryptedData = "".join(str(x) for x in asciiEncryptedData)
    #asciiEncryptedData = int(asciiEncryptedData, 2)
    #asciiEncryptedData = asciiEncryptedData.to_bytes((asciiEncryptedData.bit_length() + 7) // 8, 'big').decode()
    
    asciiDecryptedData = [int(round(abs(c))) for c in decryptedData[0]]
    asciiDecryptedData = "".join(str(x) for x in asciiDecryptedData)
    #asciiDecryptedData = int(asciiDecryptedData, 2)
    #asciiDecryptedData = asciiDecryptedData.to_bytes((asciiDecryptedData.bit_length() + 7) // 8, 'big').decode()

    print("".join(str(int(x)) for x in mathBin))
    print(asciiEncryptedData)
    print(asciiDecryptedData)    


if __name__ == '__main__':  # I don't know what this does
    main()

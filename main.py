import tensorflow as tf
from nets import generateData, Trio
import matplotlib.pyplot as plt


def main():
    in_length = 40  # mesage of length 20, so expect output of length 20
    conv_params = [
        [[4, 1, 2], 1], 
        [[2, 2, 4], 2], 
        [[1, 4, 4], 1], 
        [[1, 4, 1], 1]
    ]
    
    sess = tf.Session()
    
    myTrio = Trio(in_length, conv_params)
    
    [bobAvgErrVect, eveAvgErrVect] = myTrio.train(sess, epochs=100000, learning_rate=0.0008, batch_size=4096, report_rate=100)
    
    plt.plot(bobAvgErrVect)
    plt.title('Average Bit Error Across 1000 iterations/epoch for Bob')
    plt.show()
    plt.plot(eveAvgErrVect)
    plt.title('Average Bit Error Across 1000 iterations/epoch for Eve')
    plt.show()
    
    plaintext = 'hello my name is josh.aa'
    
    #Pick a key
    key = generateData(plaintext_len=in_length//2, batch_size=1)
    #key = [float(i) for i in key[0, :]]
    
    print(plaintext)
    encryptedData = myTrio.encryptPlaintext(sess, plaintext, key, output_bits_or_chars='chars')
    print(encryptedData)
    decryptedData = myTrio.decryptBob(sess, encryptedData, key, output_bits_or_chars='chars')
    print(decryptedData)    


if __name__ == '__main__':  # I don't know what this does
    main()

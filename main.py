import tensorflow as tf
from nets import generateData, Trio
import matplotlib.pyplot as plt


def main():
    tf.reset_default_graph()
    in_length = 40  # mesage of length 20, so expect output of length 20
    conv_params = [
        [[4, 1, 2], 1], 
        [[2, 2, 4], 2], 
        [[1, 4, 4], 1], 
        [[1, 4, 1], 1]
    ]
    
    sess = tf.Session()
    
    myTrio = Trio(in_length, conv_params)
    
    [bobAvgErrVect, eveAvgErrVect] = myTrio.train(sess, epochs=30000, learning_rate=0.0008, batch_size=4096, report_rate=1000)

    saver = tf.train.Saver()
    saver.save(sess, "./modelcheckpoints/modelchkpt.chk")

    plt.plot(bobAvgErrVect)
    plt.title('Average Bit Error Across 1000 iterations/epoch for Bob')
    plt.show()
    plt.plot(eveAvgErrVect)
    plt.title('Average Bit Error Across 1000 iterations/epoch for Eve')
    plt.show()
    
    return sess, myTrio, saver


if __name__ == '__main__':  # I don't know what this does
    sess, trio, saver = main()

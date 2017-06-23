import tensorflow as tf
from nets import generateData, Trio
import matplotlib.pyplot as plt


def trained_trio(checkpoint):
    # checkpoint is a string
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
    
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()

    saver.restore(sess,checkpoint)
 
    return sess, myTrio


#if __name__ == '__main__':  # I don't know what this does
sess, trio = trained_trio('./modelcheckpoints/modelchkpt.chk')

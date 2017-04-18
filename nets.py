import tensorflow as tf
import numpy as np

class Net:
    def __init__(self, name, in_length, conv_params):
        self.name = name  # a string
        self.conv_params = conv_params  # format: [filter_shape, stride]
        self.in_length = in_length
        self.fc_weights = tf.get_variable(
            name=self.name + '_fc_weights', shape=[in_length, in_length],
            initializer=tf.contrib.layers.xavier_initializer()
        )  # do we want to include biases? others don't
        self.conv_weights = [tf.get_variable(
            self.name + 'conv_weights'+str(conv_params.index(param)),
            shape=param[0], initializer=tf.contrib.layers.xavier_initializer()
        )
                             for param in self.conv_params]

    def fc_layer(self, in_tensor):
        # input: a tensor.
        # if we generate inputs as lists of binary
        # digits like [1,0,1,1,...], then process
        # as tf.Variable(binary_message, dtype=tf.float32)

        # in_tensor is one of
        #     plaintext + key for alice
        #     ciphertext + key for bob
        #     ciphertext for eve

        in_tensor = tf.expand_dims(in_tensor, 1)
        if self.name == 'eve':
            in_tensor = tf.concat(0, [in_tensor, tf.constant(0., shape=in_tensor.get_shape())])
        return tf.nn.sigmoid(tf.matmul(self.fc_weights, in_tensor))

    def conv_layer(self, in_tensor):
        # input: a tensor of shape [in_length, 1]
        # conv1d needs two 3-dimensional tensors as as input
        out_tensor = tf.expand_dims(in_tensor, 0)

        # for all but the last layers we use relu
        for weights in self.conv_weights[:-1]:
            stride = self.conv_params[self.conv_weights.index(weights)][1]
            # dictionary would be nicer
            out_tensor = tf.nn.relu(
                tf.nn.conv1d(out_tensor, weights, stride, padding='SAME')
            )  # if problems later, maybe I should have used placeholders?

        # for the last layer use a tanh
        weights = self.conv_weights[-1]
        stride = self.conv_params[self.conv_weights.index(weights)][1]
        out_tensor = tf.nn.tanh(
           tf.nn.conv1d(out_tensor, self.conv_weights[-1], stride, padding='SAME')
        )  # if problems later, maybe I should have used placeholders?

        out_tensor = tf.squeeze(out_tensor)

        # now round to 0's and 1's
        # warning: was getting -0.'s instead of 0's. prob ok but...
        
        # Note: Can't use ceil during training, actually, because it doesn't have a defined gradient so SGD breaks
        #out_tensor = tf.ceil(out_tensor)

        return out_tensor
    
    def loss_func(self, name, plaintext, eve_output, bob_output):
        # All networks need to use Eve's loss function as a building block
        
        # This should be the L1-Loss of the original plaintext and what Eve decrypted
        eveLoss_L1 = tf.reduce_sum(tf.abs(plaintext - eve_output))
        #eveLoss_L1 = tf.reduce_mean(tf.abs(plaintext - eve_output))
        
        # Then, perform the modification Abadi & Andersen describe in section 2.5 of (N/2 - Eve_L1)^2 / (N/2)^2
        # This should cause the network to drive Eve towards a 50% bit error rate
        eveLoss = ((plaintext.get_shape().as_list()[0]/2 - eveLoss_L1) ** 2) / ((plaintext.get_shape().as_list()[0]/2) ** 2)
        
        # Alice & Bob use the same loss function
        if self.name == 'alice' or self.name == 'bob':
            # Alice and Bob's loss is the L1-loss of [originalPlaintext, key] and what Bob recovered minus eveLoss
            aliceBobLoss = tf.reduce_sum(tf.abs(bob_output - plaintext)) - eveLoss
            #aliceBobLoss = tf.reduce_mean(tf.abs(bob_output - plaintext)) - eveLoss
            return aliceBobLoss
        else:
            return eveLoss

def generateData(plaintext_len=4):
    return np.random.randint(0, 2, size=int(plaintext_len))

class Trio():
    def __init__(self, in_length, conv_params):
        self.this = None
        self.nets = [Net(name, in_length, conv_params) for name in ['alice', 'bob', 'eve']]
        self.learning_rate = 0.0008

        self.plaintext_len = in_length/2

    def train(self, iterations=100000):
        # define a training function
        
        plaintext = tf.placeholder('float', [self.plaintext_len])
        key = tf.placeholder('float', [self.plaintext_len])
        
        tensor_in = tf.concat(0, [plaintext, key])
        
        alice_output = self.nets[0].conv_layer(self.nets[0].fc_layer(tensor_in))
        bob_output = self.nets[1].conv_layer(self.nets[1].fc_layer(tf.concat(0, [alice_output, key])))
        eve_output = self.nets[2].conv_layer(self.nets[2].fc_layer(alice_output))
        
        loss_bob = self.nets[1].loss_func('bob', plaintext, eve_output, bob_output)
        loss_eve = self.nets[2].loss_func('eve', plaintext, eve_output, bob_output)
        
        # Stolen from https://github.com/ankeshanand/neural-cryptography-tensorflow/blob/master/src/model.py#L79
        t_vars = tf.trainable_variables()
        # When training alice/bob, we only want to update their variables
        alice_bob_vars = [var for var in t_vars if 'alice' in var.name or 'bob' in var.name]        
        # And when training eve, we only want to update her variables
        eve_vars = [var for var in t_vars if 'eve' in var.name]
        
        # Optimizers used for training
        abOpt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='a_b_optimizer').minimize(loss_bob, var_list=alice_bob_vars)
        eOpt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='e_optimizer').minimize(loss_eve, var_list=eve_vars)
        
        # Begin the training loop
        sess = tf.Session()
        bobLossArr = np.zeros([iterations])
        eveLossArr = np.zeros([iterations])
        tf.initialize_all_variables().run(session=sess)
        for i in range(iterations):
            msg_training = generateData(plaintext_len=self.plaintext_len)
            key_training = generateData(plaintext_len=self.plaintext_len)
            # First, run alice and bob           
            _, bobLossArr[i] = sess.run([abOpt, loss_bob], feed_dict={plaintext:msg_training, key:key_training})
            # Then, run eve
            msg_training = generateData(plaintext_len=self.plaintext_len)
            key_training = generateData(plaintext_len=self.plaintext_len)
            _, eveLossArr[i] = sess.run([eOpt, loss_eve], feed_dict={plaintext:msg_training, key:key_training})
            
            if i % 1000 == 0:
                print("bob loss: " + str(bobLossArr[i]))
                print("eve loss: " + str(eveLossArr[i]))
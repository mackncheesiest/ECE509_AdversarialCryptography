import tensorflow as tf
import numpy as np
from encoder import Encoder

class Net:
    def __init__(self, name, in_length, conv_params):
        self.name = name  # a string
        self.conv_params = conv_params  # format: [filter_shape, stride]
        self.in_length = in_length
        
        if name == 'eve':
            self.eve_expand_fc_weights = tf.get_variable(
                name=name + '_expand_fc_weights', shape=[in_length, in_length//2],
                initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=10)
            )
        self.fc_weights = tf.get_variable(
            name=self.name + '_fc_weights', shape=[in_length, in_length],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=10)
        )  # do we want to include biases? others don't
        self.conv_weights = [tf.get_variable(
            self.name + 'conv_weights'+str(conv_params.index(param)),
            shape=param[0], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=10)
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
            expandFC = tf.nn.sigmoid(tf.matmul(self.eve_expand_fc_weights, in_tensor))
            return tf.nn.sigmoid(tf.matmul(self.fc_weights, expandFC))
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
        self.plaintext_len = in_length//2
        self.myEnc = Encoder()

    def train(self, sess, iterations=50000, learning_rate=0.0008):
        # define a training function
        plaintext = tf.placeholder('float', [self.plaintext_len])
        key = tf.placeholder('float', [self.plaintext_len])
        
        tensor_in = tf.concat(0, [plaintext, key])
        
        alice_output = self.nets[0].conv_layer(self.nets[0].fc_layer(tensor_in))
        bob_output = self.nets[1].conv_layer(self.nets[1].fc_layer(tf.concat(0, [alice_output, key])))
        eve_output = self.nets[2].conv_layer(self.nets[2].fc_layer(alice_output))
        
        loss_bob = self.nets[1].loss_func('bob', plaintext, eve_output, bob_output)
        loss_eve = self.nets[2].loss_func('eve', plaintext, eve_output, bob_output)
        
        bitErrors_bob = tf.reduce_sum(tf.abs(tf.round(bob_output) - tf.round(plaintext)))
        bitErrors_eve = tf.reduce_sum(tf.abs(tf.round(eve_output) - tf.round(plaintext)))
        
        # Stolen from https://github.com/ankeshanand/neural-cryptography-tensorflow/blob/master/src/model.py#L79
        t_vars = tf.trainable_variables()
        # When training alice/bob, we only want to update their variables
        alice_bob_vars = [var for var in t_vars if 'alice' in var.name or 'bob' in var.name]        
        # And when training eve, we only want to update her variables
        eve_vars = [var for var in t_vars if 'eve' in var.name]
        
        # Optimizers used for training
        abOpt = tf.train.AdamOptimizer(learning_rate=learning_rate, name='a_b_optimizer').minimize(loss_bob, var_list=alice_bob_vars)
        eOpt = tf.train.AdamOptimizer(learning_rate=learning_rate, name='e_optimizer').minimize(loss_eve, var_list=eve_vars)
        
        # Begin the training loop
        bobLossArr = np.zeros([int(iterations)])
        eveLossArr = np.zeros([int(iterations)])
        bobMeansVect = np.zeros([int(iterations/1000)])
        eveMeansVect = np.zeros([int(iterations/1000)])
        #tf.initialize_all_variables().run(session=sess)
        tf.global_variables_initializer().run(session=sess)
        for i in range(iterations):
            msg_training = generateData(plaintext_len=self.plaintext_len)
            key_training = generateData(plaintext_len=self.plaintext_len)
            # First, run alice and bob           
            _, bobLossArr[i] = sess.run([abOpt, bitErrors_bob], feed_dict={plaintext:msg_training, key:key_training})
            # Then, run eve twice
            msg_training = generateData(plaintext_len=self.plaintext_len)
            key_training = generateData(plaintext_len=self.plaintext_len)
            _, _ = sess.run([eOpt, bitErrors_eve], feed_dict={plaintext:msg_training, key:key_training})
            
            msg_training = generateData(plaintext_len=self.plaintext_len)
            key_training = generateData(plaintext_len=self.plaintext_len)
            _, eveLossArr[i] = sess.run([eOpt, bitErrors_eve], feed_dict={plaintext:msg_training, key:key_training})
            
            if i % 1000 == 0 and i > 0:
                bobMeansVect[int(i/1000)] = np.mean(bobLossArr[(i-1000):i])
                eveMeansVect[int(i/1000)] = np.mean(eveLossArr[(i-1000):i])
                print(str(int(i/iterations * 100)) + "% done (" + str(i) + " out of " + str(iterations) + " iterations")
                print("bob average loss: " + str(np.mean(bobLossArr[(i-1000):i])))
                print("eve average loss: " + str(np.mean(eveLossArr[(i-1000):i])))
                print("bob loss this iteration: " + str(bobLossArr[i]))
                print("eve loss this iteration: " + str(eveLossArr[i]))
                
        return bobMeansVect, eveMeansVect
    
    # Note that plaintext is a legitimate "plaintext" i.e. 'hello'
    # Key is a binary list generated with something like generateData
    # And sess is the tensorflow session used to train the model or something
    def encryptPlaintext(self, sess, plaintext, key, output_bits_or_chars='chars'):
        
        # Check that we'll be able to encrypt without padding or anything (because I don't know how to remove the padding in decoding...what is padding and what isn't?)
        if 5*len(plaintext) % self.plaintext_len != 0:
            print("Cannot encrypt\nPlease either pad your plaintext to be a multiple of network block size\nOr train a network with a different block size")
            return plaintext
        
        # Turn the plaintext string into a list of floating point bits that can be used as a tensor
        # First by turning it into an encoded string
        plaintextBin = self.myEnc.encode(plaintext)
        # Note that we don't need to zfill here because of the check above
        # And then by converting that string into a list of floating point bits
        plaintextBin = [float(c) for c in plaintextBin]
            
        # Now, iterate over all of the blocks and encrypt them
        if output_bits_or_chars == 'chars':
            result = ''
        else:
            result = []
            
        for i in range(0, len(plaintextBin), self.plaintext_len):
            # Concatenate the message with the key
            tensor_in = tf.concat(0, [plaintextBin[i:(i+self.plaintext_len)], key])
            # Get the resulting bit vector
            tf_result = sess.run([self.nets[0].conv_layer(self.nets[0].fc_layer(tensor_in))])[0]
            # Compress it back to a sequence of characters, decode them, and append to the result
            if output_bits_or_chars == 'chars':
                result += self.myEnc.decode("".join(str(int(round(abs(x)))) for x in tf_result))
            else:
                result.append(tf_result)
            
        return result
    
    def decryptBob(self, sess, ciphertext, key, output_bits_or_chars='chars'):
        
        # Check that we'll be able to encrypt without padding or anything (because I don't know how to remove the padding in decoding...what is padding and what isn't?)
        if 5*len(ciphertext) % self.plaintext_len != 0:
            print("Cannot decrypt\nPlease either pad your plaintext to be a multiple of network block size\nOr train a network with a different block size")
            return ciphertext
        
        # Turn the ciphertext string into a list of floating point bits that can be used as a tensor
        # Same process as encryptPlaintext
        ciphertextBin = self.myEnc.encode(ciphertext)
        ciphertextBin = [float(c) for c in ciphertextBin]
        
        # Now, iterate over all the blocks and decrypt them
        if output_bits_or_chars == 'chars':
            result = ''
        else:
            result = []

        for i in range(0, len(ciphertextBin), self.plaintext_len):
            # Concatenate the ciphertext with the key
            tensor_in = tf.concat(0, [ciphertextBin[i:(i+self.plaintext_len)], key])
            # Get the resulting bit vector
            tf_result = sess.run([self.nets[1].conv_layer(self.nets[1].fc_layer(tensor_in))])[0]
            # Compress it back to a sequence of characters, decode them, and append to the result
            if output_bits_or_chars == 'chars':
                result += self.myEnc.decode("".join(str(int(round(abs(x)))) for x in tf_result))
            else:
                result.append(tf_result)
        
        return result
    
    def decryptEve(self, sess, ciphertext, output_bits_or_chars='chars'):
        
        # Check that we'll be able to encrypt without padding or anything (because I don't know how to remove the padding in decoding...what is padding and what isn't?)
        if 5*len(ciphertext) % self.plaintext_len != 0:
            print("Please either pad your plaintext to be a multiple of network block size\nOr train a network with a different block size")
            return ciphertext
        
        # Turn the ciphertext string into a list of floating point bits that can be used as a tensor
        # Same process as encryptPlaintext
        ciphertextBin = self.myEnc.encode(ciphertext)
        ciphertextBin = [float(c) for c in ciphertextBin]
        
        # Now, iterate over all the blocks and decrypt them
        if output_bits_or_chars == 'chars':
            result = ''
        else:
            result = []
            
        for i in range(0, len(ciphertextBin), self.plaintext_len):
            # Concatenate the ciphertext with the key
            tensor_in = tf.concat(0, [ciphertextBin[i:(i+self.plaintext_len)]])
            # Get the resulting bit vector
            tf_result = sess.run([self.nets[2].conv_layer(self.nets[2].fc_layer(tensor_in))])[0]
            # Compress it back to a sequence of characters, decode them, and append to the result
            if output_bits_or_chars == 'chars':
                result += self.myEnc.decode("".join(str(int(round(abs(x)))) for x in tf_result))
            else:
                result.append(tf_result)
        
        return result
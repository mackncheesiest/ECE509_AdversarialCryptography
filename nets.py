import tensorflow as tf


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
        out_tensor = tf.ceil(out_tensor)

        return out_tensor

class Trio(N):
    def __init__(self):
        self.this = None
    self.nets = [Net(name) for name in ['alice', 'bob', 'eve']]

    def train(self):
        # define a training function

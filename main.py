import tensorflow as tf
import nets


def main():
    in_length = 8  # mesage of length 4, so expect output of length 4
    conv_params = [
        [[4, 1, 2], 1], 
        [[2, 2, 4], 2], 
        [[1, 4, 4], 1], 
        [[1, 4, 1], 1]
    ]
    
    Trio = nets.Trio(in_length, conv_params)
    tf.global_variables_initializer()
    Trio.train()


if __name__ == '__main__':  # I don't know what this does
    main()

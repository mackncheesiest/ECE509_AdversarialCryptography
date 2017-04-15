import tensorflow as tf
import nets


def main():
    sess = tf.session()  # not sure where to invoke sessions
    Trio = nets.Trio()
    tf.global_variables_initializer()
    Trio.train()


if __name__ == '__main__':  # I don't know what this does
    main()

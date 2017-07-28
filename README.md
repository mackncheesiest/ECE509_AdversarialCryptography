# ECE509_AdversarialCryptography

- Encoder.py: Class containing methods used for encoding and decoding strings of text into bit vectors usable by the network
- Loader.py: Module containing `trained_trio` method that returns a saved Tensorflow session and Trio object from previous training
- Main.py: Module containing the main function that demonstrates configuration of network convolutional parameters, instantiation of a `Trio` object, and plotting/saving of results
- Map_Cipherspace.py: Module containing a script that picks a random key, loads a trained model, and computes every possible ciphertext using that key for the purposes of getting an idea for the structure of this space and any biases it may have
- Neighbors.py: Module containing helper functions in calculating things like the average number of bits that change in a ciphertext when a single input bit changes (i.e. does the network exhibit diffusion/confusion)
- Nets.py: Module containing the actual network definition as well as all the tensorflow training logic
- PCA_Cipherspace: Script that computes 2D, 3D, and 20D PCA representations of data from `Map_Cipherspace.py` for the purposes of visualizing or otherwise analyzing the structure of this space (i.e. is it lopsided? Is there a cluster of points in one area such that decoding points outside that cluster could be easier?)

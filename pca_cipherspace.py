import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ciphertexts_filename = './cipherspace_values.txt'
# Load the matrix back from a file
print("loading dataset...")
ciphertexts_arr = np.loadtxt(ciphertexts_filename)
print("dataset loaded...")
print("performing PCA...")
# There should be at most 20 components in this data since it's 20 dimensional
myPCA = PCA(n_components=20)
myPCA.fit(ciphertexts_arr)
print("PCA finished...")

###############################################################################################

pcaComponents = myPCA.components_
pcaExplainedVarRatio = myPCA.explained_variance_ratio_
pcaMean = myPCA.mean_

###############################################################################################

myPCA2 = PCA(n_components=2)
myPCA2.fit(ciphertexts_arr)
ciphertexts_2D = myPCA2.transform(ciphertexts_arr)

plt.scatter(ciphertexts_2D[:, 0], ciphertexts_2D[:, 1], c='k')
plt.show()

################################################################################################

myPCA3 = PCA(n_components=3)
myPCA3.fit(ciphertexts_arr)
ciphertexts_3D = myPCA3.transform(ciphertexts_arr)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(ciphertexts_3D[:, 0], ciphertexts_3D[:, 1], ciphertexts_3D[:, 2], c='k')
plt.show()
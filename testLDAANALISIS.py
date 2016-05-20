"""
=======================================================
Comparison of LDA and PCA 2D projection of Iris dataset
=======================================================

The Iris dataset represents 3 kind of Iris flowers (Setosa, Versicolour
and Virginica) with 4 attributes: sepal length, sepal width, petal length
and petal width.

Principal Component Analysis (PCA) applied to this data identifies the
combination of attributes (principal components, or directions in the
feature space) that account for the most variance in the data. Here we
plot the different samples on the 2 first principal components.

Linear Discriminant Analysis (LDA) tries to identify attributes that
account for the most variance *between classes*. In particular,
LDA, in contrast to PCA, is a supervised method, using known class labels.
"""
print(__doc__)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy
import utils.defaults as defaults

samples_train = numpy.load('samples_train.npy')
labels_train = numpy.load('labels_train.npy')
samples_test = numpy.load('samples_test.npy')
labels_test = numpy.load('labels_test.npy')
X=samples_train
y=labels_train
print "len(X)", len(X), "len(X[0])", len(X[0]), X[0]
print "len(y)", len(y)

pca = PCA(n_components=3)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=8, solver='eigen')
X_proj_train = lda.fit(X, y).transform(X)
X_proj_test = lda.transform(samples_test)

print "X_proj_train", len(X_proj_train), len(X_proj_train[0])
print "X_proj_test", len(X_proj_test), len(X_proj_test[0])
numpy.save('wlda_train.npy', X_proj_train)
numpy.save('wlda_test.npy', X_proj_test)

#X_r3 = X_r2*X'
# Percentage of variance explained for each components
print 'explained variance ratio (first 3 components): %s \n %s'%( str(pca.explained_variance_ratio_) , str(lda.explained_variance_ratio_))



plt.figure()
for c, i, target_name in zip("rgbkcm", [0, 1, 2,3,4,5], defaults.emotions):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('PCA of emotion dataset')

plt.figure()
for c, i, target_name in zip("rgbkcm", [0, 1, 2,3,4,5], defaults.emotions):
    plt.scatter(X_proj_train[y == i, 0], X_proj_train[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('LDA of emotion dataset')





from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
fig = plt.figure()
ax = fig.gca(projection='3d')

for c, i, target_name in zip("rgbkcm", [0, 1, 2,3,4,5], defaults.emotions):
    ax.scatter(X_proj_train[y == i, 0], X_proj_train[y == i, 1], X_proj_train[y == i, 2],zdir='y',c=c, label=target_name)
plt.legend()

ax.legend()
#ax.set_xlim3d(0, 1)
#ax.set_ylim3d(0, 1)
#ax.set_zlim3d(0, 1)

plt.show()

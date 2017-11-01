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
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!


#samples_test = numpy.load('samples_test.npy')
#labels_test = numpy.load('labels_test.npy')
Xunion=numpy.load(defaults.union_1318_nolog)
Yunion=numpy.load(defaults.union_labels_1318_nolog)
Xkdef  = numpy.load(defaults.kdef_nolog_1318)
Ykdef  = numpy.load(defaults.kdef_labels_nolog_1318)

indx = numpy.load('features.npy')
Xunion=Xunion[:,indx]
Xkdef=Xkdef[:,indx]
itemindex = numpy.where(Xunion==0)
itemindex2 = numpy.where(Xkdef==0)
cols = set(itemindex[1])
cols2 = set(itemindex2[1])
print cols
print cols2

#X= numpy.delete(X, list(cols), 1)

pca = PCA(n_components=3)
XPCA = pca.fit(Xunion).transform(Xunion)

lda = LinearDiscriminantAnalysis(n_components=32)
XLDA = lda.fit(Xunion, Yunion).transform(Xunion)
XldaKdef = lda.transform(Xkdef)

#numpy.save('wlda_train.npy', X_proj_train)
#numpy.save('wlda_test.npy', X_proj_test)

# Percentage of variance explained for each components
print 'explained pca variance ratio: %s '%( str(pca.explained_variance_ratio_))
#print 'explained variance ratio (first 3 components): %s '%( str(lda.explained_variance_ratio_)) 

### LDA KDEF
figK = plt.figure()
axK = figK.gca(projection='3d')

for c, i, target_name in zip("rgbkcm", [0, 1, 2,3,4,5], defaults.emotions):
    axK.scatter(XldaKdef[Ykdef == i, 0], XldaKdef[Ykdef == i, 1], XldaKdef[Ykdef == i, 2],zdir='y',c=c, label=target_name)
plt.legend()

axK.legend()
#axK.set_xlim3d(0, 1)
#axK.set_ylim3d(0, 1)
#axK.set_zlim3d(0, 1)
axK.set_title("LDA 3D 1318 distancies")
axK.set_xlabel("primera dimensio")
axK.set_ylabel("segona dimensio")
axK.set_zlabel("tercera dimensio")

################################################################################
################### LDA 
## 2D
plt.figure()
for c, i, target_name in zip("rgbkcm", [0, 1, 2,3,4,5], defaults.emotions):
    plt.scatter(XLDA[Yunion == i, 0], XLDA[Yunion == i, 1], c=c, label=target_name)
plt.legend()
plt.title('LDA of emotion dataset')

## 3D
fig = plt.figure()
ax = fig.gca(projection='3d')

for c, i, target_name in zip("rgbkcm", [0, 1, 2,3,4,5], defaults.emotions):
    ax.scatter(XLDA[Yunion == i, 0], XLDA[Yunion == i, 1], XLDA[Yunion == i, 2],zdir='y',c=c, label=target_name)
plt.legend()

ax.legend()
#ax.set_xlim3d(0, 1)
#ax.set_ylim3d(0, 1)
#ax.set_zlim3d(0, 1)
ax.set_title("LDA 3D 1318 distancies")
ax.set_xlabel("primera dimensio")
ax.set_ylabel("segona dimensio")
ax.set_zlabel("tercera dimensio")
#plt.show()
################################################################################
################### PCA 

## 2D
plt.figure()
for c, i, target_name in zip("rgbkcm", [0, 1, 2,3,4,5], defaults.emotions):
    plt.scatter(XPCA[Yunion == i, 0], XPCA[Yunion == i, 1], c=c, label=target_name)
plt.legend()
plt.title('PCA of emotion dataset')

##PCA 3D
fig1 = plt.figure()
ax1 = fig.gca(projection='3d')

for c, i, target_name in zip("rgbkcm", [0, 1, 2,3,4,5], defaults.emotions):
    ax1.scatter(XPCA[Yunion == i, 0], XPCA[Yunion == i, 1], XPCA[Yunion == i, 2],zdir='y',c=c, label=target_name)
plt.legend()

ax1.legend()
#ax1.set_xlim3d(0, 1)
#ax1.set_ylim3d(0, 1)
#ax1.set_zlim3d(0, 1)
ax1.set_title("PCA 3D 1318 distancies")
ax1.set_xlabel("primera dimensio")
ax1.set_ylabel("segona dimensio")
ax1.set_zlabel("tercera dimensio")

plt.show()




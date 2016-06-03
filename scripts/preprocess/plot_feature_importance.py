print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import numpy
import utils.defaults as defaults
'''
# Build a classification task using 3 informative features
X, y = make_classification(n_samples=6000,
                           n_features=2331,
                           n_informative=32,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2 )#,random_state=0,shuffle=False)
                           
'''                           
X  = numpy.load(defaults.KDEF_data)
y  = numpy.load(defaults.KDEF_labels)
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250)#,random_state=0)
                              

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

'''
# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
'''
nFeatures=int(X.shape[1]*1)
threshold=int(X.shape[1]*0.4)
print "nFeatures", nFeatures
indices = indices[0:nFeatures]
# Plot the feature importances of the forest
plt.figure()
plt.title("Importancia de les caracteristiques")
plt.bar(range(0,nFeatures), importances[indices],color="k", align="center")#,yerr=std[indices])
#plt.plot(range(nFeatures), importances[indices],color="r")
plt.plot([threshold, threshold], [0, 0.0045], "k--")
plt.xticks(range(0,nFeatures,100), indices[range(0,nFeatures,100)], rotation=-60)
plt.xlim([-1, nFeatures])

plt.xlabel("Caracteristica")
plt.ylabel("Importancia")

numpy.save('features_importance.npy',indices[0:threshold])
plt.show()


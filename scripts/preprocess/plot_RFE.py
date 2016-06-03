print(__doc__)
from sklearn.linear_model import LogisticRegression
import sys
import time
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
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

if sys.argv[1]=='train':
    t0=  time.time()
    print t0
    # create a badatetimese classifier used to evaluate a subset of attributes
    model = LogisticRegression()
    # create the RFE model and select 3 attributes
    rfe = RFE(model, 32)
    rfe = rfe.fit(Xkdef, Ykdef)
    # summarize the selection of the attributes
    print "RFE: 32"
    t1=time.time()
    print (t1-t0)

    numpy.save('rank32.npy', rfe.ranking_)
    numpy.save('sup32.npy', rfe.support_)
    importances=rfe.ranking_
else:
    importances=numpy.load('rank32.npy')


indices = np.argsort(importances)[::-1]

'''
# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
'''
nFeatures=int(X.shape[1]*1)
threshold=int(X.shape[1]*0.4)
print "t",threshold
print "nFeatures", nFeatures
indices = indices[0:nFeatures]
# Plot the feature importances of the forest
plt.figure()
plt.title("Ranking")
plt.bar(range(0,nFeatures), importances[indices],color="k", align="center")#,yerr=std[indices])
#plt.plot(range(nFeatures), importances[indices],color="r")
plt.plot([threshold, threshold], [0, 2500], "k--")
plt.xticks(range(0,nFeatures,100), indices[range(0,nFeatures,100)], rotation=-60)
plt.xlim([-1, nFeatures])

plt.xlabel("Caracteristica")
plt.ylabel("Posicio")

numpy.save('RFE_50.npy',indices[0:50])
plt.show()


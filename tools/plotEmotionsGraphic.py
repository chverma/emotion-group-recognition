## table 1
import numpy as np
import matplotlib.pyplot as plt
# TPR
TPR=[ 96.60, 93.51, 99.18, 90.05, 94.64, 92.47, 99.85]
TPR=[98.03, 93.97, 99.12, 89.48, 96.04, 90.40, 99.50]
TPR_mix=[98.37, 97.05, 99.76, 91.31, 97.05, 94.93, 99.88]

# SPC
SPC=[ 99.42, 98.22, 99.94, 99.25, 99.02, 98.84, 99.82]
SPC=[99.57, 98.21, 99.83, 99.31, 98.87, 99.02, 99.76]
SPC_mix=[99.60, 99.01, 99.87, 99.57, 99.10, 99.53, 99.92]

# PPV
PPV=[ 96.69, 89.65, 99.66, 94.63, 94.27, 93.08, 98.97]
PPV=[97.57, 89.65, 98.97, 94.97, 93.56, 93.98, 98.62]
PPV_mix=[97.75, 93.68, 99.25, 96.86, 94.85, 97.17, 99.56]

# NPV
NPV=[ 99.40, 98.92, 99.86, 98.55, 99.08, 98.73, 99.97]
NPV=[99.65, 99.00, 99.85, 98.47, 99.32, 98.38, 99.91]
NPV_mix=[99.71, 99.51, 99.96, 98.74, 99.49, 99.14, 99.98]

# FPR
FPR=[ 0.58, 1.78, 0.06, 0.75, 0.98, 1.16, 0.18]
FPR=[0.43, 1.79, 0.17, 0.69, 1.13, 0.98, 0.24]
FPR_mix=[0.40, 1.08, 0.13, 0.43, 0.90, 0.47, 0.08]

# FDR
FDR=[ 3.31, 10.35, 0.34, 5.37, 5.73, 6.92, 1.03]
FDR=[2.43, 10.35, 1.03, 5.03, 6.44, 6.02, 1.38]
FDR_mix=[2.25, 6.32, 0.75, 3.14, 5.15, 2.83, 0.44]

# FNR
FNR=[ 3.40, 6.49, 0.82, 9.95, 5.36, 7.53, 0.15]
FNR=[1.97, 6.03, 0.88, 10.52, 3.96, 9.60, 0.50]
FNR_mix=[1.63, 2.95, 0.24, 8.69, 2.95, 5.07, 0.12]

# ACC
ACC=[ 99.00, 97.56, 99.83, 98.08, 98.38, 97.91, 99.83]
ACC=[99.34, 97.61, 99.73, 98.05, 98.46, 97.77, 99.72]
ACC_mix=[99.42, 98.66, 99.86, 98.51, 98.80, 98.87, 99.92]

## table monolitic best model
#felicitat & neutralitat & disgust & por & sorpresa & tristesa & ira  \\ \cline{1-9}

SVM_TPR = [98.52, 95.96, 99.87, 92.93, 95.05, 92.70, 100]
MLP_TPR = [98.25, 97.35, 99.64, 90.36, 97.43, 94.46, 99.69]
#MAX_TPR = [98.52, 97.35, 99.87,92.93,97.43,94.46,100]
SVM = [99.44, 98.49, 99.93, 99.60, 99.26, 99.16, 99.97]
MLP = [99.61, 99.04, 99.88, 99.54, 99.03, 99.36, 99.85] 

SVM_PPV = [96.86, 91.62, 99.63, 97.15, 95.68, 91.97, 98.87]
MLP_PPV = [97.72, 94.39, 99.35, 96.63, 94.77, 96.10, 99.15] 

SVM_NPV = [99.74, 99.30, 99.97, 98.96, 99.15, 98.76, 100]
MLP_NPV = [99.70, 99.56, 99.93,98.60, 99.53, 99.07, 99.94] 

SVM_FPR = [0.55, 1.50, 0.06, 0.39, 0.73, 0.83, 0.02]
MLP_FPR = [0.38, 0.95, 0.11, 0.45, 0.96, 0.63, 0.14]

SVM_FDR = [3.13, 8.37, 0.36, 2.84, 4.31, 5.02, 0.12]
MLP_FDR = [2.27, 5.60, 0.64, 3.36, 5.22, 3.89, 0.84] 

SVM_FNR = [1.47, 4.03, 0.12, 7.06, 4.94, 7.29, 0]
MLP_FNR = [1.74, 2.64, 0.35, 9.63, 2.56, 5.53, 0.30] 

SVM_ACC = [99.31, 98.12, 99.93, 98.74, 98.65, 98.22, 99.98]
MLP_ACC = [99.41, 98.80, 99.85, 98.37, 98.79, 98.66, 99.83] 

svmX =[0, 10, 20, 30, 40, 50,60]
mlpX =[1, 11, 21, 31, 41, 51,61]
mixX =[2, 12, 22, 32, 42, 52,62]
por =[]
sorpresa =[]
tristesa =[]
ira=[]

emocions = ["felicitat", "neutralitat", "disgust", "por", "sorpresa", "tristesa", "ira"]
# Plot the feature importances of the forest
plt.figure()
plt.title("Taxa de vertaders positius (TPR)",fontsize = 25)
svm = plt.bar(svmX, SVM_TPR ,color="darkgrey", align="center", width=1, label='SVM')
mlp = plt.bar(mlpX, MLP_TPR ,color="gainsboro", align="center", width=1, label='MLP')
mix = plt.bar(mixX, TPR_mix ,color="grey", align="center", width=1, label='MIX')
plt.legend(loc=4,handles=[svm, mlp, mix])
#,yerr=std[indices])
#,yerr=std[indices])
plt.xticks(mlpX, emocions, rotation=+40, fontsize = 20)
#plt.plot(range(nFeatures), importances[indices],color="r")
#plt.xticks(range(0,nFeatures,100), indices[range(0,nFeatures,100)], rotation=-60)
plt.ylim([0, 102])

plt.xlabel("Expressio facial",fontsize = 20)
plt.ylabel("% TPR",fontsize = 20)

plt.show()


# Plot the feature importances of the forest
plt.figure()
plt.title("Valor predictiu positiu (PPV)", fontsize = 25)
svm = plt.bar(svmX, SVM_PPV ,color="darkgrey", align="center", width=1, label='SVM')
mlp = plt.bar(mlpX, MLP_PPV ,color="gainsboro", align="center", width=1, label='MLP')
mix = plt.bar(mixX, PPV_mix ,color="grey", align="center", width=1, label='MIX')
plt.legend(loc=4,handles=[svm, mlp, mix])
#,yerr=std[indices])
#,yerr=std[indices])

#plt.plot(range(nFeatures), importances[indices],color="r")
plt.xticks(mlpX, emocions, rotation=40, fontsize = 20)
plt.ylim([0, 102])

plt.xlabel("Expressio facial")
plt.ylabel("% PPV", fontsize = 20)

plt.show()


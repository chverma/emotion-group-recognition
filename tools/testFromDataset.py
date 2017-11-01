from webModel import webModel
import utils.defaults as defaults
import sys
import os
import numpy
from loaddata.LoadAndShuffleData import LoadAndShuffleData as loadData
#rootFiles=  '/home/chverma/UPV/TFG/database/Aberdeen/'
rootFiles='/home/chverma/UPV/TFG/Emotion-Recognition-DOF/datasets/Still-Images/Emotion-Recogition/Anthony/Angry/'
#rootFiles='/home/chverma/UPV/TFG/Emotion-Recognition-DOF/datasets/Still-Images/Emotion-Recogition/Anthony/Neutral/'
#rootFiles='/home/chverma/UPV/TFG/Emotion-Recognition-DOF/datasets/Still-Images/Emotion-Recogition/Anthony/Shocked/'
#rootFiles='/home/chverma/UPV/TFG/Emotion-Recognition-DOF/datasets/Still-Images/Emotion-Recogition/Anthony/Happy/'
rootFiles='/home/chverma/UPV/TFG/myapp/cameraCaptureWeb/uploads/sad/'
rootFiles='/home/chverma/UPV/TFG/myapp/cameraCaptureWeb/uploads/surprised/'
#rootFiles='/home/chverma/UPV/TFG/myapp/cameraCaptureWeb/uploads/fear/'
#rootFiles='/home/chverma/UPV/TFG/myapp/cameraCaptureWeb/uploads/disgust/'
#rootFiles='/home/chverma/UPV/TFG/myapp/cameraCaptureWeb/uploads/neutral/'
#rootFiles='/home/chverma/UPV/TFG/myapp/cameraCaptureWeb/uploads/happy/'

#emotion = sys.argv[1]
models = ['svm','knn','mlp','boost','rtrees']
models = ['svm']
resErr = [0]*len(models)
matrix = True
samples_train, labels_train, samples_test, labels_test  = loadData().shuffleData(  numpy.load(defaults.file_dataset), numpy.load(defaults.file_labels))

samples_train = numpy.vstack((samples_train,samples_test))
labels_train = numpy.hstack((labels_train,labels_test))


for indx,model in enumerate(models):
    nfiles = 0
    err=0
    wM = webModel(model,samples_train, labels_train)
    
    wM.evaluate(numpy.load(defaults.file_datasetKDEF), numpy.load(defaults.file_labelsKDEF))
    wM.evaluate(numpy.load(defaults.file_dataset12), numpy.load(defaults.file_labels12))
#print "Error: %f"%((float(err)/nfiles)*100)   
print "resErr", resErr 
#print defaults.emotions[webModel.predictImage(rootFiles+str(sys.argv[1]))]

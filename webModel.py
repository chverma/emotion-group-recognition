#!/usr/bin/env python

###ENRTRAR ACI http://www.shervinemami.info/faceRecognition.html
'''
     
'''

# built-in modules
from PIL import Image
import cv2
import cv
import math
import numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import dlib
# classes
from classes.SVM import SVM
from classes.MLP import MLP
from classes.KNearest import KNearest
from classes.Boost import Boost
from classes.RTrees import RTrees
# loaddata
from loaddata.LoadAndShuffleData import LoadAndShuffleData as loadData
from loaddata.processImage import getCamFrame
from loaddata.processImage import get_significant_points
from loaddata.processImage import get_distance
from loaddata.processImage import get_landmarks
from loaddata.processImage import annotate_landmarks
# defaults
import utils.defaults as defaults

class webModel(object):
    def __init__(self, classifier, samples_train, labels_train):
        self.classifier=classifier
        
        if classifier=='svm':
            self.model = SVM(7,0.0000056 )
            self.model.set_params(dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_SVC,
                            nu=0.3,
                            degree = 1
                            ))  
        elif classifier=='mlp':
            self.model = MLP(31,8)
           
        elif classifier=='boost':
            self.model = Boost(5,None)
        elif classifier=='rtrees':
            self.model = RTrees(29,None)
        elif classifier=='knn':
            self.model = KNearest(1,None)
        elif classifier == 'svm_svc':
            self.model = SVM(7,0.0000056 )
            self.model.set_params(dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            nu=0.3,
                            degree = 1
                            )) 
        elif classifier == 'svm_nu_svc':
            self.model = SVM(7,0.0000056 )
            self.model.set_params(dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_NU_SVC,
                            nu=0.3,
                            degree = 1
                            )) 
            
        elif classifier=='knnLda':
            self.model = KNearest(1,None)
            samples_train=numpy.load(defaults.file_dataset)
            labels_train= numpy.load(defaults.file_labels)

            self.lda = LinearDiscriminantAnalysis(n_components=4, solver='eigen')
            samples_train = numpy.float32(self.lda.fit(samples_train, labels_train).transform(samples_train))


            print 'explained variance ratio (first 3 components): %s'%( str(self.lda.explained_variance_ratio_))

        self.model.train(samples_train,labels_train)
        self.model.evaluate(samples_train,labels_train)
        
    def evaluate(self,samples, labels):
        if self.classifier=='knnLda':
            samples = numpy.float32(self.lda.transform(samples))
        self.model.evaluate(samples,labels)
        
    def predictFromMatrix(self,img):
        landmark, obtained = get_landmarks(img)
        
        if obtained:
            #print "lendmark obtained from  matrix method"
            significant_points = get_significant_points(landmark)
            distance_between_points =  get_distance(significant_points, False)

            if self.classifier=='knnLda':
                distance_between_points = self.lda.transform([distance_between_points])
            #print distance_between_points
            result = int(self.model.predict(numpy.float32([distance_between_points])))

            return result
        else:
            return -1
            
    def predictImage(self,imgfile):
        img = cv2.imread(imgfile,0)
        #print "predict:", imgfile
                
        return self.predictFromMatrix(img)

    def predictFromModel(self, distance_between_points):
        print "entre1"
        print "Entree2", distance_between_points
        result = int(self.model.predict(numpy.float32([distance_between_points])))
        print result
        return result
    

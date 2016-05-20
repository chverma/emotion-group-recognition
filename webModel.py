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

import dlib
# classes
from classes.SVM import SVM
from classes.KNearest import KNearest
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
    def __init__(self):
        self.model = cv2.SVM()
        self.model.load(defaults.model_svm_xml)
        
    def predictFromMatrix(self,img):
        landmark, obtained = get_landmarks(img)
        
        if obtained:
            print "lendmark obtained from  matrix method"
            significant_points = get_significant_points(landmark)
            distance_between_points =  get_distance(significant_points)

            log_dist = map(lambda x: math.log10(x), distance_between_points)
            log_dist=numpy.asarray(log_dist, dtype=numpy.float32)

            #print distance_between_points
            result = int(self.model.predict(numpy.float32([log_dist])))
            print "result3", result
            return result
        else:
            return -1
            
    def predictImage(self,imgfile):
        img = cv2.imread(imgfile,0)
        print "predict:", imgfile
                
        return self.predictFromMatrix(img)
    

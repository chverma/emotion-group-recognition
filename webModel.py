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
# trainer
from trainers.trainerSVM import trainerSVM
import Image

class webModel(object):
    def __init__(self):
        self.model = cv2.SVM()
        self.model.load(defaults.model_svm_xml)
     
    def predictImage(self,imgfile):
        img = cv2.imread(imgfile,0)
        #b = cv.fromarray(img)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(b,'OpenCV',(10,500), font, 4,(255,255,255))#,2,cv2.LINE_AA)
        #img = numpy.asrray(b)
                
        landmark, obtained = get_landmarks(img)
        
        if obtained:
            significant_points = get_significant_points(landmark)
            distance_between_points =  get_distance(significant_points)

            log_dist = map(lambda x: math.log10(x), distance_between_points)
            log_dist=numpy.asarray(log_dist, dtype=numpy.float32)
            return int(self.model.predict(log_dist))
        else:
            return 0

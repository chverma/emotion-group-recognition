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

           
win=None
if __name__ == '__main__':

    load_model = raw_input("Load model? y/n")
    win = dlib.image_window()
    if load_model=='n':
        samples_train, labels_train, samples_test, labels_test = loadData().getData()
        model = trainerSVM()
        model.train(samples_train, labels_train)
        model.save(defaults.model_svm_xml)
        ##Does not work
        model.evaluate(samples_test, labels_test)
    else:
        #model = cv2.SVM()
        model = trainerSVM()
        model.load(defaults.model_svm_xml)
        

    ##predict from camera
    camera=cv2.VideoCapture(0)
    resp=0
    while True:
        
        img=getCamFrame(True,camera)
        cv2.putText(img, defaults.emotions[resp], (10,50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255,255,255), 2, cv2.CV_AA)
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
            
            win.set_image(img)
            ##Print Points
            #win.set_image(annotate_landmarks(img, numpy.matrix(significant_points)))

            resp = int(model.predict(log_dist))
            
            #print "response",defaults.emotions[resp]

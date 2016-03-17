#!/usr/bin/env python

###ENRTRAR ACI http://www.shervinemami.info/faceRecognition.html
'''
    https://fossies.org/diffs/opencv/2.4.11_vs_3.0.0/samples/python2/digits.py-diff.html

    SVM and KNearest digit recognition.

    Sample loads a dataset of handwritten digits from '../data/digits.png'.
    Then it trains a SVM and KNearest classifiers on it and evaluates
    their accuracy.

    Following preprocessing is applied to the dataset:
    - Moment-based image 
    deskew (see deskew())
    - Digit images are split into 4 10x10 cells and 16-bin
    histogram of oriented gradients is computed for each cell
    - Transform histograms to space with Hellinger metric (see [1] (RootSIFT))


    [1] R. Arandjelovic, A. Zisserman
    "Three things everyone should know to improve object retrieval"
    http://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
    Usage:
    digits.py    
'''

# built-in modules
from multiprocessing.pool import ThreadPool
from PIL import Image
import cv2
import cv
import math
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
import itertools as it
from trainerSVM import loadTrainingData
from trainerSVM import getCamFrame
from trainerSVM import get_significant_points
from trainerSVM import get_distance
from trainerSVM import get_landmarks
from trainerSVM import annotate_landmarks
import defaults
import dlib
import datetime

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class KNearest(StatModel):
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        self.model = cv2.KNearest()
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, self.k)
        return results.ravel()

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            C=C,
                            gamma=gamma )
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model = cv2.SVM()
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        #Thanks a lot http://stackoverflow.com/questions/8687885/python-opencv-svm-implementation
        return np.float32( [self.model.predict(s) for s in samples])
        #return self.model.predict_all(samples).ravel()
        #return self.model.predict(samples).ravel()
   
def evaluate_model(model, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print 'error: %.2f %%' % (err*100)

    confusion = np.zeros((defaults.CLASS_N, defaults.CLASS_N), np.int32)
    for i, j in zip(labels, resp):
        confusion[i, j] += 1
    print 'confusion matrix:'
    print confusion
    print
win=None
if __name__ == '__main__':

    load_model = raw_input("Load model? y/n")
    win = dlib.image_window()
    if load_model=='n':
        print "Obtaining data..."
        t1 = datetime.datetime.now()
        ## Obtain digits and labels array:digits, array:labels
        data, labels = loadTrainingData()
        t2 = datetime.datetime.now()
        print "Total time loading:", (t2-t1)
    
        print("Total dataset size:")
        print("n_samples: %d" % len(data))
        print("n_features: %d" % defaults.dim)
        print("n_classes: %d" % defaults.CLASS_N)
        
        ## shuffle data
        rand = np.random.RandomState(321)
        shuffle = rand.permutation(len(data))
        data, labels = data[shuffle], labels[shuffle]

        print "higth:%d; width:%d"%(len(data),len(data[0]))
        train_n = int(0.9*len(data))
        print "training_n:%d; total_n:%d"%(train_n,len(data))
         
        samples_train, samples_test = np.split(data, [train_n])
        labels_train, labels_test = np.split(labels, [train_n])
        

        print 'training KNearest...'
        model = KNearest(k=2)
        model.train(samples_train, labels_train)
        vis = evaluate_model(model, samples_test, labels_test)

        #model.save("emotion_KNearest.xml")
        
        print 'training SVM...'
        model = SVM(C=2.67, gamma=5.383)
        model.train(samples_train, labels_train)
        vis = evaluate_model(model, samples_test, labels_test)

        print 'saving SVM as "emotions_svm.dat"...'
        model.save('emotions_svm.xml')
    else:
        model = cv2.SVM()
        model.load('emotions_svm.xml')

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
        #img = np.asrray(b)
        landmark = get_landmarks(img)
        if landmark!=None:
            significant_points = get_significant_points(landmark)
            distance_between_points =  get_distance(significant_points)

            log_dist = map(lambda x: math.log10(x), distance_between_points)
            log_dist=np.asarray(log_dist, dtype=np.float32)
            
            win.set_image(img)
            ##Print Points
            #win.set_image(annotate_landmarks(img, np.matrix(significant_points)))
            resp = int(model.predict(log_dist))
            
            #print "response",defaults.emotions[resp]

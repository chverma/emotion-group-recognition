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
import time
import dlib
# models classes
from classes.SVM import SVM
from classes.KNearest import KNearest
from classes.MLP import MLP
from classes.RTrees import RTrees
from classes.Boost import Boost

# loaddata
from loaddata.LoadAndShuffleData import LoadAndShuffleData as loadData
from loaddata.processImage import getCamFrame
from loaddata.processImage import get_significant_points
from loaddata.processImage import get_distance
from loaddata.processImage import get_landmarks
from loaddata.processImage import annotate_landmarks
# defaults
import utils.defaults as defaults
from scripts.preprocess.drawPointsAndLines import drawPointsAndLines as drawPointsAndLines
indx=None
cols=None
def predictFromCamera(model):
    win = dlib.image_window()
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
            distance_between_points =  get_distance(significant_points, defaults.use_log)
            
            #win.set_image(img)
            ##Print Points
            win.set_image(annotate_landmarks(img, numpy.matrix(landmark)))
            #dlib.hit_enter_to_continue()
            nDistance = len(distance_between_points)
            #print nDistance
            #print len(distance_between_points)
            if nDistance:
                
                ob = numpy.float32([distance_between_points])[:,indx]
                #print "1",len(ob[0])
                if cols!=None:
                    ob = numpy.delete(ob, list(cols), 1)
                #print "2",len(ob[0])
                resp = int(model.predict(ob)) # This predict does not work with Knearest
            #else:
            #    resp = int(model.predict(numpy.float32([distance_between_points])))
            #print "response",defaults.emotions[resp]           
def main(parameters, samples, labels):
    import getopt

    models = [SVM, MLP]

    models = dict( [(cls.__name__.lower(), cls) for cls in models] )
    '''
    print 'USAGE: emotionExperts.py [--camera <on/off> [--eval <y/n>] [--draw <y/n>]'
    print 'Models: ', ', '.join(models)
    print
    '''
    args, dummy = getopt.getopt(parameters, '', ['camera=', 'eval='])
    args = dict(args)
    args.setdefault('--camera', 'on')
    args.setdefault('--eval', 'y')
   
    modelSVM = SVM(None, None)
    modelSVM.set_params(None)
    modelMLP = MLP(15, 76)
    
    modelFearSVM = SVM(None, None)
    modelFearSVM.set_params(None,True)
    '''modelFearSVM.set_params(dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_NU_SVC,
                            degree = 1,
                            nu=0.3
                            ))
    '''
    samples_train=None
    labels_train=None
    samples_test=None
    labels_test = None
    
    test_rateMix=0.
    test_rateSVM=0.
    test_rateMLP=0.
    cFea=0
    cErr=0
    confusion = numpy.matrix(numpy.zeros((defaults.CLASS_N, defaults.CLASS_N), numpy.int32))
    for _ in xrange(100):
        samples_train, labels_train, samples_test, labels_test  = loadData().shuffleData(samples, labels)
        
        from numpy import copy

        labels_train_Fear = copy(labels_train)
        labels_train_Fear[labels_train_Fear != 3] = -1
        labels_train_Fear[labels_train_Fear == 3] = 0
        print len(labels_train_Fear[labels_train_Fear == 1])
        print len(labels_train_Fear[labels_train_Fear == 0])
        
        modelSVM.train(samples_train, labels_train)
        
        modelMLP.train(samples_train, labels_train)
        modelFearSVM.train(samples_train, labels_train_Fear)
        #modelFearSVM.save("fear.xml")
        
        count=0
        results=[]
        
        for indx,s in enumerate(samples_test):
            resSVM = int(modelSVM.predict([s]))
            resMLP = int(modelMLP.predict(numpy.asarray([s])))
            resFEAR = int(modelFearSVM.predict([s]))
            
            if resFEAR==0:
                results.append(3)
                cFea=cFea+1
                if labels_test[indx]!=3:
                    print "resFEAR: label:%i"%(int(labels_test[indx]))
                    cErr=cErr+1
            else:
                if resSVM==resMLP:
                    results.append(resSVM)
                    if labels_test[indx]!=resSVM:
                        print "norm: pred: %i; label:%i"%(int(resSVM),int(labels_test[indx]))
                else:
                    if resMLP in [0,2,4,5]:
                        if resSVM in [3]:
                            results.append(resSVM)
                        else:
                            results.append(resMLP)
                            if labels_test[indx]!=resMLP:
                                print "resMLP: pred: %i; label:%i"%(int(resMLP),int(labels_test[indx]))
                    elif resSVM in [1,3,6]:
                        results.append(resSVM)
                        if labels_test[indx]!=resSVM:
                            print "resSVM: pred: %i; label:%i"%(int(resSVM),int(labels_test[indx]))
                    else:
                        if resMLP in [1]:
                            results.append(resMLP)
                            if labels_test[indx]!=resMLP:
                                print "OUCH1: resSVM: %i; resMLP:%i; label:%i"%(int(resSVM),int(resMLP),int(labels_test[indx]))
                        else:
                            results.append(resSVM)
                            if labels_test[indx]!=resSVM:
                                print "OUCH2: resSVM: %i; resMLP:%i; label:%i"%(int(resSVM),int(resMLP),int(labels_test[indx]))
        
        test_rateMix  = test_rateMix+numpy.mean(numpy.asarray(results) != labels_test)
        test_rateSVM  = test_rateSVM+numpy.mean(modelSVM.predict(samples_test) != labels_test)
        test_rateMLP  = test_rateMLP+numpy.mean(modelMLP.predict(samples_test) != labels_test)
        confusion=confusion+numpy.matrix(modelMLP.evaluate(None, labels_test, numpy.asarray(results)))
        
    print 100-(float(cErr)/cFea*100)
    print "RESULTS"
    confusionSum=confusion.sum(axis=1)
    print "confusion",confusion
    print "confusionSum",confusionSum
    print "res",numpy.diagonal((numpy.float32(confusion)/confusionSum)*100)
    test_rateMix=test_rateMix/100
    test_rateSVM=test_rateSVM/100
    test_rateMLP=test_rateMLP/100
    print "test_rateMix", test_rateMix
    print "test_rateSVM", test_rateSVM
    print "test_rateMLP", test_rateMLP
    
    
    if '--camera' in args:
        fn = args['--camera']
        if fn=='on': 
            predictFromCamera(model)
            
    return None
            
if __name__ == '__main__':
    import sys
    #print 'loading images from data file: npy'
    #If not exists, pass null

    samples = numpy.load(defaults.file_dataset)
    labels = numpy.load(defaults.file_labels)
    indx = numpy.load(defaults.model_feautures)
    nInd=len(indx)*0.4
    indx=indx[0:nInd]

    samples = samples[:,indx]
    nSamples= len(samples)
    if defaults.use_log:
        itemindex = numpy.where(samples==0)
        cols = set(itemindex[1])
        samples = numpy.delete(samples, list(cols), 1)
        samples= numpy.asarray(map(lambda x: math.log10(x), list(samples.reshape(-1,))), dtype=numpy.float32)
        samples = samples.reshape(nSamples,-1)
    '''
    samples=None
    labels=None
    '''
    #print("Total dataset size:")
    #print("n_samples: %d" % len(labels_train))
    #print("n_test: %d" % len(labels_test))
    main(sys.argv[1:], samples, labels)
    
        
    

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
    test_rateFEAR=0.
    cFea=0
    cErr=0
    confSVM=confFEAR=confMLP = numpy.matrix(numpy.zeros((defaults.CLASS_N, defaults.CLASS_N), numpy.int32))
    for _ in xrange(100):
        samples_train, labels_train, samples_test, labels_test  = loadData().shuffleData(samples, labels)
        
        from numpy import copy

        labels_train_Fear = copy(labels_train)
        labels_train_Fear[labels_train_Fear != 3] = -1
        labels_train_Fear[labels_train_Fear == 3] = 0
        
        labels_test_Fear = copy(labels_test)
        labels_test_Fear[labels_test_Fear != 3] = -1
        labels_test_Fear[labels_test_Fear == 3] = 0
        
        modelSVM.train(samples_train, labels_train)
        
        modelMLP.train(samples_train, labels_train)
        modelFearSVM.train(samples_train, labels_train_Fear)
        #modelFearSVM.save("fear.xml")
        
        
        resSVM = modelSVM.predict(samples_test)
        resMLP = modelMLP.predict(samples_test)
        resFEAR = modelFearSVM.predict(samples_test)
  
        
        #test_rateMix  = test_rateMix+numpy.mean(numpy.asarray(results) != labels_test)
        test_rateSVM  = test_rateSVM+numpy.mean(resSVM != labels_test)
        test_rateFEAR  = test_rateFEAR+numpy.mean(resFEAR != labels_test_Fear)
        test_rateMLP  = test_rateMLP+numpy.mean(resMLP != labels_test)
        
        #confusion=confusion+numpy.matrix(modelMLP.evaluate(None, labels_test, numpy.asarray(results)))
        confSVM=confSVM+numpy.matrix(modelSVM.evaluate(None, labels_test, resSVM))
        confFEAR=confFEAR+numpy.matrix(modelSVM.evaluate(None, labels_test_Fear, resFEAR))
        confMLP=confMLP+numpy.matrix(modelSVM.evaluate(None, labels_test, resMLP))
    
    print 
    print "RESULTS SVM:"
    print confSVM
    
    
    print "TP_svm:"
    TP_svm=numpy.diagonal(confSVM)
    print TP_svm
    print
    
    print "FN_svm:"
    FN_svm=(confSVM.sum(axis=1).transpose()-TP_svm)[0]
    print FN_svm
    print
    
    print "FP_svm"
    FP_svm=confSVM.sum(axis=0)-TP_svm
    print FP_svm
    print

    print "TN_svm"
    TN_svm=confSVM.sum()*numpy.ones((1, defaults.CLASS_N), numpy.int32)-FP_svm-FN_svm-TP_svm
    print TN_svm
    print
    
    print "TPR or Sensitivity: TPR=TP/(TP+FN)"
    TPR_svm=numpy.float32(TP_svm)/(TP_svm+FN_svm)
    print TPR_svm
    print
    
    print "Specificity (SPC) or true negative rate (TNR): TPR=TN/(FP+TN)"
    SPC_svm=numpy.float32(TN_svm)/(TN_svm+FP_svm)
    print SPC_svm
    print
    
    print "precision or positive predictive value (PPV): PPV=TP/(TP+FP))"
    PPV_svm=numpy.float32(TP_svm)/(TP_svm+FP_svm)
    print PPV_svm
    print
    
    print "negative predictive value (NPV): NPV=TN/(TN+FN))"
    NPV_svm=numpy.float32(TN_svm)/(TN_svm+FN_svm)
    print NPV_svm
    print
    
    print "fall-out or false positive rate (FPR): FPR=1-SPC"
    FPR_svm= 1-SPC_svm
    print FPR_svm
    print
    
    print "false discovery rate (FDR) FDR=1-PPV"
    FDR_svm=1-PPV_svm
    print FDR_svm
    print
    
    print "miss rate or false negative rate (FNR) FNR=FN/(FN+TP)"
    FNR_svm=numpy.float32(FN_svm)/(FN_svm+TP_svm)
    print FNR_svm
    print
    
    print "accuracy (ACC) ACC=TP+TN/(TP+TN+FP+FN)"
    ACC_svm=numpy.float32(TP_svm+TN_svm)/(FN_svm+TP_svm+FP_svm+TN_svm)
    print ACC_svm
    print
    
    test_rateSVM=100-test_rateSVM/100
    print "test_rateSVM", test_rateSVM
    
    
    ##################################################
    ## MLP
    print 
    print "RESULTS MLP:"
    print confMLP
    
    
    print "TP:"
    TP_mlp=numpy.diagonal(confMLP)
    print TP_mlp
    print
    
    print "FN:"
    FN_mlp=(confMLP.sum(axis=1).transpose()-TP_mlp)[0]
    print FN_mlp
    print
    
    print "FP"
    FP_mlp=confMLP.sum(axis=0)-TP_mlp
    print FP_mlp
    print

    print "TN"
    TN_mlp=confMLP.sum()*numpy.ones((1, defaults.CLASS_N), numpy.int32)-FP_mlp-FN_mlp-TP_mlp
    print TN_mlp
    print
    
    print "TPR or Sensitivity: TPR=TP/(TP+FN)"
    TPR_mlp=numpy.float32(TP_mlp)/(TP_mlp+FN_mlp)
    print TPR_mlp
    print
    
    print "Specificity (SPC) or true negative rate (TNR): TPR=TN/(FP+TN)"
    SPC_mlp=numpy.float32(TN_mlp)/(TN_mlp+FP_mlp)
    print SPC_mlp
    print
    
    print "precision or positive predictive value (PPV): PPV=TP/(TP+FP))"
    PPV_mlp=numpy.float32(TP_mlp)/(TP_mlp+FP_mlp)
    print PPV_mlp
    print
    
    print "negative predictive value (NPV): NPV=TN/(TN+FN))"
    NPV_mlp=numpy.float32(TN_mlp)/(TN_mlp+FN_mlp)
    print NPV_mlp
    print
    
    print "fall-out or false positive rate (FPR): FPR=1-SPC"
    FPR_mlp= 1-SPC_mlp
    print FPR_mlp
    print
    
    print "false discovery rate (FDR) FDR=1-PPV"
    FDR_mlp=1-PPV_mlp
    print FDR_mlp
    print
    
    print "miss rate or false negative rate (FNR) FNR=FN/(FN+TP)"
    FNR_mlp=numpy.float32(FN_mlp)/(FN_mlp+TP_mlp)
    print FNR_mlp
    print
    
    print "accuracy (ACC) ACC=TP+TN/(TP+TN+FP+FN)"
    ACC_mlp=numpy.float32(TP_mlp+TN_mlp)/(FN_mlp+TP_mlp+FP_mlp+TN_mlp)
    print ACC_mlp
    print
    
    test_rateMLP=100-test_rateMLP/100
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
    
        
    
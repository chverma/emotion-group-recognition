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
            resp = int(model.predict(numpy.float32([distance_between_points]))) # This predict does not work with Knearest
            
            #print "response",defaults.emotions[resp]           
def main(parameters, samples, labels):
    import getopt

    models = [KNearest, SVM, MLP, Boost, RTrees]
    # Decision Trees, Gradient Boosted Trees,Extremely randomized trees,Expectation Maximization
    # Dtrees, GBTrees, ERTrees, EM
    models = dict( [(cls.__name__.lower(), cls) for cls in models] )
    '''
    print 'USAGE: emotionDetection.py [--model <model>] [--param1 <k,C,nh value>] [--param2 <gamma value>] [--imgFiles] [--load <model fn>] [--save <model fn>] [--camera <on/off> [--eval <y/n>]'
    print 'Models: ', ', '.join(models)
    print
    '''
    args, dummy = getopt.getopt(parameters, '', ['model=', 'imgFiles=', 'param1=', 'param2=','load=', 'imgFiles=','load=','save=', 'camera=', 'eval='])
    args = dict(args)
    args.setdefault('--camera', 'on')
    args.setdefault('--model', 'svm')
    args.setdefault('--param1', None)
    args.setdefault('--param2', None)
    args.setdefault('--eval', 'y')

    Model = models[args['--model']]
    
    model = Model(args['--param1'], args['--param2'])
    if args['--model']=='svm':
        #http://docs.opencv.org/2.4/modules/ml/doc/support_vector_machines.html#cvsvmparams-cvsvmparams
        model.set_params(dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_NU_SVC,
                            nu=0.3,
                            degree = 1
                            ))
        #POLY: degree = 1, NU_SVC: nu=0.3
    samples_train=None
    labels_train=None
    samples_test=None
    labels_test = None
    if '--imgFiles' in args:
        print 'loading images from %s ...' % defaults.img_directory
        samples, labels=loadData().getData()
        samples_train, labels_train, samples_test, labels_test = loadData().shuffleData(samples, labels)
    else:
        samples_train, labels_train, samples_test, labels_test  = loadData().shuffleData(samples, labels)
        
    if '--load' in args:
        fn = args['--load']
        #print 'loading model from %s ...' % fn
        model.load(fn)
    else:
        #print 'training %s ...' % Model.__name__
        model.train(samples_train, labels_train)

    #print 'testing...'
    #print 'predicting train'
    train_rate = numpy.mean(model.predict(samples_train) != labels_train)
    #print 'predicting test'
    test_rate  = numpy.mean(model.predict(samples_test) != labels_test)
    if '--eval' in args:
        if args['--eval']=='y':
            print 'train error: %f  test error: %f' % (train_rate*100, test_rate*100)
            model.evaluate(samples_test, labels_test)
    
    if '--save' in args:
        fn = args['--save']
        #print 'saving model to %s ...' % fn
        model.save(fn)

    if '--camera' in args:
        fn = args['--camera']
        if fn=='on': 
            predictFromCamera(model)
            
    return test_rate*100
            
if __name__ == '__main__':
    import sys
    #print 'loading images from data file: npy'
    samples = numpy.load(defaults.file_dataset)
    labels = numpy.load(defaults.file_labels)
    #print("Total dataset size:")
    #print("n_samples: %d" % len(labels_train))
    #print("n_test: %d" % len(labels_test))
    main(sys.argv[1:], samples, labels)
    
        
    

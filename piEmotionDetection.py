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
import datetime
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
from picamera.array import PiRGBArray
from picamera import PiCamera

def predictFromCamera(model):
    #win = dlib.image_window()
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    resp=0
    count=0
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	    # grab the raw NumPy array representing the image, then initialize the timestamp
	    # and occupied/unoccupied text
	    if True:
	        print "Take photo"
	        t1 = datetime.datetime.now()
        
	        count=0
	        img = frame.array
	        #cv2.putText(img, defaults.emotions[resp], (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.CV_AA)
	        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	        t2 = datetime.datetime.now()
	        print "Time taking photo:", (t2-t1)
	        landmark, obtained = get_landmarks(img)
	        t3 = datetime.datetime.now()
	        print "Time geting landmark:", (t3-t2)
	        if obtained:
	                significant_points = get_significant_points(landmark)
	                distance_between_points =  get_distance(significant_points)
	            
	                log_dist = map(lambda x: math.log10(x), distance_between_points)
	                log_dist=numpy.asarray(log_dist, dtype=numpy.float32)
	                ##Print Points
                    #win.set_image(annotate_landmarks(img, numpy.matrix(significant_points)))
	                #win.set_image(img)
	                t4 = datetime.datetime.now()
	                print "Time geting points:", (t4-t3)
	                resp = int(model.predict(numpy.float32([log_dist])))
	                print "response",defaults.emotions[resp]
	                t5 = datetime.datetime.now()
	                print "Time predicting:", (t5-t4)
	    count=count+1
	    # clear the stream in preparation for the next frame
	    rawCapture.truncate(0)

if __name__ == '__main__':
    import getopt
    import sys

    models = [KNearest, SVM, MLP, Boost, RTrees] # RTrees, Boost, NBayes
    models = dict( [(cls.__name__.lower(), cls) for cls in models] )

    print 'USAGE: emotionDetection.py [--model <model>] [--param1 <k,C,nh value>] [--param2 <gamma value>] [--imgFiles] [--load <model fn>] [--save <model fn>] [--camera <on/off>]'
    print 'Models: ', ', '.join(models)
    print
    
    args, dummy = getopt.getopt(sys.argv[1:], '', ['model=', 'imgFiles=', 'param1=', 'param2=', 'processImages=','load=', 'save=', 'camera='])
    args = dict(args)
    args.setdefault('--camera', 'on')
    args.setdefault('--model', 'svm')
    args.setdefault('--param1', None)
    args.setdefault('--param2', None)

    Model = models[args['--model']]
    
    model = Model(args['--param1'], args['--param2'])


        
    if '--load' in args:
        fn = args['--load']
        print 'loading model from %s ...' % fn
        model.load(fn)

    predictFromCamera(model)
        
    

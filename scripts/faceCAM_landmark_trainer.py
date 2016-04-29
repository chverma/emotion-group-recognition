#!/usr/bin/python

import sys
import os
import dlib
import glob
from skimage import io
import cv2
import numpy

def getCamFrame(color,camera):
    retval,frame=camera.read()
    if not color:
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #frame=numpy.rot90(frame)
    return frame


if len(sys.argv) != 2:
    print(
        "Give the path to the trained shape predictor model as the first "
        "Execute this program by running:\n"
        "    ./face_landmark_detection_fromCamera.py shape_predictor_68_face_landmarks.dat\n")
    exit()

predictor_path = sys.argv[1]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()
win2 = dlib.image_window()
camera=cv2.VideoCapture(0)
while True:
    
    img=getCamFrame(True,camera)
    #print "type", type(img),type(img[0])
    win.clear_overlay()
    win.set_image(img)
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        #for i in xrange(shape.num_parts):
        points =  shape.parts()
        #for i in xrange(len(points)):
        #        print points[i]
        #shape=shape[:20]
        print "TotalPoints:" ,len(points)
        #class_user = raw_input("Which class?")
        
        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
        #                                          shape.part(1)))
        # Draw the face landmarks on the screen.
        win.add_overlay(shape)

        #dlib.hit_enter_to_continue()
	
    win.add_overlay(dets)
    
    #dlib.hit_enter_to_continue()

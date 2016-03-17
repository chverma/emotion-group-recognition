#!/usr/bin/python

import sys
import os
import dlib
import glob
from skimage import io
import cv2
import numpy
import matplotlib.pyplot as plt
import math
from PIL import Image
import csv
import defaults
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

def plot_data(data):
    plt.plot(data)
    #emotion =  raw_input("Enter the emotion")
    #plt.ylabel(emotion)
    #plt.savefig(emotion+".png")
    plt.show()
    
def getCamFrame(color,camera):
    retval,frame=camera.read()
    if not color:
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #frame=numpy.rot90(frame)
    return frame

def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1 or len(rects) == 0:
        return None

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def get_significant_pointsFirst(landmark):
    significant_points = []        
        
    #Center Left Eye: P1
    significant_points.append( [(landmark[40, 0]+landmark[37, 0])/2, (landmark[40, 1]+landmark[37, 1])/2 ])
        
    #Center Right Eye: P2
    significant_points.append( [(landmark[47, 0]+landmark[44, 0])/2, (landmark[47, 1]+landmark[44, 1])/2 ])
        
    #Center EyeBrown Left: P3
    significant_points.append( [landmark[19, 0], landmark[19, 1]] )
        
    #Center EyeBrown Right: P4
    significant_points.append( [landmark[24, 0], landmark[24, 1]] )
        
    #Midpoint of Eyes: P5
    significant_points.append( [landmark[27, 0], landmark[27, 1]] )
        
    #Nose Tip: P6
    significant_points.append( [landmark[33, 0], landmark[33, 1]] )
        
    #Center mouth: P7
    significant_points.append( [landmark[66, 0], landmark[66, 1]] )
    
    return significant_points

def get_significant_points(landmark):
    significant_points = []        
             
    #Center EyeBrown Left: P1
    significant_points.append( [landmark[19, 0], landmark[19, 1]] )
        
    #Center EyeBrown Right: P2
    significant_points.append( [landmark[24, 0], landmark[24, 1]] )
        
    #Point EyeBrown Left: P3
    significant_points.append( [landmark[21, 0], landmark[21, 1]] )
    
    #Point EyeBrown Right: P4
    significant_points.append( [landmark[22, 0], landmark[22, 1]] )
    
    #Midpoint of Eyes: P5
    significant_points.append( [landmark[27, 0], landmark[27, 1]] )
    
    #Top Eye Left: P6
    significant_points.append( [landmark[37, 0], landmark[37, 1]] )
    
    #Top Eye Right: P7
    significant_points.append( [landmark[44, 0], landmark[44, 1]] )
    
    #Bottom Eye Left: P8
    significant_points.append( [landmark[41, 0], landmark[41, 1]] )
    
    #Bottom Eye Right: P9
    significant_points.append( [landmark[46, 0], landmark[46, 1]] )
        
    #Nose Tip: P10
    significant_points.append( [landmark[33, 0], landmark[33, 1]] )
    
    #Top left mouth: P11
    significant_points.append( [landmark[48, 0], landmark[48, 1]] )
        
    #Top middle mouth: P12
    significant_points.append( [landmark[51, 0], landmark[51, 1]] )
    
    #Top right mouth: P13
    significant_points.append( [landmark[54, 0], landmark[54, 1]] )
    
    #Bottom middle mouth: P14
    significant_points.append( [landmark[57, 0], landmark[57, 1]] )
    
    return significant_points

#####################################
## This function returns the distance between certain points. This points and 
## and distances corresponds of the second draw
def get_distance(significant_points):
    distance = []        
    def calc_dist(a,b):
        return ( ((b[0]-a[0])**2) + ((b[1]-a[1])**2) )**(0.5)
        
    #P1-P5
    distance.append( calc_dist(significant_points[0],significant_points[4]) )
        
    #P3-P5
    distance.append( calc_dist(significant_points[2],significant_points[4] ))
        
    #P5-P4
    distance.append( calc_dist(significant_points[4],significant_points[2] ))
    
    #P5-P2
    distance.append( calc_dist(significant_points[4],significant_points[1] ))
    
    #-#
    #P6-P8
    distance.append( calc_dist(significant_points[5],significant_points[7] ))
    
    #P7-P9
    distance.append( calc_dist(significant_points[6],significant_points[8] ))
    
    #-#
    #P10-P11
    distance.append( calc_dist(significant_points[9],significant_points[10] ))
    
    #P10-P12
    distance.append( calc_dist(significant_points[9],significant_points[11] ))
    
    #P10-P13
    distance.append( calc_dist(significant_points[0],significant_points[5] ))
        
    #P11-P12
    distance.append( calc_dist(significant_points[10],significant_points[11] ))
    
    #P12-P13
    distance.append( calc_dist(significant_points[11],significant_points[12] ))
        
    #P12-P14
    distance.append( calc_dist(significant_points[11],significant_points[13] ))
    
    
    return distance
    
def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx+1), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(255, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def loadTrainingData():
    """
    load training data
    """
    # create a list for filenames of happy pictures
    happyfiles = []
    with open(defaults.happy_csv, 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            happyfiles += rec

    # create a list for filenames of neutral pictures
    neutralfiles = []
    with open(defaults.neutral_csv, 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            neutralfiles += rec

    # create a list for filenames of disgust pictures
    disgustfiles = []
    with open(defaults.disgust_csv, 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            disgustfiles += rec
    
    # create a list for filenames of disgust pictures
    fearfiles = []
    with open(defaults.fear_csv, 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            fearfiles += rec
    
    # create a list for filenames of disgust pictures
    surprisedfiles = []
    with open(defaults.surprised_csv, 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            surprisedfiles += rec
            
    # create a list for filenames of disgust pictures
    sadfiles = []
    with open(defaults.sad_csv, 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            sadfiles += rec
    

    # N x dim matrix to store the vectorized data (aka feature space)       
    phi = numpy.zeros((len(happyfiles) + len(neutralfiles) + len(disgustfiles)+len(fearfiles) + len(surprisedfiles) + len(sadfiles), defaults.dim),numpy.float32)

    # 1 x N vector to store binary labels of the data: 1 for smile and 0 for neutral
    labels = []

    # load happy data
    for idx, filename in enumerate(happyfiles):
        phi[idx] = process_image(defaults.happy_imgs + filename)
        labels.append(0)
    print "loaded happy"
    
    # load neutral data    
    offset = idx + 1
    for idx, filename in enumerate(neutralfiles):
        phi[idx + offset] = process_image(defaults.neutral_imgs + filename)
        labels.append(1)
    print "loaded neutral"

    # load disgust data
    offset = offset+idx + 1
    for idx, filename in enumerate(disgustfiles):
        phi[idx + offset] = process_image(defaults.disgust_imgs + filename)
        labels.append(2)
    
    # load fear data
    offset = offset+idx + 1
    for idx, filename in enumerate(fearfiles):
        phi[idx + offset] = process_image(defaults.fear_imgs + filename)
        labels.append(3)
        
    # load surprised data
    offset = offset+idx + 1
    for idx, filename in enumerate(surprisedfiles):
        phi[idx + offset] = process_image(defaults.surprised_imgs + filename)
        labels.append(4)
        
    # load sad data
    offset = offset+idx + 1
    for idx, filename in enumerate(sadfiles):
        phi[idx + offset] = process_image(defaults.sad_imgs + filename)
        labels.append(5)

    return phi , numpy.asarray(labels)
    
def process_image(im_path):
    img = io.imread(im_path)
    #win.clear_overlay()
    #win.set_image(img)
    landmark = get_landmarks(img)
    if landmark.any():
        significant_points = get_significant_points(landmark)
        #print significant_points
            
        distance_between_points =  get_distance(significant_points)
        #print distance_between_points
        #plot_data(distance_between_points)
            
        #Instead of mapping, the distance must be logarized before
        log_dist = map(lambda x: math.log10(x), distance_between_points)
        #plot_data(log_dist)
            
        ##Print Points
        #win.set_image(annotate_landmarks(img, numpy.matrix(significant_points)))
        #cv2.imwrite('output.jpg', annotate_landmarks(img, numpy.matrix(significant_points))) 
            
        return log_dist
        
    print "File %s where it cannot obtain landmark"%(im_path)
    return None
          
def main():
    data, labels = loadTrainingData()


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(sys.argv[1])##This arg is the shape predictor path         


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Give the path to the trained shape predictor model as the first "
            "Execute this program by running:\n"
            "    ./trainerSVM.py shape_predictor_68_face_landmarks.dat\n")
        exit()
    main()



    








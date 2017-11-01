#!/usr/bin/python

import sys
import os
import dlib
import cv2
import numpy
import matplotlib.pyplot as plt
import math
from PIL import Image

import utils.defaults as defaults
import datetime
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(defaults.model_shape)##This arg is the shape predictor path

def plot_data(data):
    plt.plot(data)
    #emotion =  raw_input("Enter the emotion")
    #plt.ylabel(emotion)
    #plt.savefig(emotion+".png")
    plt.show()

def getCamFrame(color,camera):
    retval,frame=camera.read()
    if not color:
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #frame=numpy.rot90(frame)
    return frame

def get_landmarks(im):
    t1 = datetime.datetime.now()
    rects = detector(im, 1)
    t2 = datetime.datetime.now()
    # print "Time detector:", (t2-t1)
    #If no obtained landmark
    if len(rects) > 1 or len(rects) == 0:
        return -1, False

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()]), True


def getAllPoints(landmark):
    significant_points = []

    for i in xrange(68):
        significant_points.append([landmark[i, 0], landmark[i, 1]])

    return significant_points


def getRawDistance(significant_points):
    '''This function returns the distance between certain points.
        This points and distances corresponds to all available points.
        The distance can be chosed between euclidian or manhattan distance
    '''
    distance = []

    def computeDistance(a, b, dist='euclidian'):
        if dist == 'euclidian':
            return computeEuclideanDistance(a, b)
        elif dist == 'manhattan':
            return computeManhattanDistance(a, b)

    def computeEuclideanDistance(a, b):
        return (((b[0]-a[0])**2) + ((b[1]-a[1])**2))**(0.5)

    def computeManhattanDistance(a, b):
        return abs(b[0]-a[0]) + abs(b[1]-a[1])

    for i in xrange(len(significant_points)):
        dreta = i+1
        for d in xrange(dreta, len(significant_points)):
            distance.append(computeDistance(significant_points[i], significant_points[d]))

    return distance


def getProcessedDistances(significant_points, log):
    '''
    if not log:
        return get_distance12Features(significant_points)
    else:
        return numpy.asarray(map(lambda x: math.log10(x), get_distance12Features(significant_points)), dtype=numpy.float32)
    '''
    if not log:
        return getRawDistance(significant_points)
    else:
        try:
            return numpy.asarray(map(lambda x: math.log10(x), getRawDistance(significant_points)), dtype=numpy.float32)
        except:
            print "Exception"
            return numpy.asarray([])


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


def annotate_distances(im, landmarks, dist):
    im = im.copy()

    for p1, p2 in dist:
        pos = (landmarks[p1, 0], landmarks[p1, 1])
        pos2 = (landmarks[p2, 0], landmarks[p2, 1])
        cv2.line(im, pos, pos2, color=(255, 255, 255))
    return im


def process_image(im_path):
    # img = io.imread(im_path)
    img = cv2.imread(im_path, 0)
    #    print "Warning: It cannot obtain landmark on %s"%(im_path)
    return process_image_matrix(img)


def process_image_matrix(img):
    # win.clear_overlay()
    # win.set_image(img)

    landmark, obtained = get_landmarks(img)

    if obtained:
        distance_between_points = getProcessedDistances(getAllPoints(landmark), defaults.use_log)
        # print distance_between_points
        # plot_data(distance_between_points)

        # plot_data(log_dist)

        # Print Points
        # win.set_image(annotate_landmarks(img, numpy.matrix(significant_points)))
        # cv2.imwrite('output.jpg', annotate_landmarks(img, numpy.matrix(significant_points)))

        return distance_between_points

    return None

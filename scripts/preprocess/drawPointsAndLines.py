
from loaddata.processImage import getAllPoints
from loaddata.processImage import getProcessedDistances
from loaddata.processImage import get_landmarks
from loaddata.processImage import annotate_landmarks
from loaddata.processImage import annotate_distances
import numpy
import utils.defaults as defaults
import sys
import cv2
import dlib
def drawPointsAndLines(name_features_npy,name_imgfile):

    img = cv2.imread(name_imgfile,0)

    win = dlib.image_window()

    landmark, obtained = get_landmarks(img)
    if obtained:
            significant_points = getAllPoints(landmark)
            distance_between_points =  getProcessedDistances(significant_points, defaults.use_log)
            totalDist=[]
            for i in xrange(len(significant_points)):
                dreta=i+1
                for d in xrange(dreta,len(significant_points)):
                    totalDist.append([i,d])
            #features = numpy.load('features_importance.npy')
            features = numpy.load(name_features_npy)
            nfea = len(features)*0.4
            features=features[0:nfea]

            dist=[]
            for f in features:
                dist.append([totalDist[f][0] ,totalDist[f][1] ])
            print dist
            ##Print Points
            im = annotate_landmarks(img, numpy.matrix(landmark))
            win.set_image(im)
            #cv2.imwrite('68points.png',im)
            #dlib.hit_enter_to_continue()
            im = annotate_distances(img, numpy.matrix(landmark), dist)
            stri= 'Distances%i.png'%(nfea)
            stri= 'Distances%i_FE.png'%(nfea)
            cv2.imwrite(stri,im)
            win.set_image(im)



    dlib.hit_enter_to_continue()

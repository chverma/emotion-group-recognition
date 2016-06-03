from loaddata.processImage import get_significant_points
from loaddata.processImage import get_distance
from loaddata.processImage import get_landmarks
from loaddata.processImage import annotate_landmarks
from loaddata.processImage import annotate_distances
import numpy
import utils.defaults as defaults
import sys
import cv2
print sys.argv[1]
img = cv2.imread(sys.argv[1],0)
import dlib
win = dlib.image_window()

landmark, obtained = get_landmarks(img)
if obtained:
            significant_points = get_significant_points(landmark)
            distance_between_points =  get_distance(significant_points, defaults.use_log)
            totalDist=[]
            for i in xrange(len(significant_points)):
                dreta=i+1
                for d in xrange(dreta,len(significant_points)):
                    totalDist.append([i,d])
            #features = numpy.load('features_importance.npy')
            features = numpy.load('RFE_50.npy')
            
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
            #cv2.imwrite('Feature_importance.png',im)
            win.set_image(im)



dlib.hit_enter_to_continue()

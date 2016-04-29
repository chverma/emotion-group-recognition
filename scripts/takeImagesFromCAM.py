#!/bin/python
import sys
from PIL import Image
import glob
from time import sleep
import cv2
from cv2 import cv
path = str(sys.argv[1])

def getCamFrame(color,camera):
    retval,frame=camera.read()
    if not color:
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #frame=numpy.rot90(frame)
    return frame
    
camera=cv2.VideoCapture(0)
cont=0
while cont<50:
    
    img=getCamFrame(True,camera)
    
    if img!=None:
        if cont:
            
            cv2.imwrite(path+"x"+str(cont)+".png", img)
            cv2.imshow('frame',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cont+=1
        
        print "write",cont   
        sleep(0.3)
    
del(camera)

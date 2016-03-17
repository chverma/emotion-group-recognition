#!/bin/python
import sys
from PIL import Image
import glob


path = str(sys.argv[1])
class_name = sys.argv[2]


image_list = []
target = open(filename, 'w')

for filename in glob.glob(path+'/*'): #assuming gif
    target.write(filename)

target.close()

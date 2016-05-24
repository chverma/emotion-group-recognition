#!/bin/python
import sys
from PIL import Image
import glob


img_path = "../data/faces/union/"
#img_path = "../data/faces/KDEF_Class/"
emotions = ["happy","sad","surprised","neutral","fear","disgust"]
csv_path = "../data/csv/"

for indx, item in enumerate(emotions):
    print item
    target = open(csv_path+item+".csv", 'w')
    list_= ""
    for filename in glob.glob(img_path+item+'/*'): #assuming gif
        list_+=filename.split("/")[-1]+"\t"
    target.write(list_[0:len(list_)-1])
    target.close()



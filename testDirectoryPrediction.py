from webModel import webModel
import utils.defaults as defaults
import sys
import os

#rootFiles=  '/home/chverma/UPV/TFG/database/Aberdeen/'
rootFiles='/home/chverma/UPV/TFG/Emotion-Recognition-DOF/datasets/Still-Images/Emotion-Recogition/Anthony/Angry/'
#rootFiles='/home/chverma/UPV/TFG/Emotion-Recognition-DOF/datasets/Still-Images/Emotion-Recogition/Anthony/Neutral/'
#rootFiles='/home/chverma/UPV/TFG/Emotion-Recognition-DOF/datasets/Still-Images/Emotion-Recogition/Anthony/Shocked/'
#rootFiles='/home/chverma/UPV/TFG/Emotion-Recognition-DOF/datasets/Still-Images/Emotion-Recogition/Anthony/Happy/'
rootFiles='/home/chverma/UPV/TFG/myapp/cameraCaptureWeb/uploads/sad/'
rootFiles='/home/chverma/UPV/TFG/myapp/cameraCaptureWeb/uploads/surprised/'
#rootFiles='/home/chverma/UPV/TFG/myapp/cameraCaptureWeb/uploads/fear/'
#rootFiles='/home/chverma/UPV/TFG/myapp/cameraCaptureWeb/uploads/disgust/'
#rootFiles='/home/chverma/UPV/TFG/myapp/cameraCaptureWeb/uploads/neutral/'
#rootFiles='/home/chverma/UPV/TFG/myapp/cameraCaptureWeb/uploads/happy/'
webModel = webModel()

[files for root, subdir, files in os.walk(rootFiles)]

for f in files:
    print rootFiles+f
    print f, defaults.emotions[webModel.predictImage(rootFiles+f)]
    
        
#print defaults.emotions[webModel.predictImage(rootFiles+str(sys.argv[1]))]

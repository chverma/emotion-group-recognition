from webModel import webModel
import utils.defaults as defaults
import sys
host = '127.0.0.1'
port = 8008

rootFiles = '/home/chverma/UPV/TFG/myapp/cameraCaptureWeb/public/uploads/'

webModel = webModel('load', 'mlp')

print defaults.emotions[webModel.predictFile(rootFiles+str(sys.argv[1]))]

from web.jsonsocket import Server
from webModel import webModel
import utils.defaults as defaults
host = '127.0.0.1'
port = 8008

rootFiles=  '/home/chverma/UPV/TFG/myapp/cameraCaptureWeb/uploads/'


webModel = webModel()
# Server code:
server = Server(host, port)
while True:
    server.accept()

    data = server.recv(1)
    print "rootFiles+str(data):", rootFiles+str(data)
    res = webModel.predictImage(rootFiles+str(data))
    print "result", defaults.emotions[res]
    server.send(defaults.emotions[res]+"\n").close()
    server = Server(host, port)




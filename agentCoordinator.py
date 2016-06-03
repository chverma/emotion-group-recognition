##################################
#   EVENTS                                                                #
##################################
'''
This is an example about how to use messages (events)
to trigger the execution of a behaviour (EventBehaviour)
'''
host = "91.134.135.40"
#host  = '127.0.0.1'
import os
import io
import sys
import time
import base64

import spade
import datetime
import numpy
import webModel
import cv2
import time
import Image
from loaddata.processImage import process_image_matrix as process_image_matrix
import cv
from collections import defaultdict
import utils.defaults as defaults
class Coordinator(spade.Agent.Agent):
   
    class RecvFromNAO(spade.Behaviour.Behaviour):
        def onStart(self):
            print "Starting behaviour RecvFromNAO. . ."
        def _process(self):
            #print "waiting messages..."
            self.msg = None
            try:
                self.msg = self._receive(block=True)
                print "I HAVE RECEIVED ANYTHING"
            except Exception: 
                 print "just pException"
            
            if self.msg:
                t0=datetime.datetime.now()
                
                #print "Received message..."
                imgBin  = base64.b64decode(self.msg.getContent())
                originalImage = cv.CreateImageHeader((640, 480), cv.IPL_DEPTH_8U, 1)
                cv.SetData(originalImage, imgBin)
                del imgBin
                
                npArray = numpy.asarray(originalImage[:,:])
                del originalImage
                distances = process_image_matrix(npArray)
                del npArray
                if distances!=None:
                    ### Distribute matrix

                    msg = spade.ACLMessage.ACLMessage()
                    msg.setPerformative("inform")
                    msg.setOntology("predict-array")
                    for agent in self.myAgent.classificators:
                        msg.addReceiver(agent)
                        msg.setContent(distances)
                        self.myAgent.send(msg)
                    t1=datetime.datetime.now()	
                    #print "Sended: ",distances, "time:", (t1-t0)
                    del msg
                
                    del distances
                    print "Deleted"
                else:
                    print "No lendmark :("
                
                del self.msg
            else:
                print "No messages"

    
    class RecvClassificators(spade.Behaviour.Behaviour):
        def onStart(self):
            print "Starting behaviour RecvFromModels. . ."
            self.maxResponses=10
            self.nResponses=0
            self.currentResponses=[]
        def _process(self):
            #print "waiting messages..."
            self.msg = None
            try:
                self.msg = self._receive(block=True)
            except Exception: 
                 print "just pException"
            
            if self.msg:
                t0=datetime.datetime.now()
                
                #print "Received message..."
                resp = self.msg.getContent()
                self.currentResponses.append(resp)
                ## When it has all the results or timeout, send results to nao 
                if self.nResponses>self.maxResponses:
                    d = defaultdict(int)
                    for word in self.currentResponses:
                        d[word] += 1
                    

                    maxVal = -1
                    winner =  -1
                    for k in d.keys():
                        if d[k]>maxVal:
                            maxVal = d[k]
                            winner = k
 
                    print "winer",winner
                        
                        
                    #print "Received message6..."
                    msg = spade.ACLMessage.ACLMessage()
                    msg.setPerformative("inform")
                    msg.setOntology("response-predict")
                    msg.addReceiver(spade.AID.aid("nao@"+host,["xmpp://nao@"+host]))
                    msg.setContent(winner)
                    self.myAgent.send(msg)
                    t1=datetime.datetime.now()	
                    print "Sended to nao: ", winner
                    print "time:", (t1-t0)
                    self.nResponses=0
                    self.currentResponses=[]
                else:
                    self.nResponses=self.nResponses+1
            else:
                print "No messages"
                
    class RecvLoginClassificators(spade.Behaviour.Behaviour):
        def onStart(self):
            print "Starting behaviour RecvLoginClassificators. . ."
        def _process(self):
            #print "waiting messages..."
            self.msg = None
            try:
                self.msg = self._receive(block=True)
            except Exception: 
                 print "just pException"
            
            if self.msg:
                s = self.msg.getSender()
                #print "apuntar", s
                self.myAgent.classificators.append(s)
            else:
                print "No messages"
    def _setup(self):
        # Create the template for the EventBehaviour: a message from myself
        template = spade.Behaviour.ACLTemplate()
        template.setOntology("predict-image")
        templNAO = spade.Behaviour.MessageTemplate(template)
        template.setOntology("result-predict")
        templModels = spade.Behaviour.MessageTemplate(template)
        template.setOntology("login-please")
        templModelsLogin = spade.Behaviour.MessageTemplate(template)

        # Add the EventBehaviour with its template
        self.addBehaviour(self.RecvFromNAO(),templNAO)
        self.addBehaviour(self.RecvClassificators(),templModels)
        self.addBehaviour(self.RecvLoginClassificators(),templModelsLogin)
        
        # Add the sender behaviour
        #self.addBehaviour(self.SendMsgBehav())

    
a = Coordinator("coordinator@"+host,"secret")
a.classificators=[]
a.start()

alive = True
import time
while alive:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        alive=False
a.stop()
sys.exit(0)

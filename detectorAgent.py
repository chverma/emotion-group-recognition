##################################
#   EVENTS                                                                #
##################################
'''
This is an example about how to use messages (events)
to trigger the execution of a behaviour (EventBehaviour)
'''

import os
import sys
import time

import spade
import datetime
host = "127.0.0.1"
import numpy
import importlib
import webModel
import cv2
import time
wM = webModel.webModel()
cont=0
import utils.defaults as defaults
class Detector(spade.Agent.Agent):
   
    class RecvMsgBehav(spade.Behaviour.Behaviour):
        def onStart(self):
            print "Starting behaviour . . ."
            self.counter = 0
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
                img = self.msg.getContent()
                
                new_file = open('new_file.out','w')
  
                img = img.split(',')
 
                for i in img:
     
                    new_file.write(i)
                new_file.close()

                img=numpy.loadtxt('new_file.out')
                #print "Received message2..."
                
                img_fname = "agents/images/buffer"+str(self.counter)+".png"
                #print "Received message3..."
                cv2.imwrite(img_fname,img)
                #print "Received message4..."
                msg = spade.ACLMessage.ACLMessage()
                msg.setPerformative("inform")
                msg.setOntology("emotion-detected")
                msg.addReceiver(spade.AID.aid("nao@"+host,["xmpp://nao@"+host]))
                #print "Received message5..."
                
                
   
                resp = defaults.emotions[wM.predictImage(img_fname)]
                #print "Received message6..."
                msg.setContent(str(resp))
                self.myAgent.send(msg)
                t1=datetime.datetime.now()
                print "Sended: ",resp, "time:", (t1-t0)
            else:
                print "No messages"
            
            if self.counter>9:
                self.counter=0
    
    def _setup(self):
        # Create the template for the EventBehaviour: a message from myself
        template = spade.Behaviour.ACLTemplate()
        template.setOntology("detect-image")
        t = spade.Behaviour.MessageTemplate(template)
        self.cont=0
        # Add the EventBehaviour with its template
        self.addBehaviour(self.RecvMsgBehav(),t)
        
        # Add the sender behaviour
        #self.addBehaviour(self.SendMsgBehav())

    
a = Detector("detector@"+host,"secret")

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

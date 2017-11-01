#!/usr/bin/python
# -*- coding: utf-8 -*-
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
import sys
import time
import spade
import numpy
from loaddata.processImage import getCamFrame
sys.path.append('..')

import cv2
import datetime
import base64
import pyttsx

class Sender(spade.Agent.Agent):
    
            
            
    class SendMsgBehav(spade.Behaviour.PeriodicBehaviour):
        def predictFromCamera(self):
            ##predict from camera
            img=getCamFrame(False,self.camera)

            return img
        def onStart(self):
            print "Starting behaviour . . ."

            
            #self.myAgent.engine.say('Hola, soc el inspector gadjet.')
            #self.myAgent.engine.say('Voy a ver que tal tu cara.')
            #self.myAgent.engine.runAndWait()
            self.counter = 0
            self.imgPath, self.imgLabels = getFilesInfo()
            self.resp=0
        def _onTick(self):
            
            #print 't0',t0
            #print "Sending...", self.counter
            self.counter = self.counter + 1
            """
            This behaviour sends a message to this same agent to trigger an EventBehaviour
            """
            msg = spade.ACLMessage.ACLMessage()
            msg.setPerformative("inform")
            msg.setOntology("predict-image")
            msg.addReceiver(spade.AID.aid("coordinator@"+host,["xmpp://coordinator@"+host]))
            #print "s1"
            #img = cv2.imread("agents/46.png",0)
            img = base64.b64encode(cv2.imread(self.imgPath[self.counter],0))
            #img = base64.b64encode(self.predictFromCamera())

            msg.setContent(img)
 
            self.myAgent.send(msg)
            #print "Sended!"
            print "t0"
            self.myAgent.t0=datetime.datetime.now()
           
    class RecvMsgBehav(spade.Behaviour.Behaviour):
        """
        This EventBehaviour gets launched when a message that matches its template arrives at the agent
        """
        def onStart(self):
            print "Starting behaviour RecvMsgBehav. . ."
            self.counter = 0
            
        def _process(self):
            #print "WAITING RESPONSE"
            self.msg = self._receive(True)

            if self.msg:
                t0=datetime.datetime.now()
                print 't1',(t0-self.myAgent.t0)
                #print "Response obtained"
                emotion  = self.msg.getContent()
                print "->",emotion
            else:
                print "No response"
                     

    
    def _setup(self):
        # Create the template for the EventBehaviour: a message from myself
        template = spade.Behaviour.ACLTemplate()
        template.setOntology("response-predict")
        t = spade.Behaviour.MessageTemplate(template)

        self.addBehaviour(self.RecvMsgBehav(),t)
        # Add the sender behaviour
        b = self.SendMsgBehav(1)
        self.addBehaviour(b, None)

    
a = Sender("nao@"+host,"secret")

time.sleep(1)
a.start()
a.t0=0
alive = True
import time
while alive:
    try:
        time.sleep(0.3)
    except KeyboardInterrupt:
        alive=False
a.stop()
sys.exit(0)

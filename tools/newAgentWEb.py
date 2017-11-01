#!/usr/bin/python
# -*- coding: utf-8 -*-
##################################
#   EVENTS                                                                #
##################################

'''
This is an example about how to use messages (events)
to trigger the execution of a behaviour (EventBehaviour)
'''
host = "164.132.107.119"
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
rootFiles=  '/home/chverma/UPV/TFG/myapp/cameraCaptureWeb/uploads/'
class Sender(spade.Agent.Agent):
    class RecvMsgBehav(spade.Behaviour.Behaviour):
        """
        This EventBehaviour gets launched when a message that matches its template arrives at the agent
        """
        def onStart(self):
            print("Starting RecvImgBehav (receiveImage). . .")
            # Register agent model on coordinator
            msg = spade.ACLMessage.ACLMessage()
            msg.setPerformative("inform")
            msg.setOntology("predict-image")
            msg.addReceiver(spade.AID.aid("coordinator@"+host,["xmpp://coordinator@"+host]))
            #print "s1"
            img = cv2.imread(rootFiles+str(sys.argv[1]),0)
            msg.setContent(img)

            self.myAgent.send(msg)
            self.myAgent.t0=datetime.datetime.now()
            print("Sended login!")

        def _process(self):
            self.msg = self._receive(True)

            if self.msg:
                t0=datetime.datetime.now()
                emotion  = self.msg.getContent()
                print emotion
            else:
                print "No response"
            #self.myAgent.stop()
            sys.exit(0)

    def _setup(self):
        # Create the template for the EventBehaviour: a message from myself
        template = spade.Behaviour.ACLTemplate()
        template.setOntology("response-predict")
        t = spade.Behaviour.MessageTemplate(template)

        self.addBehaviour(self.RecvMsgBehav(),t)


a = Sender("nao@"+host,"secret")

time.sleep(1)
a.start()
a.t0=0
alive = True
import time
while alive:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        alive=False
a.stop()
sys.exit(0)

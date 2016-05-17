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
import numpy
from vision_imageNAO import getNAO_image_PIL as getNAO_image_PIL
sys.path.append('..')
#host = "127.0.0.1"
#host = '37.61.152.135'
host = '91.134.135.40'
IP = "127.0.0.1"
PORT = 9559
import cv2
import datetime
from naoqi import ALProxy
tts = ALProxy("ALTextToSpeech", IP, PORT)

class Sender(spade.Agent.Agent):
    class SendMsgBehav(spade.Behaviour.PeriodicBehaviour):
        def onStart(self):
            print "Starting behaviour . . ."
            self.counter = 0
            self.resp=0
        def _onTick(self):
            t0=datetime.datetime.now()
            print 't0',t0
            #print "Sending...", self.counter
            self.counter = self.counter + 1
            """
            This behaviour sends a message to this same agent to trigger an EventBehaviour
            """
            msg = spade.ACLMessage.ACLMessage()
            msg.setPerformative("inform")
            msg.setOntology("detect-image")
            msg.addReceiver(spade.AID.aid("detector@"+host,["xmpp://detector@"+host]))
            im = getNAO_image_PIL(IP,PORT)

            msg.setContent(im)
            t2=datetime.datetime.now()
            self.myAgent.send(msg)
            t3=datetime.datetime.now()
            print "timeSend: ",(t3-t2)
            t1=datetime.datetime.now()
            print "Sended!",(t1-t0)

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
                print 't1',t0
                #print "Response obtained"
                tts.say(self.msg.getContent())
                print "->",self.msg.getContent()
            else:
                print "No response"
                     

    
    def _setup(self):
        # Create the template for the EventBehaviour: a message from myself
        template = spade.Behaviour.ACLTemplate()
        template.setOntology("emotion-detected")
        t = spade.Behaviour.MessageTemplate(template)

        self.addBehaviour(self.RecvMsgBehav(),t)
        # Add the sender behaviour
        b = self.SendMsgBehav(1)
        self.addBehaviour(b, None)

    
a = Sender("nao@"+host,"secret")

time.sleep(1)
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

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
sys.path.append('..')
host = "127.0.0.1"
import cv2
class Sender(spade.Agent.Agent):

    class SendMsgBehav(spade.Behaviour.PeriodicBehaviour):
        def onStart(self):
            print "Starting behaviour . . ."
            self.counter = 0
        def _onTick(self):
            print "Sending...", self.counter
            self.counter = self.counter + 1
            """
            This behaviour sends a message to this same agent to trigger an EventBehaviour
            """
            msg = spade.ACLMessage.ACLMessage()
            msg.setPerformative("inform")
            msg.setOntology("detect-image")
            msg.addReceiver(spade.AID.aid("detector@"+host,["xmpp://detector@"+host]))

            img = cv2.imread("agents/46.png",0)

            numpy.savetxt('test.out', img, fmt='%i')
            img_file = open("test.out",'r')
            raw = ""
            for i in xrange(480):
                line=img_file.readline()
                raw = raw+','+line

            msg.setContent(raw)
 
            self.myAgent.send(msg)
            print "Sended!"
           
    class RecvMsgBehav(spade.Behaviour.Behaviour):
        """
        This EventBehaviour gets launched when a message that matches its template arrives at the agent
        """
        def onStart(self):
            print "Starting behaviour RecvMsgBehav. . ."
            self.counter = 0
            
        def _process(self):
            print "WAITING RESPONSE"
            self.msg = self._receive(True)
            print "Just passinggg"
            if self.msg:
                print "Response obtained"
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
        b = self.SendMsgBehav(30)
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

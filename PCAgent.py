#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import time
import spade
import numpy
from loaddata.processImage import getCamFrame
import cv2
import datetime
import base64
import pyttsx
sys.path.append('..')

host = "192.168.0.2"


class Sender(spade.Agent.Agent):
    class SendMsgBehav(spade.Behaviour.PeriodicBehaviour):
        """
        This behaviour sends an image message to the coordinator to request the facial emotion
        """
        def predictFromCamera(self):
            return getCamFrame(False, self.camera)

        def onStart(self):
            print "Starting SendMsgBehav behaviour . . ."
            # self.myAgent.engine.say('Hola, soc el inspector gadjet.')
            # self.myAgent.engine.say('Voy a ver que tal tu cara.')
            # self.myAgent.engine.runAndWait()
            self.counter = 0
            self.camera = cv2.VideoCapture(0)
            self.resp = 0

        def _onTick(self):
            self.counter = self.counter + 1
            msg = spade.ACLMessage.ACLMessage()
            msg.setPerformative("inform")
            msg.setOntology("predict-image")
            msg.addReceiver(spade.AID.aid("coordinator@"+host, ["xmpp://coordinator@"+host]))

            # Example: how to predict from image
            # img = cv2.imread("agents/46.png",0)
            # img = base64.b64encode(cv2.imread("../data/faces/UNION/neutral/23.png",0))
            img = base64.b64encode(self.predictFromCamera())

            msg.setContent(img)
            self.myAgent.send(msg)
            print "Image sended!"
            self.myAgent.t0 = datetime.datetime.now()

    class RecvMsgBehav(spade.Behaviour.Behaviour):
        """
        This EventBehaviour receives the response containing the facial emotion detection
        """
        def onStart(self):
            print "Starting behaviour RecvMsgBehav. . ."
            self.counter = 0

        def _process(self):
            self.msg = self._receive(True)

            if self.msg:
                t0 = datetime.datetime.now()
                print 'Received in %d seconds' % (t0-self.myAgent.t0).seconds
                # print "Response obtained"
                emotion = self.msg.getContent()
                print "-->", emotion
            else:
                print "No response"

    def _setup(self):
        # Create the template for the EventBehaviour: a message from myself
        template = spade.Behaviour.ACLTemplate()
        template.setOntology("response-predict")
        t = spade.Behaviour.MessageTemplate(template)

        self.addBehaviour(self.RecvMsgBehav(), t)
        # Add the sender behaviour
        b = self.SendMsgBehav(1)
        self.addBehaviour(b, None)


def main():
    a = Sender("camera@"+host, "secret")

    time.sleep(1)
    a.start()
    a.t0 = 0
    alive = True

    while alive:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            alive = False
    a.stop()
    sys.exit(0)


if __name__ == "__main__":
    main()

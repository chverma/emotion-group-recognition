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
import json
import hashlib

sys.path.append('..')

# Import config
with open('config.json') as data_file:
    localConfig = json.load(data_file)

# Define the IP server that contains a running spade instance to connect it as an agent
spadeServerIP = localConfig['spade']['ip_address']
RGBimages = localConfig['images']['RGB']


class Sender(spade.Agent.Agent):
    class SendImgBehav(spade.Behaviour.PeriodicBehaviour):
        """
        This behaviour sends an image message to the coordinator to request the facial emotion
        """
        def predictFromCamera(self):
            return getCamFrame(toRGB=RGBimages, camera=self.camera)

        def onStart(self):
            print "Starting SendImgBehav behaviour . . ."
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
            msg.setOntology("img")
            msg.addReceiver(spade.AID.aid("coordinator@"+spadeServerIP, ["xmpp://coordinator@"+spadeServerIP]))

            # Example: how to predict from image
            # img = cv2.imread("agents/46.png",0)
            # img = base64.b64encode(cv2.imread("../data/faces/UNION/neutral/23.png",0))
            npImg = self.predictFromCamera()
            base64img = base64.b64encode(npImg)
            try:
                print "dtype: ", npImg.dtype
                print "shape: ", npImg.shape
                print "strides: ", npImg.strides
                print "type: ", type(npImg)
                print "sha224Numpy", hashlib.sha224(npImg).hexdigest()
                print "sha224base64", hashlib.sha224(base64img).hexdigest()
                # cv2.imwrite("{}C.png".format(str(hashlib.sha224(npImg).hexdigest())), npImg)
            except Exception as e:
                print "ErrorType:", e

            msg.setContent(base64img)
            self.myAgent.send(msg)
            print "Image sended!"
            self.myAgent.t0 = datetime.datetime.now()

    class RecvEmotionBehav(spade.Behaviour.Behaviour):
        """
        This EventBehaviour receives the response containing the facial emotion detection
        """
        def onStart(self):
            print "Starting behaviour RecvEmotionBehav. . ."
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
        template.setOntology("emotion")
        t = spade.Behaviour.MessageTemplate(template)

        self.addBehaviour(self.RecvEmotionBehav(), t)
        # Add the sender behaviour
        b = self.SendImgBehav(1)
        self.addBehaviour(b, None)


def main():
    a = Sender("camera@"+spadeServerIP, "secret")

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

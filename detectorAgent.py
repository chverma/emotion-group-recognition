##################################
#   EVENTS                                                                #
##################################
'''
This is an example about how to use messages (events)
to trigger the execution of a behaviour (EventBehaviour)
'''

import os
import io
import sys
import time
import base64

import spade
import datetime
import numpy
import importlib
import webModel
import cv2
import time
import Image
import math
import cv
import utils.defaults as defaults

cont = 0
host = "192.168.0.2"

KDEF_data = numpy.load(defaults.KDEF_data)
KDEF_labels = numpy.load(defaults.KDEF_labels)
UNION_samples = numpy.load(defaults.UNION_data)
UNION_labels = numpy.load(defaults.UNION_labels)
indx = numpy.load(defaults.model_feautures)
samples = UNION_samples[:, indx]
nSamples = len(samples)
if defaults.use_log:
        # CLEAN THE ZEROS OF LOG OP
        itemindex = numpy.where(samples == 0)
        cols = set(itemindex[1])
        samples = numpy.delete(samples, list(cols), 1)
        samples = numpy.asarray(map(lambda x: math.log10(x), list(samples.reshape(-1,))), dtype=numpy.float32)
        samples = samples.reshape(nSamples, -1)
wM = webModel.webModel('svm', samples, UNION_labels)


class Detector(spade.Agent.Agent):
    class RecvMsgBehav(spade.Behaviour.Behaviour):
        def onStart(self):
            print "Starting behaviour . . ."
            self.counter = 0

        def _process(self):
            print "waiting messages..."
            self.msg = None
            try:
                self.msg = self._receive(block=True)
                print "--------RECEIVED", self.counter
            except Exception:
                print "just pException"

            if self.msg:
                t0 = datetime.datetime.now()

                # print "Received message..."
                imgBin = base64.b64decode(self.msg.getContent())
                originalImage = cv.CreateImageHeader((640, 480), cv.IPL_DEPTH_8U, 1)
                cv.SetData(originalImage, imgBin)
                npArray = numpy.asarray(originalImage[:, :])

                msg = spade.ACLMessage.ACLMessage()
                msg.setPerformative("inform")setOntology
                msg.setOntology("response-predict")
                msg.addReceiver(spade.AID.aid("nao@"+host, ["xmpp://nao@"+host]))

                indxEmo = wM.predictFromMatrix(npArray, indx, cols)
                if indxEmo > -1:
                    resp = defaults.emotions[indxEmo]
                else:
                    resp = 'No lendmark :('
                # print "Received message6..."
                msg.setContent(resp)
                self.myAgent.send(msg)
                t1 = datetime.datetime.now()
                print "Sended: ", resp, "time:", (t1-t0)
                self.counter = self.counter+1
            else:
                print "No messages"

            if self.counter > 9:
                self.counter = 0

    def _setup(self):
        # Create the template for the EventBehaviour: a message from myself
        template = spade.Behaviour.ACLTemplate()
        template.setOntology("predict-image")
        t = spade.Behaviour.MessageTemplate(template)
        self.cont = 0
        # Add the EventBehaviour with its template
        self.addBehaviour(self.RecvMsgBehav(), t)


a = Detector("coordinator@"+host, "secret")
a.start()

alive = True

while alive:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        alive = False
a.stop()
sys.exit(0)

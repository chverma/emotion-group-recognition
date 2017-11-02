#!/usr/bin/python
# -*- coding: utf-8 -*-
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
import json
import hashlib
import requests
# Import config
with open('config.json') as data_file:
    localConfig = json.load(data_file)

# Define the IP server that contains a running spade instance to connect it as an agent
spadeServerIP = localConfig['spade']['ip_address']
RGBimages = localConfig['images']['RGB']
from flask import send_file
class Coordinator(spade.Agent.Agent):
    class RecvFromCameraAgent(spade.Behaviour.Behaviour):
        def sendImgToPersonalIdentityAgent(self, msgStr):
            '''
                if len(self.myAgent.identity) > 0:
                    msg = spade.ACLMessage.ACLMessage()
                    print "Preparing sending"
                    msg.setPerformative("inform")
                    msg.setOntology("img")
                    print "Preparing sending1", self.myAgent.identity
                    msg.addReceiver(self.myAgent.identity[0])
                    print "Preparing sending2"
                    msg.setContent(msgStr)
                    print "Preparing sending3"
                    self.myAgent.send(msg)
                    print "Preparing sending4"
            '''

        def onStart(self):
            print "Starting behaviour RecvFromCameraAgent. . ."

        def _process(self):
            self.msg = None
            try:
                self.msg = self._receive(block=True)
                print "Received from Camera Agent"
            except Exception:
                print "just pException"

            if self.msg is not None:
                t0 = datetime.datetime.now()
                # self.sendImgToPersonalIdentityAgent(self.msg.getContent())
                # print "Content", hashlib.sha224(self.msg.getContent()).hexdigest()
                # Decode from base64 string to opencv img creating the header
                imgBin = base64.b64decode(self.msg.getContent())
                if RGBimages:
                    originalImage = cv.CreateImageHeader((640, 480), cv.IPL_DEPTH_8U, 3)
                else:
                    originalImage = cv.CreateImageHeader((640, 480), cv.IPL_DEPTH_8U, 1)
                cv.SetData(originalImage, imgBin)
                del imgBin
                # Cast the opencv img to numpy array
                npArray = numpy.asarray(originalImage[:, :])
                try:
                    cv2.imwrite("imageDeProva.png", npArray)
                    url = 'http://localhost:5001/checkIdentity'
                    files = {'file': open('imageDeProva.png', 'rb')}
                    r = requests.post(url, files=files)
                    print r.text
                except Exception as e:
                    print "Error request:", e
                del originalImage
                # Obtain the distances from numpy array
                distances = process_image_matrix(npArray)
                del npArray

                if distances is not None:
                    # Distribute matrix to all classificators
                    # Build the template message
                    msg = spade.ACLMessage.ACLMessage()
                    msg.setPerformative("inform")
                    msg.setOntology("distances")
                    # For each classificator agent send it the distances
                    for agent in self.myAgent.classificators:
                        print "Sending computed distances to %s" % str(agent)
                        msg.addReceiver(agent)
                        msg.setContent(distances)
                        self.myAgent.send(msg)

                    t1 = datetime.datetime.now()
                    # print "Sended: ",distances, "time:", (t1-t0)
                    del msg

                    del distances
                    print "Distance sended!"
                else:
                    print "There're not found distances. Check the posture :("

                del self.msg
            else:
                print "No messages"

    '''Defines the behaviour to interact with Classificator agents'''
    class RecvClassificators(spade.Behaviour.Behaviour):
        def onStart(self):
            print "Starting behaviour RecvFromModels. . ."
            # Defines how many responses take it to perform a response
            self.maxResponses = 2
            # Defines the number of current responses in that round
            self.nResponses = 0
            self.currentResponses = []

        def _process(self):
            self.msg = None
            try:
                self.msg = self._receive(block=True)
                print "Received from Classificator"
            except Exception:
                print "just pException"

            if self.msg:
                t0 = datetime.datetime.now()

                resp = self.msg.getContent()
                self.currentResponses.append(resp)

                # When it has all the results or timeout, send results to Camera agent
                if self.nResponses > self.maxResponses:
                    d = defaultdict(int)
                    for word in self.currentResponses:
                        d[word] += 1

                    maxVal = -1
                    winner = -1
                    for k in d.keys():
                        if d[k] > maxVal:
                            maxVal = d[k]
                            winner = k

                    print "Emotion: %s" % winner

                    # Build and send the emotion response to the source, the camera agent
                    msg = spade.ACLMessage.ACLMessage()
                    msg.setPerformative("inform")
                    msg.setOntology("emotion")
                    msg.addReceiver(self.myAgent.clients[0])
                    msg.setContent(winner)
                    self.myAgent.send(msg)
                    t1 = datetime.datetime.now()
                    print "Sended to Camera agent: %s" % winner
                    print "time:", (t1-t0)
                    self.nResponses = 0
                    self.currentResponses = []
                else:
                    self.nResponses = self.nResponses+1
            else:
                print "No messages"

    class RecvLogin(spade.Behaviour.Behaviour):
        def onStart(self):
            print "Starting behaviour RecvLogin. . ."

        def _process(self):
            self.msg = None
            try:
                self.msg = self._receive(block=True)
            except Exception:
                print "just pException"

            if self.msg:
                s = self.msg.getSender()
                print '%s has logged in' % (s)
                if (self.msg.getContent() == 'classificator'):
                    self.myAgent.classificators.append(s)
                elif (self.msg.getContent() == 'identity'):
                    self.myAgent.identity.append(s)
                elif (self.msg.getContent() == 'clients'):
                    self.myAgent.clients.append(s)
            else:
                print "No messages"

    '''Defines the behaviour to interact with PersonalIdentity agents'''
    class RecvPersonalIdentity(spade.Behaviour.Behaviour):
        def onStart(self):
            print "Starting behaviour RecvPersonalIdentity. . ."

        def _process(self):
            self.msg = None
            try:
                self.msg = self._receive(block=True)
                print "Received from PersonalIdentity"
            except Exception:
                print "just pException"

            if self.msg:
                t0 = datetime.datetime.now()

                resp = self.msg.getContent()
                print "Identity is: %s" % resp
            else:
                print "No messages"

    def _setup(self):
        # Create the template for the EventBehaviour: a message from myself
        template = spade.Behaviour.ACLTemplate()
        template.setOntology("img")
        templCamera = spade.Behaviour.MessageTemplate(template)
        template.setOntology("emotion")
        templModels = spade.Behaviour.MessageTemplate(template)
        template.setOntology("login")
        templModelsLogin = spade.Behaviour.MessageTemplate(template)
        template.setOntology("identity")
        templPersonalIdentity = spade.Behaviour.MessageTemplate(template)

        # Add the EventBehaviour with its template
        self.addBehaviour(self.RecvFromCameraAgent(), templCamera)
        self.addBehaviour(self.RecvClassificators(), templModels)
        self.addBehaviour(self.RecvPersonalIdentity(), templPersonalIdentity)
        self.addBehaviour(self.RecvLogin(), templModelsLogin)


def main():
    a = Coordinator("coordinator@"+spadeServerIP, "secret")
    a.classificators = []
    a.identity = []
    a.clients = []
    a.start()

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

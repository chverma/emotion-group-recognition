#!/usr/bin/env python2
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
from loaddata.processImage import process_image_matrix as process_image_matrix
import cv
from collections import defaultdict
import utils.defaults as defaults
import json
import hashlib
import requests
import traceback
import logging
logging.basicConfig(filename='agentSender.log', level=logging.DEBUG)
# Import config
with open('config.json') as data_file:
    localConfig = json.load(data_file)

# Define the IP server that contains a running spade instance to connect it as an agent
spadeServerIP = sys.argv[1]  # localConfig['spade']['ip_address']
logging.info("spadeServerIP: {}".format(spadeServerIP))
RGBimages = localConfig['images']['RGB']


class Coordinator(spade.Agent.Agent):
    class RecvFromCameraAgent(spade.Behaviour.Behaviour):
        def sendImgToFaceRecognitionServer(self, npArray):
            try:
                cv2.imwrite(localConfig['face_recognizer_server']['tmp_dir'], npArray)
                url = localConfig['face_recognizer_server']['put_image_url']
                files = {'file': open(localConfig['people']['tmp_dir'], 'rb')}
                r = requests.post(url, files=files)
                return json.loads(r.text)
            except Exception as e:
                logging.info("Error request: {}".format(e))

        def sendDistancesToEmotionRecognitionAgents(self, distances, personId):
            if distances is not None:
                # Distribute matrix to all classificators
                # Build the template message
                msg = spade.ACLMessage.ACLMessage()
                if not personId:
                    personId = 'No'
                msg.setPerformative(personId)
                msg.setOntology("distances")
                # For each classificator agent send it the distances
                for agent in self.myAgent.classificators:
                    logging.info("Sending computed distances of %s to %s" % (personId, agent.getName()))
                    msg.addReceiver(agent)
                    logging.info(personId)
                    msg.setContent(distances)
                    self.myAgent.send(msg)

                t1 = datetime.datetime.now()
                # print "Sended: ",distances, "time:", (t1-t0)
                del msg
                logging.info("Distance sended!")
            else:
                logging.info("There're not found distances. Check the posture :(")

        def onStart(self):
            logging.info("Starting behaviour RecvFromCameraAgent. . .")

        def _process(self):
            self.msg = None
            try:
                self.msg = self._receive(block=True)
                logging.info("Received from Camera Agent")
            except Exception:
                self.msg = None
                logging.info("just pException")

            if self.msg is not None:
                t0 = datetime.datetime.now()

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
                personId = self.sendImgToFaceRecognitionServer(npArray)['face_recognised']
                print personId
                del originalImage
                # Obtain the distances from numpy array
                distances = process_image_matrix(npArray)
                del npArray

                self.sendDistancesToEmotionRecognitionAgents(distances, personId)

                del self.msg
                del distances
            else:
                logging.info("No messages")

    '''Defines the behaviour to interact with Classificator agents'''
    class RecvClassificators(spade.Behaviour.Behaviour):
        def onStart(self):
            logging.info("Starting behaviour RecvFromModels. . .")
            # Defines how many responses take it to perform a response
            self.maxResponses = 2
            # Defines the number of current responses in that round
            self.nResponses = 0
            self.currentResponses = []

        def sendEmotionToFaceRecognitionServer(self, personId, emotion):
            url = localConfig['face_recognizer_server']['set_emotion_url']
            logging.info("sending {} to {}".format(emotion, personId))
            r = requests.post(url, params={'name': personId, 'emotion': emotion})

        def _process(self):
            self.msg = None
            try:
                self.msg = self._receive(block=True)
                logging.info("Received from Classificator")
            except Exception:
                logging.info("just pException")

            if self.msg:
                logging.info("RECEIVED FROM CLASSI")
                t0 = datetime.datetime.now()
                try:
                    content = self.msg.getContent()
                    personId = self.msg.getPerformative()
                    print content
                    resp = json.loads(content)
                    logging.info("----------- {}".format(resp['emotion']))
                    self.currentResponses.append(resp['emotion'])

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

                        logging.info("Emotion winner: %s" % winner)

                        self.sendEmotionToFaceRecognitionServer(personId, winner)
                        # Build and send the emotion response to the source, the camera agent
                        msg = spade.ACLMessage.ACLMessage()
                        msg.setPerformative("inform")
                        msg.setOntology("emotion")
                        msg.addReceiver(self.myAgent.clients[0])
                        msg.setContent(winner)
                        self.myAgent.send(msg)
                        del msg
                        t1 = datetime.datetime.now()
                        logging.info("Sended to Camera agent: %s" % winner)
                        logging.info("time: {}".format(t1-t0))
                        self.nResponses = 0
                        self.currentResponses = []
                    else:
                        self.nResponses = self.nResponses+1
                except Exception as e:
                    logging.info("Bad emotion")
                    logging.info(traceback.format_exc())
            else:
                logging.info("No messages")

    class RecvLogin(spade.Behaviour.Behaviour):
        def onStart(self):
            logging.info("Starting behaviour RecvLogin. . .")

        def _process(self):
            self.msg = None
            try:
                self.msg = self._receive(block=True)
            except Exception:
                logging.info("just pException")

            if self.msg:
                s = self.msg.getSender()

                if (self.msg.getContent() == 'classificator'):
                    if s not in self.myAgent.classificators:
                        self.myAgent.classificators.append(s)
                        logging.info('%s has logged in' % (s.getName()))
                elif (self.msg.getContent() == 'identity'):
                    self.myAgent.identity.append(s)
                    logging.info('%s has logged in' % (s.getName()))
                elif (self.msg.getContent() == 'clients'):
                    self.myAgent.clients.append(s)
                    logging.info('%s has logged in' % (s.getName()))
            else:
                logging.info("No messages")

    '''Defines the behaviour to interact with PersonalIdentity agents'''
    class RecvPersonalIdentity(spade.Behaviour.Behaviour):
        def onStart(self):
            logging.info("Starting behaviour RecvPersonalIdentity. . .")

        def _process(self):
            self.msg = None
            try:
                self.msg = self._receive(block=True)
                logging.info("Received from PersonalIdentity")
            except Exception:
                logging.info("just pException")

            if self.msg:
                t0 = datetime.datetime.now()

                resp = self.msg.getContent()
                logging.info("Identity is: %s" % resp)
            else:
                logging.info("No messages")

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
    coordinatorId = "coordinator@{}".format(spadeServerIP)

    a = Coordinator(coordinatorId, "secret")

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

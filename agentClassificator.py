#!/usr/bin/python
# -*- coding: utf-8 -*-
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
from loaddata.LoadAndShuffleData import LoadAndShuffleData as loadData
import utils.defaults as defaults
import cv
import utils.defaults as defaults
import sys
import json
# Import config
with open('config.json') as data_file:
    localConfig = json.load(data_file)

# Define the IP server that contains a running spade instance to connect it as an agent
spadeServerIP = localConfig['spade']['ip_address']

samples_train, labels_train, samples_test, labels_test = loadData().shuffleData(
    numpy.load(defaults.file_dataset),
    numpy.load(defaults.file_labels)
)

samples_train = numpy.vstack((samples_train, samples_test))
labels_train = numpy.hstack((labels_train, labels_test))
print "Training models with %d features" % (len(samples_train))
# Model definition and training
# SVMModels
# modelSVM_SVC = webModel.webModel('svm_svc', samples_train, labels_train)
# modelSVM_NU_SVC = webModel.webModel('svm_nu_svc',samples_train, labels_train)

# MLP model
modelMLP = webModel.webModel('mlp', samples_train, labels_train)

# KNEAREST model
# modelKNN = webModel.webModel('knn', samples_train, labels_train)

# RTREES model
# modelRTrees = webModel.webModel('rtrees',samples_train, labels_train)
print "Trained models"
# Put the trained model in that list in order to create an agent for each model
# models = [modelSVM_SVC, modelSVM_NU_SVC, modelMLP, modelKNN, modelRTrees]
models = [modelMLP]


class Classificator(spade.Agent.Agent):
    class RecvMsgBehav(spade.Behaviour.Behaviour):
        def onStart(self):
            print "Starting behaviour . . ."
            # Request login to coordinator agent onStart event
            msg = spade.ACLMessage.ACLMessage()
            msg.setPerformative("inform")
            msg.setOntology("login-please")
            msg.addReceiver(spade.AID.aid("coordinator@"+spadeServerIP, ["xmpp://coordinator@"+spadeServerIP]))
            msg.setContent('')
            self.myAgent.send(msg)
            print "Sended login!"

        def _process(self):
            # Wait for messages
            print "Waiting messages ..."
            self.msg = None
            try:
                self.msg = self._receive(block=True)
            except Exception:
                print "just pException"

            if self.msg:
                t0 = datetime.datetime.now()
                try:
                    # Try to recompose from string to string that can be passed to numpy
                    content = str(self.msg.getContent())\
                        .replace('[', '')\
                        .replace(']', '')\
                        .replace('  ', ',')\
                        .replace(',,,,', ',')\
                        .replace(',,', ',')\
                        .replace(' ', '')\
                        .replace('\n', '')
                except Exception:
                    print "just pException2"
                # Cast string to numpy array. It defines the distances computed by Coordinator agent.
                distances = numpy.fromstring(content, dtype=numpy.float32, sep=',')

                # Build the reply to the Coordinator
                rep = self.msg.createReply()
                rep.setOntology("result-predict")

                # Predict the distances array
                indxEmo = self.myAgent.model.predictFromModel(distances)
                # Try to get the emotion string by index of that class
                if indxEmo > -1:
                    resp = defaults.emotions[indxEmo]
                else:
                    resp = 'No lendmark :('
                # Put the response to reply and send the message
                rep.setContent(resp)
                self.myAgent.send(rep)
                t1 = datetime.datetime.now()
                print "Sended: %s in  %d microseconds" % (resp, (t1-t0).microseconds)
            else:
                print "No messages"

    def _setup(self):
        # Create the template for the EventBehaviour: a message from myself
        template = spade.Behaviour.ACLTemplate()
        template.setOntology("predict-array")
        t = spade.Behaviour.MessageTemplate(template)
        # Add the EventBehaviour with its template
        self.addBehaviour(self.RecvMsgBehav(), t)


def main():
    modelAgents = []

    for n in range(len(models)):
        agent = "classificator"+str(n)+"@"+spadeServerIP
        classificator = Classificator(agent, "secret")
        classificator.model = models[n]
        modelAgents.append(classificator)
        classificator.start()
        print "Launched classificator "+str(n)

    alive = True
    while alive:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            alive = False

    for b in modelAgents:
        b.stop()

    sys.exit(0)


if __name__ == "__main__":
    main()

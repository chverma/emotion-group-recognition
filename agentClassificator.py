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
samples_train, labels_train, samples_test, labels_test  = loadData().shuffleData(  numpy.load(defaults.file_dataset), numpy.load(defaults.file_labels))

samples_train = numpy.vstack((samples_train,samples_test))
labels_train = numpy.hstack((labels_train,labels_test))

#SVMModels
modelSVM_SVC = webModel.webModel('svm_svc',samples_train, labels_train)
modelSVM_NU_SVC = webModel.webModel('svm_nu_svc',samples_train, labels_train)
# MLP models
modelMLP = webModel.webModel('mlp',samples_train, labels_train)
# KNEAREST models
modelKNN = webModel.webModel('knn',samples_train, labels_train)
# RTREES models
modelRTrees = webModel.webModel('rtrees',samples_train, labels_train)

models = [modelSVM_SVC, modelSVM_NU_SVC, modelMLP, modelKNN, modelRTrees]


import utils.defaults as defaults
class Classificator(spade.Agent.Agent):
   
    class RecvMsgBehav(spade.Behaviour.Behaviour):
        def onStart(self):
            print "Starting behaviour . . ."
            ##Register agent model on coordinator
            msg = spade.ACLMessage.ACLMessage()
            msg.setPerformative("inform")
            msg.setOntology("login-please")
            msg.addReceiver(spade.AID.aid("coordinator@"+host,["xmpp://coordinator@"+host]))
            msg.setContent('')
            self.myAgent.send(msg)
            print "Sended login!"
            
        def _process(self):
            print "waiting messages..."
            self.msg = None
            try:
                self.msg = self._receive(block=True)

            except Exception: 
                 print "just pException"
            
            if self.msg:
                t0=datetime.datetime.now()
                print "ENTRE pero falla\n<<%s>>"%(self.msg.getContent())
                try:
                    content = str(self.msg.getContent()).replace('[', '').replace(']', '').replace('  ', ',').replace(',,,,',',').replace(',,',',').replace(' ','').replace('\n','')
                except Exception: 
                    print "just pException2"
                print "content '%s'"%(content)
                distances = numpy.fromstring(content, dtype=numpy.float32, sep=',')

                rep = self.msg.createReply()
                rep.setOntology("result-predict")
                
                print "Predicting..."
                indxEmo = self.myAgent.model.predictFromModel(distances)
                print "Predicted..."
                if indxEmo>-1:
                    resp = defaults.emotions[indxEmo]
                else:
                    resp = 'No lendmark :('
                #print "Received message6..."
                rep.setContent(resp)
                self.myAgent.send(rep)
                t1=datetime.datetime.now()	
                print "Sended: ",resp, "time:", (t1-t0)
            else:
                print "No messages"
            
    
    def _setup(self):
        # Create the template for the EventBehaviour: a message from myself
        template = spade.Behaviour.ACLTemplate()
        template.setOntology("predict-array")
        t = spade.Behaviour.MessageTemplate(template)
        # Add the EventBehaviour with its template
        self.addBehaviour(self.RecvMsgBehav(),t)
        
        # Add the sender behaviour
        #self.addBehaviour(self.SendMsgBehav())

    
modelAgents = []

for n in range(len(models)):
    agent = "classificator"+str(n)+"@"+host
    classificator = Classificator(agent,"secret")
    classificator.model = models[n]
    modelAgents.append(classificator)
    classificator.start()
    print "Launched classificator "+str(n)

alive =True
while alive:
	try:
	    time.sleep(1)
	except KeyboardInterrupt:
	    alive=False

for b in modelAgents:
    b.stop()
        
import sys
sys.exit(0)

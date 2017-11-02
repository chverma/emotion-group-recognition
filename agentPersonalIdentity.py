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
import sys
import json
import face_recognition
import hashlib

# Import config
with open('config.json') as data_file:
    localConfig = json.load(data_file)

# Define the IP server that contains a running spade instance to connect it as an agent
spadeServerIP = localConfig['spade']['ip_address']
RGBimages = localConfig['images']['RGB']

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("/home/chverma/UPV/TFG/face_recognition/examples/obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]


nata_image = face_recognition.load_image_file("/home/chverma/UPV/TFG/data/faces/chverma/neutral/web1461950635617.png")
nata_face_encoding = face_recognition.face_encodings(nata_image)[0]

chris_image = face_recognition.load_image_file("/home/chverma/UPV/TFG/data/faces/chverma/neutral/x11.png")
chris_face_encoding = face_recognition.face_encodings(chris_image)[0]

# Initialize some variables
initialFaces = [obama_face_encoding, nata_face_encoding, chris_face_encoding]
initialNames = ["Obama", "Nata", "Christian"]
face_locations = []
face_encodings = []
face_names = []
countUnknown = 0


class Identity(spade.Agent.Agent):
    class RecvImgFromCoordinator(spade.Behaviour.Behaviour):
        def base64toNumpy(self, obj):
            """
            Returns an 2d array of dlib rects of human faces in a image using the cnn face detector
            :param obj: base64 encoded string
            :return: numpy.darray representing the image
            """
            # Decode from base64 string to numpy
            r = base64.decodestring(obj)
            q = numpy.frombuffer(r, dtype=numpy.uint8)
            if RGBimages:
                q = numpy.lib.stride_tricks.as_strided(q, (480, 640, 3), (1920, 3, 1))
            else:
                q = numpy.lib.stride_tricks.as_strided(q, (480, 640), (640, 1))
            try:
                print "dtype: ", q.dtype
                print "shape: ", q.shape
                print "strides: ", q.strides
                print "type: ", type(q)
                print "sha224Numpy", hashlib.sha224(q).hexdigest()
                print "sha224Base64", hashlib.sha224(obj).hexdigest()
            except Exception as e:
                print "ErrorType:", e
            return q

        def onStart(self):
            print "Starting behaviour RecvFromCameraAgent. . ."
            # Request login to coordinator agent onStart event
            msg = spade.ACLMessage.ACLMessage()
            msg.setPerformative("inform")
            msg.setOntology("login")
            msg.addReceiver(spade.AID.aid("coordinator@"+spadeServerIP, ["xmpp://coordinator@"+spadeServerIP]))
            msg.setContent('identity')
            self.myAgent.send(msg)
            print "Sended login!"

        def _process(self):
            self.msg = None
            try:
                self.msg = self._receive(block=True)
                print "Anything (img?) received from Coordinator agent"
            except Exception as e:
                print "just pException", e

            if self.msg is not None:
                t0 = datetime.datetime.now()

                msgContent = self.msg.getContent()
                img = self.base64toNumpy(msgContent).copy()

                del msgContent
                try:
                    # Find all the faces and face encodings in the current frame of video

                    cv2.imwrite("faces/{}P.png".format(hashlib.sha224(img).hexdigest()), img)
                    img = face_recognition.load_image_file("faces/{}P.png".format(hashlib.sha224(img).hexdigest()))
                    face_locations = face_recognition.face_locations(img)
                    print "face_locations", face_locations
                    face_encodings = face_recognition.face_encodings(img, face_locations)
                    del img
                    face_names = []
                    print "face_encodings", face_encodings
                    for face_encoding in face_encodings:
                        print "Encoding"
                        # See if the face is a match for the known face(s)
                        match = face_recognition.compare_faces(initialFaces, face_encoding)
                        name = "-1"

                        for i in range(len(match)):
                            if match[i]:
                                name = initialNames[i]

                        if name == "-1":
                            name = "Unknown %d" % (countUnknown)
                            initialFaces.append(face_encoding)
                            initialNames.append(name)
                            countUnknown += 1
                        print "NAME: ", name
                        face_names.append(name)

                except Exception as e:
                    print "Error3: ", e

                if len(face_names):
                    # Build the template message
                    msg = spade.ACLMessage.ACLMessage()
                    msg.setPerformative("inform")
                    msg.setOntology("identity")
                    # For each classificator agent send it the distances
                    msg.addReceiver(spade.AID.aid("coordinator@"+spadeServerIP, ["xmpp://coordinator@"+spadeServerIP]))
                    msg.setContent(face_names)
                    self.myAgent.send(msg)

                    t1 = datetime.datetime.now()
                    # print "Sended: ",distances, "time:", (t1-t0)
                    del msg

                    print "Identity sended!"
                del self.msg
            else:
                print "No messages"

    def _setup(self):
        # Create the template for the EventBehaviour: a message from myself
        template = spade.Behaviour.ACLTemplate()
        template.setOntology("img")
        t = spade.Behaviour.MessageTemplate(template)
        # Add the EventBehaviour with its template
        self.addBehaviour(self.RecvImgFromCoordinator(), t)


def main():
    agent = "identity@"+spadeServerIP
    identity = Identity(agent, "secret")
    identity.start()

    alive = True
    while alive:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            alive = False

    identity.stop()
    sys.exit(0)


if __name__ == "__main__":
    main()

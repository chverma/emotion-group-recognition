# This is a _very simple_ example of a web service that recognizes faces in uploaded images.
# Upload an image file and it will check if the image contains a picture of Barack Obama.
# The result is returned as json. For example:
#
# $ curl -F "file=@obama2.jpg" http://127.0.0.1:5001
#
# Returns:
#
# {
#  "face_found_in_image": true,
#  "is_picture_of_obama": true
# }
#
# This example is based on the Flask file upload example: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

# NOTE: This example requires flask to be installed! You can install it with pip:
# $ pip3 install flask

import face_recognition
from flask import Flask, jsonify, request, redirect
import hashlib
import numpy
import pprint
import json
from pymongo import MongoClient
# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# Import config
with open('config.json') as data_file:
    localConfig = json.load(data_file)
app = Flask(__name__)


class Queue:
    def __init__(self, maxElem=5):
        self.load()
        self.maxElem = maxElem

    def load(self):
        self.facesList = numpy.load("initialFaces.npy")
        self.namesList = numpy.load("initialNames.npy")

    def save(self):
        npArray = numpy.asarray(self.facesList)
        numpy.save("initialFaces.npy", npArray)
        npArray = numpy.asarray(self.namesList)
        numpy.save("initialNames.npy", npArray)

    def append(self, elem, name):
        if self.maxElem > len(self.namesList):
            self.facesList = numpy.append(self.facesList, elem, axis=0)
            self.namesList = numpy.append(self.namesList, name, axis=0)
        else:
            self.facesList[:-1] = self.facesList[1:]
            print type(self.facesList[-1])
            print type(elem)
            self.facesList[-1] = numpy.asarray(elem)

            self.namesList[:-1] = self.namesList[1:]
            print type(self.namesList[-1]), self.namesList[-1]
            print type(name), name
            self.namesList[-1] = name

    def getNames(self):
        return self.namesList

    def getFaces(self):
        return self.facesList

    def getLength(self):
        return len(self.namesList)


queue = Queue(localConfig['people']['max_number'])




def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/checkIdentity', methods=['GET', 'POST'])
def upload_image():

    print "We have %d stored faces" % queue.getLength()
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # The image file seems valid! Detect faces and return the result.
            if 'name' in request.args:
                nameProvided = request.args['name']
            else:
                nameProvided = file.filename

            return detect_faces_in_image(file, nameProvided)

    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Wfo are in that picture?</title>
    <h1>Upload a picture and see who are in that picture!</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''


def detect_faces_in_image(file_stream, nameProvided):
    global queue
    facesList = queue.getFaces()
    namesList = queue.getNames()
    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    # Get face encodings for any faces in the uploaded image
    unknown_face_encodings = face_recognition.face_encodings(img)

    face_found = False
    names_found = []
    face_recognised = False
    for unkown_face in unknown_face_encodings:
        match = face_recognition.compare_faces(facesList, unkown_face)
        name = "-1"

        for i in range(len(match)):
            face_found = True
            if match[i]:
                name = namesList[i]
                face_recognised = name

        if name == "-1":
            name = nameProvided
            queue.append([unkown_face], name)

        names_found.append(name)
    # Return the result as json
    result = {
        "face_found_in_image": face_found,
        "names": names_found,
        "face_recognised": face_recognised
    }
    return jsonify(result)


@app.route('/save', methods=['GET'])
def save_model():
    queue.save()
    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Save model</title>
    <h1>Model saved</h1>
    '''

def getData():
    global queue
    if queue is None:
        try:
            initialFaces = numpy.load("initialFaces.npy")  # [localConfig['people']['max_number']]
            initialNames = numpy.load("initialNames.npy")  # [localConfig['people']['max_number']]

        except Exception as e:
            print e
            # Load a sample picture and learn how to recognize it.
            obama_image = face_recognition.load_image_file("/home/chverma/UPV/TFG/face_recognition/examples/obama.jpg")
            obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

            nata_image = face_recognition.load_image_file("/home/chverma/UPV/TFG/data/faces/chverma/neutral/web1461950635617.png")
            nata_face_encoding = face_recognition.face_encodings(nata_image)[0]

            chris_image = face_recognition.load_image_file("/home/chverma/UPV/TFG/data/faces/chverma/neutral/x11.png")
            chris_face_encoding = face_recognition.face_encodings(chris_image)[0]

            initialFaces = [obama_face_encoding, nata_face_encoding, chris_face_encoding]
            initialNames = ["Obama", "Nata", "Christian"]

def getDataMongo(query=None):
    client = MongoClient('localhost', 27017)
    db = client.test_database
    posts = db.posts
    if query is None:
        return posts.find()
    else:
        return posts.find(query)


def insertData(personName, npArray=[]):
    client = MongoClient('localhost', 27017)
    db = client.test_database
    collection = db.test_collection
    if personName is None or personName == '':
        return False
    import datetime
    post = {"name": personName,
            "face_encoding": npArray,
            "insertedAt": datetime.datetime.utcnow()
            }
    posts = db.posts
    post_id = posts.insert_one(post).inserted_id
    return True


@app.route('/checkMongo', methods=['GET', 'POST'])
def check_mongo():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # The image file seems valid! Detect faces and return the result.
            if 'name' in request.args:
                nameProvided = request.args['name']
            else:
                nameProvided = file.filename

            return detect_faces_in_image(file, nameProvided)

    print getData()[:]
    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Mongo check</title>
    <h1>Mongo check ok!</h1>
    ''' + str(getData()[:])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)

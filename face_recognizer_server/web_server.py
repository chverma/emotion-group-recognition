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
from flask import Flask, jsonify, request, redirect, url_for
import hashlib
import numpy
import json

# Import config
with open('../config.json') as data_file:
    localConfig = json.load(data_file)
app = Flask(__name__)
queue = None


class Queue:
    def __init__(self, maxElem=20, cleared=False):
        if not cleared:
            self.load()
        else:
            self.facesList = []
            self.namesList = []
            self.emotionList = []

        self.maxElem = maxElem

    def load(self):
        try:
            self.facesList = numpy.load(localConfig['people']['face_encodings'])
            self.namesList = [x.lower() for x in numpy.load(localConfig['people']['face_names'])]
            self.emotionList = ['No']*len(self.namesList)
            print self.emotionList
        except Exception:
            self.facesList = []
            self.namesList = []
            self.emotionList = []

    def save(self):
        npArray = numpy.asarray(self.facesList)
        numpy.save(localConfig['people']['face_encodings'], npArray)
        npArray = numpy.asarray(self.namesList)
        numpy.save(localConfig['people']['face_names'], npArray)

    def append(self, elem, name):
        numElem = len(self.namesList)
        if numElem == 0:
            self.facesList = numpy.array(elem)
            self.namesList = numpy.asarray([name])
            self.emotionList = ['']
        elif self.maxElem > numElem:
            print type(self.namesList), type(name)
            self.facesList = numpy.append(self.facesList, elem, axis=0)
            self.namesList = numpy.append(self.namesList, [name], axis=0)
            self.emotionList.append('')
        else:
            self.facesList[:-1] = self.facesList[1:]
            print type(self.facesList[-1])
            print type(elem)
            self.facesList[-1] = numpy.asarray(elem)

            self.namesList[:-1] = self.namesList[1:]
            print type(self.namesList[-1]), self.namesList[-1]
            print type(name), name
            self.namesList[-1] = name

            self.emotionList[:-1] = self.emotionList[1:]
            self.emotionList[-1] = ''

    def getNames(self):
        return self.namesList

    def getFaces(self):
        return self.facesList

    def getLength(self):
        return len(self.namesList)

    def getEmotions(self):
        return self.emotionList

    def setEmotion(self, name, emotion):
        try:
            x = self.namesList.index(name)
            self.emotionList[x] = emotion
            return name, emotion
        except ValueError:
            pass



def initQueue(cleared):
    global queue
    queue = Queue(localConfig['people']['max_number'], cleared)


initQueue(False)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in localConfig['images']['allowed_ext']


def adaptName(name, maxLen=9):
    if len(name) >= maxLen:
        return name[:maxLen]
    return name


@app.route('/')
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
            if 'name' in request.form:
                nameProvided = request.form['name']
            else:
                nameProvided = file.filename

            return detect_faces_in_image(file, nameProvided)
        else:
            # Return the result as json
            result = {
                "error": True,
                "msg": "Not valid extension"
            }
            return jsonify(result)

    # If no valid image file was uploaded, show the file upload form:
    namesHeader = '<th scope="col">' + '</th><th scope="col">'.join(map(adaptName, queue.getNames())) + '</th>'
    emotionValues = '<td>' + '</td><td>'.join(queue.getEmotions()) + '</td>'

    return '''
    <!doctype html>
    <head>
    <link rel="stylesheet" href="'''+url_for('static', filename='css/bootstrap.min.css')+'''">
    </head>
    <body>
        <title>Who are in that picture?</title>
        <h1>Upload a picture and see who are in that picture!</h1>
        <h4>Now we have %d face encodings</h4>
        <table class="table-bordered">
            <tr>
                %s
            </tr>
            <tr>
                %s
            </tr>
        </table>

        <form class="form-control" method="POST" enctype="multipart/form-data" action='checkIdentity'>
          <input class="form-inline" type="file" name="file">
          <div class="form-inline" >
            <label>Name if will be saved: </label>
            <input type="text" name="name">
          </div>
          <input class="btn btn-primary btn-lg form-inline" type="submit" value="Upload">
        </form>
    </body>
    ''' % (queue.getLength(), namesHeader, emotionValues)


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
    face_recognised = []
    nFaces = 0
    for unkown_face in unknown_face_encodings:
        match = face_recognition.compare_faces(facesList, unkown_face)
        name = "-1"

        for i in range(len(match)):
            face_found = True
            if match[i]:
                name = namesList[i]
                face_recognised.append(name)

        if name == "-1":
            name = nameProvided
            if name not in queue.getNames():
                queue.append([unkown_face], name)
            else:
                name = '{}{}'.format(name, nFaces)
                queue.append([unkown_face], name)
                nFaces += 1

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


@app.route('/clear', methods=['GET'])
def clear_model():
    initQueue(True)
    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Clear model</title>
    <h1>Model cleared</h1>
    '''


@app.route('/emotion', methods=['GET', 'POST'])
def save_emotion():
    global queue
    print "We have %d stored faces" % queue.getLength()

    name = None
    emotion = None
    if 'name' in request.args:
        name = request.args['name']
    if 'emotion' in request.args:
        emotion = request.args['emotion']

    print "set emotion",  name, emotion
    queue.setEmotion(name.lower(), emotion)
    return name + ' ' + emotion


@app.route('/group-model', methods=['GET', 'POST'])
def group_model():
    # Return the result as json
    result = {
        "nsamples": queue.getLength(),
        "data": [],
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)

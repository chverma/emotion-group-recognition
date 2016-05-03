from classes.StatModel import StatModel
import cv2
import numpy
class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            C=C,
                            gamma=gamma )
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model = cv2.SVM()
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        #Thanks a lot http://stackoverflow.com/questions/8687885/python-opencv-svm-implementation
        #return numpy.float32( [self.model.predict(s) for s in samples]) #last
        #return self.model.predict_all(samples).ravel()
        #return self.model.predict(samples).ravel()
        return self.model.predict(samples);

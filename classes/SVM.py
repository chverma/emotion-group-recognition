from classes.StatModel import StatModel
import utils.defaults as defaults
import cv2
import numpy
class SVM(StatModel):
    def __init__(self, C, gamma): #Gastava C=2.67, gamma=5.383
        if C == None:
            C=2.67
        if gamma == None:
            gamma = 5.383
            
        self.params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            C=float(C),
                            gamma=float(gamma) )
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        #Thanks a lot http://stackoverflow.com/questions/8687885/python-opencv-svm-implementation
        return numpy.float32( [self.model.predict(s) for s in samples]) #last
        #return self.model.predict_all(samples).ravel()
        #return self.model.predict(samples).ravel()
        return self.model.predict(samples);
    def evaluate(self, samples, labels):
        print "type model: ", type(self.model)
        resp =  numpy.float32( [self.model.predict(s) for s in samples])
        #resp = self.model.predict(samples)
        err = (labels != resp).mean()
        print 'error: %.2f %%' % (err*100)

        confusion = numpy.zeros((defaults.CLASS_N, defaults.CLASS_N), numpy.int32)
        for i, j in zip(labels, resp):
            confusion[i, j] += 1
        print 'confusion matrix:'
        print confusion
        print

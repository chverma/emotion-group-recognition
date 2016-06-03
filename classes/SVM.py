from classes.StatModel import StatModel
import utils.defaults as defaults
import cv2
import numpy
class SVM(StatModel):
    def __init__(self, C,gamma): #Gastava C=2.67, gamma=5.383
        if C == None:
            C=2.67
        if gamma == None:
            gamma = 5.383

        self.model = cv2.SVM()
        
    def set_params(self, params=None):
        if not params:
            self.params = dict(kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_NU_SVC,
                            #gamma=3.3750000000000002e-02, ##UNION
                            gamma=5.0625000000000009e-01, ##KDEF
                            #nu=1.0000000000000000e-02) ##UNION
                            nu=8.9999999999999997e-02) ##KDEF
                            
            self.auto=False
        else:
            #kernel_type = cv2.SVM_LINEAR, cv2.SVM_POLY, cv2.SVM_RBF, 
            # default: dict( kernel_type = cv2.SVM_RBF,
            #                svm_type = cv2.SVM_C_SVC)
            self.params = params
            self.auto=True
            
    def train(self, samples, responses):
        #self.model.train(samples, responses, params = self.params)
        varInd = None
        sampleInd =None
        if self.auto:
            self.model.train_auto(samples, responses, varInd, sampleInd, params = self.params)
        else:
            self.model.train(samples, responses, params = self.params)
        

    def train_auto(self, samples, responses):
        self.model.train_auto(samples, responses, params = self.params)
        
    def predict(self, samples):
        #Thanks a lot http://stackoverflow.com/questions/8687885/python-opencv-svm-implementation
        return numpy.float32( [self.model.predict(s) for s in samples]) #last
        #return self.model.predict_all(samples).ravel()
        #return self.model.predict(samples).ravel()
        return self.model.predict(samples);
    def evaluate(self, samples, labels):
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

from classes.StatModel import StatModel
import utils.defaults as defaults
import cv2
import numpy

class SVM(StatModel):
    def __init__(self, C, gamma):  # last used is C=2.67, gamma=5.383
        if C is None:
            C = 2.67
        if gamma is None:
            gamma = 5.383

        self.model = cv2.SVM()

    def set_params(self, params=None, mod=False):
        if not params:
            self.params = dict(kernel_type=cv2.SVM_RBF,
                               svm_type=cv2.SVM_NU_SVC,
                               # svm_type = cv2.SVM_NU_SVR,
                               gamma=1.0000000000000001e-05,  # UNION
                               # gamma=1.0000000000000001e-05,  # KDEF
                               # gamma=1.0000000000000001e-05,  # JAFFE
                               # nu=1.0000000000000000e-02)  # UNION
                               # nu=1.0000000000000000e-02)  # JAFFE
                               nu=1.0000000000000000e-02)  # KDEF

            self.auto = False
        else:
            # kernel_type = cv2.SVM_LINEAR, cv2.SVM_POLY, cv2.SVM_RBF,
            # default: dict( kernel_type = cv2.SVM_RBF,
            #                svm_type = cv2.SVM_C_SVC)
            self.params = params
            self.auto = True

        if mod:
            self.params = dict(kernel_type=cv2.SVM_RBF,
                               svm_type=cv2.SVM_NU_SVC,
                               gamma=1.0000000000000001e-05,
                               nu=2.9999999999999999e-02)
            self.auto = False

    def train(self, samples, responses):
        # self.model.train(samples, responses, params = self.params)
        varInd = None
        sampleInd = None
        if self.auto:
            self.model.train_auto(samples, responses, varInd, sampleInd, params=self.params)
        else:
            self.model.train(samples, responses, params=self.params)

    def train_auto(self, samples, responses):
        self.model.train_auto(samples, responses, params=self.params)

    def predict(self, samples):
        # Thanks a lot http://stackoverflow.com/questions/8687885/python-opencv-svm-implementation
        return numpy.float32([self.model.predict(s) for s in samples])  # last
        # return self.model.predict_all(samples).ravel()
        # return self.model.predict(samples).ravel()
        return self.model.predict(samples)

    def evaluate(self, samples, labels, resp=None):
        if resp is None:
            resp = numpy.float32([self.model.predict(s) for s in samples])
        # resp = self.model.predict(samples)
        err = (labels != resp).mean()
        print 'error: %.2f %%' % (err*100)

        confusion = numpy.zeros((defaults.CLASS_N, defaults.CLASS_N), numpy.int32)
        for i, j in zip(labels, resp):
            confusion[i, j] += 1
        print 'confusion matrix:'
        print confusion
        print
        return confusion

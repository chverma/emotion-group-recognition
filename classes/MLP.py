import numpy as np
import utils.defaults as defaults
import cv2
from classes.StatModel import StatModel


class MLP(StatModel):
    def __init__(self, nh, nh2):
        self.model = cv2.ANN_MLP()
        if nh is None:
            self.nhidden = 100
        else:
            self.nhidden = int(nh)
        if nh2 is None:
            nh2 = nh
        self.nhidden2 = int(nh2)

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses).reshape(-1, defaults.CLASS_N)
        layer_sizes = np.int32([var_n, self.nhidden, self.nhidden2, defaults.CLASS_N])
        # layer_sizes = np.int32([var_n, self.nhidden, defaults.CLASS_N])
        print "layer_sizes", layer_sizes
        self.model.create(layer_sizes)

        # CvANN_MLP_TrainParams::BACKPROP,0.001
        params = dict(term_crit=(cv2.TERM_CRITERIA_COUNT, 300, 0.01),
                      train_method=cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                      bp_dw_scale=0.001,
                      bp_moment_scale=0.0)
        self.model.train(samples, np.float32(new_responses), None, params=params)

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)

    def getActivationValues(self, samples, lab="hola"):
        ret, resp = self.model.predict(samples)
        import numpy
        res = []
        for s in resp:
            maxVals = sorted(s, reverse=True)[0:2]
            cValues = [numpy.where(s == maxVals[0])[0], numpy.where(s == maxVals[1])[0]]
            res.append(cValues)

        return res

    def evaluate(self, samples, labels, resp=None):
        if resp is None:
            resp = self.predict(samples)
        # resp = self.model.predict(samples)
        err = (labels != resp).mean()
        print 'error: %.2f %%' % (err*100)

        confusion = np.zeros((defaults.CLASS_N, defaults.CLASS_N), np.int32)
        for i, j in zip(labels, resp):
            confusion[i, j] += 1
        print 'confusion matrix:'
        print confusion
        print
        return confusion

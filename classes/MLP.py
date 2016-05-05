import numpy as np
import utils.defaults as defaults
import cv2
from classes.StatModel import StatModel
class MLP(StatModel):
    def __init__(self, nh, dummy):
        self.model = cv2.ANN_MLP()
        if nh==None:
            self.nhidden=100
        else:
            self.nhidden=nh

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses).reshape(-1, defaults.CLASS_N)
        layer_sizes = np.int32([var_n, self.nhidden, self.nhidden, defaults.CLASS_N])
        print "layer_sizes", layer_sizes
        self.model.create(layer_sizes)
        
        # CvANN_MLP_TrainParams::BACKPROP,0.001
        params = dict( term_crit = (cv2.TERM_CRITERIA_COUNT, 300, 0.01),
                       train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP, 
                       bp_dw_scale = 0.001,
                       bp_moment_scale = 0.0 )
        self.model.train(samples, np.float32(new_responses), None, params = params)

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        print "predictRESULT"
        print resp[0]
        print resp[1]
        print resp[3]
        print "ARGMAX",resp.argmax(-1)[0:20]
        return resp.argmax(-1)
    def evaluate(self, samples, labels):
        print

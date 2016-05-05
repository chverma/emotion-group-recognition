import numpy as np
import utils.defaults as defaults
import cv2
from classes.StatModel import StatModel
class Boost(StatModel):
    def __init__(self,dummy,dummy1):
        self.model = cv2.Boost()
    
    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_samples = self.unroll_samples(samples)
        new_responses = self.unroll_responses(responses)
        var_types = np.array([cv2.CV_VAR_NUMERICAL] * var_n + [cv2.CV_VAR_CATEGORICAL, cv2.CV_VAR_CATEGORICAL], np.uint8)
        #CvBoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0 )
        params = dict(max_depth=5) #, use_surrogates=False)
        self.model.train(new_samples, cv2.CV_ROW_SAMPLE, new_responses, varType = var_types, params=params)

    def predict(self, samples):
        new_samples = self.unroll_samples(samples)
        pred = np.array( [self.model.predict(s, returnSum = True) for s in new_samples] )
        pred = pred.reshape(-1, defaults.CLASS_N).argmax(1)
        return pred
    
    def evaluate(self, samples, labels):
        print

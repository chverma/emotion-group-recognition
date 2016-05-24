import numpy as np
from classes.StatModel import StatModel
import cv2
class RTrees(StatModel):
    def __init__(self,maxDepth, dummy1):
        self.model = cv2.RTrees()
	self.maxDepth = maxDepth

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        var_types = np.array([cv2.CV_VAR_NUMERICAL] * var_n + [cv2.CV_VAR_CATEGORICAL], np.uint8)
        #CvRTParams(10,10,0,false,15,0,true,4,100,0.01f,CV_TERMCRIT_ITER));
        params = dict(max_depth=int(self.maxDepth) )
        self.model.train(samples, cv2.CV_ROW_SAMPLE, responses, varType = var_types, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples] )
    def evaluate(self, samples, labels):
        print

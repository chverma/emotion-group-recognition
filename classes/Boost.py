import numpy as np
import utils.defaults as defaults
import cv2
# Note:
# In case of LogitBoost and Gentle AdaBoost, each weak predictor is a regression tree, rather than a classification tree
# Even in case of Discrete AdaBoost and Real AdaBoost, the CvBoostTree::predict return value (CvDTreeNode::value) is not
# an output class label. A negative value "votes" for class #0, a positive value - for class #1. The votes are weighted.
# The weight of each individual tree may be increased or decreased using the method CvBoostTree::scale.
from classes.StatModel import StatModel


class Boost(StatModel):
    def __init__(self, maxDepth, dummy1):
        self.model = cv2.Boost()
        # Params:
        #   boost_type: DISCRETE, REAL, LOGIT, GENTLE
        #   weak_count: number of weak classificators
        self.params = dict(max_depth=int(maxDepth))  # , use_surrogates=False)

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_samples = self.unroll_samples(samples)
        new_responses = self.unroll_responses(responses)
        varTypes = np.array([cv2.CV_VAR_NUMERICAL] * var_n + [cv2.CV_VAR_CATEGORICAL, cv2.CV_VAR_CATEGORICAL], np.uint8)
        # CvBoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0 )

        self.model.train(new_samples, cv2.CV_ROW_SAMPLE, new_responses, varType=varTypes, params=self.params)

    def predict(self, samples):
        new_samples = self.unroll_samples(samples)
        pred = np.array([self.model.predict(s, returnSum=True) for s in new_samples])
        pred = pred.reshape(-1, defaults.CLASS_N).argmax(1)
        return pred

    def evaluate(self, samples, labels):
        resp = self.predict(samples)
        err = (labels != resp).mean()
        print 'error: %.2f %%' % (err*100)

        confusion = np.zeros((defaults.CLASS_N, defaults.CLASS_N), np.int32)
        for i, j in zip(labels, resp):
            confusion[i, j] += 1
        print 'confusion matrix:'
        print confusion
        print
        return confusion

    def getVotes(self, samples):
        new_samples = self.unroll_samples(samples)
        pred = np.array([self.model.predict(s, returnSum=True) for s in new_samples])
        pred = pred.reshape(-1, defaults.CLASS_N).argmax(1)
        print pred
        emm = [0]*7
        for i in xrange(7):
            emm[i] = len(pred[pred == i])
        print emm
        return pred

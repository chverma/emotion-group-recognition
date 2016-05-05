from classes.StatModel import StatModel
import numpy
import utils.defaults as defaults
import cv2
class KNearest(StatModel):
    def __init__(self, k, dummy):
        if k==None:
            self.k = 1
        else:
            self.k = int(k)
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        self.model = cv2.KNearest()
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, self.k)
        return results.ravel()
        
    def evaluate(self, samples, labels):
        #resp =  numpy.float32( [self.model.predict(s) for s in samples])
        resp = self.predict(samples)
        err = (labels != resp).mean()
        print 'error: %.2f %%' % (err*100)

        confusion = numpy.zeros((defaults.CLASS_N, defaults.CLASS_N), numpy.int32)
        for i, j in zip(labels, resp):
            confusion[i, j] += 1
        print 'confusion matrix:'
        print confusion
        print

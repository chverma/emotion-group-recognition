import numpy
class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)
    def evaluate(self, samples, labels):
        resp = self.model.predict(samples)
        err = (labels != resp).mean()
        print 'error: %.2f %%' % (err*100)

        confusion = numpy.zeros((defaults.CLASS_N, defaults.CLASS_N), np.int32)
        for i, j in zip(labels, resp):
            confusion[i, j] += 1
        print 'confusion matrix:'
        print confusion
        print
    

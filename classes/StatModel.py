import numpy
import utils.defaults as defaults
class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)
    
    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape
        new_samples = numpy.zeros((sample_n * defaults.CLASS_N, var_n+1), np.float32)
        new_samples[:,:-1] = numpy.repeat(samples, defaults.CLASS_N, axis=0)
        new_samples[:,-1] = numpy.tile(np.arange(defaults.CLASS_N), sample_n)
        return new_samples
    
    def unroll_responses(self, responses):
        sample_n = len(responses)
        new_responses = numpy.zeros(sample_n*defaults.CLASS_N, numpy.int32)
        resp_idx = numpy.int32( responses + numpy.arange(sample_n)*defaults.CLASS_N )
        new_responses[resp_idx] = 1
        
        return new_responses

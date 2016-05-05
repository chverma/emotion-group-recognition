from classes.StatModel import StatModel
import cv2
import numpy
import utils.defaults as defaults
class ANN(StatModel):
    def __init__(self,nh):
        self.model = cv2.ANN_MLP()
        self.nhidden = nh

    '''def train(self, samples, responses):
        ninputs=len(samples[0])
        noutput=defaults.CLASS_N
        # Create an array of desired layer sizes for the neural network
        layers = numpy.array([ninputs, self.nhidden, noutput])
        print "layers", layers
        self.model=cv2.ANN_MLP(layers)
        # Some parameters for learning.  Step size is the gradient step size
        # for backpropogation.
        step_size = 0.01
        # Momentum can be ignored for this example because we use backProp
        momentum = 0.0

        # Max steps of training
        nsteps = 10000

        # Error threshold for halting training
        max_err = 0.0001

        # When to stop: whichever comes first, count or error
        condition = cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS

        # Tuple of termination criteria: first condition, then # steps, then
        # error tolerance second and third things are ignored if not implied
        # by condition
        criteria = (condition, nsteps, max_err)

        # params is a dictionary with relevant things for NNet training.
        self.params = dict( term_crit = criteria, 
               train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP, 
               bp_dw_scale = step_size, 
               bp_moment_scale = momentum )

        # Train our network
        targets = -1 * numpy.ones( (len(samples), noutput), 'float' )
        
        for i in xrange(len(targets)):
            targets[i][responses[i]]=1
        
        print "LENinputs",len(samples), len(samples[0])
        print "LENtargets",len(targets), len(targets[0])
        #
        return  self.model.train(samples, targets, None, params=self.params)
    '''
    '''def train(self, samples, labels):
        ninputs=len(samples[0])
        noutput=defaults.CLASS_N
        
        # Create an array of desired layer sizes for the neural network
        layers = numpy.array([ninputs, self.nhidden, self.nhidden, noutput])
        
        self.model = cv2.ANN_MLP()
        self.model.create(layers)
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001)
        criteria2 = (cv2.TERM_CRITERIA_COUNT, 100, 0.001)
        params = dict(term_crit = criteria,
                       train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                       bp_dw_scale = 0.1,
                       bp_moment_scale = 0.0 )
        nSamples = len(samples)
        labels_matrix = -1*numpy.ones((nSamples, defaults.CLASS_N), 'float')
        for i in range(nSamples):
            labels_matrix[i, labels[i]] = 1
        
        print 'Training MLP ... with layers', layers
        print labels_matrix
        return self.model.train(samples, labels_matrix, None, params = params)
    '''
    def train(self, samples, labels):
        ninputs=len(samples[0])
        noutput=defaults.CLASS_N
        nSamples  = len(samples)

        new_responses = self.unroll_responses(labels).reshape(-1, defaults.CLASS_N)
        
        # Create an array of desired layer sizes for the neural network
        layers = numpy.array([ninputs, self.nhidden, self.nhidden, noutput])
        self.model.create(layers)
        
        # CvANN_MLP_TrainParams::BACKPROP,0.001
        params = dict( term_crit = (cv2.TERM_CRITERIA_COUNT, 10000, 0.01),
                       train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP, 
                       bp_dw_scale = 0.001,
                       bp_moment_scale = 0.0 )
            
        print "num_iter",self.model.train(samples, numpy.float32(new_responses), None, params = params)

    def predict(self, samples):
        #return self.model.predict(samples,predictionArray)
        ret, resp = self.model.predict(samples)
        print "predictRESULT",resp
        
        print "predictRESMAX",resp.argmax(-1)
        exit()
        return resp.argmax(-1)
    def evaluate(self, samples, labels):
        
        
        #targets = -1 * numpy.ones( (len(samples), defaults.CLASS_N), 'float' )
        #targets = -1 * numpy.ones( (len(samples), len(samples)), 'float' )
        nSamples = len(samples)
        labels_matrix = -1*numpy.ones((nSamples, defaults.CLASS_N), 'float')
        for i in range(nSamples):
            labels_matrix[i, labels[i]] = 1
        

        # See how the network did.
        #predictions  = self.model.predict(samples)
        #print "predictions[0]", predictions[0], "correctlabel:", labels[0]
        #print "predictions[1]", predictions[1], "correctlabel:", labels[1]
        #print "predictions[2]", predictions[2], "correctlabel:", labels[2]
        #print "predictions[3]", predictions[3], "correctlabel:", labels[3]
        #print 'testing...'
        prediction  = self.model.predict(samples)
        print "prediction",prediction
        print "labels",labels
        test_rate  = numpy.mean(prediction == labels[:])

        print 'test rate: %f' % (test_rate*100)
        '''
        # Compute sum of squared errors
        sse = numpy.sum( (labels_matrix - predictions)**2 )
        
        # Compute # correct
        true_labels = numpy.argmax( labels_matrix, axis=1)
        pred_labels = numpy.argmax( predictions, axis=1)
        num_correct = numpy.sum( labels_matrix == pred_labels )

        print "true_labels", true_labels
        print "pred_labels", pred_labels

        print 'num_correct', num_correct
        print 'sum sq. err:', sse
        print 'accuracy:', float(num_correct) / nSamples
        '''
        '''print "labels ",labels, len(labels)
        ret, resp = self.model.predict(samples)
        print "resp ",resp, len(resp)
        print "ret ",ret
        prediction = resp.argmax(-1)
        print 'Prediction:', prediction, len(prediction)
        true_labels = labels.argmax(-1)
        print 'True labels:', true_labels

        print 'Testing...'
        print prediction == true_labels
        train_rate = numpy.mean(prediction == true_labels)
        print 'Test rate: %f:' % (train_rate*100)'''
        
        print "--------------------------------------"
        print 
        print

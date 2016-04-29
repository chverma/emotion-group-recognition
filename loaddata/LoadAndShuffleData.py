import csv
import numpy
import datetime

import utils.defaults as defaults
from loaddata.processImage import process_image

class LoadAndShuffleData():
    def __init__(self):
        return 
    def loadTrainingData(self):
        """
        load training data
        """
        # create a list for filenames of happy pictures
        happyfiles = []
        with open(defaults.happy_csv, 'rb') as csvfile:
            for rec in csv.reader(csvfile, delimiter='	'):
                happyfiles += rec

        # create a list for filenames of neutral pictures
        neutralfiles = []
        with open(defaults.neutral_csv, 'rb') as csvfile:
            for rec in csv.reader(csvfile, delimiter='	'):
                neutralfiles += rec

        # create a list for filenames of disgust pictures
        disgustfiles = []
        with open(defaults.disgust_csv, 'rb') as csvfile:
            for rec in csv.reader(csvfile, delimiter='	'):
                disgustfiles += rec
        
        # create a list for filenames of disgust pictures
        fearfiles = []
        with open(defaults.fear_csv, 'rb') as csvfile:
            for rec in csv.reader(csvfile, delimiter='	'):
                fearfiles += rec
        
        # create a list for filenames of disgust pictures
        surprisedfiles = []
        with open(defaults.surprised_csv, 'rb') as csvfile:
            for rec in csv.reader(csvfile, delimiter='	'):
                surprisedfiles += rec
                
        # create a list for filenames of disgust pictures
        sadfiles = []
        with open(defaults.sad_csv, 'rb') as csvfile:
            for rec in csv.reader(csvfile, delimiter='	'):
                sadfiles += rec
        

        # N x dim matrix to store the vectorized data (aka feature space)       
        phi = numpy.zeros((len(happyfiles) + len(neutralfiles) + len(disgustfiles)+len(fearfiles) + len(surprisedfiles) + len(sadfiles), defaults.dim),numpy.float32)

        # 1 x N vector to store binary labels of the data: 1 for smile and 0 for neutral
        labels = []

        # load happy data
        for idx, filename in enumerate(happyfiles):
            phi[idx] = process_image(defaults.happy_imgs + filename)
            labels.append(0)
        print "loaded happy"
        
        # load neutral data    
        offset = idx + 1
        for idx, filename in enumerate(neutralfiles):
            phi[idx + offset] = process_image(defaults.neutral_imgs + filename)
            labels.append(1)
        print "loaded neutral"

        # load disgust data
        offset = offset+idx + 1
        for idx, filename in enumerate(disgustfiles):
            phi[idx + offset] = process_image(defaults.disgust_imgs + filename)
            labels.append(2)
        
        # load fear data
        offset = offset+idx + 1
        for idx, filename in enumerate(fearfiles):
            phi[idx + offset] = process_image(defaults.fear_imgs + filename)
            labels.append(3)
            
        # load surprised data
        offset = offset+idx + 1
        for idx, filename in enumerate(surprisedfiles):
            phi[idx + offset] = process_image(defaults.surprised_imgs + filename)
            labels.append(4)
            
        # load sad data
        offset = offset+idx + 1
        for idx, filename in enumerate(sadfiles):
            phi[idx + offset] = process_image(defaults.sad_imgs + filename)
            labels.append(5)

        return phi , numpy.asarray(labels)
    def getData(self):
        print "Obtaining data..."
        t1 = datetime.datetime.now()
        data, labels = self.loadTrainingData()
        t2 = datetime.datetime.now()
        print "Total time loading:", (t2-t1)
    
        print("Total dataset size:")
        print("n_samples: %d" % len(data))
        print("n_features: %d" % defaults.dim)
        print("n_classes: %d" % defaults.CLASS_N)
        
        ## shuffle data
        rand = numpy.random.RandomState(321)
        shuffle = rand.permutation(len(data))
        data, labels = data[shuffle], labels[shuffle]

        print "higth:%d; width:%d"%(len(data),len(data[0]))
        train_n = int(0.9*len(data))
        print "training_n:%d; total_n:%d"%(train_n,len(data))
         
        samples_train, samples_test = numpy.split(data, [train_n])
        labels_train, labels_test = numpy.split(labels, [train_n])
        
        return samples_train, labels_train, samples_test, labels_test

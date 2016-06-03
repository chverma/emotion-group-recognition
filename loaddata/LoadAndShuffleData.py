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
        
        # create a list for filenames of disgust pictures
        angryfiles = []
        with open(defaults.angry_csv, 'rb') as csvfile:
            for rec in csv.reader(csvfile, delimiter='	'):
                angryfiles += rec
                
        # N x dim matrix to store the vectorized data (aka feature space)       
        phi = numpy.zeros((len(happyfiles) + len(neutralfiles) + len(disgustfiles)+len(fearfiles) + len(surprisedfiles) + len(sadfiles)+ len(angryfiles), defaults.dim),numpy.float32)

        # 1 x N vector to store binary labels of the data: 1 for smile and 0 for neutral
        labels = []
        rem_indx=[]
        # load happy data
        for idx, filename in enumerate(happyfiles):
            distances = process_image(defaults.happy_imgs + filename)
            if distances==None:
                rem_indx.append(idx)
                
            phi[idx] = distances
            labels.append(0)
        print "loaded happy"
        
        # load neutral data    
        offset = idx + 1
        for idx, filename in enumerate(neutralfiles):
            distances=process_image(defaults.neutral_imgs + filename)
            if distances==None:
                rem_indx.append(idx+offset)
            phi[idx + offset] = distances
            labels.append(1)
        print "loaded neutral"

        # load disgust data
        offset = offset+idx + 1
        for idx, filename in enumerate(disgustfiles):
            distances=process_image(defaults.disgust_imgs + filename)
            if distances==None:
                rem_indx.append(idx+offset)
            phi[idx + offset] = distances
            labels.append(2)
        print "loaded disgust"
        
        # load fear data
        offset = offset+idx + 1
        for idx, filename in enumerate(fearfiles):
            distances=process_image(defaults.fear_imgs + filename)
            if distances==None:
                rem_indx.append(idx+offset)
            phi[idx + offset] = distances
            labels.append(3)
        print "loaded fear"
        
        # load surprised data
        offset = offset+idx + 1
        for idx, filename in enumerate(surprisedfiles):
            distances=process_image(defaults.surprised_imgs + filename)
            if distances==None:
                rem_indx.append(idx+offset)
            phi[idx + offset] = distances
            labels.append(4)
        print "loaded surprised"
        
        # load sad data
        offset = offset+idx + 1
        for idx, filename in enumerate(sadfiles):
            distances=process_image(defaults.sad_imgs + filename)
            if distances==None:
                rem_indx.append(idx+offset)
            phi[idx + offset] = distances
            labels.append(5)
        print "loaded sad"
        
        # load angry data
        offset = offset+idx + 1
        for idx, filename in enumerate(angryfiles):
            distances=process_image(defaults.angry_imgs + filename)
            if distances==None:
                rem_indx.append(idx+offset)
            phi[idx + offset] = distances
            labels.append(6)
        print "loaded angry"
        
        rem = 0 
        if len(rem_indx)>0:
            print "rem_indx", numpy.asarray(rem_indx)
            
            for i in rem_indx:
                del labels[i-rem]
                phi = numpy.delete(phi, (i-rem), axis=0)
                rem=rem+1

        return phi , numpy.asarray(labels)
        
    def shuffleData(self, data, labels):
        ## shuffle data
        nsamples = len(data)
        #rand = numpy.random.RandomState(321)
        shuffle = numpy.random.permutation(nsamples)
        data, labels = data[shuffle], labels[shuffle]
        
        train_n = int(0.9*nsamples)

        samples_train, samples_test = numpy.split(data, [train_n])
        labels_train, labels_test = numpy.split(labels, [train_n])
        
        return samples_train, labels_train, samples_test, labels_test
        
    def getData(self):
        print "Obtaining data..."
        t1 = datetime.datetime.now()
        data, labels = self.loadTrainingData()
        nsamples = len(data)
        numpy.save(defaults.file_dataset, data)
        numpy.save(defaults.file_labels, labels)
        t2 = datetime.datetime.now()
        print "Total time loading:", (t2-t1)
    
        print("Total dataset size:")
        print("n_samples: %d" % nsamples)
        print("n_features: %d" % len(data[0]))
        print("n_classes: %d" % defaults.CLASS_N)
        

        
        

        return data, labels

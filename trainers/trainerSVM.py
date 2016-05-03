from classes.SVM import SVM
import cv2
class trainerSVM():
    def __init__(self):
        self.model=None

    def train(self, samples_train, labels_train):
        print 'training SVM...'
        self.model = SVM(C=2.67, gamma=5.383)
        self.model.train(samples_train, labels_train)
        
    def save(self,destFile):
        self.model.save(destFile)
    
    def evaluate(self,samples_test, labels_test):
        return self.model.evaluate(samples_test, labels_test)
        
    def predict(self, samples):
        return self.model.predict(samples)
        
    def load(self, inputFile):
        self.model = cv2.SVM()
        return self.model.load(inputFile)

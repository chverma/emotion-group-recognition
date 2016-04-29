class KNearest():
    def __init__(self):
        self.model=None

    def train(self, samples_train, labels_train):
        print 'training KNearest...'
        self.model = KNearest(k=2)
        self.model.train(samples_train, labels_train)
        
    def save(self,destFile):
        self.model.save(destFile)
    
    def evaluate(self, samples_test, labels_test):
        return self.model.evaluate(samples_test, labels_test)
        
    def predict(self, samples):
        return self.model.predict(samples)

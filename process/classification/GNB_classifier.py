import sys,os
sys.path.append('C:\\Users\\806707\\Documents\\Python\\AutomatedLearning')
from sklearn.naive_bayes import GaussianNB
from model.Classifier import Classifier
from model.dataset import Dataset

class GNB(Classifier):
    def __init__(self,**kwargs):
        
        super().__init__()
        self.name = 'GNB'
        self.hyperparameters = {}
        self.clf = GaussianNB()
    
    def __str__(self):
        return 'GNB'

    
    def train(self,dataset):        
        super().train(dataset,model=self.clf)

    def test(self,dataset):
        super().test(dataset,model = self.clf)

    def save(self,dataset):
        super().save_results(dataset,classifier_model = self)
    

if(__name__ == '__main__'):
    clf = GNB()
    dat = Dataset(path=r'C:\Users\806707\Downloads\hill.csv')
    print('Training')
    clf.train(dat)
    print('Testing')
    clf.test(dat)
    print('Saving')
    clf.save(dat)

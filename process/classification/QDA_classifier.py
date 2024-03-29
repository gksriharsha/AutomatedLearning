import sys,os
sys.path.append('C:\\Users\\806707\\Documents\\Python\\AutomatedLearning')

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from model.Classifier import Classifier
from model.dataset import Dataset

class QDA(Classifier):
    def __init__(self,**kwargs):
        
        super().__init__()
        self.name = 'QDA'
        self.hyperparameters = {}
        self.clf = QuadraticDiscriminantAnalysis()
    
    def __str__(self):
        return 'QDA'

    
    def train(self,dataset):        
        super().train(dataset,model=self.clf)

    def test(self,dataset):
        super().test(dataset,model = self.clf)

    def save(self,dataset):
        super().save_results(dataset,classifier_model = self)
    

if(__name__ == '__main__'):
    clf = QDA()
    dat = Dataset(path=r'C:\Users\806707\Downloads\hill.csv')
    print('Training')
    clf.train(dat)
    print('Testing')
    clf.test(dat)
    print('Saving')
    clf.save(dat)

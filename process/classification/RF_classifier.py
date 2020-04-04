import sys,os
sys.path.append('C:\\Users\\806707\\Documents\\Python\\AutomatedLearning')
from model.Classifier import Classifier
from sklearn.ensemble import RandomForestClassifier  
from model.dataset import Dataset
class RF(Classifier):
    def __init__(self,**kwargs):
        Trees = kwargs.pop('Trees',100)
        max_depth = kwargs.pop('max_depth',None)
        
        super().__init__()
        self.name = 'RF'
        self.hyperparameters = {}
        self.hyperparameters['Trees'] = Trees
        self.hyperparameters['max_depth'] = max_depth
        self.clf = RandomForestClassifier(max_depth=max_depth,n_estimators=Trees)
    
    def __str__(self):
        return 'KNN'

    
    def train(self,dataset):        
        super().train(dataset,model=self.clf)

    def test(self,dataset):
        super().test(dataset,model = self.clf)

    def save(self,dataset):
        super().save_results(dataset,classifier_model = self)
    

if(__name__ == '__main__'):
    clf = RF()
    dat = Dataset(path=r'C:\Users\806707\Downloads\hill.csv')
    print('Training')
    clf.train(dat)
    print('Testing')
    clf.test(dat)
    print('Saving')
    clf.save(dat)

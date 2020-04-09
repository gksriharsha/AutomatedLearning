import sys,os
sys.path.append('C:\\Users\\806707\\Documents\\Python\\AutomatedLearning')
from model.Classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier  
from model.dataset import Dataset
class KNN(Classifier):
    def __init__(self,**kwargs):
        K = kwargs.pop('K',5)
        weights = kwargs.pop('weights','uniform')
        algorithm = kwargs.pop('algorithm','auto')
        super().__init__()
        self.name = 'KNN'
        self.hyperparameters = {}
        self.hyperparameters['K'] = K
        self.hyperparameters['weights'] = weights
        self.hyperparameters['algorithm'] = algorithm
        self.clf = KNeighborsClassifier(n_neighbors=K,weights=weights,algorithm=algorithm)
    
    def __str__(self):
        return 'KNN'

    
    def train(self,dataset):        
        super().train(dataset,model=self.clf)

    def test(self,dataset):
        super().test(dataset,model = self.clf)

    def save(self,dataset):
        super().save_results(dataset,classifier_model = self)
    

if(__name__ == '__main__'):
    clf = KNN(K=3)
    dat = Dataset(path=r'C:\Users\806707\Downloads\hill.csv')
    print('Training')
    clf.train(dat)
    print('Testing')
    clf.test(dat)
    print('Saving')
    clf.save(dat)

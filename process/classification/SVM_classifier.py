import sys,os
sys.path.append('C:\\Users\\806707\\Documents\\Python\\AutomatedLearning')
from sklearn.svm import SVC
from model.Classifier import Classifier
from model.dataset import Dataset

class SVM(Classifier):
    def __init__(self,**kwargs):
        C = kwargs.pop('C',1)
        kernel = kwargs.pop('Kernel', 'rbf')
        degree = kwargs.pop('Degree',3)
        tolerance = kwargs.pop('Tolerance',0.0001)
        super().__init__()
        self.name = 'SVM'
        self.hyperparameters = {}
        self.hyperparameters['C'] = C
        self.hyperparameters['Kernel'] = kernel
        self.hyperparameters['Degree'] = degree
        self.hyperparameters['Tolerance'] = tolerance
        self.clf = SVC(tol=tolerance,C=C,degree=degree,kernel=kernel,gamma='auto')
    
    def __str__(self):
        return 'SVM'

    
    def train(self,dataset):        
        super().train(dataset,model=self.clf)

    def test(self,dataset):
        super().test(dataset,model = self.clf)

    def save(self,dataset):
        super().save_results(dataset,classifier_model = self)
    

if(__name__ == '__main__'):
    clf = SVM(C=3)
    dat = Dataset(path=r'C:\Users\806707\Downloads\hill.csv')
    print('Training')
    clf.train(dat)
    print('Testing')
    clf.test(dat)
    print('Saving')
    clf.save(dat)

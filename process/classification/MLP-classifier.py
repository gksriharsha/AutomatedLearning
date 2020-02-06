import sys,os
sys.path.append('C:\\Users\\806707\\Documents\\Python\\AutomatedLearning')
from model.Classifier import Classifier
from sklearn.neural_network import MLPClassifier
from model.dataset import Dataset
class MLP(Classifier):
    def __init__(self,**kwargs):
        hidden_layer = kwargs.pop('Hidden_layer_neurons',None)
        solver = kwargs.pop('solver','lbfgs')
        activation = kwargs.pop('Activation fn','tanh')
        max_iterations = kwargs.pop('Iterations',10000)
        tolerance = kwargs.pop('Tolerance',None)
        
        super().__init__()
        self.name = 'MLP'
        self.hyperparameters = {}
        self.hyperparameters['Hidden layer neurons'] = hidden_layer
        self.hyperparameters['Solver'] = solver
        self.hyperparameters['Activation fn'] = activation
        self.hyperparameters['Max_Iterations'] = max_iterations
        self.hyperparameters['Tolerance'] = tolerance
        self.clf = MLPClassifier(solver=solver, max_iter=max_iterations,hidden_layer_sizes=tuple(hidden_layer),activation=activation,tol=tolerance)
    
    def __str__(self):
        return 'MLP'

    
    def train(self,dataset):        
        super().train(dataset,model=self.clf)

    def test(self,dataset):
        super().test(dataset,model = self.clf)

    def save(self,dataset):
        super().save_results(dataset,classifier_model = self)
    

if(__name__ == '__main__'):

    clf = MLP(Tolerance=0.00001,Hidden_layer_neurons = [15,7,2])
    dat = Dataset(path=r'C:\Users\806707\Downloads\hill.csv')
    print('Training')
    clf.train(dat)
    print('Testing')
    clf.test(dat)
    print('Saving')
    clf.save(dat)

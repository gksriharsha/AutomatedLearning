from model.Classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier  
from model.Dataset import Dataset
class KNN(Classifier):
    def __init__(self,**kwargs):
        K = kwargs.pop('K',None)
        weights = kwargs.pop('weights',None)
        self.name = 'KNN'
        self.hyperparameters = {}
        self.hyperparameters['K'] = K
        self.hyperparameters['weights'] = weights
        self.clf = KNeighborsClassifier(n_neighbors=K,weights=weights)
    
    def __str__(self):
        return 'KNN'

    @Classifier.time_watch
    def train(self,dataset):
        self.clf.fit(dataset.X_train,dataset.y_train)

    def test(self,dataset):
        clf_props = super.test(dataset,model = self.clf)

    def save(self,dataset):
        super.save_results(dataset,classifier_model = self)
    

if(__name__ == '__main__'):
    clf = KNN(K=3)
    #dataset = Da

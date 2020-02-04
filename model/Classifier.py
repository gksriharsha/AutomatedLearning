from ttictoc import TicToc
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import csv
import uuid
class Classifier():
    _time = 0
    """  def base_decorator(self,function):
        def wrapper():
            self.model = function()
        return wrapper """
    def __init__(self):
        try:
            f = open('../results/all_results.csv','r')
            f.close()
        except:           
            row = ['UUID','Classifier','Rows','Columns','Classes','Accuracy','F1 Score','Precision','Recall','time']
            with open('../results/all_results.csv','w') as file:
                writer = csv.writer(file)
                writer.writerow(row)
    
    def time_watch(self, function):        
        def timer():
            global _time
            t = TicToc()
            t.tic()
            function()
            t.toc()
            _time = t.elapsed
        return timer
    
    """ def metrics_watch(self,function):
        def metrics():
            function() """

    @property
    def time(self):
        return self._time
    
    def test(self,dataset,model):
        self.clf = model
        pred = self.clf.predict(dataset.X_test)
        self.results = {}
        self.results['accuracy'] = accuracy_score(dataset.y_test,pred)
        self.results['f1_score'] = f1_score(dataset.y_test,pred)
        self.results['precision'] = precision_score(dataset.y_test,pred)
        self.results['recall'] = recall_score(dataset.y_test,pred)
    
    def save_results(self,dataset,classifier_model):    
        name = classifier_model.name
        info = dataset()
        id = uuid.uuid4()
        row = [str(id),name,info['rows'],info['columns'],info['Unique classes']]+list(self.results.values())+[_time]   
        with open('../results/all_results.csv','a',newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
        
        try:
            with open(f'../results/{name}','a',newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
        except:
            with open(f'../results/{name}','w',newline='') as file:
                writer = csv.writer(file)
                row = ['UUID']+list(classifier_model.hyperparameters.keys())
                writer.writerow(row)
                writer.writerow([id]+list(classifier_model.hyperparameters.values()))
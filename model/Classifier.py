from ttictoc import TicToc
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import csv
import uuid
from pathlib import Path
import os
class Classifier():
    _time = 0
    
    def __init__(self):
        path = Path(__file__)
        try:
            f = open(os.path.join(path.parent.parent,'results','results.csv'),'r',newline='')
            f.close()
        except:           
            row = ['UUID','Dask Used','Classifier','Rows','Columns','Classes','Accuracy','F1 Score','Precision','Recall','time']
            with open(os.path.join(path.parent.parent,'results','results.csv'),'w',newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
    
    class time_watch(object):
        def __call__(self,function):        
            def timer(*args,**kwargs):
                global _time
                t = TicToc()
                t.tic()
                function(self,*args,**kwargs)
                t.toc()
                _time = t.elapsed
            return timer
    

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
    
    #@time_watch()
    def train(self,dataset,model):
        t = TicToc()
        self.clf = model
        t.tic()
        self.clf.fit(dataset.X_train,dataset.y_train)
        t.toc()
        self._time = t.elapsed

    def save_results(self,dataset,classifier_model):    
        name = classifier_model.name
        info = dataset()
        id = uuid.uuid4()
        row = [str(id),str(dataset.use_dask),name,info['rows'],info['columns'],info['Unique classes']]+list(self.results.values())+[self._time]   
        path = Path(__file__)
        with open(os.path.join(path.parent.parent,'results','results.csv'),'a',newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
        
        try:            
            with open(os.path.join(path.parent.parent,'results',f'{name}.csv'),'r',newline='') as file:
                pass
            with open(os.path.join(path.parent.parent,'results',f'{name}.csv'),'a',newline='') as file:
                writer = csv.writer(file)
                row = [id]+list(classifier_model.hyperparameters.values())
                writer.writerow(row)
        except:
            with open(os.path.join(path.parent.parent,'results',f'{name}.csv'),'w',newline='') as file:
                writer = csv.writer(file)   
                row = ['UUID']+list(classifier_model.hyperparameters.keys())             
                writer.writerow(row)
                writer.writerow([id]+list(classifier_model.hyperparameters.values()))
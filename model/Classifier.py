from ttictoc import tic,toc
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import csv
import uuid
from pathlib import Path
import os
import sys

sys.path.append(r'V:\Python\AutomatedLearning')
class Classifier():
    _time = 0
    
    def __init__(self):
        try:
            f = open('results/results.csv','r',newline='')
            f.close()
            
        except:           
            row = ['fID','eID','Dask Used',"Preprocessing Technique",'Classifier','Rows','Columns','Classes','Accuracy','F1 Score','Precision','Recall','time']
            with open('results/results.csv','w+',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            
        

    class time_watch(object):
        def __call__(self,function):        
            def timer(*args,**kwargs):
                global _time
                tic()
                function(self,*args,**kwargs)
                _time = toc()
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
        self.clf = model
        tic()
        self.clf.fit(dataset.X_train,dataset.y_train)
        self._time = toc()

    def save_results(self,dataset,classifier_model):    
        name = classifier_model.name
        info = dataset()
        fid = dataset.meta_id
        eid = uuid.uuid4()
        row = [str(fid),str(eid),str(dataset.use_dask),dataset.preprocessing,name,info['rows'],info['columns'],info['Unique classes']]+list(self.results.values())+[self._time]   
        with open('results/results.csv','a',newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
        if(not dataset.preprocessing == 'None'):
            try:
                with open(f'results/{dataset.preprocessing}.csv','r',newline='') as fi:
                    pass
                with open(f'results/{dataset.preprocessing}.csv','a',newline='') as fi:
                    writer = csv.writer(fi)
                    row = [str(eid)]+list(dataset.preprocess_meta.values())
                    writer.writerow(row)
            except:
                with open(f'results/{dataset.preprocessing}.csv','w',newline='') as file:
                    writer = csv.writer(file)
                    row = ['eID']+list(dataset.preprocess_meta.keys())             
                    writer.writerow(row)
                    writer.writerow([eid]+list(dataset.preprocess_meta.values()))
        
        try:            
            with open(f'results/{name}.csv','r',newline='') as file:
                pass
            with open(f'results/{name}.csv','a',newline='') as file:
                writer = csv.writer(file)
                row = [fid,eid]+list(classifier_model.hyperparameters.values())
                writer.writerow(row)
        
        except:
            with open(f'results/{name}.csv','w',newline='') as file:
                writer = csv.writer(file)   
                row = ['fID','eID']+list(classifier_model.hyperparameters.keys())             
                writer.writerow(row)
                writer.writerow([fid,eid]+list(classifier_model.hyperparameters.values()))
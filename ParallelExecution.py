from csv import DictReader
import sys
sys.path.append(r'V:\Python\AutomatedLearning')
from util.csv_downloader import  download_csv
from dataset_fetch.Openml_fetch import fetch
from model.dataset import Dataset
import concurrent.futures
import multiprocessing
from sklearn.metrics import accuracy_score
from process.classification.KNN_classifier import KNN
from process.classification.MLP_classifier import MLP
from process.classification.GNB_classifier import GNB
from process.classification.QDA_classifier import QDA
from process.classification.RF_classifier import RF
from process.classification.SVM_classifier import SVM
from process.classification.Adaboost_classifier import ABC
import warnings
import itertools
import os
from sklearn import datasets

#warnings.filterwarnings("ignore")
# Incomplete Python file ##############################################
clf_list = {
    KNN:
    {
        'K':3,
        'weights':'uniform',
        'algorithm':'auto'
    },
    MLP:
    {
        'Solver':'lbfgs',
        'Max_Iterations':10000,
        'Tolerance':1e-4,
        'Activation fn': 'tanh',
        'Hidden_layer_neurons':[100]
    },
    SVM:
    {
        'C':1,
        'Kernel':'rbf',
        'Degree':3,
        'Tolerance':1e-4
    },
    GNB:
    {},
    QDA:
    {},
    RF:
    {
        'Trees':100,
        'max_depth':None
    },
    ABC:
    {}
}
combinations = {
    KNN:
    {
        'K':list(range(3,47,2)),
        'weights':['uniform','distance'],
        'algorithm':['auto','ball_tree','brute']
    },
    MLP:
    {
        #'Solver':['lbfgs','adam'],
        #'Max_Iterations':10000,
        'Tolerance':[1e-4,1e-3],
        'Activation fn': ['tanh','relu'],
        'Hidden_layer_neurons':[[100],[200],[500],[1000],[100,20],[200,20],[500,20],[1000,20]]
    },
    SVM:
    {
        'C':list(range(1,50,10)),
        'Kernel':['rbf','linear'],
        'Degree':list(range(3,9)),
        'Tolerance':[1e-4,1e-3]
    },
    RF:
    {
        'Trees':list(range(100,500,100)),
        #'max_depth':None
    }
}

def calculate_combinations(combinations):
    prod = 1
    for val in combinations.values():        
        prod = prod * len(val)   
    return prod

set_inputs = []
set_inputs2 = []
def calculate_hyperparameters(combinations):    
    global set_inputs
    array_values = []
    for val in combinations.values():
        array_values.append(val)
    set_inputs.extend(list(itertools.product(*array_values)))
    
def change_hyperparameters(combinations,number):
    if(set_inputs == []):
        calculate_hyperparameters(combinations)
    required_set = set_inputs[number]
    hyperparameter_set = {}
    i = 0
    for key in combinations.keys():
        hyperparameter_set[key] = required_set[i]
        i = i+1
    return hyperparameter_set

def change_hyperparameters2(combinations,number):
    if(set_inputs2 == []):
        calculate_hyperparameters(combinations)
    required_set = set_inputs2[number]
    hyperparameter_set = {}
    i = 0
    for key in combinations.keys():
        hyperparameter_set[key] = required_set[i]
        i = i+1
    return hyperparameter_set
def data_load():        
    with open('dataset_fetch/meta/Data.csv') as f:
        Reader = DictReader(f)
        for row in Reader:
            
            print(row['Metadata']+'  ')
    
            Meta_data = row
                
            try:
                d = Dataset(url_path = Meta_data['csv_url'],meta_id =Meta_data['Metadata'])
            except ValueError:
                file_path = download_csv(Meta_data['csv_url'])
                d = Dataset(path = file_path,meta_id = Meta_data['Metadata'])            
            if(d()['Contains NANs'] == 'Yes'):
                d._impute_data(imputation_method = 'KNN')                
            return d

def Classifier(iterable):
    #global dataset
    clf = iterable[0]
    dataset = iterable[1].df
    print(clf[0])
    print(clf[1])
    c = clf[0](**clf[1])
    c.train(dataset)
    c.test(dataset)
    return (c,dataset)
def parallelexec():
    #global clf_list
    global dataset
    results = []    
    dataset = data_load()
    dataset.encode()
    dataset.split()
    mgr = multiprocessing.Manager()
    ns = mgr.Namespace()
    ns.df = dataset
    iterable = [(item,ns) for item in clf_list.items()]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(Classifier,iterable)
   
    for (result,dataset) in results:
        result.save(dataset)
    # processes = []
    # for clf in clf_list.items():
    #     p = multiprocessing.Process(target=Classifier,args=(clf,))
    #     p.start()
    #     processes.append(p)
    
    # for process in processes:
    #     process.join()

if(__name__ == '__main__'):
    global dataset
    dataset = []
    
    parallelexec()
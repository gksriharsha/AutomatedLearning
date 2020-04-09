
from csv import DictReader
import sys

sys.path.append(r'V:\Python\AutomatedLearning')
from util.csv_downloader import  download_csv
from dataset_fetch.Openml_fetch import fetch
from model.dataset import Dataset
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
#warnings.filterwarnings("ignore")
def fetch_check():
    pass
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
   
        
def exec():
    
    global set_inputs
    started = False
    progress_data = []
    subprogress_data = []
    partial_progress = 0
    try:
        with open('progress.csv','r') as f:
            progress_data = [line.rstrip() for line in f]
        started = True
    except:
        f = open('progress.csv','w+')
        f.close()
        started = False
    
    try:
        with open('sub_progress.csv','r') as f:
            subprogress_data = [line.rstrip() for line in f]
    except:
        f = open('sub_progress.csv','w+')
        f.close()

    try:
        with open('partial_progress.csv','r') as f:
            partial_progress = len([line.rstrip() for line in f])
    except:
        f = open('partial_progress.csv','w+')
        f.close()

    fetch(2000)
    Meta_data = {}
    i = 0
    with open('dataset_fetch/meta/Data.csv') as f:
        Reader = DictReader(f)
        for row in Reader:
            if((started) and (row['Metadata'] in progress_data)) :
                i = i+1
                continue

            print(row['Metadata']+'  '+ str(i))
            
            if i >=50:
                break
            else:
                i = i + 1  
            Meta_data = row
            def data_load():           
                try:
                    d = Dataset(url_path = Meta_data['csv_url'],meta_id =Meta_data['Metadata'])
                except ValueError:
                    file_path = download_csv(Meta_data['csv_url'])
                    d = Dataset(path = file_path,meta_id = Meta_data['Metadata'])            
                if(d()['Contains NANs'] == 'Yes'):
                    d._impute_data(imputation_method = 'KNN')                
                return d
            #d.split()
            d = data_load()
            if(d.contains_text_check()):
                ret_type = d.encode()
                if(ret_type == False):
                    continue
            NumberOfCols = 0
            with open('dataset_fetch/meta/MetaData.csv') as g:
                Reader2 = DictReader(g)
                for line in Reader2:
                    if(line['fID'] == row['Metadata']):
                        NumberOfCols = line['NumberOfFeatures']
            def classification_block(d):
                global set_inputs
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
                for clf in clf_list.items():
                    set_inputs = []
                    if(str(clf[0]) in subprogress_data):
                        continue
                    if(clf[0] in combinations.keys()):
                        f = open('partial_progress.csv','w+')
                        f.close()
                        for combination in range(calculate_combinations(combinations[clf[0]])):                    
                            #print(change_hyperparameters(combinations[clf[0]],combination))
                            #clf[1] = 
                            if(combination < partial_progress):
                                continue
                            try:
                                classifier = clf[0](**change_hyperparameters(combinations[clf[0]],combination))
                                print('Training ',str(clf[0]), 'classifier - ', str(i),' dataset')
                                classifier.train(d)
                                print('Testing ',str(clf[0]), ' classifier - ', str(i),' dataset')
                                classifier.test(d)
                                print('Saving results')
                                classifier.save(d)
                                f = open('partial_progress.csv','a')
                                f.write(''.join(clf[1]))
                                f.write('\n')
                                f.close()
                            except:
                                pass
                    else:
                        try:
                            classifier = clf[0](**clf[1])
                            print('Training ',str(clf[0]), 'classifier')
                            classifier.train(d)
                            print('Testing ',str(clf[0]), ' classifier')
                            classifier.test(d)
                            print('Saving results')
                            classifier.save(d)
                        except:
                            pass
                    f = open('sub_progress.csv','a')
                    f.write(str(clf[0]))
                    f.write('\n')
                    f.close()
            if(int(NumberOfCols) < 1000):
                d.split()
                classification_block(d)            
            else:
                Parameters_preprocessing = {
                    'RSFS' : {
                        'Dummy feats' : 100,
                        'fn' : 'sqrt',
                        'cutoff' : 0.99,
                        'Threshold' : 1000,
                    },
                    'PCA':{
                        'pca_ratio' : 0.95,
                    }
                }
                States_preprocessing = {
                    'RSFS':{
                        'Dummy feats' : list(range(100,d.data.shape[1],100)),
                        'fn' : ['sqrt','10log'],
                        'Threshold' : list(range(500,2100,200)),
                        'cutoff': [0.95,0.99,0.997]
                    },
                    'PCA':
                    {
                        'pca_ratio':[0.90,0.95,0.99,0.997]
                    }
                }
                global set_inputs2
                for Parameter in Parameters_preprocessing.items():
                    set_inputs2 = []
                    for State in range(calculate_combinations(States_preprocessing[Parameter[0]])):                        
                        d = data_load()
                        d.preprocess_data(Parameter[0],**change_hyperparameters2(States_preprocessing[Parameter[0]],State))
                        classification_block(d)
            
            f = open('progress.csv','a')
            f.write(str(row['Metadata']))
            f.write('\n')
            f.close()
            f = open('sub_progress.csv','w+')
            f.close()
            subprogress_data = []
            partial_progress = 0
            progress_data = []
            #os.system('aws s3 cp -r AutomatedLearning s3://amlprojectbucket/project')
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    exec()
#clf = KNN(K=3)
#clf.train(d)
#clf.test(d)
#clf.save(d)
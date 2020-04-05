
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
#warnings.filterwarnings("ignore")
def fetch_check():
    pass
def calculate_combinations(combinations):
    prod = 1
    for val in combinations.values():        
        prod = prod * len(val)   
    return prod

set_inputs = []
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

    #fetch(100)
    Meta_data = {}
    i = 0
    with open('dataset_fetch/meta/Data.csv') as f:
        Reader = DictReader(f)
        for row in Reader:
            if((started) and (row['Metadata'] in progress_data)) :
                i = i+1
                continue

            print(row['Metadata']+'  '+ str(i))
            
            if i >=5:
                break
            else:
                i = i + 1  
            Meta_data = row           
            try:
                d = Dataset(url_path = Meta_data['csv_url'],meta_id =Meta_data['Metadata'])
            except ValueError:
                file_path = download_csv(Meta_data['csv_url'])
                d = Dataset(path = file_path,meta_id = Meta_data['Metadata'])
            if(d.contains_text_check()):
                d.encode()
            d.split()
            clf_list = {
                KNN:
                {
                    'K':3,
                    'weights':'uniform',
                },
                MLP:
                {
                    'Solver':'lbfgs',
                    'Max_Iterations':10000,
                    'Tolerance':1e-4,
                    'Activation fn': 'tanh',
                    'Hidden layer neurons':[100]
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
                    'K':list(range(3,21,2)),
                    'weights':['uniform','distance']
                },
                MLP:
                {
                    #'Solver':['lbfgs','adam'],
                    #'Max_Iterations':10000,
                    #'Tolerance':[1e-4,1e-3],
                    'Activation fn': ['tanh','relu'],
                    #'Hidden layer neurons':[[100],[200],[500],[1000],[100,20]]
                },
                SVM:
                {
                    'C':list(range(1,50,10)),
                    #'Kernel':['rbf','linear'],
                    #'Degree':list(range(3,9)),
                    #'Tolerance':[1e-4,1e-3]
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
                        classifier = clf[0](**change_hyperparameters(combinations[clf[0]],combination))
                        print('Training ',str(clf[0]), 'classifier')
                        classifier.train(d)
                        print('Testing ',str(clf[0]), ' classifier')
                        classifier.test(d)
                        print('Saving results')
                        classifier.save(d)
                        f = open('partial_progress.csv','a')
                        f.write(''.join(clf[1]))
                        f.write('\n')
                        f.close()
                else:
                    classifier = clf[0](**clf[1])
                    print('Training ',str(clf[0]), 'classifier')
                    classifier.train(d)
                    print('Testing ',str(clf[0]), ' classifier')
                    classifier.test(d)
                    print('Saving results')
                    classifier.save(d)
                f = open('sub_progress.csv','a')
                f.write(str(clf[0]))
                f.write('\n')
                f.close()
            f = open('progress.csv','a')
            f.write(str(row['Metadata']))
            f.write('\n')
            f.close()
            f = open('sub_progress.csv','w+')
            f.close()
            subprogress_data = []
            partial_progress = 0
            progress_data = []
with warnings.catch_warnings():
    #warnings.filterwarnings('ignore')
    exec()
#clf = KNN(K=3)
#clf.train(d)
#clf.test(d)
#clf.save(d)